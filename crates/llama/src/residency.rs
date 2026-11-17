//! How much of a file-backed range is in RAM right now (mincore), for the
//! disk tier: weights left in the mapped model file may be dropped from RAM
//! and re-read at any time. Ported from the placement spike.

use std::path::Path;

/// Mappings of `path` in this process: (address start, end, file offset).
pub fn mappings(path: &Path) -> Vec<(usize, usize, u64)> {
    let Some(canon) = std::fs::canonicalize(path)
        .ok()
        .and_then(|c| c.to_str().map(String::from))
    else {
        return vec![];
    };
    std::fs::read_to_string("/proc/self/maps")
        .unwrap_or_default()
        .lines()
        .filter(|l| l.ends_with(canon.as_str()))
        .filter_map(|l| {
            let mut f = l.split_whitespace();
            let (s, e) = f.next()?.split_once('-')?;
            let _perms = f.next()?;
            let off = u64::from_str_radix(f.next()?, 16).ok()?;
            Some((
                usize::from_str_radix(s, 16).ok()?,
                usize::from_str_radix(e, 16).ok()?,
                off,
            ))
        })
        .collect()
}

/// Resident bytes of an address range.
fn resident(addr: usize, len: usize) -> u64 {
    if len == 0 {
        return 0;
    }
    let ps = aie_sys::page_size();
    let start = addr / ps * ps;
    let pages = (addr + len).div_ceil(ps) * ps / ps - start / ps;
    let Ok(vec) = aie_sys::mincore(start, pages * ps) else {
        return 0;
    };
    let n = vec.iter().filter(|r| **r).count() as u64;
    (n * ps as u64).min(len as u64)
}

/// Resident bytes of the given file ranges (file offset, length) of `path`,
/// through this process's mappings of it. Ranges not mapped count as 0.
pub fn resident_file_bytes(path: &Path, ranges: &[(u64, u64)]) -> u64 {
    let maps = mappings(path);
    ranges
        .iter()
        .map(|&(off, len)| {
            maps.iter()
                .find(|(s, e, foff)| off >= *foff && off + len <= foff + (e - s) as u64)
                .map(|(s, _, foff)| resident(s + (off - foff) as usize, len as usize))
                .unwrap_or(0)
        })
        .sum()
}

/// What the kernel holds for an address range, from /proc/self/smaps (bytes).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Smaps {
    /// In RAM (normal and reserved huge pages).
    pub resident: u64,
    /// Swapped out.
    pub swapped: u64,
    /// Locked (mlock).
    pub locked: u64,
    /// On transparent huge pages.
    pub thp: u64,
    /// On reserved huge pages (hugetlb).
    pub hugetlb: u64,
}

/// Give an anonymous mapping its own name (shown in smaps as
/// `[anon:NAME]`). Differently named mappings are never merged by the
/// kernel, so `smaps()` sees exactly this range. Needs Linux 5.17+ with
/// CONFIG_ANON_VMA_NAME; returns false where it isn't available.
pub fn keep_separate(map: &aie_sys::AnonMap, name: &str) -> bool {
    map.set_name(name).is_ok()
}

/// Sum smaps fields over the mappings that lie inside [base, base + len).
pub fn smaps(base: usize, len: usize) -> Smaps {
    let text = std::fs::read_to_string("/proc/self/smaps").unwrap_or_default();
    let mut cap: u64 = 0;
    let mut out = Smaps::default();
    for line in text.lines() {
        if let Some((range, _)) = line.split_once(' ')
            && let Some((s, e)) = range.split_once('-')
            && let (Ok(s), Ok(e)) = (usize::from_str_radix(s, 16), usize::from_str_radix(e, 16))
        {
            // Bytes of this mapping that fall in the range. A mapping the
            // kernel merged with a neighbour only partly overlaps; its
            // counts are capped at the overlap (an upper bound) rather than
            // dropped, which would read as "nothing resident, nothing
            // swapped". keep_separate() prevents the merge in the first place.
            cap = e.min(base + len).saturating_sub(s.max(base)) as u64;
            continue;
        }
        if cap == 0 {
            continue;
        }
        let mut f = line.split_whitespace();
        let (Some(k), Some(v)) = (f.next(), f.next()) else {
            continue;
        };
        let Ok(kib) = v.parse::<u64>() else { continue };
        let b = (kib * 1024).min(cap);
        match k {
            "Rss:" | "Private_Hugetlb:" | "Shared_Hugetlb:" => out.resident += b,
            "Swap:" => out.swapped += b,
            "Locked:" => out.locked += b,
            "AnonHugePages:" => out.thp += b,
            _ => {}
        }
        if matches!(k, "Private_Hugetlb:" | "Shared_Hugetlb:") {
            out.hugetlb += b;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smaps_sees_a_fresh_mapping() {
        let len = 8 << 20;
        let mut m = aie_sys::AnonMap::new(len).unwrap();
        // Without a name the kernel may merge this mapping with a neighbour
        // (another test thread's), and smaps would see a bigger area.
        let named = keep_separate(&m, "aie-test");
        m.as_mut_slice().fill(1);
        let s = smaps(m.addr(), len);
        if named {
            assert_eq!(s.resident, len as u64);
        } else {
            assert!(s.resident <= len as u64);
        }
        assert_eq!(s.swapped, 0);
    }

    #[test]
    fn smaps_handles_a_merged_mapping() {
        // One 16 MiB mapping stands in for our 8 MiB buffer merged with an
        // 8 MiB neighbour: no mapping lies wholly inside the buffer.
        let len = 8 << 20;
        let mut m = aie_sys::AnonMap::new(2 * len).unwrap();
        m.as_mut_slice().fill(1);
        // Merged: capped at the overlap, never silently zero.
        let merged = smaps(m.addr(), len);
        assert!(merged.resident > 0 && merged.resident <= len as u64);
        // Named: split off by the kernel, so exact.
        if m.set_name_range(0, len, "aie-test-merged").is_ok() {
            assert_eq!(smaps(m.addr(), len).resident, len as u64);
        }
    }
}
