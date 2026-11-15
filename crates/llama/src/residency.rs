//! How much of a file-backed range is in RAM right now (mincore), for the
//! disk tier: weights left in the mapped model file may be dropped from RAM
//! and re-read at any time. Ported from the placement spike.

use std::ffi::c_void;
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

fn page_size() -> usize {
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
}

/// Resident bytes of an address range.
fn resident(addr: usize, len: usize) -> u64 {
    if len == 0 {
        return 0;
    }
    let ps = page_size();
    let start = addr / ps * ps;
    let pages = (addr + len).div_ceil(ps) * ps / ps - start / ps;
    let mut vec = vec![0u8; pages];
    if unsafe { libc::mincore(start as *mut c_void, pages * ps, vec.as_mut_ptr()) } != 0 {
        return 0;
    }
    let n = vec.iter().filter(|b| **b & 1 == 1).count() as u64;
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

/// Sum smaps fields over the mappings that lie inside [base, base + len).
pub fn smaps(base: usize, len: usize) -> Smaps {
    let text = std::fs::read_to_string("/proc/self/smaps").unwrap_or_default();
    let mut inside = false;
    let mut out = Smaps::default();
    for line in text.lines() {
        if let Some((range, _)) = line.split_once(' ')
            && let Some((s, e)) = range.split_once('-')
            && let (Ok(s), Ok(e)) = (usize::from_str_radix(s, 16), usize::from_str_radix(e, 16))
        {
            inside = s >= base && e <= base + len;
            continue;
        }
        if !inside {
            continue;
        }
        let mut f = line.split_whitespace();
        let (Some(k), Some(v)) = (f.next(), f.next()) else {
            continue;
        };
        let Ok(kib) = v.parse::<u64>() else { continue };
        let b = kib * 1024;
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
        let p = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert_ne!(p, libc::MAP_FAILED);
        unsafe { std::ptr::write_bytes(p as *mut u8, 1, len) };
        let s = smaps(p as usize, len);
        assert_eq!(s.resident, len as u64);
        assert_eq!(s.swapped, 0);
        unsafe { libc::munmap(p, len) };
    }
}
