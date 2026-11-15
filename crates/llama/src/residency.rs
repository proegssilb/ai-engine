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
