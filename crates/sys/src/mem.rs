//! Queries about memory in this process.

use crate::{einval, last_error};
use std::ffi::{c_int, c_void};
use std::io;

/// The system page size in bytes.
pub fn page_size() -> usize {
    rustix::param::page_size()
}

/// Which pages of `[addr, addr + len)` are resident in RAM (`mincore`), one
/// entry per page. `addr` must be page-aligned (else EINVAL); `len` is
/// rounded up to whole pages. An error (ENOMEM) if any part of the range
/// isn't mapped in this process.
///
/// This is safe for any address: the kernel only inspects the page tables
/// and writes to the vector this function owns, never to the range.
pub fn mincore(addr: usize, len: usize) -> io::Result<Vec<bool>> {
    let ps = page_size();
    if !addr.is_multiple_of(ps) {
        return Err(einval());
    }
    let pages = len.div_ceil(ps);
    let mut vec = vec![0u8; pages];
    // SAFETY: mincore(addr, length, vec) writes one byte per page of the
    // range into `vec`, which has exactly ceil(len / page size) bytes, and is
    // exclusively borrowed for the call. It never reads or writes the memory
    // in `[addr, addr + len)` itself (it reports the page cache state), so an
    // arbitrary or unmapped address can only produce an error, not unsoundness.
    let r = unsafe { libc::mincore(addr as *mut c_void, len, vec.as_mut_ptr()) };
    if r != 0 {
        return Err(last_error());
    }
    Ok(vec.into_iter().map(|b| b & 1 == 1).collect())
}

/// Where the kernel keeps each page at the given addresses, without moving
/// anything (`move_pages` with no destination nodes). Entry `i` is the NUMA
/// node of the page containing `addrs[i]`, or a negative errno: `-ENOENT`
/// for a page not in memory yet, `-EFAULT` for an unmapped address.
/// Addresses need not be aligned.
///
/// Safe for any addresses: with a null `nodes` array `move_pages` only
/// reports status and never migrates pages or dereferences the addresses in
/// this process.
pub fn move_pages_status(addrs: &[usize]) -> io::Result<Vec<i32>> {
    let mut pages: Vec<*mut c_void> = addrs.iter().map(|a| *a as *mut c_void).collect();
    let mut status: Vec<c_int> = vec![0; addrs.len()];
    // SAFETY: move_pages(pid = 0 (this process), count, pages, nodes, status,
    // flags). `pages` is an array of `count` pointers and `status` an array of
    // `count` ints, both owned locals that stay alive and unaliased for the
    // call; the kernel reads the first and writes the second. `nodes` is null,
    // which makes the call a pure query (no pages are migrated; flags = 0
    // adds no MPOL_MF_MOVE*). The page pointers are only used as numbers.
    let r = unsafe {
        libc::syscall(
            libc::SYS_move_pages,
            0 as libc::c_int,
            addrs.len() as libc::c_ulong,
            pages.as_mut_ptr(),
            std::ptr::null::<c_int>(),
            status.as_mut_ptr(),
            0 as libc::c_int,
        )
    };
    if r != 0 {
        return Err(last_error());
    }
    Ok(status)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AnonMap;

    #[test]
    fn page_size_is_a_power_of_two() {
        let _g = crate::serial();
        let p = page_size();
        assert!(p >= 4096 && p.is_power_of_two());
    }

    #[test]
    fn mincore_sees_touched_pages() {
        let _g = crate::serial();
        let ps = page_size();
        let mut m = AnonMap::new(8 * ps).unwrap();
        let none = mincore(m.addr(), m.len()).unwrap();
        assert_eq!(none.len(), 8);
        assert!(none.iter().all(|r| !r), "{none:?}");
        m.as_mut_slice()[2 * ps] = 1;
        m.as_mut_slice()[5 * ps] = 1;
        let r = mincore(m.addr(), m.len()).unwrap();
        assert!(r[2] && r[5]);
        assert_eq!(r.iter().filter(|x| **x).count(), 2, "{r:?}");
    }

    #[test]
    fn mincore_rejects_bad_ranges() {
        let _g = crate::serial();
        let m = AnonMap::new(page_size()).unwrap();
        assert_eq!(
            mincore(m.addr() + 1, 1).unwrap_err().raw_os_error(),
            Some(libc::EINVAL)
        );
        // Page zero is never mapped.
        assert_eq!(
            mincore(0, page_size()).unwrap_err().raw_os_error(),
            Some(libc::ENOMEM)
        );
        assert_eq!(mincore(m.addr(), 0).unwrap(), Vec::<bool>::new());
    }

    #[test]
    fn move_pages_reports_nodes_of_touched_pages() {
        let _g = crate::serial();
        let ps = page_size();
        let mut m = AnonMap::new(4 * ps).unwrap();
        let addrs: Vec<usize> = (0..4).map(|i| m.addr() + i * ps).collect();
        // Untouched: not in memory.
        let s = move_pages_status(&addrs).unwrap();
        assert!(s.iter().all(|n| *n == -libc::ENOENT), "{s:?}");
        m.as_mut_slice().fill(1);
        let s = move_pages_status(&addrs).unwrap();
        assert!(s.iter().all(|n| *n >= 0), "{s:?}");
        // An unmapped address is reported per page, not as a call failure.
        let s = move_pages_status(&[addrs[0], 0x1000]).unwrap();
        assert!(s[0] >= 0 && s[1] == -libc::EFAULT, "{s:?}");
        assert!(move_pages_status(&[]).unwrap().is_empty());
    }
}
