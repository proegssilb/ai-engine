//! Owned anonymous mappings.

use crate::{einval, last_error};
use rustix::mm::{self, MapFlags, ProtFlags};
use std::ffi::{CString, c_void};
use std::io;
use std::mem::ManuallyDrop;
use std::ptr::NonNull;

/// `madvise` advice that cannot change the contents of the mapping.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Advice {
    /// MADV_HUGEPAGE: transparent huge pages wanted.
    HugePage,
    /// MADV_NOHUGEPAGE: transparent huge pages off, whatever the system setting.
    NoHugePage,
}

/// An anonymous, private, read-write mapping, unmapped on drop.
///
/// The kernel zero-fills it. [`len`](Self::len) is the true mapping length
/// (the request rounded up to the page size), so it is exactly what
/// `munmap` needs. The mapping is owned: nothing else can unmap it, and
/// `as_mut_slice` hands out the memory under Rust's usual exclusivity rules.
///
/// To give the mapping to code that frees it later (a C callback), use
/// [`into_raw`](Self::into_raw) and, when taking it back,
/// [`from_raw`](Self::from_raw).
#[derive(Debug)]
pub struct AnonMap {
    ptr: NonNull<u8>,
    len: usize,
}

// SAFETY: AnonMap owns plain memory with no thread affinity. Moving it to
// another thread moves only the right to access and unmap it.
unsafe impl Send for AnonMap {}
// SAFETY: through `&AnonMap` the bytes are only read (`as_slice`), which is
// fine from several threads at once, as for `&[u8]`; writing needs `&mut self`
// (`as_mut_slice`). The other shared methods act on the mapping through system
// calls that never change its contents, and are safe from any thread.
unsafe impl Sync for AnonMap {}

impl AnonMap {
    /// Map `len` bytes (rounded up to the page size) of normal pages.
    /// `len == 0` is an error (EINVAL), as for `mmap`.
    pub fn new(len: usize) -> io::Result<Self> {
        if len == 0 {
            return Err(einval());
        }
        let len = len
            .checked_next_multiple_of(crate::page_size())
            .ok_or_else(einval)?;
        Self::map(len, MapFlags::PRIVATE)
    }

    /// Map `len` bytes backed by reserved huge pages of `huge_page` bytes
    /// (a power of two, 2 MiB or 1 GiB on x86-64). `len` must be a non-zero
    /// multiple of `huge_page`: `munmap` of a huge-page mapping fails
    /// otherwise, which would leak it. Fails (typically ENOMEM) when the
    /// pages aren't reserved.
    pub fn new_hugetlb(len: usize, huge_page: u64) -> io::Result<Self> {
        // The kernel takes log2(page size) in bits 26..32 of the flags.
        const MAP_HUGE_SHIFT: u32 = 26;
        let ok = huge_page.is_power_of_two()
            && huge_page >= crate::page_size() as u64
            && len != 0
            && (len as u64).is_multiple_of(huge_page);
        if !ok {
            return Err(einval());
        }
        let shift = huge_page.trailing_zeros();
        let flags = MapFlags::PRIVATE
            | MapFlags::HUGETLB
            | MapFlags::from_bits_retain(shift << MAP_HUGE_SHIFT);
        Self::map(len, flags)
    }

    fn map(len: usize, flags: MapFlags) -> io::Result<Self> {
        // SAFETY: a null hint with no MAP_FIXED lets the kernel choose a fresh
        // address, so no existing mapping is replaced. The mapping is
        // anonymous, so no file or fd is involved. `len` is non-zero.
        let p = unsafe {
            mm::mmap_anonymous(
                std::ptr::null_mut(),
                len,
                ProtFlags::READ | ProtFlags::WRITE,
                flags,
            )
        }?;
        // mmap never returns null on success (address 0 is unmappable by default).
        let ptr = NonNull::new(p.cast::<u8>()).ok_or_else(einval)?;
        Ok(AnonMap { ptr, len })
    }

    /// Base address.
    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Base address as a number, for reports and range arithmetic.
    pub fn addr(&self) -> usize {
        self.ptr.as_ptr() as usize
    }

    /// Mapping length in bytes (a multiple of the page size).
    pub fn len(&self) -> usize {
        self.len
    }

    /// Never true: a mapping is at least one page.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The memory, for writing (touching it faults the pages in).
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: `ptr` is the base of a live read-write mapping of exactly
        // `len` bytes that we own (it is unmapped only in Drop or by
        // into_raw, which consume `self`). Every byte is initialised (the
        // kernel zero-fills anonymous pages). `&mut self` makes the slice
        // exclusive for its lifetime; any other pointers to this memory come
        // from `as_ptr` and are the caller's responsibility to not use meanwhile.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// The memory, for reading.
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: as for as_mut_slice; a shared reference to initialised bytes
        // of a live mapping we own, with `&self` preventing a concurrent `&mut`.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Give the mapping a name, shown in `/proc/self/smaps` as `[anon:NAME]`.
    /// Differently named mappings are never merged by the kernel. Needs
    /// Linux 5.17+ with CONFIG_ANON_VMA_NAME (otherwise EINVAL). The name
    /// can't contain NUL and the kernel limits it to 79 bytes.
    pub fn set_name(&self, name: &str) -> io::Result<()> {
        self.set_name_range(0, self.len, name)
    }

    /// Like [`set_name`](Self::set_name) for the part of the mapping at
    /// `offset..offset + len` (bytes; `offset` must be page-aligned). The
    /// kernel splits the mapping at the edges, so the part becomes its own
    /// area. The range must lie inside the mapping, else EINVAL.
    pub fn set_name_range(&self, offset: usize, len: usize, name: &str) -> io::Result<()> {
        const PR_SET_VMA: libc::c_int = 0x5356_4d41;
        const PR_SET_VMA_ANON_NAME: libc::c_ulong = 0;
        let end = offset.checked_add(len).ok_or_else(einval)?;
        if end > self.len {
            return Err(einval());
        }
        let name = CString::new(name).map_err(|_| einval())?;
        // SAFETY: prctl(PR_SET_VMA, PR_SET_VMA_ANON_NAME, addr, len, name):
        // the range `base + offset .. + len` was checked above to lie inside
        // our live mapping (the kernel itself rejects an unaligned start),
        // and `name` is a pointer to a NUL-terminated string that the kernel
        // copies before returning and that outlives the call. The call only
        // labels the area; it never reads or writes its contents.
        let r = unsafe {
            libc::prctl(
                PR_SET_VMA,
                PR_SET_VMA_ANON_NAME,
                (self.ptr.as_ptr() as usize + offset) as libc::c_ulong,
                len as libc::c_ulong,
                name.as_ptr() as libc::c_ulong,
            )
        };
        if r == 0 { Ok(()) } else { Err(last_error()) }
    }

    /// `madvise` with advice that only tunes page size.
    pub fn advise(&self, advice: Advice) -> io::Result<()> {
        let advice = match advice {
            Advice::HugePage => mm::Advice::LinuxHugepage,
            Advice::NoHugePage => mm::Advice::LinuxNoHugepage,
        };
        // SAFETY: the range is exactly our live mapping. MADV_HUGEPAGE and
        // MADV_NOHUGEPAGE change only how the kernel backs the pages, never
        // their contents or validity.
        unsafe { mm::madvise(self.ptr.as_ptr().cast(), self.len, advice) }?;
        Ok(())
    }

    /// Bind the mapping to one NUMA node (`mbind` with MPOL_BIND and
    /// MPOL_MF_STRICT; no existing pages are moved). Pages faulted in later
    /// are allocated on `node`; fails with EIO if pages already resident
    /// violate the policy. `node` must be below [`MAX_NODES`](crate::MAX_NODES).
    pub fn mbind_node(&self, node: u32) -> io::Result<()> {
        const MPOL_BIND: libc::c_long = 2;
        const MPOL_MF_STRICT: libc::c_ulong = 1;
        let mask = crate::sched::node_mask(node)?;
        // SAFETY: mbind(addr, len, mode, nodemask, maxnode, flags): `addr`/`len`
        // are our live mapping; `mask` is a local array of `mask.len()` words
        // that the kernel reads (maxnode = its bit count) and outlives the
        // call. Without MPOL_MF_MOVE* the call only sets policy and
        // never relocates or invalidates memory. The syscall return is a long.
        let r = unsafe {
            libc::syscall(
                libc::SYS_mbind,
                self.ptr.as_ptr() as *mut c_void,
                self.len,
                MPOL_BIND,
                mask.as_ptr(),
                (mask.len() * 64) as libc::c_ulong,
                MPOL_MF_STRICT,
            )
        };
        if r == 0 { Ok(()) } else { Err(last_error()) }
    }

    /// `mlock` the whole mapping (faults every page in now, under the
    /// current memory policy). Fails with ENOMEM or EPERM past RLIMIT_MEMLOCK.
    pub fn mlock(&self) -> io::Result<()> {
        // SAFETY: the range is exactly our live mapping; mlock only pins pages
        // in RAM and doesn't change their contents.
        unsafe { mm::mlock(self.ptr.as_ptr().cast(), self.len) }?;
        Ok(())
    }

    /// Give up ownership without unmapping: returns (base, length). The
    /// mapping stays alive until someone passes the pair to
    /// [`from_raw`](Self::from_raw) (or calls `munmap` on it themselves).
    pub fn into_raw(self) -> (*mut u8, usize) {
        let this = ManuallyDrop::new(self);
        (this.ptr.as_ptr(), this.len)
    }

    /// Take back ownership of a mapping released by [`into_raw`](Self::into_raw).
    ///
    /// # Safety
    ///
    /// `ptr` and `len` must be exactly the pair a previous `into_raw`
    /// returned, `from_raw` must not have been called on it before, and no
    /// other code may use or unmap the memory afterwards (the returned
    /// `AnonMap` unmaps it on drop).
    pub unsafe fn from_raw(ptr: *mut u8, len: usize) -> AnonMap {
        AnonMap {
            // SAFETY: by the caller's contract `ptr` came from a live
            // AnonMap, whose pointer is non-null.
            ptr: unsafe { NonNull::new_unchecked(ptr) },
            len,
        }
    }
}

impl Drop for AnonMap {
    fn drop(&mut self) {
        // SAFETY: `ptr`/`len` describe a mapping we created and still own
        // (ownership ends only here or in into_raw, which skips Drop). `len`
        // is page-aligned (rounded in `new`, a multiple of the huge page in
        // `new_hugetlb`), as munmap needs. The only Rust references into it
        // are borrows of `self`, which have ended.
        let r = unsafe { mm::munmap(self.ptr.as_ptr().cast(), self.len) };
        debug_assert!(r.is_ok(), "munmap of an owned mapping failed: {r:?}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn smaps_has(addr: usize, needle: &str) -> bool {
        let text = std::fs::read_to_string("/proc/self/smaps").unwrap();
        text.lines()
            .filter(|l| {
                l.split_once('-')
                    .and_then(|(s, _)| usize::from_str_radix(s, 16).ok())
                    == Some(addr)
            })
            .any(|l| l.contains(needle))
    }

    #[test]
    fn maps_rounds_up_and_zero_fills() {
        let _g = crate::serial();
        let mut m = AnonMap::new(1).unwrap();
        assert_eq!(m.len(), crate::page_size());
        assert_eq!(m.addr() % crate::page_size(), 0);
        assert!(m.as_slice().iter().all(|b| *b == 0));
        m.as_mut_slice()[5] = 7;
        assert_eq!(m.as_slice()[5], 7);
        assert!(AnonMap::new(0).is_err());
    }

    #[test]
    fn drop_unmaps() {
        let _g = crate::serial();
        // Large, so that a thread stack mapped meanwhile by another test
        // can't cover the whole range again.
        let m = AnonMap::new(64 << 20).unwrap();
        let (a, l) = (m.addr(), m.len());
        drop(m);
        // mincore on an unmapped range reports ENOMEM.
        let e = crate::mincore(a, l).unwrap_err();
        assert_eq!(e.raw_os_error(), Some(libc::ENOMEM));
    }

    #[test]
    fn raw_round_trip_keeps_it_mapped() {
        let _g = crate::serial();
        let mut m = AnonMap::new(64 << 20).unwrap();
        m.as_mut_slice()[0] = 9;
        let (p, l) = m.into_raw();
        assert!(crate::mincore(p as usize, l).is_ok());
        // SAFETY: (p, l) came from into_raw just above and is used once.
        let m = unsafe { AnonMap::from_raw(p, l) };
        assert_eq!(m.as_slice()[0], 9);
        let (a, l) = (m.addr(), m.len());
        drop(m);
        assert!(crate::mincore(a, l).is_err());
    }

    #[test]
    fn names_the_mapping() {
        let _g = crate::serial();
        let m = AnonMap::new(1 << 20).unwrap();
        match m.set_name("aie-sys-test") {
            Ok(()) => assert!(smaps_has(m.addr(), "[anon:aie-sys-test]")),
            // Older kernel or no CONFIG_ANON_VMA_NAME.
            Err(e) => assert_eq!(e.raw_os_error(), Some(libc::EINVAL)),
        }
        assert!(m.set_name("nul\0inside").is_err());
        // A named sub-range becomes its own area; one outside the mapping is refused.
        let ps = crate::page_size();
        let big = AnonMap::new(8 * ps).unwrap();
        if big.set_name_range(2 * ps, 2 * ps, "aie-sys-part").is_ok() {
            assert!(smaps_has(big.addr() + 2 * ps, "[anon:aie-sys-part]"));
        }
        assert!(big.set_name_range(7 * ps, 2 * ps, "x").is_err());
        assert!(big.set_name_range(usize::MAX, 2, "x").is_err());
    }

    #[test]
    fn advice_and_mbind_and_mlock() {
        let _g = crate::serial();
        let mut m = AnonMap::new(1 << 20).unwrap();
        m.advise(Advice::HugePage).unwrap();
        m.advise(Advice::NoHugePage).unwrap();
        // Node 0 exists everywhere NUMA is compiled in; without it (or in a
        // sandbox that blocks mbind) the error is returned, not a panic.
        let bound = m.mbind_node(0);
        if let Err(e) = &bound {
            assert!(
                matches!(
                    e.raw_os_error(),
                    Some(libc::ENOSYS | libc::EPERM | libc::EINVAL)
                ),
                "{e}"
            );
        }
        // Past the mask: rejected before the call.
        assert!(m.mbind_node(crate::MAX_NODES as u32).is_err());
        // mlock of 1 MiB fits the usual 8 MiB default; if the limit is lower
        // it fails with an error.
        match m.mlock() {
            Ok(()) => {
                m.as_mut_slice()[0] = 1;
                let s = std::fs::read_to_string("/proc/self/status").unwrap();
                assert!(s.contains("VmLck"));
            }
            Err(e) => assert!(
                matches!(e.raw_os_error(), Some(libc::ENOMEM | libc::EPERM)),
                "{e}"
            ),
        }
    }

    #[test]
    fn hugetlb_arguments_are_checked_and_failure_is_an_error() {
        let _g = crate::serial();
        let page = crate::page_size();
        assert!(AnonMap::new_hugetlb(0, 2 << 20).is_err());
        assert!(AnonMap::new_hugetlb(page, 3 << 20).is_err()); // not a power of two
        assert!(AnonMap::new_hugetlb(1 << 20, 2 << 20).is_err()); // not a multiple
        // With no reserved pages this fails (ENOMEM); with some it succeeds.
        match AnonMap::new_hugetlb(2 << 20, 2 << 20) {
            Ok(m) => assert_eq!(m.len(), 2 << 20),
            Err(e) => assert!(
                matches!(
                    e.raw_os_error(),
                    Some(libc::ENOMEM | libc::EPERM | libc::EINVAL)
                ),
                "{e}"
            ),
        }
    }
}
