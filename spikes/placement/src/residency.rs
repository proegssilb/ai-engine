//! Ground-truth checks from the kernel's side: where are a tensor's bytes,
//! and are they actually in RAM right now?

use crate::sys;
use std::ffi::{CStr, CString, c_void};
use std::path::Path;

pub struct FileTensor {
    pub name: String,
    pub file_offset: usize,
    pub size: usize,
}

/// Tensor byte ranges inside the GGUF file, via ggml's own gguf reader.
pub fn gguf_layout(path: &Path) -> Vec<FileTensor> {
    let c = CString::new(path.to_str().unwrap()).unwrap();
    unsafe {
        let ctx = sys::gguf_init_from_file(c.as_ptr(), sys::gguf_init_params { no_alloc: true, ctx: std::ptr::null_mut() });
        assert!(!ctx.is_null(), "gguf_init_from_file failed");
        let data = sys::gguf_get_data_offset(ctx);
        let out = (0..sys::gguf_get_n_tensors(ctx))
            .map(|i| FileTensor {
                name: CStr::from_ptr(sys::gguf_get_tensor_name(ctx, i)).to_string_lossy().into_owned(),
                file_offset: data + sys::gguf_get_tensor_offset(ctx, i),
                size: sys::gguf_get_tensor_size(ctx, i),
            })
            .collect();
        sys::gguf_free(ctx);
        out
    }
}

/// Where llama.cpp mmap'd `path` in our address space: (addr of file offset 0, mapping start, mapping end).
pub fn find_file_mapping(path: &Path) -> Option<(usize, usize, usize)> {
    let canon = std::fs::canonicalize(path).ok()?;
    let canon = canon.to_str()?;
    let maps = std::fs::read_to_string("/proc/self/maps").ok()?;
    let mut found: Option<(usize, usize, usize)> = None;
    for line in maps.lines() {
        if !line.ends_with(canon) {
            continue;
        }
        let mut f = line.split_whitespace();
        let (range, _perms, off) = (f.next()?, f.next()?, f.next()?);
        let (s, e) = range.split_once('-')?;
        let (s, e) = (usize::from_str_radix(s, 16).ok()?, usize::from_str_radix(e, 16).ok()?);
        let off = usize::from_str_radix(off, 16).ok()?;
        let origin = s - off;
        found = Some(match found {
            None => (origin, s, e),
            Some((o, s0, e0)) => (o, s0.min(s), e0.max(e)),
        });
    }
    found
}

fn page_range(addr: usize, len: usize) -> (usize, usize) {
    let ps = unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize };
    let start = addr / ps * ps;
    let end = (addr + len).div_ceil(ps) * ps;
    (start, (end - start) / ps)
}

/// Fraction of the range's pages that are resident in RAM (mincore).
pub fn resident_fraction(addr: usize, len: usize) -> f64 {
    if len == 0 {
        return 1.0;
    }
    let (start, pages) = page_range(addr, len);
    let mut vec = vec![0u8; pages];
    let ps = unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize };
    let r = unsafe { libc::mincore(start as *mut c_void, pages * ps, vec.as_mut_ptr()) };
    if r != 0 {
        return f64::NAN;
    }
    vec.iter().filter(|b| **b & 1 == 1).count() as f64 / pages as f64
}

/// NUMA node of a few sample pages (first / middle / last), via move_pages(nodes=NULL).
/// Only meaningful for resident pages; non-resident report -ENOENT.
pub fn numa_nodes(addr: usize, len: usize) -> Vec<i32> {
    if len == 0 {
        return vec![];
    }
    let (start, pages) = page_range(addr, len);
    let ps = unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize };
    let mut ptrs: Vec<*mut c_void> = [0, pages / 2, pages - 1].iter().map(|p| (start + p * ps) as *mut c_void).collect();
    ptrs.dedup();
    let mut status = vec![0i32; ptrs.len()];
    let r = unsafe {
        libc::syscall(libc::SYS_move_pages, 0, ptrs.len() as libc::c_ulong, ptrs.as_mut_ptr(), std::ptr::null::<i32>(), status.as_mut_ptr(), 0)
    };
    if r != 0 {
        return vec![];
    }
    status
}

/// Ask the kernel to reclaim these pages now (MADV_PAGEOUT). Clean file-backed
/// pages are simply dropped; the next access reads them from disk again.
/// mlock'd pages are skipped by the kernel.
pub fn page_out(addr: usize, len: usize) -> i32 {
    let (start, pages) = page_range(addr, len);
    let ps = unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize };
    unsafe { libc::madvise(start as *mut c_void, pages * ps, libc::MADV_PAGEOUT) }
}

pub fn proc_status(field: &str) -> String {
    std::fs::read_to_string("/proc/self/status")
        .unwrap_or_default()
        .lines()
        .find(|l| l.starts_with(field))
        .map(|l| l[field.len() + 1..].trim().to_string())
        .unwrap_or_default()
}
