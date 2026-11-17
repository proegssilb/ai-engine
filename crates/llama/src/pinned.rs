//! A ggml buffer type implemented in Rust.
//!
//! llama.cpp copies any tensor whose override points at a non-default buffer
//! type into memory that the buffer type allocates (see
//! `llama_model::load_tensors`: only the device's *default* buffer type is
//! backed by the mmap'd file). So a custom host buffer type is how we decide,
//! per tensor, "this lives in RAM we own" vs "this stays a view of the file".
//!
//! Our allocations are anonymous mmaps that we can mbind() to a NUMA node,
//! advise toward hugepages, and mlock() so the kernel can't evict them.
//!
//! Ported from spikes/placement/src/pinned.rs.

use crate::sys;
use std::ffi::{CStr, CString, c_void};
use std::ptr;
use std::sync::{Mutex, PoisonError};

#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PinOptions {
    /// mbind the allocation to this NUMA node (MPOL_BIND).
    pub numa_node: Option<u32>,
    /// mlock the allocation so it can never be paged out.
    pub mlock: bool,
    /// Transparent huge pages: true = MADV_HUGEPAGE, false = MADV_NOHUGEPAGE
    /// (explicitly off, even when the system's THP setting is "always").
    pub hugepages: bool,
    /// Back the buffer with reserved huge pages (hugetlb) of this size in bytes,
    /// 2 MiB or 1 GiB. Needs pages reserved (vm.nr_hugepages or
    /// /sys/kernel/mm/hugepages); falls back to normal pages if there aren't
    /// enough, and records why.
    #[serde(default)]
    pub hugetlb: Option<u64>,
    /// cudaHostRegister the allocation (page-locked for DMA), so GPU backends
    /// can copy from it at full PCIe speed when they offload ops on these weights.
    pub cuda_register: bool,
}

#[derive(Clone, Debug)]
pub struct BufferRecord {
    pub base: usize,
    pub len: usize,
    /// (name, data address, bytes) for every tensor placed in this buffer.
    pub tensors: Vec<(String, usize, usize)>,
    pub mbind_result: Option<Result<(), i32>>,
    pub mlock_result: Option<Result<(), i32>>,
    /// Ok, or the cudaError_t code / -1 if libcudart couldn't be loaded.
    pub cuda_register_result: Option<Result<(), i32>>,
    /// Ok(page size) if backed by reserved huge pages, Err(errno) if that
    /// was asked for and failed (normal pages were used instead).
    pub hugetlb_result: Option<Result<u64, i32>>,
    /// Bytes actually mapped (rounded up to the page size used).
    pub map_len: usize,
    pub freed: bool,
}

/// Every buffer this process allocated, with the tensors placed in it, so
/// placement can be verified and reported.
pub static REGISTRY: Mutex<Vec<BufferRecord>> = Mutex::new(Vec::new());

struct BuftCtx {
    name: CString,
    opts: PinOptions,
}

struct BufCtx {
    base: *mut u8,
    map_len: usize,
    idx: usize,
}

const ALIGNMENT: usize = 64;

fn errno(e: &std::io::Error) -> i32 {
    e.raw_os_error().unwrap_or(-1)
}

/// Create a buffer type usable in `llama_model_tensor_buft_override`.
/// It's leaked on purpose: it must outlive every model that references it.
pub(crate) fn new_buffer_type(name: &str, opts: PinOptions) -> sys::ggml_backend_buffer_type_t {
    let ctx = Box::new(BuftCtx {
        name: CString::new(name).unwrap(),
        opts,
    });
    // SAFETY: Looks up a registered device; returns null if none (checked below).
    let cpu_dev = unsafe {
        sys::ggml_backend_dev_by_type(sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_CPU)
    };
    assert!(!cpu_dev.is_null(), "no CPU device registered");
    let buft = Box::new(sys::ggml_backend_buffer_type {
        iface: sys::ggml_backend_buffer_type_i {
            get_name: Some(buft_get_name),
            alloc_buffer: Some(buft_alloc_buffer),
            alloc_buffer_n: None,
            get_alignment: Some(buft_get_alignment),
            get_max_size: None,
            get_alloc_size: None,
            get_alloc_size_n: None,
            // "host" = plain memory with standard ggml layout, so the CPU
            // backend computes on it directly via tensor->data.
            is_host: Some(buft_is_host),
        },
        device: cpu_dev,
        context: Box::into_raw(ctx) as *mut c_void,
    });
    Box::into_raw(buft)
}

unsafe fn buft_ctx<'a>(buft: sys::ggml_backend_buffer_type_t) -> &'a BuftCtx {
    // SAFETY: Only called with buffer types created by `new_buffer_type`, whose `context` is a
    // leaked Box<BuftCtx> that is never freed, so the reference is valid for any lifetime.
    unsafe { &*((*buft).context as *const BuftCtx) }
}

unsafe extern "C" fn buft_get_name(buft: sys::ggml_backend_buffer_type_t) -> *const libc::c_char {
    // SAFETY: ggml calls this with our buffer type (the iface is only installed on types from
    // `new_buffer_type`); the name is a CString inside the leaked BuftCtx, so the pointer stays
    // valid.
    unsafe { buft_ctx(buft).name.as_ptr() }
}

unsafe extern "C" fn buft_get_alignment(_: sys::ggml_backend_buffer_type_t) -> usize {
    ALIGNMENT
}

unsafe extern "C" fn buft_is_host(_: sys::ggml_backend_buffer_type_t) -> bool {
    true
}

unsafe extern "C" fn buft_alloc_buffer(
    buft: sys::ggml_backend_buffer_type_t,
    size: usize,
) -> sys::ggml_backend_buffer_t {
    // SAFETY: As above: `buft` is one of ours.
    let opts = unsafe { buft_ctx(buft).opts.clone() };
    // Reserved huge pages first, if asked; else (or on failure) normal pages.
    // size can be 0 (llama.cpp's no_alloc dummy buffers); still map one page.
    let mut hugetlb_result = None;
    let mut mapped = None;
    if let Some(hp) = opts.hugetlb.filter(|_| size > 0) {
        let len = size.div_ceil(hp as usize) * hp as usize;
        match aie_sys::AnonMap::new_hugetlb(len, hp) {
            Ok(m) => {
                hugetlb_result = Some(Ok(hp));
                mapped = Some(m);
            }
            Err(e) => hugetlb_result = Some(Err(errno(&e))),
        }
    }
    let map = match mapped {
        Some(m) => m,
        // AnonMap rounds the length up to whole pages.
        None => match aie_sys::AnonMap::new(size.max(1)) {
            Ok(m) => m,
            Err(_) => return ptr::null_mut(),
        },
    };
    let (base, map_len) = (map.as_ptr(), map.len());

    // Its own name, so the kernel never merges it with a neighbouring
    // mapping (another RAM-tier buffer, say) and the RAM tier's residency
    // and swap figures (from smaps) cover exactly this buffer.
    {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static N: AtomicUsize = AtomicUsize::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        crate::residency::keep_separate(&map, &format!("aie-ram-tier-{n}"));
    }

    let mut rec = BufferRecord {
        base: map.addr(),
        len: size,
        tensors: vec![],
        mbind_result: None,
        mlock_result: None,
        cuda_register_result: None,
        hugetlb_result,
        map_len,
        freed: false,
    };

    if !matches!(rec.hugetlb_result, Some(Ok(_))) {
        let advice = if opts.hugepages {
            aie_sys::Advice::HugePage
        } else {
            aie_sys::Advice::NoHugePage
        };
        let _ = map.advise(advice);
    }
    if let Some(node) = opts.numa_node {
        // Must happen before first touch: the policy applies to pages as they fault in.
        rec.mbind_result = Some(map.mbind_node(node).map_err(|e| errno(&e)));
    }
    if opts.mlock {
        // mlock also faults every page in now, under the NUMA policy above.
        rec.mlock_result = Some(map.mlock().map_err(|e| errno(&e)));
    }
    if opts.cuda_register {
        rec.cuda_register_result = Some(match cudart() {
            // cudaHostRegisterPortable: page-locked for every CUDA context, so
            // all GPUs (not only the current one) copy from it at full speed.
            // SAFETY: `base` is the start of a live mapping of `map_len` bytes that `map` still
            // owns; cudaHostRegister only pins it. The function pointer was resolved by dlsym
            // with the right signature (see `cudart`).
            Some(c) => match unsafe { (c.register)(base as *mut c_void, map_len, 1) } {
                0 => Ok(()),
                e => Err(e),
            },
            None => Err(-1),
        });
    }

    let idx = {
        let mut reg = REGISTRY.lock().unwrap_or_else(PoisonError::into_inner);
        reg.push(rec);
        reg.len() - 1
    };
    // From here llama.cpp owns the mapping; buf_free takes it back.
    let (base, map_len) = map.into_raw();
    let ctx = Box::new(BufCtx { base, map_len, idx });
    // SAFETY: `buft` is ours; `ctx` is a fresh Box whose ownership passes to the buffer and
    // comes back exactly once in `buf_free`; `size` <= `map_len`, so every offset ggml derives
    // from the base stays inside the mapping.
    unsafe {
        sys::ggml_backend_buffer_init(buft, BUFFER_IFACE, Box::into_raw(ctx) as *mut c_void, size)
    }
}

const BUFFER_IFACE: sys::ggml_backend_buffer_i = sys::ggml_backend_buffer_i {
    free_buffer: Some(buf_free),
    get_base: Some(buf_get_base),
    init_tensor: Some(buf_init_tensor),
    memset_tensor: Some(buf_memset_tensor),
    set_tensor: Some(buf_set_tensor),
    get_tensor: Some(buf_get_tensor),
    set_tensor_2d: None,
    get_tensor_2d: None,
    cpy_tensor: Some(buf_cpy_tensor),
    clear: Some(buf_clear),
    reset: None,
    set_tensor_n: None,
    get_tensor_n: None,
    cpy_tensor_n: None,
};

unsafe fn buf_ctx<'a>(buf: sys::ggml_backend_buffer_t) -> &'a BufCtx {
    // SAFETY: Only called with buffers created by `buft_alloc_buffer`, whose `context` is a
    // Box<BufCtx> that stays alive until `buf_free`, which ggml calls last.
    unsafe { &*((*buf).context as *const BufCtx) }
}

unsafe extern "C" fn buf_free(buf: sys::ggml_backend_buffer_t) {
    // SAFETY: ggml calls free_buffer exactly once per buffer, last. `context` is the
    // Box<BufCtx> made in `buft_alloc_buffer`; (base, map_len) is the pair AnonMap::into_raw
    // gave up there, and this is the only place it is taken back. CUDA is unregistered before
    // the memory is unmapped.
    unsafe {
        let ctx = Box::from_raw((*buf).context as *mut BufCtx);
        let registered = matches!(
            REGISTRY.lock().unwrap_or_else(PoisonError::into_inner)[ctx.idx].cuda_register_result,
            Some(Ok(()))
        );
        if registered && let Some(c) = cudart() {
            (c.unregister)(ctx.base as *mut c_void);
        }
        // Ownership returns from llama.cpp: (base, map_len) is the pair
        // into_raw gave up in buft_alloc_buffer, and this is the only place
        // it is taken back (buf_free runs once per buffer).
        drop(aie_sys::AnonMap::from_raw(ctx.base, ctx.map_len));
        REGISTRY.lock().unwrap_or_else(PoisonError::into_inner)[ctx.idx].freed = true;
    }
}

unsafe extern "C" fn buf_get_base(buf: sys::ggml_backend_buffer_t) -> *mut c_void {
    // SAFETY: `buf` is one of ours (see `buf_ctx`).
    unsafe { buf_ctx(buf).base as *mut c_void }
}

unsafe extern "C" fn buf_init_tensor(
    buf: sys::ggml_backend_buffer_t,
    t: *mut sys::ggml_tensor,
) -> sys::ggml_status {
    // SAFETY: `buf` is one of ours and `t` a tensor ggml is placing in it; its name is a NUL-
    // terminated array inside the tensor; ggml_nbytes only reads the tensor's shape.
    unsafe {
        // Views share their parent's storage; only record real allocations.
        if (*t).view_src.is_null() {
            let name = CStr::from_ptr((*t).name.as_ptr())
                .to_string_lossy()
                .into_owned();
            let entry = (name, (*t).data as usize, sys::ggml_nbytes(t));
            REGISTRY.lock().unwrap_or_else(PoisonError::into_inner)[buf_ctx(buf).idx]
                .tensors
                .push(entry);
        }
    }
    sys::ggml_status_GGML_STATUS_SUCCESS
}

unsafe extern "C" fn buf_memset_tensor(
    _: sys::ggml_backend_buffer_t,
    t: *mut sys::ggml_tensor,
    value: u8,
    offset: usize,
    size: usize,
) {
    // SAFETY: ggml's memset_tensor contract: `t` is a tensor in this buffer and offset + size
    // <= ggml_nbytes(t) (asserted by ggml_backend_tensor_memset), so the range lies inside our
    // mapping.
    unsafe { ptr::write_bytes(((*t).data as *mut u8).add(offset), value, size) }
}

unsafe extern "C" fn buf_set_tensor(
    _: sys::ggml_backend_buffer_t,
    t: *mut sys::ggml_tensor,
    data: *const c_void,
    offset: usize,
    size: usize,
) {
    // SAFETY: ggml's set_tensor contract: `t` is a tensor in this buffer, offset + size <=
    // ggml_nbytes(t) (asserted by ggml_backend_tensor_set), and `data` is readable for `size`
    // bytes; the source is caller memory, which can't overlap our mapping.
    unsafe { ptr::copy_nonoverlapping(data as *const u8, ((*t).data as *mut u8).add(offset), size) }
}

unsafe extern "C" fn buf_get_tensor(
    _: sys::ggml_backend_buffer_t,
    t: *const sys::ggml_tensor,
    data: *mut c_void,
    offset: usize,
    size: usize,
) {
    // SAFETY: ggml's get_tensor contract: `t` is a tensor in this buffer, offset + size <=
    // ggml_nbytes(t) (asserted by ggml_backend_tensor_get), and `data` is writable for `size`
    // bytes and doesn't overlap our mapping.
    unsafe { ptr::copy_nonoverlapping(((*t).data as *const u8).add(offset), data as *mut u8, size) }
}

unsafe extern "C" fn buf_cpy_tensor(
    _: sys::ggml_backend_buffer_t,
    src: *const sys::ggml_tensor,
    dst: *mut sys::ggml_tensor,
) -> bool {
    // SAFETY: ggml's cpy_tensor contract: `dst` is a tensor in this buffer with the same layout
    // as `src` (asserted by ggml_backend_tensor_copy), so both are ggml_nbytes(src) long. We
    // only copy when `src` lives in host memory, where `data` is a plain pointer; two distinct
    // tensors don't overlap.
    unsafe {
        if !(*src).buffer.is_null() && sys::ggml_backend_buffer_is_host((*src).buffer) {
            ptr::copy_nonoverlapping(
                (*src).data as *const u8,
                (*dst).data as *mut u8,
                sys::ggml_nbytes(src),
            );
            return true;
        }
    }
    false
}

unsafe extern "C" fn buf_clear(buf: sys::ggml_backend_buffer_t, value: u8) {
    // SAFETY: `buf` is one of ours; ggml's `size` is the size we passed to
    // ggml_backend_buffer_init, at most `map_len`, so the whole range is inside our mapping.
    unsafe {
        let ctx = buf_ctx(buf);
        ptr::write_bytes(ctx.base, value, (*buf).size);
    }
}

/// cudaHostRegister / cudaHostUnregister, looked up at runtime so the crate
/// builds and runs without CUDA. llama.cpp's CUDA backend already loaded
/// libcudart if it's in use, so this finds the same copy.
struct Cudart {
    register: unsafe extern "C" fn(*mut c_void, usize, libc::c_uint) -> i32,
    unregister: unsafe extern "C" fn(*mut c_void) -> i32,
}

fn cudart() -> Option<&'static Cudart> {
    static CUDART: std::sync::OnceLock<Option<Cudart>> = std::sync::OnceLock::new();
    CUDART
        // SAFETY: dlopen/dlsym take NUL-terminated names (each CString lives until the end of
        // its dlopen call). The CUDA runtime's symbols have exactly the signatures transmuted
        // to, cudaError_t being a C int: cudaHostRegister(void*, size_t, unsigned) and
        // cudaHostUnregister(void*). The library is never closed, so the function pointers stay
        // valid.
        .get_or_init(|| unsafe {
            let h = ["libcudart.so.12", "libcudart.so", "libcudart.so.13"]
                .iter()
                .map(|n| {
                    libc::dlopen(
                        CString::new(*n).unwrap().as_ptr(),
                        libc::RTLD_NOW | libc::RTLD_GLOBAL,
                    )
                })
                .find(|h| !h.is_null())?;
            let reg = libc::dlsym(h, c"cudaHostRegister".as_ptr());
            let unreg = libc::dlsym(h, c"cudaHostUnregister".as_ptr());
            if reg.is_null() || unreg.is_null() {
                return None;
            }
            Some(Cudart {
                register: std::mem::transmute::<
                    *mut c_void,
                    unsafe extern "C" fn(*mut c_void, usize, libc::c_uint) -> i32,
                >(reg),
                unregister: std::mem::transmute::<
                    *mut c_void,
                    unsafe extern "C" fn(*mut c_void) -> i32,
                >(unreg),
            })
        })
        .as_ref()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Our buffer type through ggml's own API, as llama.cpp uses it: allocate,
    /// clear, place a tensor, write and read it back, free; many times over,
    /// at sizes including zero. Each buffer must be recorded as freed.
    #[test]
    fn buffers_allocate_use_and_free_repeatedly() {
        crate::init();
        let buft = new_buffer_type("RustPinnedTest", PinOptions::default());
        let mut bases = Vec::new();
        for _round in 0..100 {
            for size in [0usize, 1, 4096, 1 << 20] {
                // SAFETY: `buft` is a buffer type from `new_buffer_type`, alive
                // for the process; ggml returns null on failure, checked next.
                let buf = unsafe { sys::ggml_backend_buft_alloc_buffer(buft, size) };
                assert!(!buf.is_null());
                if size == 0 {
                    // ggml hands out a dummy buffer without calling us.
                    // SAFETY: `buf` is a live buffer, freed exactly once here.
                    unsafe { sys::ggml_backend_buffer_free(buf) };
                    continue;
                }
                // SAFETY: `buf` is a live buffer from our type.
                let base = unsafe { sys::ggml_backend_buffer_get_base(buf) } as usize;
                assert_eq!(base % ALIGNMENT, 0);
                bases.push(base);
                // SAFETY: `buf` is live; clear writes its `size` bytes.
                unsafe { sys::ggml_backend_buffer_clear(buf, 0xAB) };
                // SAFETY: the buffer's memory is `size` readable bytes at `base`.
                let bytes = unsafe { std::slice::from_raw_parts(base as *const u8, size) };
                assert!(bytes.iter().all(|b| *b == 0xAB));

                if size >= 4096 {
                    let params = sys::ggml_init_params {
                        // SAFETY: returns a constant; no preconditions.
                        mem_size: unsafe { sys::ggml_tensor_overhead() } * 2,
                        mem_buffer: ptr::null_mut(),
                        no_alloc: true,
                    };
                    // SAFETY: `params` asks ggml to allocate its own metadata
                    // pool (null buffer); null on failure, checked next.
                    let ctx = unsafe { sys::ggml_init(params) };
                    assert!(!ctx.is_null());
                    // SAFETY: `ctx` is live with room for this tensor's metadata.
                    let t =
                        unsafe { sys::ggml_new_tensor_1d(ctx, sys::ggml_type_GGML_TYPE_F32, 1024) };
                    assert!(!t.is_null());
                    // SAFETY: `t` (4096 bytes) is placed at the base of `buf`,
                    // which has at least 4096 bytes.
                    let st = unsafe { sys::ggml_backend_tensor_alloc(buf, t, base as *mut c_void) };
                    assert_eq!(st, sys::ggml_status_GGML_STATUS_SUCCESS);
                    let src: Vec<f32> = (0..1024).map(|i| i as f32 * 0.5).collect();
                    let mut dst = vec![0f32; 1024];
                    // SAFETY: `t` lives in `buf`; `src`/`dst` are 4096-byte
                    // host buffers matching the tensor's size.
                    unsafe {
                        sys::ggml_backend_tensor_set(t, src.as_ptr().cast(), 0, 4096);
                        sys::ggml_backend_tensor_get(t, dst.as_mut_ptr().cast(), 0, 4096);
                    }
                    assert_eq!(src, dst);
                    // SAFETY: `ctx` is live and freed once; `t` isn't used after.
                    unsafe { sys::ggml_free(ctx) };
                }
                // SAFETY: `buf` is live and freed exactly once.
                unsafe { sys::ggml_backend_buffer_free(buf) };
                let reg = REGISTRY.lock().unwrap_or_else(PoisonError::into_inner);
                let rec = reg.iter().rev().find(|r| r.base == base).unwrap();
                assert!(rec.freed, "buffer at {base:#x} not recorded as freed");
            }
        }
        assert_eq!(bases.len(), 300);
    }
}
