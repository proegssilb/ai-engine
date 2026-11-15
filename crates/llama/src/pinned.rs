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
use std::sync::Mutex;

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

fn errno() -> i32 {
    std::io::Error::last_os_error().raw_os_error().unwrap_or(-1)
}

fn page_size() -> usize {
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
}

/// Create a buffer type usable in `llama_model_tensor_buft_override`.
/// It's leaked on purpose: it must outlive every model that references it.
pub(crate) fn new_buffer_type(name: &str, opts: PinOptions) -> sys::ggml_backend_buffer_type_t {
    let ctx = Box::new(BuftCtx {
        name: CString::new(name).unwrap(),
        opts,
    });
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
    unsafe { &*((*buft).context as *const BuftCtx) }
}

unsafe extern "C" fn buft_get_name(buft: sys::ggml_backend_buffer_type_t) -> *const libc::c_char {
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
    let opts = unsafe { buft_ctx(buft).opts.clone() };
    let anon = |len: usize, extra: libc::c_int| unsafe {
        libc::mmap(ptr::null_mut(), len, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | extra, -1, 0)
    };
    // Reserved huge pages first, if asked; else (or on failure) normal pages.
    // size can be 0 (llama.cpp's no_alloc dummy buffers); still map one page.
    let mut hugetlb_result = None;
    let mut mapped = None;
    if let Some(hp) = opts.hugetlb.filter(|_| size > 0) {
        let len = size.div_ceil(hp as usize) * hp as usize;
        const MAP_HUGE_SHIFT: libc::c_int = 26;
        let flags = libc::MAP_HUGETLB | ((hp.trailing_zeros() as libc::c_int) << MAP_HUGE_SHIFT);
        let b = anon(len, flags);
        if b == libc::MAP_FAILED {
            hugetlb_result = Some(Err(errno()));
        } else {
            hugetlb_result = Some(Ok(hp));
            mapped = Some((b, len));
        }
    }
    let (base, map_len) = match mapped {
        Some(m) => m,
        None => {
            let ps = page_size();
            let len = size.max(1).div_ceil(ps) * ps;
            let b = anon(len, 0);
            if b == libc::MAP_FAILED {
                return ptr::null_mut();
            }
            (b, len)
        }
    };

    let mut rec = BufferRecord {
        base: base as usize,
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
        let advice = if opts.hugepages { libc::MADV_HUGEPAGE } else { libc::MADV_NOHUGEPAGE };
        unsafe { libc::madvise(base, map_len, advice) };
    }
    if let Some(node) = opts.numa_node {
        // Must happen before first touch: the policy applies to pages as they fault in.
        let mask: [libc::c_ulong; 16] = {
            let mut m = [0; 16];
            m[(node / 64) as usize] = 1 << (node % 64);
            m
        };
        const MPOL_BIND: libc::c_long = 2;
        const MPOL_MF_STRICT: libc::c_ulong = 1;
        let r = unsafe {
            libc::syscall(
                libc::SYS_mbind,
                base,
                map_len,
                MPOL_BIND,
                mask.as_ptr(),
                (mask.len() * 64) as libc::c_ulong,
                MPOL_MF_STRICT,
            )
        };
        rec.mbind_result = Some(if r == 0 { Ok(()) } else { Err(errno()) });
    }
    if opts.mlock {
        // mlock also faults every page in now, under the NUMA policy above.
        let r = unsafe { libc::mlock(base, map_len) };
        rec.mlock_result = Some(if r == 0 { Ok(()) } else { Err(errno()) });
    }
    if opts.cuda_register {
        rec.cuda_register_result = Some(match cudart() {
            Some(c) => match unsafe { (c.register)(base, map_len, 0) } {
                0 => Ok(()),
                e => Err(e),
            },
            None => Err(-1),
        });
    }

    let idx = {
        let mut reg = REGISTRY.lock().unwrap();
        reg.push(rec);
        reg.len() - 1
    };
    let ctx = Box::new(BufCtx {
        base: base as *mut u8,
        map_len,
        idx,
    });
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
    unsafe { &*((*buf).context as *const BufCtx) }
}

unsafe extern "C" fn buf_free(buf: sys::ggml_backend_buffer_t) {
    unsafe {
        let ctx = Box::from_raw((*buf).context as *mut BufCtx);
        let registered = matches!(
            REGISTRY.lock().unwrap()[ctx.idx].cuda_register_result,
            Some(Ok(()))
        );
        if registered && let Some(c) = cudart() {
            (c.unregister)(ctx.base as *mut c_void);
        }
        libc::munmap(ctx.base as *mut c_void, ctx.map_len);
        REGISTRY.lock().unwrap()[ctx.idx].freed = true;
    }
}

unsafe extern "C" fn buf_get_base(buf: sys::ggml_backend_buffer_t) -> *mut c_void {
    unsafe { buf_ctx(buf).base as *mut c_void }
}

unsafe extern "C" fn buf_init_tensor(
    buf: sys::ggml_backend_buffer_t,
    t: *mut sys::ggml_tensor,
) -> sys::ggml_status {
    unsafe {
        // Views share their parent's storage; only record real allocations.
        if (*t).view_src.is_null() {
            let name = CStr::from_ptr((*t).name.as_ptr())
                .to_string_lossy()
                .into_owned();
            let entry = (name, (*t).data as usize, sys::ggml_nbytes(t));
            REGISTRY.lock().unwrap()[buf_ctx(buf).idx]
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
    unsafe { ptr::write_bytes(((*t).data as *mut u8).add(offset), value, size) }
}

unsafe extern "C" fn buf_set_tensor(
    _: sys::ggml_backend_buffer_t,
    t: *mut sys::ggml_tensor,
    data: *const c_void,
    offset: usize,
    size: usize,
) {
    unsafe { ptr::copy_nonoverlapping(data as *const u8, ((*t).data as *mut u8).add(offset), size) }
}

unsafe extern "C" fn buf_get_tensor(
    _: sys::ggml_backend_buffer_t,
    t: *const sys::ggml_tensor,
    data: *mut c_void,
    offset: usize,
    size: usize,
) {
    unsafe { ptr::copy_nonoverlapping(((*t).data as *const u8).add(offset), data as *mut u8, size) }
}

unsafe extern "C" fn buf_cpy_tensor(
    _: sys::ggml_backend_buffer_t,
    src: *const sys::ggml_tensor,
    dst: *mut sys::ggml_tensor,
) -> bool {
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
