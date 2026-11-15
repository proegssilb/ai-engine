//! llama.cpp's log, captured. llama.cpp logs through one process-global
//! callback; lines are kept in a bounded buffer and optionally echoed.
//!
//! Don't parse placement from these lines: they've been wrong before (D4).
//! Use [`crate::verify`].

use crate::sys;
use std::collections::VecDeque;
use std::ffi::{CStr, c_void};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

const KEEP: usize = 4096;

static LINES: Mutex<VecDeque<String>> = Mutex::new(VecDeque::new());
static ECHO: AtomicBool = AtomicBool::new(false);
/// llama.cpp sends partial lines; this holds the unfinished one.
static PENDING: Mutex<String> = Mutex::new(String::new());

pub(crate) unsafe fn install() {
    unsafe { sys::llama_log_set(Some(callback), std::ptr::null_mut()) };
}

unsafe extern "C" fn callback(
    _level: sys::ggml_log_level,
    text: *const libc::c_char,
    _: *mut c_void,
) {
    let s = unsafe { CStr::from_ptr(text) }.to_string_lossy();
    if ECHO.load(Ordering::Relaxed) {
        eprint!("{s}");
    }
    let mut pending = PENDING.lock().unwrap();
    pending.push_str(&s);
    while let Some(i) = pending.find('\n') {
        let line: String = pending.drain(..=i).collect();
        let mut lines = LINES.lock().unwrap();
        if lines.len() == KEEP {
            lines.pop_front();
        }
        lines.push_back(line.trim_end().to_string());
    }
}

/// Echo llama.cpp's log to stderr as it arrives.
pub fn set_echo(on: bool) {
    ECHO.store(on, Ordering::Relaxed);
}

/// The most recent captured lines (up to 4096).
pub fn recent() -> Vec<String> {
    LINES.lock().unwrap().iter().cloned().collect()
}

/// Recent lines containing `needle`, newest last.
pub fn grep(needle: &str) -> Vec<String> {
    LINES
        .lock()
        .unwrap()
        .iter()
        .filter(|l| l.contains(needle))
        .cloned()
        .collect()
}

/// Last few lines, for error messages.
pub fn tail(n: usize) -> String {
    let lines = LINES.lock().unwrap();
    let skip = lines.len().saturating_sub(n);
    lines
        .iter()
        .skip(skip)
        .cloned()
        .collect::<Vec<_>>()
        .join("\n")
}
