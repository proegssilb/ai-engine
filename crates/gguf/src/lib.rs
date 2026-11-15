//! Read GGUF model files: header, metadata and tensor table, without loading
//! the weights.
//!
//! The checks mirror llama.cpp's own reader (`ggml/src/gguf.cpp` at the pinned
//! version) so that a file this crate accepts is one llama.cpp will load:
//! versions 2–3, a power-of-two alignment, at most 4 dimensions, unique tensor
//! names shorter than 64 bytes, and tensor data laid out contiguously in table
//! order.
//!
//! [`write`] produces GGUF files, mainly for tests.

mod read;
mod types;
pub mod write;

pub use read::Error;
pub use types::GgmlType;

use std::collections::BTreeMap;
use std::path::Path;

/// File magic, `GGUF` read as a little-endian u32.
pub const MAGIC: u32 = 0x4655_4747;
/// Newest GGUF version supported (llama.cpp's `GGUF_VERSION`).
pub const VERSION: u32 = 3;
/// Alignment used when `general.alignment` is absent.
pub const DEFAULT_ALIGNMENT: u64 = 32;
/// ggml's `GGML_MAX_NAME`: tensor names must be shorter than this.
pub const MAX_NAME: usize = 64;
/// ggml's `GGML_MAX_DIMS`.
pub const MAX_DIMS: usize = 4;

/// A metadata value.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    /// Element type id and elements. Nested arrays are allowed by the format.
    Array(u32, Vec<Value>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl Value {
    /// The value as an unsigned integer, if it is any non-negative integer type.
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::U8(v) => Some(v.into()),
            Value::U16(v) => Some(v.into()),
            Value::U32(v) => Some(v.into()),
            Value::U64(v) => Some(v),
            Value::I8(v) => u64::try_from(v).ok(),
            Value::I16(v) => u64::try_from(v).ok(),
            Value::I32(v) => u64::try_from(v).ok(),
            Value::I64(v) => u64::try_from(v).ok(),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Value::F32(v) => Some(v.into()),
            Value::F64(v) => Some(v),
            _ => self.as_u64().map(|v| v as f64),
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(_, v) => Some(v),
            _ => None,
        }
    }
}

/// One entry of the tensor table.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorInfo {
    pub name: String,
    /// Dimensions, innermost first (ggml's `ne`), without trailing 1s added.
    pub dims: Vec<u64>,
    pub ty: GgmlType,
    /// Offset of the tensor's data from the start of the data section.
    pub offset: u64,
    /// Size of the tensor's data in bytes, unpadded.
    pub size: u64,
}

impl TensorInfo {
    pub fn elements(&self) -> u64 {
        self.dims.iter().product()
    }
}

/// A parsed GGUF header.
#[derive(Clone, Debug)]
pub struct Gguf {
    pub version: u32,
    /// Metadata in file order is not significant; keys are unique.
    pub metadata: BTreeMap<String, Value>,
    /// Tensor table in file order.
    pub tensors: Vec<TensorInfo>,
    pub alignment: u64,
    /// File offset where tensor data starts.
    pub data_offset: u64,
}

impl Gguf {
    /// Read the header of a GGUF file. The tensor data isn't read, but the file
    /// must be long enough to hold it.
    pub fn open(path: impl AsRef<Path>) -> Result<Gguf, Error> {
        read::open(path.as_ref(), true)
    }

    /// Read only the header, accepting a file that stops after it (for example
    /// the first few MiB of a remote model fetched with a range request).
    pub fn open_header(path: impl AsRef<Path>) -> Result<Gguf, Error> {
        read::open(path.as_ref(), false)
    }

    /// Parse a header from any reader, without checking the data section's length.
    pub fn from_reader(r: impl std::io::Read) -> Result<Gguf, Error> {
        read::parse(r)
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.metadata.get(key)
    }

    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(Value::as_str)
    }

    pub fn get_u64(&self, key: &str) -> Option<u64> {
        self.get(key).and_then(Value::as_u64)
    }

    /// `general.architecture`, e.g. `qwen3moe`.
    pub fn architecture(&self) -> Option<&str> {
        self.get_str("general.architecture")
    }

    /// An architecture-scoped key such as `qwen3moe.block_count`.
    pub fn arch_u64(&self, suffix: &str) -> Option<u64> {
        let arch = self.architecture()?;
        self.get_u64(&format!("{arch}.{suffix}"))
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Size of the data section (the last tensor's end).
    pub fn data_size(&self) -> u64 {
        self.tensors.last().map(|t| t.offset + t.size).unwrap_or(0)
    }

    /// Absolute file offset of a tensor's data.
    pub fn file_offset(&self, t: &TensorInfo) -> u64 {
        self.data_offset + t.offset
    }
}

pub(crate) fn pad(x: u64, align: u64) -> u64 {
    x.div_ceil(align) * align
}
