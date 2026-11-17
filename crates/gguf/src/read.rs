use crate::{
    DEFAULT_ALIGNMENT, GgmlType, Gguf, MAGIC, MAX_DIMS, MAX_NAME, TensorInfo, VERSION, Value, pad,
};
use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;

/// Why a file couldn't be read as GGUF.
#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    /// Not a GGUF file.
    BadMagic(u32),
    /// GGUF v1, or newer than [`VERSION`], or the wrong endianness.
    UnsupportedVersion(u32),
    Malformed(String),
    /// The file ends before the tensor data does.
    Truncated {
        needed: u64,
        actual: u64,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "I/O error: {e}"),
            Error::BadMagic(m) => write!(f, "not a GGUF file (magic {m:#010x})"),
            Error::UnsupportedVersion(v) => {
                write!(f, "unsupported GGUF version {v} (supported: 2-{VERSION})")
            }
            Error::Malformed(m) => write!(f, "malformed GGUF: {m}"),
            Error::Truncated { needed, actual } => {
                write!(f, "file is truncated: needs {needed} bytes, has {actual}")
            }
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        if e.kind() == io::ErrorKind::UnexpectedEof {
            Error::Malformed("unexpected end of file in header".into())
        } else {
            Error::Io(e)
        }
    }
}

fn malformed<T>(msg: impl Into<String>) -> Result<T, Error> {
    Err(Error::Malformed(msg.into()))
}

pub(crate) fn open(path: &Path, check_len: bool) -> Result<Gguf, Error> {
    let file = File::open(path)?;
    let len = file.metadata()?.len();
    let g = parse(BufReader::with_capacity(1 << 20, file))?;
    let needed = g.data_offset + g.data_size();
    if check_len && len < needed {
        return Err(Error::Truncated {
            needed,
            actual: len,
        });
    }
    Ok(g)
}

/// Counts bytes consumed so the data section's start can be computed.
struct Counted<R> {
    inner: R,
    pos: u64,
}

impl<R: Read> Counted<R> {
    fn bytes<const N: usize>(&mut self) -> Result<[u8; N], Error> {
        let mut b = [0u8; N];
        self.inner.read_exact(&mut b)?;
        self.pos += N as u64;
        Ok(b)
    }
    fn u32(&mut self) -> Result<u32, Error> {
        Ok(u32::from_le_bytes(self.bytes()?))
    }
    fn u64(&mut self) -> Result<u64, Error> {
        Ok(u64::from_le_bytes(self.bytes()?))
    }
    fn string(&mut self) -> Result<String, Error> {
        let len = self.u64()?;
        // Read through `take` so a corrupt length can't force a huge allocation.
        let mut buf = Vec::new();
        let got = (&mut self.inner).take(len).read_to_end(&mut buf)? as u64;
        self.pos += got;
        if got != len {
            return malformed(format!(
                "string of {len} bytes runs past the end of the file"
            ));
        }
        String::from_utf8(buf).or_else(|_| malformed("string is not valid UTF-8"))
    }

    fn value(&mut self, ty: u32, depth: u32) -> Result<Value, Error> {
        Ok(match ty {
            0 => Value::U8(self.bytes::<1>()?[0]),
            1 => Value::I8(self.bytes::<1>()?[0] as i8),
            2 => Value::U16(u16::from_le_bytes(self.bytes()?)),
            3 => Value::I16(i16::from_le_bytes(self.bytes()?)),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(i32::from_le_bytes(self.bytes()?)),
            6 => Value::F32(f32::from_le_bytes(self.bytes()?)),
            7 => match self.bytes::<1>()?[0] {
                0 => Value::Bool(false),
                1 => Value::Bool(true),
                b => return malformed(format!("bool value {b}")),
            },
            8 => Value::String(self.string()?),
            9 => {
                if depth > 8 {
                    return malformed("arrays nested too deeply");
                }
                let elem = self.u32()?;
                let n = self.u64()?;
                let mut v = Vec::with_capacity(n.min(1 << 16) as usize);
                for _ in 0..n {
                    v.push(self.value(elem, depth + 1)?);
                }
                Value::Array(elem, v)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(i64::from_le_bytes(self.bytes()?)),
            12 => Value::F64(f64::from_le_bytes(self.bytes()?)),
            t => return malformed(format!("unknown metadata value type {t}")),
        })
    }
}

pub(crate) fn parse(r: impl Read) -> Result<Gguf, Error> {
    let mut r = Counted { inner: r, pos: 0 };

    let magic = r.u32()?;
    if magic != MAGIC {
        return Err(Error::BadMagic(magic));
    }
    let version = r.u32()?;
    // A version with zero low bits is a byte-swapped (big-endian) file.
    if !(2..=VERSION).contains(&version) || version & 0xFFFF == 0 {
        return Err(Error::UnsupportedVersion(version));
    }
    let n_tensors = r.u64()?;
    let n_kv = r.u64()?;

    let mut metadata = BTreeMap::new();
    for _ in 0..n_kv {
        let key = r.string()?;
        let ty = r.u32()?;
        let value = r.value(ty, 0)?;
        if metadata.insert(key.clone(), value).is_some() {
            return malformed(format!("duplicate metadata key {key:?}"));
        }
    }

    let alignment = match metadata.get("general.alignment") {
        None => DEFAULT_ALIGNMENT,
        Some(Value::U32(a)) if *a != 0 && a.is_power_of_two() => (*a).into(),
        Some(Value::U32(a)) => return malformed(format!("alignment {a} is not a power of 2")),
        Some(v) => return malformed(format!("general.alignment must be u32, got {v:?}")),
    };

    let mut tensors = Vec::with_capacity(n_tensors.min(1 << 16) as usize);
    let mut seen = HashSet::new();
    for i in 0..n_tensors {
        let name = r.string()?;
        if name.len() >= MAX_NAME {
            return malformed(format!(
                "tensor {i} name is {} bytes (limit {})",
                name.len(),
                MAX_NAME - 1
            ));
        }
        if !seen.insert(name.clone()) {
            return malformed(format!("duplicate tensor name {name:?}"));
        }
        let n_dims = r.u32()? as usize;
        if n_dims > MAX_DIMS {
            return malformed(format!(
                "tensor {name:?} has {n_dims} dimensions (max {MAX_DIMS})"
            ));
        }
        let mut dims = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            let d = r.u64()?;
            if d > i64::MAX as u64 {
                return malformed(format!("tensor {name:?} has a negative dimension"));
            }
            dims.push(d);
        }
        let ty = GgmlType(r.u32()?);
        if !ty.is_known() {
            return malformed(format!("tensor {name:?} has unknown type id {}", ty.0));
        }
        let size = ty.tensor_bytes(&dims).ok_or_else(|| {
            Error::Malformed(format!(
                "tensor {name:?}: shape {dims:?} doesn't fit type {ty} (block size or overflow)"
            ))
        })?;
        let offset = r.u64()?;
        tensors.push(TensorInfo {
            name,
            dims,
            ty,
            offset,
            size,
        });
    }

    // Data must be contiguous in table order, each tensor padded to the alignment.
    let mut expected = 0u64;
    for t in &tensors {
        if t.offset != expected {
            return malformed(format!(
                "tensor {:?} has offset {}, expected {expected}",
                t.name, t.offset
            ));
        }
        expected = t
            .offset
            .checked_add(pad(t.size, alignment))
            .ok_or_else(|| Error::Malformed(format!("tensor {:?} offset overflows", t.name)))?;
    }

    let data_offset = if tensors.is_empty() {
        r.pos
    } else {
        pad(r.pos, alignment)
    };
    Ok(Gguf {
        version,
        metadata,
        tensors,
        alignment,
        data_offset,
    })
}
