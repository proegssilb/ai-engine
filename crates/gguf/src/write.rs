//! Write GGUF files. Meant for tests and fixtures: it builds the whole header in
//! memory and fills tensor data with zeros unless data is given.

use crate::{DEFAULT_ALIGNMENT, GgmlType, MAGIC, VERSION, Value, pad};
use std::io::{self, Write};
use std::path::Path;

struct Tensor {
    name: String,
    dims: Vec<u64>,
    ty: GgmlType,
    data: Option<Vec<u8>>,
}

/// Builds a GGUF file.
#[derive(Default)]
pub struct Writer {
    kv: Vec<(String, Value)>,
    tensors: Vec<Tensor>,
}

impl Writer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn kv(mut self, key: impl Into<String>, value: Value) -> Self {
        self.kv.push((key.into(), value));
        self
    }

    pub fn str(self, key: impl Into<String>, s: impl Into<String>) -> Self {
        self.kv(key, Value::String(s.into()))
    }

    pub fn u32(self, key: impl Into<String>, v: u32) -> Self {
        self.kv(key, Value::U32(v))
    }

    /// Add a tensor whose data will be zeros.
    pub fn tensor(mut self, name: impl Into<String>, ty: GgmlType, dims: &[u64]) -> Self {
        self.tensors.push(Tensor { name: name.into(), dims: dims.to_vec(), ty, data: None });
        self
    }

    /// Add a tensor with explicit data (must be exactly the tensor's size).
    pub fn tensor_data(mut self, name: impl Into<String>, ty: GgmlType, dims: &[u64], data: Vec<u8>) -> Self {
        self.tensors.push(Tensor { name: name.into(), dims: dims.to_vec(), ty, data: Some(data) });
        self
    }

    fn alignment(&self) -> u64 {
        self.kv
            .iter()
            .find(|(k, _)| k == "general.alignment")
            .and_then(|(_, v)| v.as_u64())
            .unwrap_or(DEFAULT_ALIGNMENT)
    }

    /// The header bytes, padded to the start of the data section.
    pub fn header(&self) -> io::Result<Vec<u8>> {
        let align = self.alignment();
        let mut h = Vec::new();
        h.extend(MAGIC.to_le_bytes());
        h.extend(VERSION.to_le_bytes());
        h.extend((self.tensors.len() as u64).to_le_bytes());
        h.extend((self.kv.len() as u64).to_le_bytes());
        for (k, v) in &self.kv {
            put_str(&mut h, k);
            h.extend(type_id(v).to_le_bytes());
            put_value(&mut h, v);
        }
        let mut offset = 0u64;
        for t in &self.tensors {
            put_str(&mut h, &t.name);
            h.extend((t.dims.len() as u32).to_le_bytes());
            for d in &t.dims {
                h.extend(d.to_le_bytes());
            }
            h.extend(t.ty.0.to_le_bytes());
            h.extend(offset.to_le_bytes());
            let size = size_of(t)?;
            offset += pad(size, align);
        }
        if !self.tensors.is_empty() {
            h.resize(pad(h.len() as u64, align) as usize, 0);
        }
        Ok(h)
    }

    /// Write a complete file: header and tensor data.
    pub fn write(&self, mut w: impl Write) -> io::Result<()> {
        let align = self.alignment();
        w.write_all(&self.header()?)?;
        let zeros = vec![0u8; 1 << 16];
        for t in &self.tensors {
            let size = size_of(t)?;
            match &t.data {
                Some(d) if d.len() as u64 == size => w.write_all(d)?,
                Some(d) => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("tensor {:?}: {} bytes of data, expected {size}", t.name, d.len()),
                    ));
                }
                None => write_zeros(&mut w, size, &zeros)?,
            }
            write_zeros(&mut w, pad(size, align) - size, &zeros)?;
        }
        Ok(())
    }

    pub fn write_file(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let f = std::fs::File::create(path)?;
        let mut w = io::BufWriter::new(f);
        self.write(&mut w)?;
        w.flush()
    }

    /// Write only the header: a file `Gguf::open_header` accepts and `Gguf::open` rejects
    /// as truncated. Lets tests describe full-size models cheaply.
    pub fn write_header_file(&self, path: impl AsRef<Path>) -> io::Result<()> {
        std::fs::write(path, self.header()?)
    }
}

fn size_of(t: &Tensor) -> io::Result<u64> {
    t.ty.tensor_bytes(&t.dims).ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, format!("tensor {:?}: shape {:?} invalid for {}", t.name, t.dims, t.ty))
    })
}

fn write_zeros(w: &mut impl Write, mut n: u64, zeros: &[u8]) -> io::Result<()> {
    while n > 0 {
        let k = n.min(zeros.len() as u64) as usize;
        w.write_all(&zeros[..k])?;
        n -= k as u64;
    }
    Ok(())
}

fn put_str(h: &mut Vec<u8>, s: &str) {
    h.extend((s.len() as u64).to_le_bytes());
    h.extend(s.as_bytes());
}

fn type_id(v: &Value) -> u32 {
    match v {
        Value::U8(_) => 0,
        Value::I8(_) => 1,
        Value::U16(_) => 2,
        Value::I16(_) => 3,
        Value::U32(_) => 4,
        Value::I32(_) => 5,
        Value::F32(_) => 6,
        Value::Bool(_) => 7,
        Value::String(_) => 8,
        Value::Array(..) => 9,
        Value::U64(_) => 10,
        Value::I64(_) => 11,
        Value::F64(_) => 12,
    }
}

fn put_value(h: &mut Vec<u8>, v: &Value) {
    match v {
        Value::U8(x) => h.push(*x),
        Value::I8(x) => h.push(*x as u8),
        Value::U16(x) => h.extend(x.to_le_bytes()),
        Value::I16(x) => h.extend(x.to_le_bytes()),
        Value::U32(x) => h.extend(x.to_le_bytes()),
        Value::I32(x) => h.extend(x.to_le_bytes()),
        Value::F32(x) => h.extend(x.to_le_bytes()),
        Value::Bool(x) => h.push(*x as u8),
        Value::String(s) => put_str(h, s),
        Value::Array(elem, items) => {
            h.extend(elem.to_le_bytes());
            h.extend((items.len() as u64).to_le_bytes());
            for i in items {
                put_value(h, i);
            }
        }
        Value::U64(x) => h.extend(x.to_le_bytes()),
        Value::I64(x) => h.extend(x.to_le_bytes()),
        Value::F64(x) => h.extend(x.to_le_bytes()),
    }
}
