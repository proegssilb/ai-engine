//! ggml tensor types and their storage sizes.
//!
//! Generated from ggml itself (`ggml_type_name`, `ggml_blck_size`,
//! `ggml_type_size`) at the pinned llama.cpp, `vendor/llama.cpp` @ a956a852c.
//! Ids missing here (4, 5, 31-33, 36-38) are retired in ggml. Regenerate when
//! the llama.cpp pin moves.

use std::fmt;

/// A ggml tensor type, by its on-disk id.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GgmlType(pub u32);

struct Traits {
    id: u32,
    name: &'static str,
    /// Elements per block.
    block: u64,
    /// Bytes per block.
    bytes: u64,
}

const TABLE: &[Traits] = &[
    Traits { id: 0, name: "f32", block: 1, bytes: 4 },
    Traits { id: 1, name: "f16", block: 1, bytes: 2 },
    Traits { id: 2, name: "q4_0", block: 32, bytes: 18 },
    Traits { id: 3, name: "q4_1", block: 32, bytes: 20 },
    Traits { id: 6, name: "q5_0", block: 32, bytes: 22 },
    Traits { id: 7, name: "q5_1", block: 32, bytes: 24 },
    Traits { id: 8, name: "q8_0", block: 32, bytes: 34 },
    Traits { id: 9, name: "q8_1", block: 32, bytes: 36 },
    Traits { id: 10, name: "q2_K", block: 256, bytes: 84 },
    Traits { id: 11, name: "q3_K", block: 256, bytes: 110 },
    Traits { id: 12, name: "q4_K", block: 256, bytes: 144 },
    Traits { id: 13, name: "q5_K", block: 256, bytes: 176 },
    Traits { id: 14, name: "q6_K", block: 256, bytes: 210 },
    Traits { id: 15, name: "q8_K", block: 256, bytes: 292 },
    Traits { id: 16, name: "iq2_xxs", block: 256, bytes: 66 },
    Traits { id: 17, name: "iq2_xs", block: 256, bytes: 74 },
    Traits { id: 18, name: "iq3_xxs", block: 256, bytes: 98 },
    Traits { id: 19, name: "iq1_s", block: 256, bytes: 50 },
    Traits { id: 20, name: "iq4_nl", block: 32, bytes: 18 },
    Traits { id: 21, name: "iq3_s", block: 256, bytes: 110 },
    Traits { id: 22, name: "iq2_s", block: 256, bytes: 82 },
    Traits { id: 23, name: "iq4_xs", block: 256, bytes: 136 },
    Traits { id: 24, name: "i8", block: 1, bytes: 1 },
    Traits { id: 25, name: "i16", block: 1, bytes: 2 },
    Traits { id: 26, name: "i32", block: 1, bytes: 4 },
    Traits { id: 27, name: "i64", block: 1, bytes: 8 },
    Traits { id: 28, name: "f64", block: 1, bytes: 8 },
    Traits { id: 29, name: "iq1_m", block: 256, bytes: 56 },
    Traits { id: 30, name: "bf16", block: 1, bytes: 2 },
    Traits { id: 34, name: "tq1_0", block: 256, bytes: 54 },
    Traits { id: 35, name: "tq2_0", block: 256, bytes: 66 },
    Traits { id: 39, name: "mxfp4", block: 32, bytes: 17 },
    Traits { id: 40, name: "nvfp4", block: 64, bytes: 36 },
    Traits { id: 41, name: "q1_0", block: 128, bytes: 18 },
    Traits { id: 42, name: "q2_0", block: 64, bytes: 18 },
    Traits { id: 43, name: "f8_e4m3", block: 1, bytes: 1 },
];

impl GgmlType {
    pub const F32: GgmlType = GgmlType(0);
    pub const F16: GgmlType = GgmlType(1);
    pub const Q4_K: GgmlType = GgmlType(12);
    pub const Q6_K: GgmlType = GgmlType(14);

    fn traits(self) -> Option<&'static Traits> {
        TABLE.iter().find(|t| t.id == self.0)
    }

    /// Whether ggml at the pinned version knows this type.
    pub fn is_known(self) -> bool {
        self.traits().is_some()
    }

    /// ggml's name for the type, e.g. `q4_K`.
    pub fn name(self) -> Option<&'static str> {
        self.traits().map(|t| t.name)
    }

    /// (elements per block, bytes per block).
    pub fn block(self) -> Option<(u64, u64)> {
        self.traits().map(|t| (t.block, t.bytes))
    }

    /// Bytes needed for a tensor of this type and shape, as ggml computes it
    /// (`ggml_row_size(type, ne0) * ne1 * ne2 * ne3`). `None` for unknown types
    /// or a first dimension that isn't a whole number of blocks.
    pub fn tensor_bytes(self, dims: &[u64]) -> Option<u64> {
        let (block, bytes) = self.block()?;
        let (&ne0, rest) = dims.split_first().unwrap_or((&1, &[]));
        if ne0 % block != 0 {
            return None;
        }
        let row = ne0 / block * bytes;
        rest.iter().try_fold(row, |acc, &d| acc.checked_mul(d))
    }
}

impl fmt::Debug for GgmlType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.name() {
            Some(n) => write!(f, "{n}"),
            None => write!(f, "type#{}", self.0),
        }
    }
}

impl fmt::Display for GgmlType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_match_ggml() {
        // 2048 x 768 q4_K: 8 blocks per row * 144 bytes * 768 rows
        assert_eq!(GgmlType::Q4_K.tensor_bytes(&[2048, 768]), Some(8 * 144 * 768));
        assert_eq!(GgmlType::F16.tensor_bytes(&[10, 3, 2]), Some(120));
        assert_eq!(GgmlType::F32.tensor_bytes(&[]), Some(4));
        // not a whole number of blocks
        assert_eq!(GgmlType::Q4_K.tensor_bytes(&[100]), None);
        assert_eq!(GgmlType(4).tensor_bytes(&[32]), None);
        assert_eq!(GgmlType(4).to_string(), "type#4");
    }

    #[test]
    fn table_ids_unique_and_sorted() {
        assert!(TABLE.windows(2).all(|w| w[0].id < w[1].id));
    }
}
