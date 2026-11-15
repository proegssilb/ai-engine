//! What users type to name a model.

use std::fmt;
use std::path::PathBuf;

/// A model reference.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelRef {
    /// `hf.co/owner/repo[:tag]`: a GGUF repo on Hugging Face; the tag picks the
    /// quantization (`Q4_K_M`).
    Hf(HfRef),
    /// A GGUF file already on disk.
    Local(PathBuf),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HfRef {
    /// `owner/name`.
    pub repo: String,
    pub tag: Option<String>,
}

impl fmt::Display for HfRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "hf.co/{}", self.repo)?;
        if let Some(t) = &self.tag {
            write!(f, ":{t}")?;
        }
        Ok(())
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct ParseError(pub String);

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ParseError {}

fn valid_part(s: &str) -> bool {
    !s.is_empty() && s != "." && s != ".." && s.chars().all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c))
}

impl ModelRef {
    /// Parse `hf.co/owner/repo[:tag]` (also `huggingface.co/...` and
    /// `https://` forms), or a path to a `.gguf` file.
    pub fn parse(s: &str) -> Result<ModelRef, ParseError> {
        let s = s.trim();
        let rest = s.strip_prefix("https://").or_else(|| s.strip_prefix("http://")).unwrap_or(s);
        if let Some(r) = rest.strip_prefix("hf.co/").or_else(|| rest.strip_prefix("huggingface.co/")) {
            let (repo, tag) = match r.rsplit_once(':') {
                Some((repo, tag)) if !tag.is_empty() => (repo, Some(tag.to_string())),
                Some(_) => return Err(ParseError(format!("{s}: empty tag after ':'"))),
                None => (r, None),
            };
            let parts: Vec<&str> = repo.trim_end_matches('/').split('/').collect();
            if parts.len() != 2 || !parts.iter().all(|p| valid_part(p)) {
                return Err(ParseError(format!("{s}: expected hf.co/OWNER/REPO[:TAG]")));
            }
            if let Some(t) = &tag
                && !valid_part(t)
            {
                return Err(ParseError(format!("{s}: tag {t:?} has characters other than letters, digits, - _ .")));
            }
            return Ok(ModelRef::Hf(HfRef { repo: parts.join("/"), tag }));
        }
        if s.ends_with(".gguf") || s.starts_with('/') || s.starts_with("./") || s.starts_with("../") {
            return Ok(ModelRef::Local(PathBuf::from(s)));
        }
        Err(ParseError(format!(
            "{s}: not a model reference; use hf.co/OWNER/REPO[:TAG] or a path to a .gguf file"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hf(repo: &str, tag: Option<&str>) -> ModelRef {
        ModelRef::Hf(HfRef { repo: repo.into(), tag: tag.map(Into::into) })
    }

    #[test]
    fn parses() {
        assert_eq!(ModelRef::parse("hf.co/Qwen/Qwen3-30B-A3B-GGUF:Q4_K_M"), Ok(hf("Qwen/Qwen3-30B-A3B-GGUF", Some("Q4_K_M"))));
        assert_eq!(ModelRef::parse("huggingface.co/unsloth/Qwen3-8B-GGUF"), Ok(hf("unsloth/Qwen3-8B-GGUF", None)));
        assert_eq!(ModelRef::parse("https://huggingface.co/a/b:UD-Q4_K_XL"), Ok(hf("a/b", Some("UD-Q4_K_XL"))));
        assert_eq!(ModelRef::parse("./m.gguf"), Ok(ModelRef::Local("./m.gguf".into())));
        assert_eq!(ModelRef::parse("/srv/x.gguf"), Ok(ModelRef::Local("/srv/x.gguf".into())));
        let Ok(ModelRef::Hf(r)) = ModelRef::parse("hf.co/a/b:Q8_0") else { panic!() };
        assert_eq!(r.to_string(), "hf.co/a/b:Q8_0");
    }

    #[test]
    fn rejects() {
        for bad in ["hf.co/a", "hf.co/a/b/c", "hf.co/a/b:", "hf.co/../b", "hf.co/a/b:Q4 K", "qwen3:30b", "hf.co/a/b:x/y"] {
            assert!(ModelRef::parse(bad).is_err(), "{bad}");
        }
    }
}
