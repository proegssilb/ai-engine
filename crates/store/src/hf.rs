//! Hugging Face: resolve a repo and tag to the GGUF files to download.

use crate::Error;
use serde::Deserialize;

/// One file in a repo.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RepoFile {
    pub path: String,
    pub size: u64,
    /// SHA-256 of the content, from Git LFS. `None` for small non-LFS files.
    pub sha256: Option<String>,
}

#[derive(Deserialize)]
struct Revision {
    sha: String,
}

#[derive(Deserialize)]
struct TreeEntry {
    #[serde(rename = "type")]
    kind: String,
    path: String,
    size: u64,
    lfs: Option<Lfs>,
}

#[derive(Deserialize)]
struct Lfs {
    oid: String,
}

/// A Hugging Face Hub endpoint.
#[derive(Clone)]
pub struct Hub {
    base: String,
    http: reqwest::Client,
    token: Option<String>,
}

impl Hub {
    /// `base` is `https://huggingface.co` in production. The token, if any,
    /// is sent for gated or private repos.
    pub fn new(base: impl Into<String>, token: Option<String>) -> Result<Hub, Error> {
        let http = reqwest::Client::builder()
            .user_agent(concat!("ai-engine/", env!("CARGO_PKG_VERSION")))
            .build()?;
        Ok(Hub {
            base: base.into().trim_end_matches('/').to_string(),
            http,
            token,
        })
    }

    /// The public hub, with `HF_TOKEN` from the environment if set.
    pub fn public() -> Result<Hub, Error> {
        Hub::new(
            "https://huggingface.co",
            std::env::var("HF_TOKEN").ok().filter(|t| !t.is_empty()),
        )
    }

    pub(crate) fn get(&self, url: &str) -> reqwest::RequestBuilder {
        let r = self.http.get(url);
        match &self.token {
            Some(t) => r.bearer_auth(t),
            None => r,
        }
    }

    /// The commit a branch or tag currently points to.
    pub async fn revision(&self, repo: &str, rev: &str) -> Result<String, Error> {
        let url = format!("{}/api/models/{repo}/revision/{rev}", self.base);
        let r: Revision = check(self.get(&url).send().await?, &url)
            .await?
            .json()
            .await?;
        Ok(r.sha)
    }

    /// All files in a repo at a commit.
    pub async fn files(&self, repo: &str, sha: &str) -> Result<Vec<RepoFile>, Error> {
        let mut url = format!("{}/api/models/{repo}/tree/{sha}?recursive=true", self.base);
        let mut out = Vec::new();
        loop {
            let resp = check(self.get(&url).send().await?, &url).await?;
            let next = next_link(resp.headers());
            let entries: Vec<TreeEntry> = resp.json().await?;
            out.extend(
                entries
                    .into_iter()
                    .filter(|e| e.kind == "file")
                    .map(|e| RepoFile {
                        path: e.path,
                        size: e.size,
                        sha256: e.lfs.map(|l| l.oid),
                    }),
            );
            match next {
                Some(n) => url = n,
                None => return Ok(out),
            }
        }
    }

    /// Download URL for a file at a commit.
    pub fn file_url(&self, repo: &str, sha: &str, path: &str) -> String {
        format!("{}/{repo}/resolve/{sha}/{path}", self.base)
    }
}

pub(crate) async fn check(resp: reqwest::Response, url: &str) -> Result<reqwest::Response, Error> {
    let status = resp.status();
    if status.is_success() {
        return Ok(resp);
    }
    Err(match status.as_u16() {
        401 | 403 => Error::Unauthorized(url.to_string()),
        404 => Error::NotFound(url.to_string()),
        _ => Error::Http(format!("{url}: HTTP {status}")),
    })
}

/// `Link: <url>; rel="next"` pagination.
fn next_link(h: &reqwest::header::HeaderMap) -> Option<String> {
    let v = h.get(reqwest::header::LINK)?.to_str().ok()?;
    v.split(',').find_map(|part| {
        let (url, params) = part.split_once(';')?;
        params.contains("rel=\"next\"").then(|| {
            url.trim()
                .trim_start_matches('<')
                .trim_end_matches('>')
                .to_string()
        })
    })
}

/// A GGUF path's name without folder, `.gguf` and split suffix:
/// `BF16/Qwen3-30B-A3B-BF16-00001-of-00002.gguf` -> `Qwen3-30B-A3B-BF16`.
pub fn stem_of(path: &str) -> Option<String> {
    let name = path.rsplit('/').next()?;
    let stem = name.strip_suffix(".gguf")?;
    Some(
        strip_split(stem)
            .map(|(s, _, _)| s)
            .unwrap_or(stem)
            .to_string(),
    )
}

/// Whether a file stem carries a tag: equal to it, or ending in `-TAG`
/// (case-insensitive). `Q2_K` doesn't match `...-Q2_K_L`.
fn has_tag(stem: &str, tag: &str) -> bool {
    let (s, t) = (stem.to_ascii_uppercase(), tag.to_ascii_uppercase());
    s == t || s.ends_with(&format!("-{t}"))
}

/// `name-00001-of-00003` -> (name, 1, 3)
fn strip_split(stem: &str) -> Option<(&str, u32, u32)> {
    let (rest, of) = stem.rsplit_once("-of-")?;
    let (name, idx) = rest.rsplit_once('-')?;
    if idx.len() != 5 || of.len() != 5 {
        return None;
    }
    Some((name, idx.parse().ok()?, of.parse().ok()?))
}

fn is_model_gguf(path: &str) -> bool {
    let name = path.rsplit('/').next().unwrap_or(path).to_ascii_lowercase();
    name.ends_with(".gguf") && !name.starts_with("mmproj") && !name.contains("-mmproj")
}

/// The tags a repo offers, sorted: model file names with the part they all
/// share removed (`Qwen3-30B-A3B-UD-Q4_K_XL` -> `UD-Q4_K_XL`).
pub fn tags(files: &[RepoFile]) -> Vec<String> {
    let mut stems: Vec<String> = files
        .iter()
        .filter(|f| is_model_gguf(&f.path))
        .filter_map(|f| stem_of(&f.path))
        .collect();
    stems.sort();
    stems.dedup();
    let cut = match stems.as_slice() {
        [] => return vec![],
        [one] => one.rfind('-').map(|i| i + 1).unwrap_or(0),
        [first, rest @ ..] => {
            // longest common prefix that ends at a '-'
            let mut n = first.len();
            for s in rest {
                n = n.min(
                    first
                        .bytes()
                        .zip(s.bytes())
                        .take_while(|(a, b)| a == b)
                        .count(),
                );
            }
            first[..n].rfind('-').map(|i| i + 1).unwrap_or(0)
        }
    };
    let mut t: Vec<String> = stems
        .iter()
        .map(|s| s[cut.min(s.len())..].to_string())
        .filter(|t| !t.is_empty())
        .collect();
    t.sort_by_key(|s| s.to_ascii_uppercase());
    t.dedup_by(|a, b| a.eq_ignore_ascii_case(b));
    t
}

/// The default when no tag is given: Q4_K_M if offered (ollama's habit), or
/// the only quantization if there's just one.
pub const DEFAULT_TAG: &str = "Q4_K_M";

/// Pick the files for a tag (case-insensitive, exact), split parts in order.
pub fn select<'a>(files: &'a [RepoFile], tag: Option<&str>) -> Result<Vec<&'a RepoFile>, Error> {
    let offered = tags(files);
    let tag = match tag {
        Some(t) => t.to_string(),
        None if offered.iter().any(|t| t.eq_ignore_ascii_case(DEFAULT_TAG)) => {
            DEFAULT_TAG.to_string()
        }
        None if offered.len() == 1 => offered[0].clone(),
        None => return Err(Error::AmbiguousTag { offered }),
    };
    let mut chosen: Vec<&RepoFile> = files
        .iter()
        .filter(|f| is_model_gguf(&f.path) && stem_of(&f.path).is_some_and(|s| has_tag(&s, &tag)))
        .collect();
    if chosen.is_empty() {
        return Err(Error::NoSuchTag { tag, offered });
    }
    // The same tag in two places (e.g. a root file and a folder) is ambiguous;
    // split parts of one file share a stem.
    let stem = |f: &RepoFile| {
        let dir = f.path.rsplit_once('/').map(|(d, _)| d).unwrap_or("");
        format!("{dir}/{}", stem_of(&f.path).unwrap_or_default())
    };
    let first = stem(chosen[0]);
    if chosen.iter().any(|f| stem(f) != first) {
        return Err(Error::Http(format!(
            "tag {tag} matches several files: {}",
            chosen
                .iter()
                .map(|f| f.path.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )));
    }
    chosen.sort_by_key(|f| f.path.clone());
    if let Some((_, _, of)) = strip_split(chosen[0].path.strip_suffix(".gguf").unwrap())
        && chosen.len() as u32 != of
    {
        return Err(Error::Http(format!(
            "tag {tag}: expected {of} split parts, found {}",
            chosen.len()
        )));
    }
    if let Some(f) = chosen.iter().find(|f| f.sha256.is_none()) {
        return Err(Error::Http(format!(
            "{}: no SHA-256 (not stored with Git LFS), can't verify it",
            f.path
        )));
    }
    Ok(chosen)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(path: &str) -> RepoFile {
        RepoFile {
            path: path.into(),
            size: 1,
            sha256: Some("00".repeat(32)),
        }
    }

    #[test]
    fn stems_and_tag_matching() {
        assert_eq!(
            stem_of("BF16/Qwen3-30B-A3B-BF16-00001-of-00002.gguf").as_deref(),
            Some("Qwen3-30B-A3B-BF16")
        );
        assert_eq!(stem_of("README.md"), None);
        assert!(has_tag("Qwen3-30B-A3B-UD-Q4_K_XL", "ud-q4_k_xl"));
        assert!(has_tag("Qwen3-30B-A3B-UD-Q4_K_XL", "Q4_K_XL"));
        assert!(!has_tag("Qwen3-30B-A3B-Q2_K_L", "Q2_K"));
        assert!(!has_tag("Qwen3-30B-A3B-Q4_K_M", "K_M"));
        let one = [f("Qwen3-8B-Q4_K_M.gguf")];
        assert_eq!(tags(&one), vec!["Q4_K_M"]);
        let unsloth = [
            f("Qwen3-30B-A3B-UD-Q4_K_XL.gguf"),
            f("Qwen3-30B-A3B-Q4_K_M.gguf"),
        ];
        assert_eq!(tags(&unsloth), vec!["Q4_K_M", "UD-Q4_K_XL"]);
    }

    #[test]
    fn selection() {
        let files = vec![
            f("Qwen3-30B-A3B-Q2_K.gguf"),
            f("Qwen3-30B-A3B-Q2_K_L.gguf"),
            f("Qwen3-30B-A3B-Q4_K_M.gguf"),
            f("BF16/Qwen3-30B-A3B-BF16-00002-of-00002.gguf"),
            f("BF16/Qwen3-30B-A3B-BF16-00001-of-00002.gguf"),
            f("mmproj-F16.gguf"),
            f("README.md"),
        ];
        assert_eq!(tags(&files), vec!["BF16", "Q2_K", "Q2_K_L", "Q4_K_M"]);
        let p = |t| {
            select(&files, t)
                .unwrap()
                .iter()
                .map(|f| f.path.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(p(Some("q2_k")), vec!["Qwen3-30B-A3B-Q2_K.gguf"]);
        assert_eq!(p(None), vec!["Qwen3-30B-A3B-Q4_K_M.gguf"]);
        assert_eq!(
            p(Some("BF16")),
            vec![
                "BF16/Qwen3-30B-A3B-BF16-00001-of-00002.gguf",
                "BF16/Qwen3-30B-A3B-BF16-00002-of-00002.gguf"
            ]
        );
        assert!(matches!(
            select(&files, Some("Q8_0")),
            Err(Error::NoSuchTag { .. })
        ));
        assert!(matches!(
            select(&files[..2], None),
            Err(Error::AmbiguousTag { .. })
        ));
        // a missing split part is caught
        assert!(select(&files[..4], Some("BF16")).is_err());
    }

    #[test]
    fn link_header() {
        let mut h = reqwest::header::HeaderMap::new();
        h.insert(
            reqwest::header::LINK,
            "<https://x/api?cursor=abc>; rel=\"next\"".parse().unwrap(),
        );
        assert_eq!(next_link(&h).as_deref(), Some("https://x/api?cursor=abc"));
    }
}
