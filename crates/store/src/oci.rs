//! Pulling from OCI registries (the distribution spec): community model
//! registries in a known format, hosted on any OCI registry (features.md).
//!
//! A model is an image manifest whose layers include GGUF files, recognised
//! by media type: ours, or ollama's model layer (ollama's registry speaks the
//! same protocol). Split models are several GGUF layers, ordered by their
//! `org.opencontainers.image.title` annotation.
//!
//! Auth: anonymous bearer tokens via the `WWW-Authenticate` challenge, which
//! is what public registries (Docker Hub, GHCR, ollama, zot, Harbor) use.

use crate::Error;
use serde::Deserialize;
use std::fmt;
use std::sync::Mutex;

/// Media types of GGUF model layers.
pub const GGUF_MEDIA_TYPES: &[&str] = &[
    "application/vnd.aie.model.gguf",
    "application/vnd.ollama.image.model",
];

const MANIFEST_ACCEPT: &str = "application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json";

/// `registry/namespace/repo:tag` or `...@sha256:digest`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OciRef {
    /// Host, with port if any (`ghcr.io`, `localhost:5000`).
    pub registry: String,
    /// Repository path (`org/model`).
    pub repository: String,
    /// Tag or `sha256:...` digest.
    pub reference: String,
}

impl fmt::Display for OciRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sep = if self.reference.starts_with("sha256:") {
            '@'
        } else {
            ':'
        };
        write!(
            f,
            "{}/{}{sep}{}",
            self.registry, self.repository, self.reference
        )
    }
}

impl OciRef {
    /// Parse a reference whose first component is a registry host: it has a
    /// dot or a port, or is `localhost`. Without a tag, `latest`.
    pub fn parse(s: &str) -> Option<OciRef> {
        let (host, rest) = s.split_once('/')?;
        if !(host.contains('.') || host.contains(':') || host == "localhost") || rest.is_empty() {
            return None;
        }
        let (repo, reference) = if let Some((r, d)) = rest.split_once('@') {
            (r, d.to_string())
        } else {
            match rest.rsplit_once(':') {
                Some((r, t)) if !t.contains('/') => (r, t.to_string()),
                _ => (rest, "latest".to_string()),
            }
        };
        let ok_repo = !repo.is_empty()
            && repo.split('/').all(|p| {
                !p.is_empty()
                    && p.chars()
                        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || "._-".contains(c))
            });
        let ok_ref = if let Some(hex) = reference.strip_prefix("sha256:") {
            hex.len() == 64 && hex.chars().all(|c| c.is_ascii_hexdigit())
        } else {
            !reference.is_empty()
                && reference.len() <= 128
                && reference
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || "._-".contains(c))
        };
        (ok_repo && ok_ref).then(|| OciRef {
            registry: host.to_string(),
            repository: repo.to_string(),
            reference,
        })
    }
}

#[derive(Deserialize)]
pub struct Descriptor {
    #[serde(rename = "mediaType")]
    pub media_type: String,
    pub digest: String,
    pub size: u64,
    #[serde(default)]
    pub annotations: std::collections::BTreeMap<String, String>,
}

#[derive(Deserialize)]
pub struct ImageManifest {
    #[serde(rename = "mediaType", default)]
    pub media_type: String,
    #[serde(default)]
    pub layers: Vec<Descriptor>,
}

/// One GGUF file to fetch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layer {
    pub file_name: String,
    pub sha256: String,
    pub size: u64,
}

/// A registry client; `http` for plain-HTTP registries (local testing).
pub struct Client {
    http: reqwest::Client,
    scheme: &'static str,
    token: Mutex<Option<String>>,
}

#[derive(Deserialize)]
struct Token {
    token: Option<String>,
    access_token: Option<String>,
}

impl Client {
    pub fn new(insecure_http: bool) -> Result<Client, Error> {
        let http = reqwest::Client::builder()
            .user_agent(concat!("ai-engine/", env!("CARGO_PKG_VERSION")))
            .build()?;
        Ok(Client {
            http,
            scheme: if insecure_http { "http" } else { "https" },
            token: Mutex::new(None),
        })
    }

    /// Plain HTTP for `localhost` registries, HTTPS otherwise.
    pub fn for_registry(registry: &str) -> Result<Client, Error> {
        let local = registry.starts_with("localhost") || registry.starts_with("127.0.0.1");
        Client::new(local)
    }

    pub(crate) fn get(&self, url: &str) -> reqwest::RequestBuilder {
        let r = self.http.get(url);
        match self.token.lock().unwrap().as_ref() {
            Some(t) => r.bearer_auth(t),
            None => r,
        }
    }

    pub fn blob_url(&self, r: &OciRef, digest: &str) -> String {
        format!(
            "{}://{}/v2/{}/blobs/{digest}",
            self.scheme, r.registry, r.repository
        )
    }

    /// GET with one round of the bearer-token challenge.
    async fn get_authed(&self, url: &str, accept: &str) -> Result<reqwest::Response, Error> {
        let resp = self
            .get(url)
            .header(reqwest::header::ACCEPT, accept)
            .send()
            .await?;
        if resp.status() != reqwest::StatusCode::UNAUTHORIZED {
            return Ok(resp);
        }
        let challenge = resp
            .headers()
            .get(reqwest::header::WWW_AUTHENTICATE)
            .and_then(|v| v.to_str().ok())
            .ok_or_else(|| Error::Unauthorized(url.to_string()))?
            .to_string();
        let params = parse_challenge(&challenge)
            .ok_or_else(|| Error::Unauthorized(format!("{url} (unsupported auth: {challenge})")))?;
        let mut token_url = reqwest::Url::parse(&params.realm)
            .map_err(|e| Error::Http(format!("token realm {}: {e}", params.realm)))?;
        {
            let mut q = token_url.query_pairs_mut();
            if let Some(s) = &params.service {
                q.append_pair("service", s);
            }
            if let Some(s) = &params.scope {
                q.append_pair("scope", s);
            }
        }
        let t: Token = crate::hf::check(
            self.http.get(token_url.as_str()).send().await?,
            token_url.as_str(),
        )
        .await?
        .json()
        .await?;
        *self.token.lock().unwrap() = t.token.or(t.access_token);
        Ok(self
            .get(url)
            .header(reqwest::header::ACCEPT, accept)
            .send()
            .await?)
    }

    /// Resolve a reference to its manifest digest and GGUF layers.
    pub async fn resolve(&self, r: &OciRef) -> Result<(String, Vec<Layer>), Error> {
        let url = format!(
            "{}://{}/v2/{}/manifests/{}",
            self.scheme, r.registry, r.repository, r.reference
        );
        let resp = crate::hf::check(self.get_authed(&url, MANIFEST_ACCEPT).await?, &url).await?;
        let digest = resp
            .headers()
            .get("docker-content-digest")
            .and_then(|v| v.to_str().ok())
            .map(String::from);
        let body = resp.bytes().await?;
        let digest = digest.unwrap_or_else(|| {
            format!(
                "sha256:{}",
                crate::hex(&<sha2::Sha256 as sha2::Digest>::digest(&body))
            )
        });
        let m: ImageManifest =
            serde_json::from_slice(&body).map_err(|e| Error::Manifest(format!("{url}: {e}")))?;
        if m.media_type.contains("index") || m.media_type.contains("manifest.list") {
            return Err(Error::Manifest(format!(
                "{r} is a multi-platform index, not a model"
            )));
        }
        let mut ggufs: Vec<&Descriptor> = m
            .layers
            .iter()
            .filter(|l| GGUF_MEDIA_TYPES.contains(&l.media_type.as_str()))
            .collect();
        if ggufs.is_empty() {
            return Err(Error::Manifest(format!(
                "{r} has no GGUF layer (media types {})",
                GGUF_MEDIA_TYPES.join(", ")
            )));
        }
        let title = |d: &Descriptor| d.annotations.get("org.opencontainers.image.title").cloned();
        if ggufs.iter().all(|d| title(d).is_some()) {
            ggufs.sort_by_key(|d| title(d));
        }
        let n = ggufs.len();
        let layers = ggufs
            .iter()
            .enumerate()
            .map(|(i, d)| {
                let sha = d.digest.strip_prefix("sha256:").ok_or_else(|| {
                    Error::Manifest(format!("layer digest {} isn't sha256", d.digest))
                })?;
                let file_name = match title(d).filter(|t| !t.contains('/') && t.ends_with(".gguf"))
                {
                    Some(t) => t,
                    None if n == 1 => "model.gguf".into(),
                    None => format!("model-{:05}-of-{n:05}.gguf", i + 1),
                };
                Ok(Layer {
                    file_name,
                    sha256: sha.to_string(),
                    size: d.size,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        Ok((digest, layers))
    }
}

struct Challenge {
    realm: String,
    service: Option<String>,
    scope: Option<String>,
}

/// `Bearer realm="...",service="...",scope="..."`
fn parse_challenge(h: &str) -> Option<Challenge> {
    let rest = h
        .trim()
        .strip_prefix("Bearer ")
        .or_else(|| h.trim().strip_prefix("bearer "))?;
    let mut realm = None;
    let mut service = None;
    let mut scope = None;
    // values are quoted and may contain commas (scope lists), so scan
    let mut s = rest;
    while let Some(eq) = s.find('=') {
        let key = s[..eq].trim().trim_start_matches(',').trim();
        let after = &s[eq + 1..];
        let (val, next) = if let Some(q) = after.strip_prefix('"') {
            let end = q.find('"')?;
            (&q[..end], &q[end + 1..])
        } else {
            let end = after.find(',').unwrap_or(after.len());
            (&after[..end], &after[end..])
        };
        match key {
            "realm" => realm = Some(val.to_string()),
            "service" => service = Some(val.to_string()),
            "scope" => scope = Some(val.to_string()),
            _ => {}
        }
        s = next;
    }
    Some(Challenge {
        realm: realm?,
        service,
        scope,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_references() {
        let r = OciRef::parse("registry.ollama.ai/library/qwen3:0.6b").unwrap();
        assert_eq!(
            (
                r.registry.as_str(),
                r.repository.as_str(),
                r.reference.as_str()
            ),
            ("registry.ollama.ai", "library/qwen3", "0.6b")
        );
        let r = OciRef::parse("localhost:5000/models/m").unwrap();
        assert_eq!(
            (r.registry.as_str(), r.reference.as_str()),
            ("localhost:5000", "latest")
        );
        let d = format!("ghcr.io/o/m@sha256:{}", "a".repeat(64));
        assert_eq!(OciRef::parse(&d).unwrap().to_string(), d);
        for bad in [
            "qwen3:0.6b",
            "hf/x:y",
            "ghcr.io/",
            "ghcr.io/O/Upper:t",
            "ghcr.io/o/m:bad tag",
            "ghcr.io/o/m@sha256:zz",
        ] {
            assert!(OciRef::parse(bad).is_none(), "{bad}");
        }
    }

    #[test]
    fn parses_challenges() {
        let c = parse_challenge(r#"Bearer realm="https://ghcr.io/token",service="ghcr.io",scope="repository:o/m:pull,push""#).unwrap();
        assert_eq!(c.realm, "https://ghcr.io/token");
        assert_eq!(c.service.as_deref(), Some("ghcr.io"));
        assert_eq!(c.scope.as_deref(), Some("repository:o/m:pull,push"));
        assert!(parse_challenge("Basic realm=\"x\"").is_none());
    }
}
