//! The model store: pulls GGUF models into content-addressed blobs.
//!
//! Layout under a store root (one root per disk; where a model's blobs live is
//! its disk tier):
//!
//! ```text
//! blobs/sha256/<hex>                         file content, verified on download
//! blobs/sha256/<hex>.partial                 an interrupted download, resumed next time
//! manifests/hf.co/<owner>/<repo>/<TAG>.json  what a reference resolved to
//! models/hf.co/<owner>/<repo>/<TAG>/<file>   symlinks with the original file names
//! ```
//!
//! The symlink folder exists because llama.cpp finds the parts of a split
//! model by file name (`-00001-of-00003.gguf`), and blobs are named by hash.
//!
//! One pull of a given blob at a time: the daemon serialises pulls.

pub mod hf;
pub mod oci;
mod reference;

pub use oci::OciRef;
pub use reference::{HfRef, ModelRef, ParseError};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;
use std::path::{Path, PathBuf};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Http(String),
    NotFound(String),
    /// Gated or private repo: set `HF_TOKEN`, and accept the model's terms on the Hub.
    Unauthorized(String),
    NoSuchTag {
        tag: String,
        offered: Vec<String>,
    },
    AmbiguousTag {
        offered: Vec<String>,
    },
    /// Downloaded content doesn't match the published hash.
    Digest {
        file: String,
        expected: String,
        actual: String,
    },
    Manifest(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "{e}"),
            Error::Http(m) => f.write_str(m),
            Error::NotFound(u) => write!(f, "not found: {u}"),
            Error::Unauthorized(u) => {
                write!(
                    f,
                    "not authorised: {u} (private or gated? for Hugging Face set HF_TOKEN; registries: public models only for now)"
                )
            }
            Error::NoSuchTag { tag, offered } => write!(
                f,
                "no {tag} in this repo; it offers: {}",
                offered.join(", ")
            ),
            Error::AmbiguousTag { offered } => {
                write!(
                    f,
                    "this repo has several quantizations; pick one with :TAG ({})",
                    offered.join(", ")
                )
            }
            Error::Digest {
                file,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "{file}: downloaded content has SHA-256 {actual}, expected {expected}"
                )
            }
            Error::Manifest(m) => write!(f, "bad manifest: {m}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

impl From<reqwest::Error> for Error {
    fn from(e: reqwest::Error) -> Self {
        Error::Http(e.to_string())
    }
}

/// Where a model came from.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Source {
    Hf {
        repo: String,
        tag: String,
        revision: String,
    },
    Oci {
        registry: String,
        repository: String,
        reference: String,
        /// Manifest digest the reference resolved to.
        digest: String,
    },
}

impl Source {
    /// (namespace path, tag) under manifests/ and models/. Hugging Face tags
    /// are case-insensitive (upper-cased); OCI tags are case-sensitive.
    fn key(&self) -> (PathBuf, String) {
        match self {
            Source::Hf { repo, tag, .. } => {
                (PathBuf::from("hf.co").join(repo), tag.to_ascii_uppercase())
            }
            Source::Oci {
                registry,
                repository,
                reference,
                ..
            } => (PathBuf::from(registry).join(repository), reference.clone()),
        }
    }
}

/// The store key a reference names, if it names one exactly.
fn ref_key(r: &ModelRef) -> Option<(PathBuf, String)> {
    match r {
        ModelRef::Hf(h) => Some((
            PathBuf::from("hf.co").join(&h.repo),
            h.tag.as_ref()?.to_ascii_uppercase(),
        )),
        ModelRef::Oci(o) => Some((
            PathBuf::from(&o.registry).join(&o.repository),
            o.reference.clone(),
        )),
        ModelRef::Local(_) => None,
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileEntry {
    /// Path in the source repo.
    pub path: String,
    pub sha256: String,
    pub size: u64,
}

/// What a reference resolved to.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    pub schema: u32,
    /// The reference as users write it, e.g. `hf.co/Qwen/Qwen3-30B-A3B-GGUF:Q4_K_M`.
    pub name: String,
    pub source: Source,
    /// Model files; split parts in order.
    pub files: Vec<FileEntry>,
    /// Seconds since the epoch.
    pub pulled_at: u64,
}

impl Manifest {
    pub fn size(&self) -> u64 {
        self.files.iter().map(|f| f.size).sum()
    }
}

/// A model in the store.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Model {
    pub manifest: Manifest,
    /// The file to hand llama.cpp (the first split part).
    pub path: PathBuf,
}

/// Download progress for one file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Progress {
    pub file: String,
    pub done: u64,
    pub total: u64,
}

/// A store rooted in one directory.
#[derive(Clone, Debug)]
pub struct Store {
    root: PathBuf,
}

impl Store {
    pub fn open(root: impl Into<PathBuf>) -> Result<Store, Error> {
        let root = root.into();
        for d in ["blobs/sha256", "manifests", "models"] {
            std::fs::create_dir_all(root.join(d))?;
        }
        Ok(Store { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn blob_path(&self, sha256: &str) -> PathBuf {
        self.root.join("blobs/sha256").join(sha256)
    }

    fn manifest_path(&self, ns: &Path, tag: &str) -> PathBuf {
        self.root
            .join("manifests")
            .join(ns)
            .join(format!("{tag}.json"))
    }

    fn model_dir(&self, ns: &Path, tag: &str) -> PathBuf {
        self.root.join("models").join(ns).join(tag)
    }

    /// Resolve a reference against the Hub and download what's missing.
    pub async fn pull(
        &self,
        hub: &hf::Hub,
        r: &HfRef,
        progress: &mut (dyn FnMut(Progress) + Send),
    ) -> Result<Model, Error> {
        let revision = hub.revision(&r.repo, "main").await?;
        let files = hub.files(&r.repo, &revision).await?;
        let chosen = hf::select(&files, r.tag.as_deref())?;
        let tag = match &r.tag {
            Some(t) => t.clone(),
            None => {
                let offered = hf::tags(&files);
                let stem = hf::stem_of(&chosen[0].path).unwrap_or_default();
                offered
                    .into_iter()
                    .find(|t| stem.to_ascii_uppercase().ends_with(&t.to_ascii_uppercase()))
                    .unwrap_or_default()
            }
        };

        let mut entries = Vec::new();
        for f in &chosen {
            let sha = f.sha256.clone().expect("select() requires hashes");
            let url = hub.file_url(&r.repo, &revision, &f.path);
            self.fetch(&|u| hub.get(u), &url, &f.path, &sha, f.size, progress)
                .await?;
            entries.push(FileEntry {
                path: f.path.clone(),
                sha256: sha,
                size: f.size,
            });
        }

        let manifest = Manifest {
            schema: 1,
            name: HfRef {
                repo: r.repo.clone(),
                tag: Some(tag.clone()),
            }
            .to_string(),
            source: Source::Hf {
                repo: r.repo.clone(),
                tag: tag.clone(),
                revision,
            },
            files: entries,
            pulled_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        };
        self.commit(&manifest)
    }

    /// Resolve a registry reference and download its GGUF layers.
    pub async fn pull_oci(
        &self,
        r: &OciRef,
        progress: &mut (dyn FnMut(Progress) + Send),
    ) -> Result<Model, Error> {
        let client = oci::Client::for_registry(&r.registry)?;
        let (digest, layers) = client.resolve(r).await?;
        let mut entries = Vec::new();
        for l in &layers {
            let url = client.blob_url(r, &format!("sha256:{}", l.sha256));
            self.fetch(
                &|u| client.get(u),
                &url,
                &l.file_name,
                &l.sha256,
                l.size,
                progress,
            )
            .await?;
            entries.push(FileEntry {
                path: l.file_name.clone(),
                sha256: l.sha256.clone(),
                size: l.size,
            });
        }
        let manifest = Manifest {
            schema: 1,
            name: r.to_string(),
            source: Source::Oci {
                registry: r.registry.clone(),
                repository: r.repository.clone(),
                reference: r.reference.clone(),
                digest,
            },
            files: entries,
            pulled_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        };
        self.commit(&manifest)
    }

    /// Download one blob unless present, resuming a partial download, and
    /// verify it against its SHA-256.
    async fn fetch(
        &self,
        get: &(dyn Fn(&str) -> reqwest::RequestBuilder + Sync),
        url: &str,
        name: &str,
        sha256: &str,
        size: u64,
        progress: &mut (dyn FnMut(Progress) + Send),
    ) -> Result<(), Error> {
        let dest = self.blob_path(sha256);
        if tokio::fs::metadata(&dest)
            .await
            .is_ok_and(|m| m.len() == size)
        {
            progress(Progress {
                file: name.into(),
                done: size,
                total: size,
            });
            return Ok(());
        }
        let partial = dest.with_extension("partial");
        let mut hasher = Sha256::new();
        let mut have = match tokio::fs::metadata(&partial).await {
            Ok(m) if m.len() <= size => m.len(),
            _ => 0,
        };
        if have > 0 {
            // hash what's already there so the final check covers the whole file
            let mut f = tokio::fs::File::open(&partial).await?;
            let mut buf = vec![0u8; 1 << 20];
            let mut left = have;
            while left > 0 {
                let want = left.min(buf.len() as u64) as usize;
                let n = f.read(&mut buf[..want]).await?;
                if n == 0 {
                    break;
                }
                hasher.update(&buf[..n]);
                left -= n as u64;
            }
        }

        let mut req = get(url);
        if have > 0 {
            req = req.header(reqwest::header::RANGE, format!("bytes={have}-"));
        }
        let mut resp = hf::check(req.send().await?, url).await?;
        let mut out = if have > 0 && resp.status() == reqwest::StatusCode::PARTIAL_CONTENT {
            tokio::fs::OpenOptions::new()
                .append(true)
                .open(&partial)
                .await?
        } else {
            // server ignored the range: start over
            have = 0;
            hasher = Sha256::new();
            tokio::fs::File::create(&partial).await?
        };
        progress(Progress {
            file: name.into(),
            done: have,
            total: size,
        });
        while let Some(chunk) = resp.chunk().await? {
            hasher.update(&chunk);
            out.write_all(&chunk).await?;
            have += chunk.len() as u64;
            progress(Progress {
                file: name.into(),
                done: have,
                total: size,
            });
        }
        out.flush().await?;
        drop(out);

        let actual = hex(&hasher.finalize());
        if have != size || actual != sha256 {
            tokio::fs::remove_file(&partial).await.ok();
            return Err(if have != size {
                Error::Http(format!("{name}: got {have} bytes, expected {size}"))
            } else {
                Error::Digest {
                    file: name.into(),
                    expected: sha256.into(),
                    actual,
                }
            });
        }
        tokio::fs::rename(&partial, &dest).await?;
        Ok(())
    }

    /// Write the manifest and the symlink folder.
    fn commit(&self, m: &Manifest) -> Result<Model, Error> {
        let (ns, tag) = m.source.key();
        let dir = self.model_dir(&ns, &tag);
        if dir.exists() {
            std::fs::remove_dir_all(&dir)?;
        }
        std::fs::create_dir_all(&dir)?;
        let depth = dir.strip_prefix(&self.root).unwrap().components().count();
        let up: PathBuf = std::iter::repeat_n("..", depth).collect();
        for f in &m.files {
            let base = f.path.rsplit('/').next().unwrap();
            std::os::unix::fs::symlink(up.join("blobs/sha256").join(&f.sha256), dir.join(base))?;
        }
        let mp = self.manifest_path(&ns, &tag);
        std::fs::create_dir_all(mp.parent().unwrap())?;
        let tmp = mp.with_extension("json.tmp");
        std::fs::write(
            &tmp,
            serde_json::to_vec_pretty(m).map_err(|e| Error::Manifest(e.to_string()))?,
        )?;
        std::fs::rename(&tmp, &mp)?;
        self.model_of(m.clone())
    }

    fn model_of(&self, manifest: Manifest) -> Result<Model, Error> {
        let (ns, tag) = manifest.source.key();
        let first = manifest
            .files
            .first()
            .ok_or_else(|| Error::Manifest(format!("{} has no files", manifest.name)))?;
        let path = self
            .model_dir(&ns, &tag)
            .join(first.path.rsplit('/').next().unwrap());
        Ok(Model { manifest, path })
    }

    /// A pulled model, if present and complete.
    pub fn get(&self, r: &ModelRef) -> Result<Option<Model>, Error> {
        let Some((ns, tag)) = ref_key(r) else {
            return Ok(None);
        };
        let p = self.manifest_path(&ns, &tag);
        if !p.exists() {
            return Ok(None);
        }
        let m: Manifest = serde_json::from_slice(&std::fs::read(&p)?)
            .map_err(|e| Error::Manifest(format!("{}: {e}", p.display())))?;
        let complete = m.files.iter().all(|f| {
            std::fs::metadata(self.blob_path(&f.sha256)).is_ok_and(|md| md.len() == f.size)
        });
        Ok(if complete {
            Some(self.model_of(m)?)
        } else {
            None
        })
    }

    /// Every pulled model.
    pub fn list(&self) -> Result<Vec<Model>, Error> {
        let mut out = Vec::new();
        let mut stack = vec![self.root.join("manifests")];
        while let Some(d) = stack.pop() {
            for e in std::fs::read_dir(&d)? {
                let p = e?.path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.extension().is_some_and(|x| x == "json") {
                    let m: Manifest = serde_json::from_slice(&std::fs::read(&p)?)
                        .map_err(|e| Error::Manifest(format!("{}: {e}", p.display())))?;
                    out.push(self.model_of(m)?);
                }
            }
        }
        out.sort_by(|a, b| a.manifest.name.cmp(&b.manifest.name));
        Ok(out)
    }

    /// Remove a model, then any blobs no remaining model uses. Returns bytes freed.
    pub fn remove(&self, r: &ModelRef) -> Result<u64, Error> {
        let Some((ns, tag)) = ref_key(r) else {
            return Err(Error::Manifest(
                "removing needs a tagged hf.co or registry reference".into(),
            ));
        };
        let mp = self.manifest_path(&ns, &tag);
        if !mp.exists() {
            return Err(Error::NotFound(format!("{}", ns.join(&tag).display())));
        }
        std::fs::remove_file(&mp)?;
        let dir = self.model_dir(&ns, &tag);
        if dir.exists() {
            std::fs::remove_dir_all(&dir)?;
        }
        self.gc()
    }

    /// Delete blobs no manifest references. Partial downloads are kept so an
    /// interrupted pull can resume.
    pub fn gc(&self) -> Result<u64, Error> {
        let used: std::collections::HashSet<String> = self
            .list()?
            .into_iter()
            .flat_map(|m| m.manifest.files.into_iter().map(|f| f.sha256))
            .collect();
        let mut freed = 0;
        for e in std::fs::read_dir(self.root.join("blobs/sha256"))? {
            let e = e?;
            let name = e.file_name().to_string_lossy().into_owned();
            if name.ends_with(".partial") {
                continue;
            }
            if !used.contains(&name) {
                freed += e.metadata()?.len();
                std::fs::remove_file(e.path())?;
            }
        }
        Ok(freed)
    }
}

pub(crate) fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}
