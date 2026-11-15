//! Several model stores, one per disk: where a model's file lives is its
//! disk tier, so users choose the disk when they pull (features.md).

use aie_store::{Error, Model, ModelRef, Store};
use std::path::{Path, PathBuf};

pub struct Stores(Vec<Store>);

impl Stores {
    pub fn open(roots: &[PathBuf]) -> Result<Stores, String> {
        if roots.is_empty() {
            return Err("no model store configured".into());
        }
        roots
            .iter()
            .map(|r| Store::open(r).map_err(|e| format!("{}: {e}", r.display())))
            .collect::<Result<Vec<_>, _>>()
            .map(Stores)
    }

    pub fn roots(&self) -> impl Iterator<Item = &Path> {
        self.0.iter().map(|s| s.root())
    }

    /// The store to pull into: by root path, or the first (default).
    pub fn pick(&self, root: Option<&str>) -> Result<&Store, String> {
        match root {
            None | Some("") => Ok(&self.0[0]),
            Some(r) => {
                let want = std::fs::canonicalize(r).unwrap_or_else(|_| PathBuf::from(r));
                self.0
                    .iter()
                    .find(|s| {
                        std::fs::canonicalize(s.root()).unwrap_or_else(|_| s.root().to_path_buf())
                            == want
                    })
                    .ok_or_else(|| {
                        let known: Vec<String> =
                            self.roots().map(|p| p.display().to_string()).collect();
                        format!("{r} isn't a configured store; stores: {}", known.join(", "))
                    })
            }
        }
    }

    /// A model from whichever store has it (first match).
    pub fn get(&self, r: &ModelRef) -> Result<Option<Model>, Error> {
        for s in &self.0 {
            if let Some(m) = s.get(r)? {
                return Ok(Some(m));
            }
        }
        Ok(None)
    }

    /// Every model in every store, with its store root.
    pub fn list_with_root(&self) -> Result<Vec<(Model, PathBuf)>, Error> {
        let mut out = Vec::new();
        for s in &self.0 {
            out.extend(s.list()?.into_iter().map(|m| (m, s.root().to_path_buf())));
        }
        out.sort_by(|a, b| a.0.manifest.name.cmp(&b.0.manifest.name));
        Ok(out)
    }

    pub fn list(&self) -> Result<Vec<Model>, Error> {
        Ok(self.list_with_root()?.into_iter().map(|(m, _)| m).collect())
    }

    /// Remove a model from every store that has it. Returns bytes freed.
    pub fn remove(&self, r: &ModelRef) -> Result<u64, Error> {
        let mut freed = 0;
        let mut found = false;
        for s in &self.0 {
            if s.get(r)?.is_some() {
                freed += s.remove(r)?;
                found = true;
            }
        }
        if !found {
            return Err(Error::NotFound(format!("{r:?}")));
        }
        Ok(freed)
    }
}
