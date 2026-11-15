//! Pulls against a local mock of the Hugging Face Hub API.

use aie_store::hf::Hub;
use aie_store::{Error, HfRef, ModelRef, Progress, Store};
use axum::Router;
use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::IntoResponse;
use axum::routing::get;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

const SHA: &str = "0123456789abcdef0123456789abcdef01234567";

#[derive(Default)]
struct Mock {
    files: HashMap<String, Vec<u8>>,
    /// published hashes (may be wrong on purpose)
    hashes: HashMap<String, String>,
    /// requests seen: (path, range header)
    log: Mutex<Vec<(String, Option<String>)>>,
}

fn sha256(b: &[u8]) -> String {
    Sha256::digest(b)
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect()
}

async fn revision() -> impl IntoResponse {
    axum::Json(serde_json::json!({ "sha": SHA, "id": "x" }))
}

async fn tree(State(m): State<Arc<Mock>>) -> impl IntoResponse {
    let mut entries: Vec<serde_json::Value> = m
        .files
        .iter()
        .map(|(p, d)| serde_json::json!({ "type": "file", "path": p, "size": d.len(), "oid": "x", "lfs": { "oid": m.hashes[p], "size": d.len() } }))
        .collect();
    entries.push(serde_json::json!({ "type": "file", "path": "README.md", "size": 3, "oid": "y" }));
    entries.push(serde_json::json!({ "type": "directory", "path": "BF16", "size": 0, "oid": "z" }));
    axum::Json(entries)
}

async fn resolve(
    State(m): State<Arc<Mock>>,
    Path((_o, _r, rev, path)): Path<(String, String, String, String)>,
    h: HeaderMap,
) -> impl IntoResponse {
    assert_eq!(rev, SHA, "downloads must pin the resolved commit");
    let range = h
        .get(header::RANGE)
        .map(|v| v.to_str().unwrap().to_string());
    m.log.lock().unwrap().push((path.clone(), range.clone()));
    let Some(data) = m.files.get(&path) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    if let Some(r) = range {
        let from: usize = r
            .strip_prefix("bytes=")
            .unwrap()
            .trim_end_matches('-')
            .parse()
            .unwrap();
        return (StatusCode::PARTIAL_CONTENT, data[from..].to_vec()).into_response();
    }
    data.clone().into_response()
}

async fn serve(mock: Mock) -> (String, Arc<Mock>) {
    let m = Arc::new(mock);
    let app = Router::new()
        .route("/api/models/{o}/{r}/revision/{rev}", get(revision))
        .route("/api/models/{o}/{r}/tree/{rev}", get(tree))
        .route("/{o}/{r}/resolve/{rev}/{*path}", get(resolve))
        .with_state(m.clone());
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = l.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(l, app).await.unwrap() });
    (format!("http://{addr}"), m)
}

fn mock_with(files: &[(&str, Vec<u8>)]) -> Mock {
    let mut m = Mock::default();
    for (p, d) in files {
        m.hashes.insert(p.to_string(), sha256(d));
        m.files.insert(p.to_string(), d.clone());
    }
    m
}

fn tmp_store(name: &str) -> Store {
    let d = std::env::temp_dir().join(format!("aie-store-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    Store::open(d).unwrap()
}

fn hf(s: &str) -> HfRef {
    match ModelRef::parse(s).unwrap() {
        ModelRef::Hf(r) => r,
        _ => panic!(),
    }
}

fn noop() -> impl FnMut(Progress) + Send {
    |_| {}
}

#[tokio::test]
async fn pulls_default_tag_and_lists() {
    let q4 = vec![7u8; 100_000];
    let (base, mock) = serve(mock_with(&[
        ("M-Q4_K_M.gguf", q4.clone()),
        ("M-Q8_0.gguf", vec![8u8; 10]),
    ]))
    .await;
    let hub = Hub::new(base, None).unwrap();
    let store = tmp_store("default");

    let mut seen = Vec::new();
    let model = store
        .pull(&hub, &hf("hf.co/o/r"), &mut |p: Progress| seen.push(p.done))
        .await
        .unwrap();
    assert_eq!(model.manifest.name, "hf.co/o/r:Q4_K_M");
    assert_eq!(
        std::fs::read(&model.path).unwrap(),
        q4,
        "symlink resolves to the blob"
    );
    assert_eq!(seen.last(), Some(&100_000));
    assert_eq!(mock.log.lock().unwrap().len(), 1);

    // second pull: nothing downloaded
    store
        .pull(&hub, &hf("hf.co/o/r:q4_k_m"), &mut noop())
        .await
        .unwrap();
    assert_eq!(mock.log.lock().unwrap().len(), 1);

    assert!(
        store
            .get(&ModelRef::Hf(hf("hf.co/o/r:Q4_K_M")))
            .unwrap()
            .is_some()
    );
    assert!(
        store
            .get(&ModelRef::Hf(hf("hf.co/o/r:Q8_0")))
            .unwrap()
            .is_none()
    );
    assert_eq!(store.list().unwrap().len(), 1);
}

#[tokio::test]
async fn split_model_keeps_names_for_llama_cpp() {
    let (a, b) = (vec![1u8; 300], vec![2u8; 200]);
    let (base, _) = serve(mock_with(&[
        ("BF16/M-BF16-00002-of-00002.gguf", b.clone()),
        ("BF16/M-BF16-00001-of-00002.gguf", a.clone()),
        ("M-Q4_K_M.gguf", vec![0u8; 5]),
    ]))
    .await;
    let store = tmp_store("split");
    let model = store
        .pull(
            &Hub::new(base, None).unwrap(),
            &hf("hf.co/o/r:bf16"),
            &mut noop(),
        )
        .await
        .unwrap();
    assert!(model.path.ends_with("M-BF16-00001-of-00002.gguf"));
    let sibling = model.path.with_file_name("M-BF16-00002-of-00002.gguf");
    assert_eq!(std::fs::read(sibling).unwrap(), b);
    assert_eq!(model.manifest.files.len(), 2);
    assert_eq!(model.manifest.size(), 500);
}

#[tokio::test]
async fn resumes_a_partial_download() {
    let data: Vec<u8> = (0..50_000u32).map(|i| (i % 251) as u8).collect();
    let (base, mock) = serve(mock_with(&[("M-Q4_K_M.gguf", data.clone())])).await;
    let store = tmp_store("resume");
    let sha = sha256(&data);
    std::fs::write(
        store.blob_path(&sha).with_extension("partial"),
        &data[..20_000],
    )
    .unwrap();

    let model = store
        .pull(
            &Hub::new(base, None).unwrap(),
            &hf("hf.co/o/r"),
            &mut noop(),
        )
        .await
        .unwrap();
    assert_eq!(std::fs::read(&model.path).unwrap(), data);
    assert_eq!(
        mock.log.lock().unwrap()[0].1.as_deref(),
        Some("bytes=20000-")
    );
}

#[tokio::test]
async fn rejects_content_that_does_not_match_its_hash() {
    let mut m = mock_with(&[("M-Q4_K_M.gguf", vec![1u8; 1000])]);
    m.hashes.insert("M-Q4_K_M.gguf".into(), "ab".repeat(32));
    let (base, _) = serve(m).await;
    let store = tmp_store("digest");
    let err = store
        .pull(
            &Hub::new(base, None).unwrap(),
            &hf("hf.co/o/r"),
            &mut noop(),
        )
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Digest { .. }), "{err}");
    assert!(store.list().unwrap().is_empty());
    assert!(
        !store
            .blob_path(&"ab".repeat(32))
            .with_extension("partial")
            .exists()
    );
}

#[tokio::test]
async fn tag_errors_list_what_is_offered() {
    let (base, _) = serve(mock_with(&[
        ("M-Q8_0.gguf", vec![0u8; 4]),
        ("M-Q6_K.gguf", vec![0u8; 4]),
    ]))
    .await;
    let hub = Hub::new(base, None).unwrap();
    let store = tmp_store("tags");
    match store
        .pull(&hub, &hf("hf.co/o/r"), &mut noop())
        .await
        .unwrap_err()
    {
        Error::AmbiguousTag { offered } => assert_eq!(offered, vec!["Q6_K", "Q8_0"]),
        e => panic!("{e}"),
    }
    assert!(matches!(
        store.pull(&hub, &hf("hf.co/o/r:Q4_K_M"), &mut noop()).await,
        Err(Error::NoSuchTag { .. })
    ));
}

#[tokio::test]
async fn remove_collects_unused_blobs_only() {
    let shared = vec![3u8; 64];
    let (base, _) = serve(mock_with(&[
        ("A-Q4_K_M.gguf", shared.clone()),
        ("B-Q8_0.gguf", vec![4u8; 32]),
    ]))
    .await;
    let hub = Hub::new(base, None).unwrap();
    let store = tmp_store("remove");
    store
        .pull(&hub, &hf("hf.co/o/r:Q4_K_M"), &mut noop())
        .await
        .unwrap();
    store
        .pull(&hub, &hf("hf.co/o/r:Q8_0"), &mut noop())
        .await
        .unwrap();
    std::fs::write(store.blob_path("feed").with_extension("partial"), b"x").unwrap();
    assert_eq!(
        store.remove(&ModelRef::Hf(hf("hf.co/o/r:Q8_0"))).unwrap(),
        32
    );
    assert!(
        store
            .get(&ModelRef::Hf(hf("hf.co/o/r:Q4_K_M")))
            .unwrap()
            .is_some()
    );
    assert!(
        store.blob_path("feed").with_extension("partial").exists(),
        "partials survive gc"
    );
    assert!(matches!(
        store.remove(&ModelRef::Hf(hf("hf.co/o/r:Q8_0"))),
        Err(Error::NotFound(_))
    ));
}

#[tokio::test]
async fn unauthorized_is_explained() {
    let app = Router::new().route(
        "/api/models/{o}/{r}/revision/{rev}",
        get(|| async { StatusCode::UNAUTHORIZED }),
    );
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", l.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(l, app).await.unwrap() });
    let err = tmp_store("auth")
        .pull(
            &Hub::new(base, None).unwrap(),
            &hf("hf.co/o/gated"),
            &mut noop(),
        )
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Unauthorized(_)));
    assert!(err.to_string().contains("HF_TOKEN"));
}

/// A mock OCI registry with bearer-token auth (the WWW-Authenticate flow).
mod oci {
    use super::*;
    use axum::http::HeaderValue;

    #[derive(Default)]
    pub struct Reg {
        pub blobs: HashMap<String, Vec<u8>>,
        pub manifest: String,
        pub addr: Mutex<String>,
    }

    async fn token() -> impl IntoResponse {
        axum::Json(serde_json::json!({ "token": "t0k" }))
    }

    #[allow(clippy::result_large_err)] // test helper; boxing would only add noise
    fn authed(h: &HeaderMap, reg: &Reg) -> Result<(), Response> {
        if h.get(header::AUTHORIZATION).and_then(|v| v.to_str().ok()) == Some("Bearer t0k") {
            return Ok(());
        }
        let addr = reg.addr.lock().unwrap().clone();
        let mut r = StatusCode::UNAUTHORIZED.into_response();
        let challenge = format!(
            "Bearer realm=\"http://{addr}/token\",service=\"test\",scope=\"repository:m/q:pull\""
        );
        r.headers_mut().insert(
            header::WWW_AUTHENTICATE,
            HeaderValue::from_str(&challenge).unwrap(),
        );
        Err(r)
    }

    async fn manifest(State(reg): State<Arc<Reg>>, h: HeaderMap) -> Response {
        if let Err(r) = authed(&h, &reg) {
            return r;
        }
        (
            [(
                header::CONTENT_TYPE,
                "application/vnd.oci.image.manifest.v1+json",
            )],
            reg.manifest.clone(),
        )
            .into_response()
    }

    async fn blob(
        State(reg): State<Arc<Reg>>,
        Path((_ns, _repo, digest)): Path<(String, String, String)>,
        h: HeaderMap,
    ) -> Response {
        if let Err(r) = authed(&h, &reg) {
            return r;
        }
        match reg.blobs.get(digest.trim_start_matches("sha256:")) {
            Some(b) => b.clone().into_response(),
            None => StatusCode::NOT_FOUND.into_response(),
        }
    }

    use axum::response::Response;

    pub async fn serve(reg: Reg) -> String {
        let reg = Arc::new(reg);
        let app = Router::new()
            .route("/token", get(token))
            .route("/v2/{ns}/{repo}/manifests/{reference}", get(manifest))
            .route("/v2/{ns}/{repo}/blobs/{digest}", get(blob))
            .with_state(reg.clone());
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = l.local_addr().unwrap().to_string();
        *reg.addr.lock().unwrap() = addr.clone();
        tokio::spawn(async move { axum::serve(l, app).await.unwrap() });
        addr.replace("127.0.0.1", "localhost")
    }
}

#[tokio::test]
async fn pulls_a_split_model_from_an_oci_registry_with_token_auth() {
    let (a, b, cfg) = (vec![5u8; 700], vec![6u8; 300], b"{}".to_vec());
    let layer = |d: &[u8], media: &str, title: &str| {
        serde_json::json!({ "mediaType": media, "digest": format!("sha256:{}", sha256(d)), "size": d.len(),
            "annotations": { "org.opencontainers.image.title": title } })
    };
    let manifest = serde_json::json!({
        "schemaVersion": 2,
        "mediaType": "application/vnd.oci.image.manifest.v1+json",
        "config": { "mediaType": "application/vnd.aie.config.v1+json", "digest": format!("sha256:{}", sha256(&cfg)), "size": 2 },
        // listed out of order: split parts are sorted by title
        "layers": [
            layer(&b, "application/vnd.aie.model.gguf", "q-00002-of-00002.gguf"),
            layer(b"template", "application/vnd.ollama.image.template", "template"),
            layer(&a, "application/vnd.aie.model.gguf", "q-00001-of-00002.gguf"),
        ]
    });
    let mut reg = oci::Reg {
        manifest: manifest.to_string(),
        ..Default::default()
    };
    reg.blobs.insert(sha256(&a), a.clone());
    reg.blobs.insert(sha256(&b), b.clone());
    let addr = oci::serve(reg).await;

    let store = tmp_store("oci");
    let r = match ModelRef::parse(&format!("{addr}/m/q:v1")).unwrap() {
        ModelRef::Oci(r) => r,
        other => panic!("{other:?}"),
    };
    let model = store.pull_oci(&r, &mut noop()).await.unwrap();
    assert!(model.path.ends_with("q-00001-of-00002.gguf"));
    assert_eq!(std::fs::read(&model.path).unwrap(), a);
    assert_eq!(
        std::fs::read(model.path.with_file_name("q-00002-of-00002.gguf")).unwrap(),
        b
    );
    assert_eq!(model.manifest.name, format!("{addr}/m/q:v1"));
    assert!(store.get(&ModelRef::Oci(r.clone())).unwrap().is_some());
    assert_eq!(store.list().unwrap().len(), 1);
    assert_eq!(store.remove(&ModelRef::Oci(r)).unwrap(), 1000);
}

#[tokio::test]
async fn oci_manifest_without_gguf_is_explained() {
    let manifest = serde_json::json!({ "schemaVersion": 2, "mediaType": "application/vnd.oci.image.manifest.v1+json", "layers": [] });
    let addr = oci::serve(oci::Reg {
        manifest: manifest.to_string(),
        ..Default::default()
    })
    .await;
    let ModelRef::Oci(r) = ModelRef::parse(&format!("{addr}/m/q")).unwrap() else {
        panic!()
    };
    let err = tmp_store("oci-empty")
        .pull_oci(&r, &mut noop())
        .await
        .unwrap_err();
    assert!(err.to_string().contains("no GGUF layer"), "{err}");
}
