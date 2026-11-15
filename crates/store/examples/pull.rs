//! Pull a model into a store directory, or show what a reference resolves to.
//!
//!     cargo run -p aie-store --example pull -- [--dry-run] STORE_DIR hf.co/OWNER/REPO[:TAG] | REGISTRY/REPO[:TAG]

use aie_store::hf::{self, Hub};
use aie_store::{ModelRef, Store};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let dry = args.first().is_some_and(|a| a == "--dry-run");
    if dry {
        args.remove(0);
    }
    let [dir, reference] = args.as_slice() else {
        eprintln!("usage: pull [--dry-run] STORE_DIR hf.co/OWNER/REPO[:TAG]");
        std::process::exit(2)
    };
    let r = match ModelRef::parse(reference)? {
        ModelRef::Hf(r) => r,
        ModelRef::Oci(o) => {
            let client = aie_store::oci::Client::for_registry(&o.registry)?;
            if dry {
                let (digest, layers) = client.resolve(&o).await?;
                println!("{o} @ {digest}");
                for l in layers {
                    println!(
                        "  {}  {:.2} GiB  sha256 {}",
                        l.file_name,
                        l.size as f64 / (1u64 << 30) as f64,
                        l.sha256
                    );
                }
                return Ok(());
            }
            let m = Store::open(dir)?.pull_oci(&o, &mut |_| {}).await?;
            println!("{} -> {}", m.manifest.name, m.path.display());
            return Ok(());
        }
        ModelRef::Local(_) => return Err("local files don't need pulling".into()),
    };
    let hub = Hub::public()?;
    if dry {
        let rev = hub.revision(&r.repo, "main").await?;
        let files = hub.files(&r.repo, &rev).await?;
        println!("{} @ {rev}", r.repo);
        println!("tags: {}", hf::tags(&files).join(" "));
        for f in hf::select(&files, r.tag.as_deref())? {
            println!(
                "  {}  {:.2} GiB  sha256 {}",
                f.path,
                f.size as f64 / (1u64 << 30) as f64,
                f.sha256.as_deref().unwrap_or("-")
            );
        }
        return Ok(());
    }
    let store = Store::open(dir)?;
    let mut last = 0u64;
    let model = store
        .pull(&hub, &r, &mut |p| {
            if p.done == p.total || p.done - last > 256 << 20 {
                last = p.done;
                eprintln!(
                    "{}: {:.0}%",
                    p.file,
                    100.0 * p.done as f64 / p.total.max(1) as f64
                );
            }
        })
        .await?;
    println!("{} -> {}", model.manifest.name, model.path.display());
    Ok(())
}
