//! Daemon configuration: defaults, overridden by `$AIE_HOME/config.toml`,
//! overridden by the environment.

use serde::Deserialize;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

#[derive(Clone, Debug)]
pub struct Config {
    /// Where the API listens. Localhost by default; no built-in auth
    /// (features.md): put a gateway in front for anything else.
    pub listen: SocketAddr,
    /// Everything the daemon keeps: store, placements, sockets, logs.
    pub home: PathBuf,
    /// Model store roots, one per disk (default `$AIE_HOME/store`). A model's
    /// store is its disk tier; pulls go to the first unless told otherwise.
    pub stores: Vec<PathBuf>,
    /// How long an idle model stays loaded (30 minutes by default; ollama's
    /// is 5). A request's own `keep_alive` overrides it.
    pub keep_alive: Duration,
    /// Context length when a request doesn't say.
    pub n_ctx: u32,
    /// Prompt tokens per batch.
    pub n_batch: u32,
    /// CPU threads per worker (0 = all cores).
    pub threads: i32,
    /// How long a worker may take to load and check its placement.
    pub load_timeout: Duration,
    /// Start workers with swap forbidden (systemd user scope) when possible,
    /// so the RAM tier stays in RAM without mlock.
    pub forbid_worker_swap: bool,
    /// The executable to start workers with (`<exe> worker ...`); the daemon's own by default.
    pub worker_exe: PathBuf,
}

/// The `config.toml` file; every field optional.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct File {
    listen: Option<SocketAddr>,
    store: Option<PathBuf>,
    stores: Option<Vec<PathBuf>>,
    keep_alive: Option<String>,
    n_ctx: Option<u32>,
    n_batch: Option<u32>,
    threads: Option<i32>,
    load_timeout: Option<String>,
    forbid_worker_swap: Option<bool>,
}

pub const DEFAULT_PORT: u16 = 11435;

impl Config {
    /// Defaults under `home`.
    pub fn new(home: PathBuf) -> Config {
        Config {
            listen: SocketAddr::from(([127, 0, 0, 1], DEFAULT_PORT)),
            stores: vec![home.join("store")],
            home,
            keep_alive: Duration::from_secs(30 * 60),
            n_ctx: 4096,
            n_batch: 512,
            threads: 0,
            load_timeout: Duration::from_secs(1800),
            forbid_worker_swap: true,
            worker_exe: std::env::current_exe().unwrap_or_else(|_| PathBuf::from("ai-engine")),
        }
    }

    /// `$AIE_HOME` (default `~/.ai-engine`), its `config.toml`, then `AIE_HOST`.
    pub fn load() -> Result<Config, String> {
        let home = std::env::var_os("AIE_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".ai-engine")))
            .ok_or("set AIE_HOME or HOME")?;
        let mut c = Config::new(home);
        let path = c.home.join("config.toml");
        if path.exists() {
            let text =
                std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            let f: File = toml::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
            if let Some(v) = f.listen {
                c.listen = v;
            }
            if let Some(v) = f.store {
                c.stores = vec![v];
            }
            if let Some(v) = f.stores {
                if v.is_empty() {
                    return Err(format!("{}: stores is empty", path.display()));
                }
                c.stores = v;
            }
            if let Some(v) = f.keep_alive {
                c.keep_alive =
                    parse_duration(&v).ok_or(format!("keep_alive: bad duration {v:?}"))?;
            }
            if let Some(v) = f.n_ctx {
                c.n_ctx = v;
            }
            if let Some(v) = f.n_batch {
                c.n_batch = v;
            }
            if let Some(v) = f.threads {
                c.threads = v;
            }
            if let Some(v) = f.forbid_worker_swap {
                c.forbid_worker_swap = v;
            }
            if let Some(v) = f.load_timeout {
                c.load_timeout =
                    parse_duration(&v).ok_or(format!("load_timeout: bad duration {v:?}"))?;
            }
        }
        if let Ok(h) = std::env::var("AIE_HOST") {
            c.listen = parse_host(&h).ok_or(format!("AIE_HOST: bad address {h:?}"))?;
        }
        Ok(c)
    }
}

/// `host:port`, `host` (default port) or `:port`.
pub fn parse_host(s: &str) -> Option<SocketAddr> {
    let s = s.trim().trim_start_matches("http://");
    if let Ok(a) = s.parse() {
        return Some(a);
    }
    if let Some(port) = s.strip_prefix(':') {
        return Some(SocketAddr::from(([127, 0, 0, 1], port.parse().ok()?)));
    }
    format!("{s}:{DEFAULT_PORT}").parse().ok()
}

/// ollama-style durations: `"5m"`, `"1h30m"`, `"90s"`, `"300"` (seconds).
pub fn parse_duration(s: &str) -> Option<Duration> {
    let s = s.trim();
    if let Ok(n) = s.parse::<f64>() {
        return (n >= 0.0).then(|| Duration::from_secs_f64(n));
    }
    let mut total = 0.0;
    let mut num = String::new();
    let mut chars = s.chars().peekable();
    let mut any = false;
    while let Some(c) = chars.next() {
        if c.is_ascii_digit() || c == '.' {
            num.push(c);
            continue;
        }
        let unit = if c == 'm' && chars.peek() == Some(&'s') {
            chars.next();
            0.001
        } else {
            match c {
                'h' => 3600.0,
                'm' => 60.0,
                's' => 1.0,
                _ => return None,
            }
        };
        total += num.parse::<f64>().ok()? * unit;
        num.clear();
        any = true;
    }
    (any && num.is_empty()).then(|| Duration::from_secs_f64(total))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn durations() {
        assert_eq!(parse_duration("5m"), Some(Duration::from_secs(300)));
        assert_eq!(parse_duration("1h30m"), Some(Duration::from_secs(5400)));
        assert_eq!(parse_duration("90"), Some(Duration::from_secs(90)));
        assert_eq!(parse_duration("250ms"), Some(Duration::from_millis(250)));
        assert_eq!(parse_duration("5x"), None);
        assert_eq!(parse_duration("5"), Some(Duration::from_secs(5)));
        assert_eq!(parse_duration("m"), None);
    }

    #[test]
    fn hosts() {
        assert_eq!(
            parse_host("0.0.0.0:8080"),
            Some(SocketAddr::from(([0, 0, 0, 0], 8080)))
        );
        assert_eq!(
            parse_host(":9000"),
            Some(SocketAddr::from(([127, 0, 0, 1], 9000)))
        );
        assert_eq!(
            parse_host("127.0.0.1"),
            Some(SocketAddr::from(([127, 0, 0, 1], DEFAULT_PORT)))
        );
        assert_eq!(
            parse_host("http://127.0.0.1:1"),
            Some(SocketAddr::from(([127, 0, 0, 1], 1)))
        );
    }
}
