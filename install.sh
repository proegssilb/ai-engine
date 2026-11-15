#!/usr/bin/env bash
# Install ai-engine from a GitHub release.
#
#   curl -fsSL https://raw.githubusercontent.com/proegssilb/ai-engine/main/install.sh | bash
#   ./install.sh [--system] [--service] [--version X.Y.Z] [--cpu-only] [--level v2|v3|v4] [--dir DIR]
#
# It picks the build that matches this machine: the x86-64 level of the CPU
# (v2, v3 = AVX2, v4 = AVX-512) and, if nvidia-smi is present, the CUDA build.
# The download is checked against the release's SHA256SUMS. Rerunning upgrades
# in place.
#
#   --system     install to /usr/local/bin (uses sudo when not root); default ~/.local/bin
#   --service    also install and start the systemd service (implies --system):
#                creates the ai-engine user, data in /var/lib/ai-engine
#   --version    release to install (default: the latest)
#   --cpu-only   skip the CUDA build even if an NVIDIA GPU is present
#   --level      override the detected CPU level
#   --dir        install directory (overrides the two defaults)
#
# Environment:
#   AIE_RELEASE_BASE   where releases live (default
#                      https://github.com/proegssilb/ai-engine/releases); expects
#                      <base>/latest/download/<asset> and <base>/download/v<ver>/<asset>.
#                      Any http(s):// or file:// location with that layout works, which
#                      is how to install from a private repo mirror.
#   AIE_CURL_AUTH      optional header, e.g. "Authorization: Bearer <token>"
#   AIE_CPUINFO        file to read CPU flags from (default /proc/cpuinfo; for tests)
set -euo pipefail

base=${AIE_RELEASE_BASE:-https://github.com/proegssilb/ai-engine/releases}
base=${base%/}
version=""
system=0
service=0
cpu_only=0
level=""
dir=""

while [ $# -gt 0 ]; do
  case "$1" in
    --system) system=1; shift ;;
    --service) service=1; system=1; shift ;;
    --version) version=${2#v}; shift 2 ;;
    --cpu-only) cpu_only=1; shift ;;
    --level) level=${2#x86-64-}; shift 2 ;;
    --dir) dir=$2; shift 2 ;;
    -h|--help) sed -n '2,/^set -e/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

say() { printf '%s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

[ "$(uname -s)" = Linux ] || die "only Linux is supported"
[ "$(uname -m)" = x86_64 ] || die "only x86-64 builds are published (this machine is $(uname -m))"
for c in curl tar sha256sum; do command -v "$c" >/dev/null || die "$c is required"; done

# The highest x86-64 level whose required flags the CPU has.
detect_level() {
  local flags f l
  flags=$(grep -m1 '^flags' "${AIE_CPUINFO:-/proc/cpuinfo}" || true)
  flags=" ${flags#*:} "
  has_all() { for f in "$@"; do case "$flags" in *" $f "*) ;; *) return 1 ;; esac; done; }
  if has_all avx512f avx512bw avx512cd avx512dq avx512vl avx avx2 bmi1 bmi2 f16c fma abm movbe cx16 popcnt sse4_1 sse4_2 ssse3; then l=v4
  elif has_all avx avx2 bmi1 bmi2 f16c fma abm movbe cx16 popcnt sse4_1 sse4_2 ssse3; then l=v3
  elif has_all cx16 popcnt sse4_1 sse4_2 ssse3; then l=v2
  else die "this CPU lacks SSE4.2; no build supports it"
  fi
  printf '%s' "$l"
}

[ -n "$level" ] || level=$(detect_level)
case "$level" in v2|v3|v4) ;; *) die "unknown CPU level '$level' (v2, v3 or v4)" ;; esac
gpu=0
if [ "$cpu_only" = 0 ] && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then gpu=1; fi
say "CPU level: x86-64-$level; NVIDIA GPU: $([ "$gpu" = 1 ] && echo yes || echo no)"

if [ -n "$version" ]; then url="$base/download/v$version"; else url="$base/latest/download"; fi
curl_get() { # url dest
  if [ -n "${AIE_CURL_AUTH:-}" ]; then curl -fsSL -H "$AIE_CURL_AUTH" -o "$2" "$1"; else curl -fsSL -o "$2" "$1"; fi
}

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
curl_get "$url/SHA256SUMS" "$tmp/SHA256SUMS" || die "could not download $url/SHA256SUMS (is there a release? private repo? see AIE_RELEASE_BASE / AIE_CURL_AUTH)"

# Candidates, best first: CUDA at this level or lower, then plain CPU the same way.
# A lower level always runs on a higher CPU.
levels_down() { case "$1" in v4) echo "v4 v3 v2" ;; v3) echo "v3 v2" ;; *) echo "v2" ;; esac; }
asset=""
for pass in cuda cpu; do
  [ "$pass" = cuda ] && [ "$gpu" = 0 ] && continue
  for l in $(levels_down "$level"); do
    suffix=$l; [ "$pass" = cuda ] && suffix="$l-cuda"
    a=$(awk -v s="-linux-x86_64-$suffix.tar.gz" '{ n=length($2); if (substr($2, n-length(s)+1) == s) print $2 }' "$tmp/SHA256SUMS" | head -n1)
    if [ -n "$a" ]; then asset=$a; break 2; fi
  done
done
[ -n "$asset" ] || die "no matching build in $url/SHA256SUMS"
say "Downloading $asset"
curl_get "$url/$asset" "$tmp/$asset"
(cd "$tmp" && grep -F " $asset" SHA256SUMS | sha256sum -c --quiet -) || die "checksum mismatch for $asset"
say "Checksum OK"
tar -C "$tmp" -xzf "$tmp/$asset"
pkg=$tmp/${asset%.tar.gz}
[ -x "$pkg/ai-engine" ] || die "unexpected archive layout"

# Where to put it.
sudo=""
if [ "$system" = 1 ] && [ "$(id -u)" != 0 ]; then
  command -v sudo >/dev/null || die "--system needs root or sudo"
  sudo=sudo
fi
if [ -z "$dir" ]; then
  if [ "$system" = 1 ]; then dir=/usr/local/bin; else dir=$HOME/.local/bin; fi
fi
$sudo mkdir -p "$dir"
# Copy next to the target, then rename: replacing a binary that is running
# fails with "text file busy", but renaming over it is fine.
$sudo install -m755 "$pkg/ai-engine" "$dir/.ai-engine.new"
$sudo mv -f "$dir/.ai-engine.new" "$dir/ai-engine"
say "Installed $dir/ai-engine"
case ":$PATH:" in *":$dir:"*) ;; *) say "Note: $dir is not on your PATH." ;; esac

if [ "$service" = 1 ]; then
  command -v systemctl >/dev/null || die "--service needs systemd"
  if ! id ai-engine >/dev/null 2>&1; then
    $sudo useradd --system --home /var/lib/ai-engine --create-home --shell /usr/sbin/nologin ai-engine
    say "Created user ai-engine"
  fi
  # GPU device access, for the groups that exist.
  for g in video render; do
    if getent group "$g" >/dev/null; then $sudo usermod -aG "$g" ai-engine; fi
  done
  $sudo mkdir -p /var/lib/ai-engine
  $sudo chown ai-engine:ai-engine /var/lib/ai-engine
  # The unit's ExecStart follows the install directory; AIE_HOME is set in the unit.
  sed "s|^ExecStart=.*ai-engine serve|ExecStart=$dir/ai-engine serve|" "$pkg/ai-engine.service" > "$tmp/ai-engine.service"
  $sudo install -m644 "$tmp/ai-engine.service" /etc/systemd/system/ai-engine.service
  $sudo systemctl daemon-reload
  $sudo systemctl enable ai-engine >/dev/null 2>&1
  $sudo systemctl restart ai-engine
  say "Service started: systemctl status ai-engine (data in /var/lib/ai-engine, API on 127.0.0.1:11435)"
else
  say "Try it: ai-engine serve   (API on 127.0.0.1:11435, data in ~/.ai-engine)"
fi
