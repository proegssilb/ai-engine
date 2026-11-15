#!/usr/bin/env bash
# Build release tarballs: the UI once, then one binary per CPU level (and
# optionally per CUDA level), plus SHA256SUMS. install.sh picks from these.
#
#   tools/release.sh [--version V] [--out DIR] [--levels "v2 v3 v4"] [--cuda "v2 v3 v4"]
#
# Assets are named ai-engine-<version>-linux-x86_64-<level>[-cuda].tar.gz, where
# <level> is the x86-64 microarchitecture level the CPU code is compiled for
# (v2 ~ 2009+, v3 = AVX2/FMA ~ 2015+, v4 = AVX-512 ~ 2017+). Each tarball holds
# ai-engine-<version>-.../ai-engine and ai-engine.service.
#
# Each variant builds in its own target directory (target-release/<variant>):
# llama.cpp's cmake cache would otherwise carry options (CUDA on, an earlier
# CPU level) from one variant into the next. CUDA builds are slow (~20 min,
# several GB each) and need a CUDA 12.x toolkit; 12.4 supports sm_70..sm_90 but
# not newer cards, hence CUDA_ARCHITECTURES below.
set -euo pipefail

cd "$(dirname "$0")/.."
version=""
out="dist-release"
levels="v2 v3 v4"
cuda=""
while [ $# -gt 0 ]; do
  case "$1" in
    --version) version=$2; shift 2 ;;
    --out) out=$2; shift 2 ;;
    --levels) levels=$2; shift 2 ;;
    --cuda) cuda=$2; shift 2 ;;
    -h|--help) sed -n '2,/^set -e/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [ -z "$version" ]; then
  version=$(git describe --tags --always --dirty 2>/dev/null || echo dev)
  version=${version#v}
fi
# Compute-capability 70 (V100) up to 89 (Ada); 12.4 can't target newer cards.
export CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-70;75;80;86;89}"

[ -d vendor/llama.cpp/ggml ] || { echo "vendor/llama.cpp is empty: git submodule update --init --recursive" >&2; exit 1; }

echo "== UI"
(cd ui && npm ci && npx svelte-check --tsconfig ./tsconfig.json && npm run build)

mkdir -p "$out"
out=$(cd "$out" && pwd)
rm -f "$out"/SHA256SUMS

build() { # level, cuda(0|1)
  local level=$1 gpu=$2 suffix=$1
  local features=()
  if [ "$gpu" = 1 ]; then suffix="$level-cuda"; features=(--features cuda); fi
  local name="ai-engine-$version-linux-x86_64-$suffix"
  echo "== $name"
  AIE_CPU_TARGET="x86-64-$level" CARGO_TARGET_DIR="target-release/$suffix" \
    cargo build --release --locked -p ai-engine "${features[@]}"
  local stage
  stage=$(mktemp -d)
  mkdir "$stage/$name"
  cp "target-release/$suffix/release/ai-engine" "$stage/$name/"
  cp contrib/systemd/ai-engine.service "$stage/$name/"
  tar -C "$stage" -czf "$out/$name.tar.gz" "$name"
  rm -rf "$stage"
}

for l in $levels; do build "$l" 0; done
for l in $cuda; do build "$l" 1; done

(cd "$out" && sha256sum ./*.tar.gz | sed 's| \./| |' > SHA256SUMS)
echo "== done"
cat "$out/SHA256SUMS"
