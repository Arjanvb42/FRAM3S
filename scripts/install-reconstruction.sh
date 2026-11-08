#!/usr/bin/env bash
# Installs the scan tools FRAM3 needs on Fedora: COLMAP (camera poses) and
# OpenMVS (dense cloud, mesh, texture). Both are built with CUDA when the NVIDIA
# driver + CUDA toolkit are present. Installs into ~/.local/fram3-tools.
set -euo pipefail
PREFIX=${PREFIX:-$HOME/.local/fram3-tools}
SRC=${SRC:-$HOME/.cache/fram3-build}
JOBS=${JOBS:-$(( $(nproc) > 2 ? $(nproc) - 1 : 1 ))}
mkdir -p "$PREFIX" "$SRC"

echo "== 1/4 Bouwpakketten installeren (vraagt om je wachtwoord) =="
sudo dnf install -y gcc-c++ cmake ninja-build git \
  boost-devel eigen3-devel flann-devel freeimage-devel metis-devel glog-devel gflags-devel \
  sqlite-devel glew-devel qt5-qtbase-devel cgal-devel ceres-solver-devel suitesparse-devel \
  openblas-devel lapack-devel opencv-devel libjpeg-turbo-devel libpng-devel libtiff-devel \
  zlib-devel vcglib-devel || true

CUDA_ARGS=()
if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
  if ! command -v nvcc >/dev/null && [ ! -x /usr/local/cuda/bin/nvcc ]; then
    echo "== CUDA-toolkit (nvcc) ontbreekt: installeren vanuit NVIDIA-repo =="
    FEDORA=$(rpm -E %fedora)
    REPO_FEDORA=$FEDORA
    until curl -fsI "https://developer.download.nvidia.com/compute/cuda/repos/fedora${REPO_FEDORA}/x86_64/cuda-fedora${REPO_FEDORA}.repo" >/dev/null 2>&1 || [ "$REPO_FEDORA" -lt 39 ]; do
      REPO_FEDORA=$((REPO_FEDORA-1))
    done
    sudo dnf config-manager addrepo --from-repofile="https://developer.download.nvidia.com/compute/cuda/repos/fedora${REPO_FEDORA}/x86_64/cuda-fedora${REPO_FEDORA}.repo" || true
    # Only the toolkit: the driver stays the RPM Fusion one.
    sudo dnf install -y cuda-toolkit --exclude='nvidia-driver*,kmod-nvidia*,*-nvidia-*' || echo "LET OP: CUDA-toolkit niet geïnstalleerd, bouwen zonder CUDA."
  fi
  export PATH=/usr/local/cuda/bin:$PATH
  if command -v nvcc >/dev/null; then
    # Turing (T2000 = sm_75); GCC 16 is newer than nvcc supports, so let nvcc accept it.
    CUDA_ARGS=(-DCUDA_ENABLED=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_FLAGS=--allow-unsupported-compiler)
    echo "CUDA gevonden: $(nvcc --version | tail -1)"
  fi
fi
[ ${#CUDA_ARGS[@]} -eq 0 ] && CUDA_ARGS=(-DCUDA_ENABLED=OFF) && echo "Geen CUDA: bouwen voor CPU."

echo "== 2/4 COLMAP bouwen =="
[ -d "$SRC/colmap" ] || git clone --depth 1 --branch 3.11.1 https://github.com/colmap/colmap.git "$SRC/colmap"
cmake -S "$SRC/colmap" -B "$SRC/colmap/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" -DGUI_ENABLED=OFF "${CUDA_ARGS[@]}"
cmake --build "$SRC/colmap/build" -j "$JOBS"
cmake --install "$SRC/colmap/build"

echo "== 3/4 OpenMVS bouwen =="
if [ ! -d "$SRC/vcglib" ]; then git clone --depth 1 https://github.com/cdcseacave/VCG.git "$SRC/vcglib"; fi
[ -d "$SRC/openMVS" ] || git clone --depth 1 --recurse-submodules --branch v2.3.0 https://github.com/cdcseacave/openMVS.git "$SRC/openMVS"
MVS_CUDA=OFF; [[ "${CUDA_ARGS[*]}" == *ON* ]] && MVS_CUDA=ON
cmake -S "$SRC/openMVS" -B "$SRC/openMVS/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" -DVCG_ROOT="$SRC/vcglib" -DOpenMVS_USE_CUDA=$MVS_CUDA \
  -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_FLAGS=--allow-unsupported-compiler -DOpenMVS_BUILD_VIEWER=OFF
cmake --build "$SRC/openMVS/build" -j "$JOBS"
cmake --install "$SRC/openMVS/build"

echo "== 4/4 FRAM3 instellen =="
MVS_BIN=$(dirname "$(find "$PREFIX" -name DensifyPointCloud -type f | head -1)")
mkdir -p "$HOME/.config/fram3"
cat > "$HOME/.config/fram3/env" <<CFG
FRAM3_COLMAP=$PREFIX/bin/colmap
FRAM3_OPENMVS_DIR=$MVS_BIN
CFG
echo "Klaar. COLMAP: $PREFIX/bin/colmap  OpenMVS: $MVS_BIN"
