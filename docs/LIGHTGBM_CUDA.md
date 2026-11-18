# LightGBM with CUDA on this machine

Build record and measurements, 2026-11-18. The build is additive: the repo's
`pyproject.toml` and `uv.lock` still pin the PyPI wheel, which is CPU only. The
CUDA wheel lives outside the repo and is installed only in its own venv.

## Repo measurement: the real cluster52 fit

`data/interim/prep52_{X,y,w}.npy` holds the 20,006,240 x 13 grouped risk set,
85/15 contiguous split, `num_leaves=63`, `learning_rate=0.05`, 400 rounds with
early stopping 50, `seed=7`.

| Run | Wall | Iterations | Validation logloss | Peak VRAM |
|---|---|---|---|---|
| CPU, 4 threads | 99.0 s | 219 | 0.013864634 | none |
| CUDA, RTX 3050 4 GB | 45.3 s | 206 | 0.013871638 | 2,193 MiB (407 idle) |
| CUDA, repeat | 43.9 s | 206 | 0.013871638 | same |
| CPU vs CUDA predictions | | | | max abs diff 0.0323 |

The 2.19x wall gain is 2.06x per iteration. The iteration count differs (219 against
206) because greedy tie-breaking changes which round early stopping picks.

Two repeat CUDA fits on the same rows agree to `4.5e-08` on 100,000 predictions,
but not bitwise. CUDA against CPU diverges by up to `0.032` on the same predictions,
with a logloss gap of `7e-06`.

Policy: the exact-first gates in `AGENTS.md` stay on the CPU wheel. A CUDA fit is
for exploratory sweeps only, and its result is reported with the tolerance above,
never as a reproduction of a committed number.

# LightGBM 4.7.0 with CUDA — build record

Built 2026-11-18 09:28 UTC on this machine, isolated in /tmp/lgb-cuda.
Nothing was installed into /home/ibrahim/uni/cache or its .venv.

## Artifact
- wheel: /tmp/lgb-cuda/wheels/lightgbm-4.7.0-py3-none-linux_x86_64.whl
- size: 11,492,202 bytes
- sha256: efab323e009145a4281f14c33272d248bf297c8094d37c6ec18f3d96f3e3191c
- installed into: /tmp/lgb-cuda/.venv

## Toolchain actually used
- CUDA toolkit: **12.4.131** at /usr/lib/nvidia-cuda-toolkit (Ubuntu package nvidia-cuda-toolkit +
  nvidia-cuda-dev). Debian layout: bin/nvcc, headers in /usr/include, libs in /usr/lib/x86_64-linux-gnu.
  /usr/local/cuda -> cuda-13.3 (13.3.73) was NOT used.
- Assembled a standard-layout CUDA root by symlink: /tmp/lgb-cuda/cu124/{bin,include,lib64,nvvm/libdevice}
- host compiler: gcc-13 / g++-13 (system default is gcc 15.2, unsupported by nvcc 12.4)
- CMAKE_CUDA_ARCHITECTURES=86 (RTX 3050 Laptop, sm_86, 4 GB)
- cmake 3.31.6 (wheel), ninja 1.13.2 (wheel), scikit-build-core 1.1.2
- CMAKE_BUILD_PARALLEL_LEVEL=4

## NCCL
LightGBM 4.x CUDA builds require NCCL (CMakeLists.txt: `find_package(NCCL REQUIRED)`).
No system NCCL, no sudo. Used the pip wheel nvidia-nccl-cu12==2.32.6 (shared libnccl.so.2 only).
Wheel ships libnccl.so.2 but no unversioned libnccl.so, so FindNCCL misses it; fixed with a symlink root:
/tmp/lgb-cuda/nccl/lib/libnccl.so -> .venv/.../nvidia/nccl/lib/libnccl.so.2
/tmp/lgb-cuda/nccl/include      -> .venv/.../nvidia/nccl/include
Built with -DBUILD_WITH_SHARED_NCCL=ON (no static lib available).
The wheel's lib_lightgbm.so has RUNPATH = the venv nccl dir, so `import lightgbm` works with a clean env.

## Second fix
src/c_api.cpp lives in target lightgbm_capi_objs, which does not link NCCL::NCCL, so nccl.h was not on
its include path (upstream relies on NCCL headers being in a default include dir). Supplied via
-DCMAKE_CXX_FLAGS=-I/tmp/lgb-cuda/nccl/include instead of editing the source.

## Command
CMAKE_ARGS="-DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DBUILD_WITH_SHARED_NCCL=ON   -DCMAKE_CXX_FLAGS=-I/tmp/lgb-cuda/nccl/include -DCMAKE_CUDA_FLAGS=-I/tmp/lgb-cuda/nccl/include   -DCMAKE_INSTALL_RPATH=<venv>/lib/python3.13/site-packages/nvidia/nccl/lib   -DCMAKE_MODULE_LINKER_FLAGS=-Wl,-rpath,<same> -DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath,<same>" \
  CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 CUDACXX=/tmp/lgb-cuda/cu124/bin/nvcc \
  CUDAToolkit_ROOT=/tmp/lgb-cuda/cu124 NCCL_ROOT=/tmp/lgb-cuda/nccl \
  CMAKE_BUILD_PARALLEL_LEVEL=4 \
  .venv/bin/python -m build --wheel --no-isolation --outdir /tmp/lgb-cuda/wheels .
(see /tmp/lgb-cuda/build.sh)

## Timings
- successful build: 135 s total (configure 4.2 s, 76 ninja targets, wheel assembly)
- earlier failed attempts: 4 s (NCCL not found), 124 s (nccl.h missing in c_api.cpp), 190 s (no install rpath)
- full compile of all 15 .cu files + 54 C/C++ objects: about 2 min at -j4

## Verification (see verify_result.json, parity.json, big.json)
- 200k x 20 binary, 200 rounds, num_leaves=63: CPU 5.47 s / CUDA 4.52 s, GPU util 99%,
  peak VRAM 672 MiB (407 MiB idle baseline)
- 1M x 50, 100 rounds: CPU 33.19 s / CUDA 4.28 s = 7.75x, peak VRAM 659 MiB, test AUC identical
  (0.945931 both)
- predictions are NOT bitwise identical to CPU: 200k/200-round max |dp| = 0.24, mean 0.017,
  corr 0.9973, held-out AUC delta 1.2e-4
- cause: the CUDA histogram uses atomics, so the CUDA learner is not run-to-run reproducible:
  two identical 20-round CUDA fits differ by 7.4e-10. First 20 trees have identical split
  feature/threshold lists vs CPU; divergence accumulates through greedy tie-breaking.
- 'deterministic=true' does not remove this (it only pins CPU histogram/feature order).
- LightGBM 4.7 prints no "Using CUDA Tree Learner" line; the failure mode of a non-CUDA build is
  LightGBMError "CUDA Tree Learner was not enabled in this build" from cuda_single_gpu_tree_learner.hpp.
- Informational log line on CUDA: "[LightGBM] [Warning] Using sparse features with CUDA is currently
  not supported." and "Metric auc is not implemented in cuda version. Fall back to evaluation on CPU."
