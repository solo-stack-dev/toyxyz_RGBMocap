# AMD Radeon build

This fork of [toyxyz/toyxyz_RGBMocap](https://github.com/toyxyz/toyxyz_RGBMocap)
runs on AMD Radeon GPUs instead of NVIDIA GPUs. Every AI model runtime uses
AMD's PyTorch build for Windows (ROCm) in place of PyTorch for CUDA. Use the
upstream repository on an NVIDIA PC.

## Requirements

- 64-bit Windows 11 and OpenGL 4.3 for the viewport.
- An AMD Radeon RX 7900 XTX, 7900 XT or 7900 GRE, or a Radeon PRO W7900/W7800
  (GPU target `gfx1100`). The pinned ROCm kernels cover this target only; see
  [other Radeon GPUs](#other-radeon-gpus).
- A current AMD Adrenalin driver. No separate ROCm or HIP SDK installation is
  needed: the ROCm runtime ships inside the pinned Python packages.
- Install into a short folder path such as `C:\RGBMoCap` or `D:\repos\RGBMoCap`.
  ROCm's kernel libraries nest about 180 characters deep below the install
  folder, and Windows rejects paths over 260 characters unless long paths are
  enabled.

## What changed from the NVIDIA version

| Area | NVIDIA upstream | This fork |
| --- | --- | --- |
| PyTorch | 2.7.1+cu128 (Python 3.10, 3.11), 2.8.0+cu128 and 2.10.0+cu128 (Python 3.12) | 2.14.0+rocm10.1.0 for every runtime, with torchvision 0.29.0a0+rocm10.1.0 |
| GPU libraries | CUDA DLLs inside the torch wheel | `rocm-sdk-core`, `rocm-sdk-libraries` and `rocm-sdk-device-gfx1100` wheels from `https://stable.repo.amd.com/rocm/whl-next/`, pinned by SHA-256 |
| `rocm` meta package | none | AMD publishes it only as source; `tools/make_rocm_meta_source.py` repacks its pure-Python `rocm_sdk` module (without running its setup script) into `installer/runtime_sources/rocm-10.1.0-py3-none-any/`, which the installer turns into a bundled wheel |
| GPU choice | CUDA device 0 | `mocap/amd_gpu.py` hides GPUs without kernels (often the Ryzen's integrated Radeon, which ROCm lists first) by setting `HIP_VISIBLE_DEVICES` at launch and in every setup script |
| GEM-X | SOMA skinning and skeleton fit on NVIDIA Warp | `mocap/soma_rocm.py` switches SOMA to its built-in PyTorch path; Warp stays installed but unused |
| Installer staging | `.setup/staging/<64-character id>` | `.setup/st/<12 characters>`, to stay under Windows' path limit |

The CUDA-named PyTorch API (`torch.cuda.*`, device `"cuda"`) maps to the Radeon
GPU under ROCm, so model code is unchanged. Error messages that named NVIDIA
now name AMD, and **Help > Check System** reports `hip_runtime`.

## Verified on the developer's PC

RX 7900 XTX (24 GB), Ryzen CPU with integrated Radeon, Windows 11, 2026-11-20.

| Check | Result |
| --- | --- |
| `install.cmd`: all six base runtimes, PEAR support, camera alignment | Installed and hash-verified; PEAR's GPU self-test passes |
| GPU ops in the base runtime | matmul (FP32, FP16, BF16), conv2d, scaled-dot-product attention, torchvision NMS and RoIAlign pass on the 7900 XTX |
| `Setup_GEMX.cmd` and GEM-X reconstruction | A 5 s, 150-frame clip in 140 s, peak 3.6 GiB VRAM, with contact correction; the result loads in the app |
| `Setup_SAM31.cmd` | Installed; native BF16 attention check passes |
| Find person, Default tracker (YOLO + OSNet) | 34 s for a 5 s clip |
| Find person, SAM 3.1 | 92 s for the same clip |
| `Setup_MAMMA.cmd` (SAM 2 `_C` and pytorch_sdf compiled with HIP) | Installed; SDF distances within 2.7e-7 of an exact CPU search on box, sphere and torus meshes; SAM 2 connected components equal SciPy |
| MAMMA masks stage | Two 60-frame views in 58 s, peak 2.1 GiB VRAM |
| `Setup_PromptHMR.cmd` with moving-camera support (LieTorch and DroidCalib compiled with HIP) | Installed; LieTorch GPU equals its CPU code within 2.1e-14; DroidCalib correlation within 1.6e-6 of a PyTorch reference |
| PromptHMR tracking, camera calibration, moving camera, 2D keypoints | 120 frames: 34.7 s, 1.4 s, 15.4 s, 7.7 s; peak 3.9 GiB VRAM |

Not verified yet, because they need registered body files the developer has not
installed: ExMR, FlowHMR and GVHMR reconstruction, MAMMA and PromptHMR beyond the
stages above, and the PEAR live preview. See the [model guides](models/README.md).

## Native builds (MAMMA, PromptHMR)

MAMMA and PromptHMR compile CUDA sources at setup. On this build they are
hipified by PyTorch and compiled with AMD's clang. `installer/rocm_build.py`
installs the pinned ROCm compiler wheels (`installer/rocm-devel-lock.json`) once
into `.setup/rocm-devel/` (about 4.5 GiB) and needs Visual Studio 2022 or newer
with **Desktop development with C++** and a Windows SDK. No CUDA Toolkit is used.
`installer/rocm/hip_cuda_compat.h` adapts 32-bit warp masks to HIP. Fixes made
for AMD GPUs: a bounded tree-build loop and memory fences in pytorch_sdf (it
hung, then returned stale distances), one template keyword in LieTorch, and a
pure-PyTorch `torch_scatter` (DroidCalib uses only `scatter_sum` and
`scatter_mean`). xFormers is dropped; its callers use PyTorch attention.
Details: [MAMMA agent setup](MAMMA_AGENT_SETUP.md), [PromptHMR](models/PROMPTHMR.md).

## Speed setting

The launcher sets `MIOPEN_FIND_MODE=FAST` unless it is already set. MIOpen's
default search picked slow convolution algorithms: PromptHMR's tracking took
381 s instead of 34.7 s, and SAM 2's network ran 27 times slower, with the same
output (within 3e-6).

### Installed size on that PC

| Part | Size |
| --- | ---: |
| Base runtimes (`runtime*` folders from `install.cmd`, Blender, FFmpeg) | 11.8 GiB |
| GEM-X runtime (its own ROCm copy; CUDA DLL sharing does not apply) | 5.0 GiB |
| SAM 3.1 overlay | 28 MiB |
| Public base and GEM-X model files | 33 GiB |

The ROCm wheels download in about 2.8 GB for the base install, against about
7.6 GB for the CUDA wheels. The model-size tables in the user guide still apply;
the GEM-X "with CUDA sharing" row does not.

## Other Radeon GPUs

AMD publishes kernel packages per GPU target. To build for another target,
download the matching `amd-torch-device-<target>`,
`amd-torchvision-device-<target>` and `rocm-sdk-device-<target>` wheels,
change `GFX` in `tools/rocm_lock.py` and `SUPPORTED_ARCHES` in
`mocap/amd_gpu.py`, and run:

```cmd
runtime\python.exe -I tools\rocm_lock.py installer\runtime-lock.json <folder with the wheels>
runtime\python.exe -I tools\rocm_lock.py installer\gemx-runtime-lock.json <folder with the wheels>
```

Then run `install.cmd`. Only `gfx1100` has been tested.
