# STEELGUARD AI — Raspberry Pi 5 (4GB RAM) Target Deployment Guide

> [!IMPORTANT]
> **Physical Hardware Status:** As of 2026-11-17, the physical Raspberry Pi 5 board has **not yet been received/purchased**.
> Development, architecture validation, and real ExecuTorch Llama 3.2 1B inference have been fully validated on Ubuntu 24.04 (WSL2 x86-64).
> The ARM64 cross-compiled package (`steelguard-pi5-bundle-20261116-v2`) has passed all static integrity and checksum verifications.
> This runbook defines the exact specifications, memory budget, transfer procedure, and hardware acceptance tests to be executed immediately upon receiving the physical Pi 5.

---

## 1. Target Hardware & Operating System Specifications

| Property | Target Specification | Validation Requirement |
| :--- | :--- | :--- |
| **Model** | Raspberry Pi 5 Model B (4 GB LPDDR4X-4267) | Verify `cat /proc/device-tree/model` |
| **Processor** | Broadcom BCM2712 (4x ARM Cortex-A76 @ 2.4 GHz) | Cryptography & DotProd extensions enabled |
| **Architecture** | AArch64 (64-bit ARMv8.2-A) | Verify `uname -m` == `aarch64` |
| **Cooling** | Official Raspberry Pi Active Cooler | Essential to prevent thermal throttling at 2.4 GHz |
| **Target OS** | Raspberry Pi OS 64-bit (Debian 13 Trixie based) | Required for GLIBC 2.38+ |
| **Dynamic Loader** | `/lib/ld-linux-aarch64.so.1` | Must resolve cleanly with `ldd` |
| **C/C++ Runtimes** | `GLIBC >= 2.38`, `GLIBCXX >= 3.4.32` | Verify `strings /lib/aarch64-linux-gnu/libc.so.6 \| grep GLIBC_2.38` |
| **Storage** | 32 GB+ High-Endurance microSD or NVMe HAT | >= 10 GB free space recommended |

---

## 2. Strict 4GB RAM Resource Budget & Headroom Analysis

The target device has **4,096 MB** total physical RAM. To guarantee operational safety and prevent Linux Out-Of-Memory (`oom-killer`) termination, the inspection pipeline runs components sequentially with explicit memory bounds:

```
Total Available RAM: 4,096 MB (4.00 GiB)
========================================================================
[System OS & Daemons]          ~500 MB  (Lightweight headless Linux)
[Python 3.12 + Stdlib HTTP]     ~60 MB  (Zero heavyweight frameworks)
[YOLOv8 Nano Vision Engine]    ~120 MB  (PyTorch CPU / ONNX Runtime)
[Severstal U-Net (Dual Model)] ~180 MB  (Pixel segmentation worker)
[ExecuTorch Llama 3.2 1B INT4] ~1,350 MB (Peak VmHWM during token decode)
------------------------------------------------------------------------
Peak Combined Memory Usage:    ~2,210 MB (2.16 GiB)
========================================================================
GUARANTEED AVAILABLE HEADROOM: ~1,886 MB (1.84 GiB / 46.0% FREE RAM)
```

### Memory Safety Controls
1. **Zero Multi-Model Race Conditions:** YOLO inference runs first; once complete, features and deterministic summaries are serialized. The LLM advisory is only triggered on-demand or as an isolated sub-process.
2. **Page-Mapped INT4 Model (`.pte`):** The 768.8 MiB quantized model file is mapped directly into memory via ExecuTorch, sharing pages with the OS cache.
3. **Bounded Context Length:** Context is strictly limited to <= 1024 tokens. Key-Value (KV) cache is sized at < 64 MiB.
4. **Deterministic Fallback on Exhaustion:** If memory allocation fails or generation exceeds the 5.0-second timeout, `AdvisoryService` immediately degrades to `RULE_BASED_FALLBACK`, maintaining 100% industrial uptime.

---

## 3. Verified Bundle Inventory (AArch64)

The cross-compiled bundle is prepared and checksum-verified at `/home/sai01/steelguard-pi5-bundle-20261116-v2/`:

| Artifact | File Size | SHA-256 Digest |
| :--- | :--- | :--- |
| `steelguard_llama32_1b_int4.pte` | 806,179,840 B (768.8 MiB) | `d7d509a435d3ece32bbe349b5042d2198cd9ea53cc73f5c28c024fd90fbac5f3` |
| `tokenizer.model` | 2,184,272 B (2.08 MiB) | `82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55` |
| `llama_main` (AArch64) | 4,218,656 B (4.02 MiB) | `a742c0f02a502035087fb64b8bb9d7594047938ba91bac9d3ee323f87b069c84` |
| `libllama_runner.so` (AArch64) | 7,651,328 B (7.30 MiB) | `406d97812d48fa9df88064393f999aff20ffe22e46d36793ea8140ca5aa13929` |
| `advisory.manifest.json` | Manifest Descriptor | Generated on-target via `phase5d_bundle.py relocate` |

---

## 4. Hardware Deployment Procedure (Upon Pi 5 Arrival)

### Step 1 — Transfer Bundle from Workstation
From the host workstation (WSL2), transfer the bundle directory to the Raspberry Pi:
```bash
# Set Pi IP/hostname
export PI_HOST="pi@192.168.1.100"

# Copy package excluding host-specific absolute manifest paths
rsync -avz --exclude="advisory.manifest.json" \
  /home/sai01/steelguard-pi5-bundle-20261116-v2/ \
  "$PI_HOST":~/steelguard-pi5-bundle/
```

### Step 2 — Relocate Manifest on Pi 5
SSH into the physical Pi 5 and adapt the manifest paths to the target directory:
```bash
ssh pi@192.168.1.100
cd ~/steelguard-pi5-bundle

# Automatically regenerates advisory.manifest.json with absolute target paths
python3 scripts/phase5d_bundle.py relocate --destination "$PWD"
```

### Step 3 — Verify Dynamic Linker & Libraries
Ensure all required shared objects resolve on the physical OS:
```bash
LD_LIBRARY_PATH="$PWD/artifacts" ldd artifacts/llama_main
```
*Expected output:* All libraries resolved (`libc.so.6`, `libm.so.6`, `libstdc++.so.6`, `libpthread.so.0`), with zero "not found" symbols.

### Step 4 — Run Strict Offline Hardware Qualification
Execute the qualification suite inside an unshared, loopback-only network namespace to prove offline independence:
```bash
sudo unshare --net python3 scripts/phase5d_qualify.py \
  --manifest "$PWD/advisory.manifest.json" \
  --output pi5-hardware-qualification.json \
  --runs 3 \
  --require-network-isolation \
  --probe-timeout
```

*Pass Criteria:*
- 3 consecutive runs returning valid action codes (`HOLD_FOR_REVIEW`, `SCRAP_IMMEDIATE`, etc.).
- Reason code `OK` with `llm_used: true`.
- Zero internet requests.
- Latency per advisory query <= 5.0 seconds.

---

## 5. Launching STEELGUARD AI Production Service on Pi 5

Once qualified, the inspection service can be started directly on the Raspberry Pi:

```bash
# Set environment variables
export STEELGUARD_EXECUTORCH_MANIFEST="/home/pi/steelguard-pi5-bundle/advisory.manifest.json"
export YOLO_OFFLINE="True"
export ULTRALYTICS_AUTOUPDATE="0"

# Start the lightweight HTTP inspection dashboard
python3 steelguard_app/run_dashboard.py --host 0.0.0.0 --port 8080
```

Access the dashboard from any browser on the local plant network:
`http://<pi-ip-address>:8080/`
