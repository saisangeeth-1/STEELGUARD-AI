# STEELGUARD AI — Portable Industrial Defect Detection & Quality-Control Assistant

**Bharat AI-SoC Hackathon | Problem Statement 5**  
**Real-Time Industrial Defect Detection & Quality-Control Assistant**

---

## 1. Overview & Capabilities

STEELGUARD AI is an offline edge quality control and inspection assistant for hot/cold rolled steel strip manufacturing. It operates **100% locally** with zero cloud APIs, zero external telemetry, and full CPU-only execution support for standard laptops and ARM64 edge devices.

### Core Capabilities
1. **Dual-Model Vision Pipeline:**
   - **YOLOv8 Nano (Surface Defect Detection):** Fine-tuned on GC10-DET augmented with NEU-DET Inclusions. Achieves **64.14% mAP@50** overall and **30.54% AP@50** on inclusions, operating at ~25 ms per frame on standard CPU (~40 FPS).
   - **Lightweight U-Net (Pixel-Level Segmentation):** Independent experimental model trained on Severstal Steel Defect dataset for pixel-level boundary localization (~220 ms on CPU).
2. **Deterministic Inspection Intelligence:**
   - Enforces immutable industrial acceptance rules (QR-01 through QR-06).
   - Never allows LLM hallucinations or generative models to alter measurement data, bounding boxes, or PASS/REJECT quality verdicts.
3. **Local LLM Advisory (ExecuTorch + Llama 3.2 1B Instruct):**
   - Offline constrained action planning via ExecuTorch 1.5.1 INT4.
   - Guaranteed deterministic rule-based fallback when offline LLM runtime is unavailable or timeouts occur.
4. **SQLite Quality Traceability:**
   - Every inspection, bounding box, quality rule trigger, and operator action is immutably logged to a local SQLite database for compliance auditing.

---

## 2. Quick Start Guide (Windows)

### Prerequisites
- Python 3.10, 3.11, or 3.12 (standard 64-bit Python).
- No GPU or CUDA required.

### Step 1 — Automated Installation
Double-click `install_windows.bat` or run in terminal:
```cmd
install_windows.bat
```
This automatically:
- Creates a clean isolated virtual environment (`.venv`).
- Installs official CPU-only PyTorch (`--index-url https://download.pytorch.org/whl/cpu`).
- Installs lightweight runtime dependencies (`ultralytics`, `opencv-python`, `numpy`, `pillow`, `psutil`).

### Step 2 — Run Pre-Flight Verification
```cmd
.venv\Scripts\python.exe scripts\verify_portable_release.py
```
This verifies:
- Cryptographic SHA-256 integrity of all vision checkpoints.
- Dataset integrity of curated test demonstration samples.
- Real CPU inference for YOLOv8 Nano (~25 ms).
- Real CPU segmentation for Lightweight U-Net (~220 ms).
- Immutable deterministic quality verdict enforcement.
- SQLite inspection database write/read persistence.

### Step 3 — Launch Dashboard
Double-click `run_demo_cpu.bat` or run:
```cmd
run_demo_cpu.bat
```
Open your browser at: **`http://127.0.0.1:8080/`**

---

## 3. Quick Start Guide (Linux / Raspberry Pi / macOS)

### Step 1 — Installation
```bash
chmod +x install_linux.sh run_demo_cpu.sh
./install_linux.sh
```

### Step 2 — Pre-Flight Verification
```bash
.venv/bin/python scripts/verify_portable_release.py
```

### Step 3 — Launch
```bash
./run_demo_cpu.sh
```
Open your browser at: **`http://127.0.0.1:8080/`**

---

## 4. Connecting the Real WSL2 ExecuTorch Llama Advisory (Optional)

On Windows host machines with WSL2 Ubuntu installed, STEELGUARD AI can connect directly to the native Linux ExecuTorch Llama 3.2 1B INT4 runtime.

### Configuration
1. Export or copy the ExecuTorch bundle into your WSL home directory (e.g. `~/steelguard-host-bundle-20261116`).
2. Run the dashboard with the manifest pointer:
```cmd
.venv\Scripts\python.exe steelguard_app\run_dashboard.py --cpu --demo --manifest /home/<your-wsl-username>/steelguard-host-bundle-20261116/advisory.manifest.json
```
3. Or set environment variables:
```cmd
set STEELGUARD_EXECUTORCH_MANIFEST=/home/<your-wsl-username>/steelguard-host-bundle-20261116/advisory.manifest.json
set STEELGUARD_WSL_DISTRO=Ubuntu-24.04
run_demo_cpu.bat
```

### Deterministic Fallback Guarantee
If WSL2 or ExecuTorch is not configured or fails to respond within 5 seconds:
- STEELGUARD AI automatically engages **`RULE_BASED_FALLBACK`**.
- The operator action plan and explanation are deterministically generated from the verified bounding box evidence.
- The quality verdict is **never** compromised, ensuring continuous 100% industrial uptime.

---

## 5. Curated Demonstration Samples

This release includes 5 verified held-out test samples in `demo_samples/`:

| Sample File | Category | Source Dataset | Split Membership | Verified Quality Verdict |
| :--- | :--- | :--- | :--- | :--- |
| `img_01_3402617700_01009.jpg` | Critical Defect | GC10-DET | Held-Out Test Split | **`REJECT`** (Crescent Gap edge tear) |
| `img_02_4402623300_00034.jpg` | Inclusion Defect | GC10-DET | Held-Out Test Split | **`REJECT`** (Non-metallic inclusion) |
| `img_01_425006600_00905.jpg` | Minor Defect | GC10-DET | Held-Out Test Split | **`REVIEW REQUIRED`** (Water spot) |
| `000789191.jpg` | Clean Surface | Severstal | Held-Out Test (`has_defect=0`) | **`PASS`** (Zero defects detected) |
| `0007a71bf.jpg` | Surface Scratch | Severstal | Held-Out Test (`has_defect=1`) | **`U-Net Segmented`** (2.91% defect) |

---

## 6. Target Hardware Status & Disclosure

> [!IMPORTANT]
> **Physical Hardware Status:**
> Physical Raspberry Pi 5 hardware and continuous camera integration have not yet been validated on a physical device. All provided benchmarks in this portable release were measured on genuine CPU hardware on a standard laptop host. An ARM64 cross-compiled package (`steelguard-pi5-bundle-20261116-v2`) with verified checksums is prepared for immediate deployment upon receipt of target hardware.
