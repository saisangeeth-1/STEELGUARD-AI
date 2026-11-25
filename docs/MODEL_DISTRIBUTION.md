# STEELGUARD AI — Model Acquisition and Checksum Verification

This document specifies the exact cryptographic hashes, architecture specifications, and distribution procedures for all vision and language model checkpoints in STEELGUARD AI.

## 1. Vision Checkpoint Inventory

| Checkpoint Name | Model Architecture | Training Dataset | Primary Metric | File Size | SHA-256 Digest |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `steelguard_gc10_neu_inclusion_best.pt` | YOLOv8 Nano (Detection) | GC10-DET + NEU Inclusion | 64.14% mAP@50 (Inclusion AP 30.54%) | 6.23 MB | `41ce4fd1099db9bce7e93b734bcfba40810fa365f1e58a82c30da8a24a3f8e2c` |
| `steelguard_gc10n_baseline_best.pt` | YOLOv8 Nano (Detection) | GC10-DET Baseline | 60.63% mAP@50 (Inclusion AP 14.98%) | 6.23 MB | `ffacba4cc1a781d6fd023baf04ccb135890cbd4edf9d78cdab5bb89cbab08ee6` |
| `severstal_unet_best.pt` | Lightweight U-Net (Segmentation) | Severstal Steel Defect | 0.3477 Defect Dice | 23.42 MB | `10666e11ecf088489fc2544d85e50cce0108b93c5cb888974f30d886029104e3` |

All vision checkpoints are distributed directly inside `steelguard_app/weights/` and automatically verified via SHA-256 upon model loading.

---

## 2. Language Model Advisory Bundle (ExecuTorch 1.5.1 + Llama 3.2 1B Instruct)

The offline edge advisory runtime utilizes Meta's Llama 3.2 1B Instruct model quantized to INT4 using ExecuTorch 1.5.1.

| Artifact | File Size | SHA-256 Digest | Distribution Location |
| :--- | :--- | :--- | :--- |
| `steelguard_llama32_1b_int4.pte` | 768.8 MiB | `d7d509a435d3ece32bbe349b5042d2198cd9ea53cc73f5c28c024fd90fbac5f3` | GitHub Release Asset / Hugging Face Hub |
| `tokenizer.model` | 2.08 MiB | `82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55` | GitHub Release Asset / Meta Llama 3.2 |
| `llama_main` (AArch64) | 4.02 MiB | `a742c0f02a502035087fb64b8bb9d7594047938ba91bac9d3ee323f87b069c84` | Pre-compiled ARM64 Edge Binary |
| `libllama_runner.so` (AArch64) | 7.30 MiB | `406d97812d48fa9df88064393f999aff20ffe22e46d36793ea8140ca5aa13929` | Pre-compiled ARM64 Edge Library |

Because `.pte` binary models exceed standard GitHub repository file size limits (>100 MB), they are distributed via GitHub Release Asset Tags (`v1.0-release`) and verified via `scripts/check_executorch_advisory.py`.
