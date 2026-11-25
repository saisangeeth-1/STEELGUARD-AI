"""
STEELGUARD AI - Part 2: Checkpoint and Resource Verification
Measures true inference latency with warm-up and CUDA synchronization.
Measures memory allocated/reserved before and after inference.
Tests model isolation, CPU fallback, and CUDA OOM simulation.
"""

import gc
import hashlib
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
BACKEND_DIR = APP_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))

os.environ["YOLO_OFFLINE"] = "True"
os.environ["ULTRALYTICS_AUTOUPDATE"] = "0"

from segmentation_service import (
    segmentation_manager,
    EXPECTED_SHA256 as UNET_EXPECTED_SHA256,
    get_sample_image_path
)
from model_manager import model_manager, REGISTERED_MODELS

EXPECTED_HASHES = {k: v["expected_sha256"] for k, v in REGISTERED_MODELS.items()}

def compute_sha256(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def verify_checkpoints():
    print("--- 1. Checkpoint SHA-256 Verification ---")
    # U-Net local & source
    local_unet = APP_DIR / "weights" / "severstal_unet_best.pt"
    src_unet = PROJECT_ROOT / "runs" / "segmentation" / "severstal_unet_v1" / "best_model.pt"
    local_unet_sha = compute_sha256(local_unet)
    src_unet_sha = compute_sha256(src_unet)
    print(f"U-Net Local SHA-256:  {local_unet_sha}")
    print(f"U-Net Source SHA-256: {src_unet_sha}")
    print(f"Expected U-Net SHA:   {UNET_EXPECTED_SHA256}")
    assert local_unet_sha == UNET_EXPECTED_SHA256 == src_unet_sha, "U-Net SHA mismatch!"

    # YOLO Candidate
    cand_path = APP_DIR / "weights" / "steelguard_gc10_neu_inclusion_best.pt"
    cand_sha = compute_sha256(cand_path)
    print(f"YOLO Candidate SHA:   {cand_sha}")
    assert cand_sha == EXPECTED_HASHES["candidate"], "YOLO candidate SHA mismatch!"

    # YOLO Baseline
    base_path = APP_DIR / "weights" / "steelguard_gc10n_baseline_best.pt"
    base_sha = compute_sha256(base_path)
    print(f"YOLO Baseline SHA:    {base_sha}")
    assert base_sha == EXPECTED_HASHES["baseline"], "YOLO baseline SHA mismatch!"
    print("All checkpoint SHA-256 hashes VERIFIED MATCH.\n")

def test_inference_benchmarks():
    print("--- 2. Inference Benchmarks with Warm-Up & CUDA Synchronization ---")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    is_cuda = device.startswith("cuda")

    # Load both models
    print("Loading models...")
    _ = segmentation_manager.load_model(force_reload=True)
    _ = model_manager.switch_model("candidate")

    yolo_img_path = PROJECT_ROOT / "datasets" / "gc10_yolo" / "images" / "test" / "img_01_3402617700_01009.jpg"
    unet_img_path = get_sample_image_path("0007a71bf.jpg")

    # Warm-up (3 passes each)
    print("Executing 3 warm-up passes for each model...")
    for _ in range(3):
        _ = model_manager.run_inference(yolo_img_path)
        _ = segmentation_manager.run_inference(unet_img_path)
        if is_cuda:
            torch.cuda.synchronize()

    # Memory state after warm-up
    if is_cuda:
        gc.collect()
        torch.cuda.empty_cache()
        mem_alloc_idle = torch.cuda.memory_allocated(0) / (1024 ** 2)
        mem_res_idle = torch.cuda.memory_reserved(0) / (1024 ** 2)
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        print(f"\nMemory State (Both Models Co-resident, Idle Post-Warmup):")
        print(f"  Allocated VRAM: {mem_alloc_idle:.2f} MB")
        print(f"  Reserved VRAM:  {mem_res_idle:.2f} MB")
        print(f"  Free Headroom:  {total_vram - mem_res_idle:.2f} MB / {total_vram:.2f} MB ({(1 - mem_res_idle/total_vram)*100:.1f}%)")

    # Benchmark YOLO inference over 10 iterations
    print("\nBenchmarking YOLOv8 Nano (10 iterations)...")
    yolo_times = []
    yolo_peak_mems = []
    for _ in range(10):
        if is_cuda:
            torch.cuda.reset_peak_memory_stats(0)
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model_manager.run_inference(yolo_img_path)
        if is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        yolo_times.append((t1 - t0) * 1000.0)
        if is_cuda:
            yolo_peak_mems.append(torch.cuda.max_memory_allocated(0) / (1024 ** 2))

    # Benchmark U-Net inference over 10 iterations
    print("Benchmarking Severstal Lightweight U-Net (10 iterations)...")
    unet_times = []
    unet_peak_mems = []
    for _ in range(10):
        if is_cuda:
            torch.cuda.reset_peak_memory_stats(0)
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = segmentation_manager.run_inference(unet_img_path)
        if is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        unet_times.append((t1 - t0) * 1000.0)
        if is_cuda:
            unet_peak_mems.append(torch.cuda.max_memory_allocated(0) / (1024 ** 2))

    yolo_mean = np.mean(yolo_times)
    yolo_std = np.std(yolo_times)
    unet_mean = np.mean(unet_times)
    unet_std = np.std(unet_times)

    print(f"\nYOLOv8 Nano Latency:        {yolo_mean:.2f} +/- {yolo_std:.2f} ms ({1000.0/yolo_mean:.1f} FPS)")
    if is_cuda:
        print(f"YOLOv8 Nano Peak Active Mem: {np.mean(yolo_peak_mems):.2f} MB")
    print(f"Severstal U-Net Latency:    {unet_mean:.2f} +/- {unet_std:.2f} ms ({1000.0/unet_mean:.1f} FPS)")
    if is_cuda:
        print(f"Severstal U-Net Peak Active: {np.mean(unet_peak_mems):.2f} MB")

    # Combined dual-model sequential execution
    print("\nBenchmarking Combined Dual-Model Pipeline (Sequential)...")
    combo_times = []
    for _ in range(10):
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model_manager.run_inference(yolo_img_path)
        _ = segmentation_manager.run_inference(unet_img_path)
        if is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        combo_times.append((t1 - t0) * 1000.0)
    combo_mean = np.mean(combo_times)
    print(f"Dual-Model Combined Latency: {combo_mean:.2f} +/- {np.std(combo_times):.2f} ms ({1000.0/combo_mean:.1f} FPS dual pipeline)\n")

def test_cpu_fallback_and_oom_handling():
    print("--- 3. CPU Fallback & Memory Safety Verification ---")
    unet_img_path = get_sample_image_path("0007a71bf.jpg")
    yolo_img_path = PROJECT_ROOT / "datasets" / "gc10_yolo" / "images" / "test" / "img_01_3402617700_01009.jpg"

    # Test explicit CPU inference for U-Net
    print("Testing U-Net forced CPU mode...")
    orig_device = segmentation_manager._device
    segmentation_manager._model.to("cpu")
    segmentation_manager._device = "cpu"
    res_cpu = segmentation_manager.run_inference(unet_img_path)
    assert res_cpu["status"] == "SUCCESS", "CPU inference failed"
    assert res_cpu["device_used"] == "cpu", "Device must be cpu"
    print(f"U-Net CPU Inference Successful! Latency: {res_cpu['inference_time_ms']:.2f} ms, Defect Pixels: {res_cpu['total_defect_pixels']}")

    # Restore to CUDA
    if torch.cuda.is_available():
        segmentation_manager._model.to("cuda:0")
        segmentation_manager._device = "cuda:0"
        print("Restored U-Net to cuda:0")

    # Test simulated CUDA OOM handling
    print("\nTesting simulated CUDA OOM fallback handling...")
    real_forward = segmentation_manager._model.forward
    called = [False]
    def mock_forward_oom(*args, **kwargs):
        if not called[0]:
            called[0] = True
            raise torch.cuda.OutOfMemoryError("Simulated CUDA OOM for resilience verification")
        return real_forward(*args, **kwargs)

    segmentation_manager._model.forward = mock_forward_oom
    try:
        # run_inference catches torch.cuda.OutOfMemoryError, clears cache, moves model to cpu, and succeeds!
        res_oom = segmentation_manager.run_inference(unet_img_path)
        assert res_oom["status"] == "SUCCESS", "OOM recovery failed"
        assert res_oom["total_defect_pixels"] > 0
        print(f"U-Net CUDA OOM Gracefully Handled! Recovered on CPU with {res_oom['total_defect_pixels']} px detected.")
    finally:
        # Restore real forward method
        segmentation_manager._model.forward = real_forward
        if torch.cuda.is_available():
            segmentation_manager._model.to("cuda:0")
            segmentation_manager._device = "cuda:0"

    # Verify YOLO remains unaffected
    yolo_res = model_manager.run_inference(yolo_img_path)
    assert yolo_res["status"] == "SUCCESS", "YOLO failed after U-Net CPU fallback"
    print(f"YOLO Verified Unaffected! Detections: {len(yolo_res['detections'])}, Latency: {yolo_res['inference_time_ms']:.2f} ms")
    print("CPU Fallback & OOM Isolation VERIFIED.\n")

if __name__ == "__main__":
    verify_checkpoints()
    test_inference_benchmarks()
    test_cpu_fallback_and_oom_handling()
    print("PART 2 CHECKPOINT AND RESOURCE VERIFICATION COMPLETE.")
