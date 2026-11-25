"""Benchmark curated demo samples on CPU, recording real detections, quality verdicts, and latencies.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

# Ensure backend directory is in path
ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT / "steelguard_app" / "backend"
sys.path.insert(0, str(BACKEND_DIR))

# Force CPU mode explicitly
os.environ["STEELGUARD_DEVICE"] = "cpu"

from model_manager import model_manager
from segmentation_service import segmentation_manager
from inspection_intelligence import summarize_inspection
from quality_engine import assess_quality

DEMO_DIR = ROOT / "demo_samples"


def run_benchmark():
    # Configure managers for CPU execution
    model_manager._device = "cpu"
    if hasattr(segmentation_manager, "_model") and segmentation_manager._model is not None:
        segmentation_manager._model.to("cpu")
    segmentation_manager._device = "cpu"

    manifest_path = DEMO_DIR / "demo_samples_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    results = []
    print("=" * 72)
    print(" STEELGUARD AI — CURATED SAMPLES CPU INFERENCE BENCHMARK")
    print(f" Target Device: CPU (Platform: {sys.platform})")
    print("=" * 72)

    for item in manifest["curated_samples"]:
        img_path = DEMO_DIR / item["filename"]
        print(f"\nProcessing Sample: {item['filename']} [{item['category']}]")
        print(f" -> Provenance: {item['dataset']} ({item['split']})")

        # 1. YOLO Inference
        t0 = time.perf_counter()
        yolo_res = model_manager.run_inference(str(img_path))
        t1 = time.perf_counter()
        yolo_latency_ms = round((t1 - t0) * 1000.0, 2)

        # 2. Inspection Intelligence & Deterministic Verdict
        summary = summarize_inspection(yolo_res)

        sample_record = {
            "id": item["id"],
            "filename": item["filename"],
            "category": item["category"],
            "dataset_provenance": {
                "dataset": item["dataset"],
                "split": item["split"],
                "sha256": item["sha256"]
            },
            "yolo_detection": {
                "model_version": yolo_res["model_version"],
                "device": "cpu",
                "latency_ms": yolo_latency_ms,
                "detection_count": len(yolo_res["detections"]),
                "detections": [
                    {
                        "class_name": d["class_name"],
                        "confidence": round(d["confidence"], 4),
                        "bbox_xyxy": d["bbox_xyxy"],
                        "area_px": d["area_px"],
                        "criticality": d["criticality"]
                    }
                    for d in yolo_res["detections"]
                ]
            },
            "deterministic_quality_assessment": {
                "verdict": summary["assessment"]["status"],
                "recommended_action": summary["recommended_action"],
                "explanation": summary["assessment"]["summary"],
                "warnings": summary["warnings"]
            }
        }

        # 3. If Segmentation Sample, run U-Net
        if item["category"] in ("segmentation_defect", "clean_sample") or "0007" in item["filename"]:
            t_seg0 = time.perf_counter()
            seg_res = segmentation_manager.run_inference(str(img_path))
            t_seg1 = time.perf_counter()
            seg_latency_ms = round((t_seg1 - t_seg0) * 1000.0, 2)

            sample_record["unet_segmentation"] = {
                "device": "cpu",
                "latency_ms": seg_latency_ms,
                "has_defect": seg_res["has_defect"],
                "active_classes": seg_res["active_classes"],
                "defect_percentage": round(seg_res["defect_percentage"], 4),
                "total_defect_pixels": seg_res["total_defect_pixels"]
            }
            print(f" -> U-Net Segmentation: Defect={seg_res['has_defect']}, Pct={seg_res['defect_percentage']:.2f}%, Latency={seg_latency_ms}ms")

        print(f" -> YOLO Detections: {len(yolo_res['detections'])} found in {yolo_latency_ms}ms")
        print(f" -> Deterministic Verdict: {summary['assessment']['status']}")
        print(f" -> Action: {summary['recommended_action'][:60]}...")

        results.append(sample_record)

    out_file = DEMO_DIR / "curated_samples_benchmark.json"
    out_file.write_text(json.dumps({"benchmark_results": results}, indent=2), encoding="utf-8")
    print("\n" + "=" * 72)
    print(f" Benchmark complete. Real results recorded in: {out_file}")
    print("=" * 72)


if __name__ == "__main__":
    run_benchmark()
