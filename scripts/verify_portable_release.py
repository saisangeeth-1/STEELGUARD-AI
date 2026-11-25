"""Automated Pre-Flight & Functional Verification for Portable Release."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "steelguard_app" / "backend"))
os.environ["STEELGUARD_DEVICE"] = "cpu"

EXPECTED_WEIGHTS = {
    "steelguard_gc10_neu_inclusion_best.pt": "41ce4fd1099db9bce7e93b734bcfba40810fa365f1e58a82c30da8a24a3f8e2c",
    "steelguard_gc10n_baseline_best.pt": "ffacba4cc1a781d6fd023baf04ccb135890cbd4edf9d78cdab5bb89cbab08ee6",
    "severstal_unet_best.pt": "10666e11ecf088489fc2544d85e50cce0108b93c5cb888974f30d886029104e3"
}

def verify():
    print("=" * 72)
    print(" STEELGUARD AI — PORTABLE RELEASE SELF-TEST")
    print("=" * 72)

    # 1. Check Weights Hashes
    weights_dir = ROOT / "steelguard_app" / "weights"
    for fname, exp_hash in EXPECTED_WEIGHTS.items():
        w_path = weights_dir / fname
        if not w_path.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {fname}")
        hasher = hashlib.sha256()
        with open(w_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        act_hash = hasher.hexdigest()
        if act_hash != exp_hash:
            raise ValueError(f"Hash mismatch on {fname}")
        print(f" [PASS] Verified Checkpoint: {fname}")

    # 2. Check Curated Demo Samples
    manifest_file = ROOT / "demo_samples" / "demo_samples_manifest.json"
    if not manifest_file.is_file():
        raise FileNotFoundError("Missing demo_samples_manifest.json")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    for s in manifest["curated_samples"]:
        s_path = ROOT / "demo_samples" / s["filename"]
        if not s_path.is_file():
            raise FileNotFoundError(f"Missing sample: {s['filename']}")
    print(f" [PASS] Verified {len(manifest['curated_samples'])} Curated Demo Samples")

    # 3. Test YOLO Inference on CPU
    from model_manager import model_manager
    model_manager._device = "cpu"
    test_img = ROOT / "demo_samples" / "img_01_3402617700_01009.jpg"
    t0 = time.perf_counter()
    res = model_manager.run_inference(str(test_img))
    t1 = time.perf_counter()
    yolo_ms = round((t1 - t0) * 1000.0, 2)
    print(f" [PASS] YOLO CPU Inference Succeeded in {yolo_ms} ms (Detections: {len(res['detections'])})")

    # 4. Test Deterministic Quality Engine
    from inspection_intelligence import summarize_inspection
    summ = summarize_inspection(res)
    if summ["assessment"]["status"] != "REJECT":
        raise AssertionError("Expected REJECT verdict for crescent gap")
    print(f" [PASS] Deterministic Quality Verdict: {summ['assessment']['status']} (Immutable Rule Enforced)")

    # 5. Test U-Net Segmentation on CPU
    from segmentation_service import segmentation_manager
    segmentation_manager._device = "cpu"
    seg_img = ROOT / "demo_samples" / "0007a71bf.jpg"
    t_seg0 = time.perf_counter()
    seg_res = segmentation_manager.run_inference(str(seg_img))
    t_seg1 = time.perf_counter()
    seg_ms = round((t_seg1 - t_seg0) * 1000.0, 2)
    if not seg_res["has_defect"]:
        raise AssertionError("Expected defect in 0007a71bf.jpg")
    print(f" [PASS] U-Net CPU Segmentation Succeeded in {seg_ms} ms (Defect: {seg_res['active_classes']})")

    # 6. Test SQLite Traceability Save & Query
    import db
    db_path = ROOT / "steelguard_app" / "data" / "steelguard.db"
    db.init_db(db_path)
    insp_id = f"TEST-SELFCHECK-{int(time.time())}"
    rec = {
        "inspection_id": insp_id,
        "source": "AI_INFERENCE",
        "filename": test_img.name,
        "image_width": res["image_width"],
        "image_height": res["image_height"],
        "detections": res["detections"],
        "assessment": {
            "status": summ["assessment"]["status"],
            "summary": summ["assessment"]["summary"],
            "rule_ids": ["QR-01"]
        },
        "model_name": "YOLOv8 Nano (CPU Self-Check)",
        "model_version": res["model_version"],
        "inference_time_ms": yolo_ms
    }
    save_res = db.save_inspection(rec, db_path=db_path)
    if not save_res.get("success"):
        raise AssertionError(f"save_inspection failed: {save_res}")
    fetched = db.get_inspection(insp_id, db_path=db_path)
    if not fetched or fetched["id"] != insp_id:
        raise AssertionError(f"Failed to retrieve saved inspection record from SQLite: {fetched}")
    # Clean up test record so database remains pristine
    with db.ManagedConnection(db_path) as conn:
        conn.execute("DELETE FROM inspections WHERE id = ?", (insp_id,))
        conn.commit()
    print(f" [PASS] SQLite Traceability Verified (Record {insp_id} saved, queried, and verified)")

    print("=" * 72)
    print(" ALL PORTABLE RELEASE VERIFICATION CHECKS PASSED (100% CPU-ONLY)")
    print("=" * 72)

if __name__ == "__main__":
    verify()
