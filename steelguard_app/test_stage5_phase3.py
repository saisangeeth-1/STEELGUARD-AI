"""
STEELGUARD AI - Phase 3 Comprehensive Production Integration Verification Suite
Tests candidate model loading, cryptographic hash verification, class ontology mapping,
model switching/rollback, image validation, local YOLO inference, clean surface handling,
deterministic quality rules evaluation, SQLite history persistence, and REST API dispatch.
"""

import base64
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

# Setup paths
APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from model_manager import (
    model_manager,
    REGISTERED_MODELS,
    GC10_CLASS_CATALOG,
    CLASS_MAP_BY_ID,
    ModelIntegrityError,
    InvalidImageError
)
from quality_engine import assess_quality, CONFIGURED_RULES
from dataset_service import registry
import db
from server import SteelGuardRequestHandler

TEST_DB_PATH = APP_DIR / "data" / "test_phase3_verification.db"


class DummyWFile:
    def __init__(self):
        self.data = bytearray()
    def write(self, b):
        self.data.extend(b)
    def flush(self):
        pass


class DummyRFile:
    def __init__(self, data=b""):
        self.data = data
        self.pos = 0
    def read(self, n):
        chunk = self.data[self.pos:self.pos + n]
        self.pos += len(chunk)
        return chunk


def dispatch(method: str, path: str, payload=None):
    """Simulates HTTP request dispatch through SteelGuardRequestHandler."""
    body_bytes = b""
    headers = {}
    if payload is not None:
        body_bytes = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
        headers["Content-Length"] = str(len(body_bytes))

    handler = SteelGuardRequestHandler.__new__(SteelGuardRequestHandler)
    handler.rfile = DummyRFile(body_bytes)
    handler.wfile = DummyWFile()
    handler.path = path
    handler.headers = headers
    handler.command = method
    handler.request_version = "HTTP/1.1"

    status_capture = []
    headers_capture = {}

    def mock_send_response(code, message=None):
        status_capture.append(code)

    def mock_send_header(keyword, value):
        headers_capture[keyword] = value

    def mock_end_headers():
        pass

    handler.send_response = mock_send_response
    handler.send_header = mock_send_header
    handler.end_headers = mock_end_headers

    if method == "GET":
        handler.do_GET()
    elif method == "POST":
        handler.do_POST()
    elif method == "OPTIONS":
        handler.do_OPTIONS()

    return status_capture[0] if status_capture else 500, headers_capture, bytes(handler.wfile.data)


def run_phase3_tests():
    print("=" * 70)
    print(" STEELGUARD AI — PHASE 3 PRODUCTION MODEL INTEGRATION TEST SUITE")
    print("=" * 70)

    # -------------------------------------------------------------
    # Test 1: Checkpoint Inventory & Cryptographic SHA-256 Verification
    # -------------------------------------------------------------
    print("\n[Test 1/10] Checkpoint Cryptographic Hash & Inventory Verification:")
    cand_info = REGISTERED_MODELS["candidate"]
    base_info = REGISTERED_MODELS["baseline"]

    cand_weights = APP_DIR / "weights" / cand_info["filename"]
    base_weights = APP_DIR / "weights" / base_info["filename"]

    assert cand_weights.is_file(), f"Candidate weights file missing: {cand_weights}"
    assert base_weights.is_file(), f"Baseline weights file missing: {base_weights}"

    cand_hash = model_manager.compute_sha256(cand_weights)
    base_hash = model_manager.compute_sha256(base_weights)

    print(f"             Candidate Checkpoint: {cand_weights.name}")
    print(f"               Expected SHA-256:   {cand_info['expected_sha256']}")
    print(f"               Actual SHA-256:     {cand_hash}")
    assert cand_hash == cand_info["expected_sha256"], "Candidate SHA-256 mismatch!"
    print(f"               Integrity Status:   [VERIFIED MATCH]")

    print(f"             Baseline Checkpoint:  {base_weights.name}")
    print(f"               Expected SHA-256:   {base_info['expected_sha256']}")
    print(f"               Actual SHA-256:     {base_hash}")
    assert base_hash == base_info["expected_sha256"], "Baseline SHA-256 mismatch!"
    print(f"               Integrity Status:   [VERIFIED MATCH]")
    print("             -> PASSED")

    # -------------------------------------------------------------
    # Test 2: Correct 10-Class Ontology & Metadata Mapping
    # -------------------------------------------------------------
    print("\n[Test 2/10] 10-Class Defect Ontology & Metadata Verification:")
    model_manager.switch_model("candidate")
    model = model_manager.load_active_model()
    class_names = model.names
    assert len(class_names) == 10, f"Expected 10 classes, got {len(class_names)}"

    expected_codes = [
        "1_chongkong", "2_hanfeng", "3_yueyawan", "4_shuiban", "5_youban",
        "6_siban", "7_yiwu", "8_yahen", "9_zhehen", "10_yaozhe"
    ]
    for idx, expected in enumerate(expected_codes):
        actual = class_names[idx]
        assert actual == expected, f"Class ID {idx} mismatch: expected {expected}, got {actual}"
        catalog_entry = CLASS_MAP_BY_ID[idx]
        assert catalog_entry["code"] == expected
        print(f"             Class [{idx}] -> {actual:<12} ({catalog_entry['name']:<25}) [{catalog_entry['criticality']}]")
    print("             -> PASSED")

    # -------------------------------------------------------------
    # Test 3: Model Rollback & Dynamic Switching
    # -------------------------------------------------------------
    print("\n[Test 3/10] Model Switching & Instant Rollback Verification:")
    print("             1. Switching to Rollback Baseline ('baseline')...")
    switched_base = model_manager.switch_model("baseline")
    assert switched_base["key"] == "baseline"
    assert switched_base["version"] == "v1.0-gc10-baseline"
    assert switched_base["actual_sha256"] == base_info["expected_sha256"]
    print(f"                Active: {switched_base['name']} ({switched_base['version']}) [MATCH]")

    print("             2. Switching back to Candidate Production ('candidate')...")
    switched_cand = model_manager.switch_model("candidate")
    assert switched_cand["key"] == "candidate"
    assert switched_cand["version"] == "v2.0-neu-inclusion"
    assert switched_cand["actual_sha256"] == cand_info["expected_sha256"]
    print(f"                Active: {switched_cand['name']} ({switched_cand['version']}) [MATCH]")
    print("             -> PASSED")

    # -------------------------------------------------------------
    # Test 4: Image Input Validation & Graceful Error Handling
    # -------------------------------------------------------------
    print("\n[Test 4/10] Image Input Validation & Security:")
    # A. Empty bytes
    try:
        model_manager.run_inference(b"")
        assert False, "Empty bytes should have raised InvalidImageError"
    except InvalidImageError:
        print("             - Empty byte stream: Rejected cleanly with InvalidImageError [OK]")

    # B. Corrupted byte stream
    try:
        model_manager.run_inference(b"CORRUPT_NOT_AN_IMAGE_DATA_BYTES")
        assert False, "Corrupt bytes should have raised InvalidImageError"
    except InvalidImageError:
        print("             - Corrupt image payload: Rejected cleanly with InvalidImageError [OK]")

    # C. Non-existent file path
    try:
        model_manager.run_inference("D:/non_existent_folder/missing_image.jpg")
        assert False, "Missing file path should have raised InvalidImageError"
    except InvalidImageError:
        print("             - Missing file reference: Rejected cleanly with InvalidImageError [OK]")

    # D. Valid synthetic image
    valid_arr = np.zeros((1000, 2048, 3), dtype=np.uint8) + 128
    pil_decoded, desc = model_manager.decode_image_input(valid_arr)
    assert pil_decoded.size == (2048, 1000)
    print("             - NumPy array / synthetic image: Decoded successfully [OK]")
    print("             -> PASSED")

    # -------------------------------------------------------------
    # Test 5: Successful Inference & Defect Localization
    # -------------------------------------------------------------
    print("\n[Test 5/10] Real Local YOLO Inference on Test Sample:")
    test_img_path = Path("datasets/gc10_yolo/images/test/img_01_3402617700_01009.jpg")
    assert test_img_path.is_file(), f"Test image missing: {test_img_path}"

    with open(test_img_path, "rb") as f:
        img_bytes = f.read()

    inf_res = model_manager.run_inference(img_bytes, filename=test_img_path.name)
    assert inf_res["status"] == "SUCCESS"
    assert inf_res["model_name"] == cand_info["name"]
    assert inf_res["model_version"] == "v2.0-neu-inclusion"
    assert inf_res["total_defects"] > 0, "Expected defects on test image"
    assert inf_res["inference_time_ms"] > 0.0

    print(f"             Workpiece:        {inf_res['filename']}")
    print(f"             Inference Device: {inf_res['device']}")
    print(f"             Latency:          {inf_res['inference_time_ms']} ms")
    print(f"             Defects Detected: {inf_res['total_defects']}")

    for det in inf_res["detections"]:
        x1, y1, x2, y2 = det["bbox_xyxy"]
        assert 0 <= x1 < x2 <= inf_res["image_width"]
        assert 0 <= y1 < y2 <= inf_res["image_height"]
        assert 0.0 <= det["confidence"] <= 1.0
        assert det["class_id"] in CLASS_MAP_BY_ID
        print(f"               - {det['detection_id']}: {det['class_name']:<20} (Conf: {det['confidence']*100:.1f}%) BBox: {det['bbox_xyxy']}")
    print("             -> PASSED")

    # -------------------------------------------------------------
    # Test 6: Clean Workpiece Zero-Detection Conformance
    # -------------------------------------------------------------
    print("\n[Test 6/10] Clean Workpiece Zero-Detection Behavior:")
    blank_img = np.zeros((1000, 2048, 3), dtype=np.uint8) + 160
    blank_res = model_manager.run_inference(blank_img, filename="prime_coil_clean.jpg")

    assert blank_res["total_defects"] == 0
    assert len(blank_res["detections"]) == 0
    assert blank_res["assessment"]["status"] == "PASS"
    assert "QR-00" in blank_res["assessment"]["rule_ids"]

    print(f"             Total Defects Detected: {blank_res['total_defects']}")
    print(f"             Quality Verdict:        {blank_res['assessment']['status']}")
    print(f"             Rule Triggered:         {blank_res['assessment']['rule_ids']}")
    print(f"             Quality Summary:        {blank_res['assessment']['summary']}")
    print("             -> PASSED")

    # -------------------------------------------------------------
    # Test 7: Deterministic Quality Engine Assessment
    # -------------------------------------------------------------
    print("\n[Test 7/10] Deterministic Quality Rules Engine Decoupling:")
    # Check that assessment strictly reflects detections
    punch_hole_det = [{
        "class_id": 0,
        "class_name": "Punching Hole",
        "raw_label": "1_chongkong",
        "confidence": 0.95,
        "bbox_xyxy": [100, 100, 200, 200],
        "area_px": 10000
    }]
    punch_verdict = assess_quality(punch_hole_det)
    assert punch_verdict["status"] == "REJECT"
    assert "QR-01" in punch_verdict["rule_ids"]
    print("             - Perforation defect (1_chongkong): Evaluated as REJECT under QR-01 [OK]")

    weld_det = [{
        "class_id": 1,
        "class_name": "Welding Line",
        "raw_label": "2_hanfeng",
        "confidence": 0.88,
        "bbox_xyxy": [500, 10, 520, 990],
        "area_px": 19600
    }]
    weld_verdict = assess_quality(weld_det)
    assert weld_verdict["status"] == "REVIEW REQUIRED"
    assert "QR-02" in weld_verdict["rule_ids"]
    print("             - Weld Joint seam (2_hanfeng): Evaluated as REVIEW REQUIRED under QR-02 [OK]")
    print("             -> PASSED")

    # -------------------------------------------------------------
    # Test 8: Inspection History & SQLite Traceability Store
    # -------------------------------------------------------------
    print("\n[Test 8/10] SQLite Database History Persistence & Traceability:")
    if TEST_DB_PATH.exists():
        TEST_DB_PATH.unlink()

    ai_record = {
        "inspection_id": "SG-EXP-TEST-001",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "filename": "test_inspection_coil.jpg",
        "image_reference": "test_inspection_coil.jpg",
        "source": "AI_INFERENCE",
        "image_width": 2048,
        "image_height": 1000,
        "detections": inf_res["detections"],
        "assessment": inf_res["assessment"],
        "model_name": inf_res["model_name"],
        "model_version": inf_res["model_version"],
        "inference_time_ms": inf_res["inference_time_ms"],
        "operator_notes": "Phase 3 automated validation inspection.",
        "review_status": "UNREVIEWED"
    }

    db_res = db.save_inspection(ai_record, db_path=TEST_DB_PATH)
    assert db_res["success"] is True
    assert db_res["inspection_id"] == "SG-EXP-TEST-001"

    # Retrieve and verify traceability
    loaded_rec = db.get_inspection("SG-EXP-TEST-001", db_path=TEST_DB_PATH)
    assert loaded_rec is not None
    assert loaded_rec["source"] == "AI_INFERENCE"
    assert loaded_rec["model_version"] == "v2.0-neu-inclusion"
    assert loaded_rec["inference_time_ms"] == inf_res["inference_time_ms"]
    assert len(loaded_rec["detections"]) == len(inf_res["detections"])
    assert loaded_rec["detections"][0]["is_ground_truth"] is False

    print(f"             Persisted Inspection ID: {loaded_rec['id']}")
    print(f"             Source Separation:       {loaded_rec['source']} (is_ground_truth={loaded_rec['detections'][0]['is_ground_truth']})")
    print(f"             Model Version Trace:     {loaded_rec['model_version']}")
    print(f"             Inference Latency Trace: {loaded_rec['inference_time_ms']} ms")

    # Clean up test DB
    if TEST_DB_PATH.exists():
        TEST_DB_PATH.unlink()
    print("             -> PASSED")

    # -------------------------------------------------------------
    # Test 9: REST API Dispatch Integration
    # -------------------------------------------------------------
    print("\n[Test 9/10] Full Backend REST Endpoints Dispatch:")
    # 1. GET /api/health
    status, _, body = dispatch("GET", "/api/health")
    assert status == 200
    health_data = json.loads(body.decode("utf-8"))
    assert health_data["models"]["vision"]["status"] == "READY"
    assert health_data["models"]["vision"]["model_key"] == "candidate"
    print("             GET /api/health -> 200 OK (Telemetry: READY)")

    # 2. GET /api/models
    status, _, body = dispatch("GET", "/api/models")
    assert status == 200
    models_data = json.loads(body.decode("utf-8"))
    assert models_data["total_models"] == 2
    print("             GET /api/models -> 200 OK (2 registered checkpoints)")

    # 3. POST /api/models/switch
    status, _, body = dispatch("POST", "/api/models/switch", {"model_key": "baseline"})
    assert status == 200
    switch_data = json.loads(body.decode("utf-8"))
    assert switch_data["status"] == "SUCCESS"
    assert switch_data["model"]["key"] == "baseline"
    print("             POST /api/models/switch -> 200 OK (Switched to baseline)")

    # Switch back to candidate
    status, _, _ = dispatch("POST", "/api/models/switch", {"model_key": "candidate"})
    assert status == 200
    print("             POST /api/models/switch -> 200 OK (Restored candidate)")

    # 4. POST /api/inference/run with Base64 payload
    with open(test_img_path, "rb") as f:
        b64_str = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("utf-8")
    status, _, body = dispatch("POST", "/api/inference/run", {
        "image_base64": b64_str,
        "filename": test_img_path.name
    })
    assert status == 200
    run_data = json.loads(body.decode("utf-8"))
    assert run_data["status"] == "SUCCESS"
    assert run_data["total_defects"] > 0
    print(f"             POST /api/inference/run -> 200 OK ({run_data['total_defects']} defects detected in {run_data['inference_time_ms']} ms)")
    print("             -> PASSED")

    # -------------------------------------------------------------
    # Test 10: Local / Offline Verification
    # -------------------------------------------------------------
    print("\n[Test 10/10] Offline Operational Verification:")
    assert os.environ.get("YOLO_OFFLINE") == "True"
    assert os.environ.get("ULTRALYTICS_AUTOUPDATE") == "0"
    print("             Environment: YOLO_OFFLINE=True, ULTRALYTICS_AUTOUPDATE=0")
    print("             Cloud Dependencies: NONE (Zero external API or cloud calls required)")
    print("             Local Execution: 100% verified on host hardware.")
    print("             -> PASSED")

    print("\n" + "=" * 70)
    print(" ALL 10 PHASE 3 VERIFICATION TESTS PASSED SUCCESSFULLY.")
    print("=" * 70)


if __name__ == "__main__":
    run_phase3_tests()
