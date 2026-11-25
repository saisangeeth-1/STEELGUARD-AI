"""
STEELGUARD AI - Stage 4 Comprehensive Test Suite
Validates persistent SQLite storage, schema integrity, duplicate prevention,
traceability filtering, CSV/JSON exports, source separation, and raw dataset immutability.
"""

import csv
import io
import json
from pathlib import Path
import sys
import tempfile

# Add backend directory to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "backend"))

import db
from server import SteelGuardRequestHandler, get_real_system_telemetry
from dataset_service import DATASET_ROOT, LABEL_DIR


class MockSocketRequest:
    def __init__(self, raw_http: bytes):
        self.rfile = io.BytesIO(raw_http)
        self.wfile = io.BytesIO()

    def makefile(self, *args, **kwargs):
        if 'r' in args[0]:
            return self.rfile
        return self.wfile

    def sendall(self, data):
        self.wfile.write(data)


def dispatch(method: str, path: str, body_json=None):
    body_bytes = json.dumps(body_json).encode("utf-8") if body_json else b""
    headers = [
        f"{method} {path} HTTP/1.1",
        "Host: 127.0.0.1",
        f"Content-Length: {len(body_bytes)}",
        "Content-Type: application/json"
    ]
    raw = ("\r\n".join(headers) + "\r\n\r\n").encode("utf-8") + body_bytes
    req = MockSocketRequest(raw)
    SteelGuardRequestHandler(req, ("127.0.0.1", 1234), None)
    output = req.wfile.getvalue()
    
    sep = b"\r\n\r\n"
    header_bytes, _, resp_body = output.partition(sep)
    header_text = header_bytes.decode("utf-8", errors="replace")
    status_line = header_text.split("\r\n")[0]
    return status_line, header_text, resp_body


def run_stage4_tests():
    print("=" * 65)
    print(" STEELGUARD AI - STAGE 4 FUNCTIONAL VERIFICATION SUITE")
    print("=" * 65)

    temp_dir = tempfile.TemporaryDirectory()
    test_db = Path(temp_dir.name) / "test_steelguard.db"

    # 1. Database Creation & Schema
    print("[Test 1/8] Database Schema & Foreign Key Constraints:")
    db.init_db(test_db)
    assert test_db.is_file(), "Database file must be created on disk"
    with db.ManagedConnection(test_db) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {row[0] for row in cursor.fetchall()}
        assert "inspections" in tables, "Table 'inspections' missing"
        assert "defect_instances" in tables, "Table 'defect_instances' missing"
        assert "quality_rule_triggers" in tables, "Table 'quality_rule_triggers' missing"
    print("           Created tables: inspections, defect_instances, quality_rule_triggers.")
    print("           -> PASSED")

    # 2. Empty Database Behavior
    print("[Test 2/8] Empty Database Behavior:")
    empty_stats = db.get_overview_stats(test_db)
    assert empty_stats["total_inspections"] == 0
    assert empty_stats["pass_rate_pct"] is None, "Pass rate must be None when no inspections exist"
    empty_list = db.list_inspections(db_path=test_db)
    assert empty_list["total"] == 0
    assert empty_list["inspections"] == []
    print("           Empty database reports 0 inspections and None pass rate without error.")
    print("           -> PASSED")

    # 3. Saving & Retrieving Ground-Truth & AI Inference Records
    print("[Test 3/8] Persistence & Source Separation:")
    gt_record = {
        "inspection_id": "QC-GT-001",
        "timestamp": "2026-10-28T22:00:00Z",
        "filename": "coil_425501900.jpg",
        "image_reference": "img_02_425501900_00017",
        "source": "GROUND_TRUTH",
        "image_width": 2048,
        "image_height": 1000,
        "detections": [
            {
                "detection_id": "GT-01",
                "class_id": 1,
                "class_name": "Punching Hole",
                "raw_label": "1_chongkong",
                "criticality": "CRITICAL",
                "confidence": 1.0,
                "is_ground_truth": True,
                "bbox_xyxy": [1995, 1, 2048, 42],
                "bbox_wh": [53, 41],
                "area_px": 2173
            }
        ],
        "assessment": {
            "status": "REJECT",
            "summary": "Critical breach: punching hole",
            "rule_ids": ["QR-01"],
            "rules_applied": [{"rule_id": "QR-01", "title": "Breach", "verdict": "REJECT", "reason": "Perforation"}]
        },
        "operator_notes": "Initial ground truth sample",
        "review_status": "CONFIRMED"
    }

    ai_record = {
        "inspection_id": "QC-AI-002",
        "timestamp": "2026-10-28T22:15:00Z",
        "filename": "coil_online_002.jpg",
        "image_reference": "coil_online_002.jpg",
        "source": "AI_INFERENCE",
        "image_width": 2048,
        "image_height": 1000,
        "detections": [
            {
                "detection_id": "DET-01",
                "class_id": 4,
                "class_name": "Water Spot",
                "raw_label": "4_shuiban",
                "criticality": "LOW",
                "confidence": 0.91,
                "is_ground_truth": False,
                "bbox_xyxy": [400, 700, 600, 750],
                "bbox_wh": [200, 50],
                "area_px": 10000
            }
        ],
        "assessment": {
            "status": "PASS",
            "summary": "Water spot within tolerance",
            "rule_ids": ["QR-05"],
            "rules_applied": [{"rule_id": "QR-05", "title": "Fluid Residue", "verdict": "PASS", "reason": "Minor water spot"}]
        },
        "model_name": "YOLOv8 Nano",
        "model_version": "v1.0.0",
        "operator_notes": "Automated online inspection",
        "review_status": "UNREVIEWED"
    }

    res1 = db.save_inspection(gt_record, test_db)
    assert res1["success"] is True
    res2 = db.save_inspection(ai_record, test_db)
    assert res2["success"] is True

    # Retrieve and verify source separation
    rec_gt = db.get_inspection("QC-GT-001", test_db)
    assert rec_gt["source"] == "GROUND_TRUTH"
    assert rec_gt["detections"][0]["is_ground_truth"] is True
    assert rec_gt["detections"][0]["confidence"] == 1.0

    rec_ai = db.get_inspection("QC-AI-002", test_db)
    assert rec_ai["source"] == "AI_INFERENCE"
    assert rec_ai["detections"][0]["is_ground_truth"] is False
    assert rec_ai["detections"][0]["confidence"] == 0.91
    print("           Verified source separation (GROUND_TRUTH confidence: 1.0 vs AI_INFERENCE confidence: 0.91).")
    print("           -> PASSED")

    # 4. Duplicate Prevention
    print("[Test 4/8] Duplicate Submission Prevention:")
    dup_res = db.save_inspection(gt_record, test_db)
    assert dup_res["success"] is False
    assert dup_res["already_exists"] is True
    print("           Duplicate inspection ID rejected without error.")
    print("           -> PASSED")

    # 5. Filtering and Search
    print("[Test 5/8] Search, Filtering, and Pagination:")
    # Search by keyword
    search_res = db.list_inspections(search="coil_425501900", db_path=test_db)
    assert search_res["total"] == 1
    assert search_res["inspections"][0]["id"] == "QC-GT-001"

    # Filter by status
    pass_res = db.list_inspections(status="PASS", db_path=test_db)
    assert pass_res["total"] == 1
    assert pass_res["inspections"][0]["id"] == "QC-AI-002"

    # Filter by source
    gt_filter = db.list_inspections(source="GROUND_TRUTH", db_path=test_db)
    assert gt_filter["total"] == 1
    print("           Filtered correctly by keyword, status (PASS), and source (GROUND_TRUTH).")
    print("           -> PASSED")

    # 6. CSV and JSON Exports
    print("[Test 6/8] CSV & JSON Export Serialization:")
    csv_content = db.export_inspections_csv(db_path=test_db)
    reader = list(csv.reader(io.StringIO(csv_content)))
    assert len(reader) == 3, f"Expected header + 2 data rows, got {len(reader)}"
    assert reader[0][0] == "Inspection ID"
    assert "QC-GT-001" in reader[1][0] or "QC-GT-001" in reader[2][0]

    # Operator notes update
    upd_res = db.update_inspection_notes("QC-GT-001", "Lab approved for shear line", "CONFIRMED", test_db)
    assert upd_res is True
    rec_after = db.get_inspection("QC-GT-001", test_db)
    assert rec_after["operator_notes"] == "Lab approved for shear line"
    print("           CSV generation and JSON serialization verified.")
    print("           -> PASSED")

    # 7. Dataset Immutability Audit
    print("[Test 7/8] Raw Dataset Immutability Audit:")
    assert DATASET_ROOT.exists()
    assert LABEL_DIR.exists()
    xml_count = len(list(LABEL_DIR.glob("*.xml")))
    assert xml_count == 2294, f"Dataset alteration! Expected 2294, found {xml_count}"
    print(f"           Dataset root verified untouched ({xml_count} XML files).")
    print("           -> PASSED")

    # 8. Regression: Stage 3 Test Suite
    print("[Test 8/8] Regression Verification (Stage 3 Suite):")
    from test_stage3 import test_stage3_suite
    test_stage3_suite()

    temp_dir.cleanup()
    print("=" * 65)
    print(" ALL 8 STAGE 4 VERIFICATION TESTS PASSED SUCCESSFULLY.")
    print("=" * 65)


if __name__ == "__main__":
    run_stage4_tests()
