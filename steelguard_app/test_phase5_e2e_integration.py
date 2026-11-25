"""
STEELGUARD AI — Phase 5 End-to-End Local Integration Test Suite
Verifies:
1. Real YOLO inference -> GC10 detection + deterministic QR-01 REJECT verdict.
2. Deterministic summary generation -> canonical evidence, severity, rules.
3. Real ExecuTorch Llama 3.2 1B Instruct CPU advisory via safe WSL bridge.
4. Truthful fallback verification (unconfigured / timeout).
5. Separate Severstal U-Net pixel segmentation and absolute model taxonomy isolation.
6. SQLite persistence and traceability for both YOLO and U-Net stores.
"""

import io
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

# Environment setup
ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "steelguard_app" / "backend"
sys.path.insert(0, str(BACKEND))

os.environ["YOLO_OFFLINE"] = "True"
os.environ["ULTRALYTICS_AUTOUPDATE"] = "0"

import db
import model_manager as mm
from advisory_service import AdvisoryService
from inspection_intelligence import summarize_inspection
import server


class Phase5E2EIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory(prefix="steelguard-p5-e2e-")
        cls.orig_db = db.DEFAULT_DB_PATH
        cls.orig_cfg = mm.CONFIG_FILE
        cls.test_db = Path(cls.temp_dir.name) / "test_history.db"
        cls.test_cfg = Path(cls.temp_dir.name) / "model_config.json"
        db.DEFAULT_DB_PATH = cls.test_db
        mm.CONFIG_FILE = cls.test_cfg
        db.init_db()

        # WSL Manifest path for real Llama 3.2 ExecuTorch runtime
        cls.wsl_manifest = "/home/sai01/steelguard-host-bundle-20261116/advisory.manifest.json"

    @classmethod
    def tearDownClass(cls):
        db.DEFAULT_DB_PATH = cls.orig_db
        mm.CONFIG_FILE = cls.orig_cfg
        cls.temp_dir.cleanup()

    def dispatch(self, path, method="GET", payload=None, manifest=None):
        h = server.SteelGuardRequestHandler.__new__(server.SteelGuardRequestHandler)
        h.path = path
        body = json.dumps(payload).encode("utf-8") if payload is not None else b""
        h.headers = {"Content-Length": str(len(body)), "Content-Type": "application/json"}
        h.rfile = io.BytesIO(body)
        h.wfile = io.BytesIO()
        status_codes = []
        h.send_response = lambda code: status_codes.append(code)
        h.send_header = lambda *args: None
        h.end_headers = lambda: None

        service = AdvisoryService(manifest)
        orig_adv = server.advisory_service
        server.advisory_service = service
        try:
            getattr(h, "do_" + method)()
        finally:
            server.advisory_service = orig_adv

        res_body = h.wfile.getvalue()
        parsed = json.loads(res_body.decode("utf-8")) if res_body else {}
        return status_codes[0] if status_codes else 200, parsed

    def test_01_yolo_inference_and_deterministic_verdict(self):
        """Test 1: Real YOLO detection on verified critical defect sample."""
        print("\n[Step 1] Running real YOLO inference on sample 'img_01_3402617700_01009'...")
        code, res = self.dispatch("/api/inference/run", method="POST", payload={"sample_id": "img_01_3402617700_01009"})
        self.assertEqual(code, 200)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertGreater(len(res["detections"]), 0)

        # Check detected labels
        raw_labels = [d["raw_label"] for d in res["detections"]]
        class_names = [d["class_name"] for d in res["detections"]]
        print(f"         Detections found: {class_names} ({raw_labels}) in {res['inference_time_ms']} ms")
        self.assertGreater(len(raw_labels), 0)

        # Check deterministic quality verdict (QR-01 Critical Defect)
        assessment = res["assessment"]
        self.assertEqual(assessment["status"], "REJECT")
        self.assertIn("QR-01", assessment["rule_ids"])
        print(f"         Verdict: {assessment['status']} | Rule: {assessment['rule_ids']}")
        self.__class__.yolo_result = res

    def test_02_deterministic_summary_contract(self):
        """Test 2: Deterministic Inspection Intelligence summary endpoint."""
        print("\n[Step 2] Generating deterministic Inspection Intelligence summary...")
        yolo_res = getattr(self.__class__, "yolo_result", None)
        self.assertIsNotNone(yolo_res, "Prerequisite Step 1 must have run")

        summary_payload = {
            "source": yolo_res["source"],
            "status": yolo_res["status"],
            "detections": yolo_res["detections"],
            "filename": yolo_res["filename"],
            "model_name": yolo_res["model_name"],
            "model_version": yolo_res["model_version"],
            "model_key": yolo_res["model_key"],
            "sha256": yolo_res["sha256"],
            "image_width": yolo_res["image_width"],
            "image_height": yolo_res["image_height"]
        }

        code, summary = self.dispatch("/api/intelligence/summarize", method="POST", payload=summary_payload)
        self.assertEqual(code, 200)
        self.assertEqual(summary["engine"], "deterministic")
        self.assertFalse(summary["llm_used"])
        self.assertFalse(summary["segmentation_fused"])
        self.assertEqual(summary["assessment"]["status"], "REJECT")
        self.assertIn("Hold", summary["recommended_action"])

        print(f"         Summary verdict: {summary['assessment']['status']}")
        print(f"         Recommended action: {summary['recommended_action']}")
        self.__class__.canonical_summary = summary

    def test_03_real_executorch_llama_advisory(self):
        """Test 3: Real ExecuTorch Llama 3.2 1B inference via safe local WSL bridge."""
        print("\n[Step 3] Querying real ExecuTorch Llama 3.2 1B Instruct runtime on CPU...")
        summary = getattr(self.__class__, "canonical_summary", None)
        self.assertIsNotNone(summary, "Prerequisite Step 2 must have run")

        code, adv = self.dispatch("/api/intelligence/advisory", method="POST",
                                  payload={"summary": summary}, manifest=self.wsl_manifest)
        self.assertEqual(code, 200)
        print(f"         Advisory Source: {adv['source']}")
        print(f"         LLM Used:        {adv['llm_used']}")
        print(f"         Reason Code:     {adv['reason_code']}")
        print(f"         Model ID:        {adv.get('model_id')}")
        print(f"         Verdict Match:   {adv['quality_verdict']} == {summary['assessment']['status']}")
        print(f"         Recommendations: {adv['recommendations']}")

        self.assertEqual(adv["source"], "EXECUTORCH")
        self.assertTrue(adv["llm_used"])
        self.assertEqual(adv["reason_code"], "OK")
        self.assertEqual(adv["model_id"], "meta-llama/Llama-3.2-1B-Instruct")
        self.assertEqual(adv["quality_verdict"], "REJECT")
        self.assertFalse(adv["segmentation_fused"])
        self.assertGreater(len(adv["recommendations"]), 0)
        self.__class__.advisory_result = adv

    def test_04_truthful_rule_based_fallback(self):
        """Test 4: Truthful deterministic fallback when ExecuTorch runtime is unconfigured."""
        print("\n[Step 4] Testing truthful fallback with unconfigured advisory...")
        summary = getattr(self.__class__, "canonical_summary", None)
        self.assertIsNotNone(summary)

        # Call with manifest=None
        code, fallback = self.dispatch("/api/intelligence/advisory", method="POST",
                                       payload={"summary": summary}, manifest=None)
        self.assertEqual(code, 200)
        self.assertEqual(fallback["source"], "RULE_BASED_FALLBACK")
        self.assertFalse(fallback["llm_used"])
        self.assertEqual(fallback["reason_code"], "NOT_CONFIGURED")
        self.assertEqual(fallback["quality_verdict"], "REJECT")
        self.assertIn("Hold", fallback["recommendations"][0])
        print("         Fallback correctly returned RULE_BASED_FALLBACK with NOT_CONFIGURED reason code.")

    def test_05_sqlite_traceability_and_persistence(self):
        """Test 5: SQLite inspection saving and history retrieval."""
        print("\n[Step 5] Persisting inspection to SQLite database...")
        yolo_res = getattr(self.__class__, "yolo_result", None)
        adv = getattr(self.__class__, "advisory_result", None)

        operator_notes = f"Verified via ExecuTorch Llama advisory ({adv['source']}: {adv['recommendations'][0]})"
        save_payload = {
            "source": "AI_INFERENCE",
            "filename": yolo_res["filename"],
            "image_reference": "img_01_3402617700_01009",
            "image_width": yolo_res["image_width"],
            "image_height": yolo_res["image_height"],
            "detections": yolo_res["detections"],
            "assessment": yolo_res["assessment"],
            "model_name": yolo_res["model_name"],
            "model_version": yolo_res["model_version"],
            "inference_time_ms": yolo_res["inference_time_ms"],
            "operator_notes": operator_notes,
            "review_status": "UNREVIEWED"
        }

        code, save_res = self.dispatch("/api/inspections", method="POST", payload=save_payload)
        self.assertEqual(code, 200)
        self.assertTrue(save_res["success"])
        insp_id = save_res["inspection_id"]
        print(f"         Saved record with Inspection ID: {insp_id}")

        # Verify in SQLite directly
        conn = sqlite3.connect(self.test_db)
        cursor = conn.cursor()
        cursor.execute("SELECT id, assessment_status, model_name, operator_notes FROM inspections WHERE id = ?", (insp_id,))
        row = cursor.fetchone()
        conn.close()

        self.assertIsNotNone(row)
        self.assertEqual(row[1], "REJECT")
        self.assertIn("ExecuTorch", row[3])
        print(f"         Verified SQLite persistence: ID={row[0]} | Status={row[1]} | Notes={row[3][:40]}...")

    def test_06_separate_severstal_unet_segmentation(self):
        """Test 6: U-Net pixel segmentation and absolute taxonomy isolation."""
        print("\n[Step 6] Running Severstal U-Net pixel segmentation on '0007a71bf.jpg'...")
        code, seg_res = self.dispatch("/api/segmentation/run", method="POST",
                                      payload={"sample_id": "0007a71bf.jpg", "save_to_db": True})
        self.assertEqual(code, 200)
        self.assertEqual(seg_res["status"], "SUCCESS")
        self.assertIn("Lightweight U-Net", seg_res["model_name"])
        self.assertGreater(seg_res["total_defect_pixels"], 0)
        self.assertTrue(seg_res.get("db_saved", False))

        print(f"         U-Net inference time: {seg_res['inference_time_ms']} ms")
        print(f"         Defect pixels: {seg_res['total_defect_pixels']} ({seg_res['defect_percentage']}%)")
        print(f"         Classes detected: {[c['name'] for c in seg_res['class_breakdown'] if c['pixel_count'] > 0 and c['class_id'] > 0]}")

        # Verify taxonomy separation: U-Net does not have GC10 classes
        for c in seg_res["class_breakdown"]:
            self.assertIn(c["class_id"], [0, 1, 2, 3, 4])
            self.assertNotIn("chongkong", c["name"])

        # Verify SQLite segmentation table is decoupled
        conn = sqlite3.connect(self.test_db)
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM segmentation_inspections")
        seg_count = cursor.fetchone()[0]
        cursor.execute("SELECT count(*) FROM inspections")
        det_count = cursor.fetchone()[0]
        conn.close()

        self.assertEqual(seg_count, 1)
        self.assertEqual(det_count, 1)
        print("         Verified separate SQLite tables: 1 YOLO inspection, 1 U-Net segmentation.")


if __name__ == "__main__":
    unittest.main()
