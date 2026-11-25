"""Run with the dedicated Python; uses no datasets or checkpoints."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))
from inspection_intelligence import summarize_inspection
from quality_engine import assess_quality


class IntelligenceTests(unittest.TestCase):
    def payload(self, label="1_chongkong", area=20):
        return {"source": "AI_INFERENCE", "status": "SUCCESS",
                "detections": [{"raw_label": label, "area_px": area}]}

    def test_critical_rule_and_no_mutation(self):
        p = self.payload()
        original = copy.deepcopy(p)
        result = summarize_inspection(p)
        self.assertEqual(result["assessment"]["status"], "REJECT")
        self.assertEqual(result["assessment"]["rule_ids"], ["QR-01"])
        self.assertEqual(p, original)
        self.assertFalse(result["llm_used"])

    def test_area_only_fluid_and_boundary(self):
        self.assertEqual(assess_quality(self.payload("4_shuiban", 30000)["detections"])["status"], "PASS")
        self.assertEqual(summarize_inspection(self.payload("4_shuiban", 30001))["assessment"]["status"], "REVIEW REQUIRED")

    def test_inclusion_boundary(self):
        for area, verdict in [(4999, "REVIEW REQUIRED"), (5000, "REJECT")]:
            self.assertEqual(summarize_inspection(self.payload("7_yiwu", area))["assessment"]["status"], verdict)

    def test_missing_or_failed_inference(self):
        for p in [{}, [], {"source": "AI_INFERENCE", "detections": []},
                  {"source": "AI_INFERENCE", "status": "SUCCESS"}]:
            with self.assertRaises(ValueError):
                summarize_inspection(p)

    def test_invalid_geometry(self):
        for d in [{"raw_label": "x", "area_px": float("nan")},
                  {"raw_label": "x", "bbox_xyxy": [4, 0, 2, 3]},
                  {"raw_label": "x", "area_px": -1}, None]:
            p = self.payload()
            p["detections"] = [d]
            with self.assertRaises(ValueError):
                summarize_inspection(p)

    def test_unknown_class_with_passing_known_class(self):
        p = self.payload("4_shuiban")
        p["detections"].append({"raw_label": "unknown", "area_px": 10})
        self.assertEqual(summarize_inspection(p)["assessment"]["status"], "REVIEW REQUIRED")

    def test_ground_truth_separation(self):
        p = self.payload()
        p["source"] = "GROUND_TRUTH"
        r = summarize_inspection(p)
        self.assertIn("Dataset annotations, not model predictions.", r["warnings"])
        self.assertFalse(r["segmentation_fused"])

    def test_empty_success_is_qualified(self):
        p = self.payload()
        p["detections"] = []
        self.assertIn("No detections does not establish a defect-free surface.", summarize_inspection(p)["warnings"])

    def test_multiple_defects_policy_severity_and_evidence(self):
        p = self.payload()
        p.update(filename="sheet.jpg", model_name="YOLO", sha256="checkpoint-hash", image_width=200)
        p["detections"] = [
            {"raw_label": "1_chongkong", "bbox_xyxy": [1, 2, 11, 12], "confidence": 0.8},
            {"raw_label": "4_shuiban", "area_px": 20, "criticality": "FABRICATED"}
        ]
        original = copy.deepcopy(p)
        r = summarize_inspection(p)
        self.assertEqual(p, original)
        self.assertEqual(r["assessment"]["status"], "REJECT")
        self.assertEqual([d["severity"] for d in r["evidence"]], ["CRITICAL", "LOW"])
        self.assertEqual(r["evidence"][0]["area_px"], 100)
        self.assertEqual(r["provenance"]["filename"], "sheet.jpg")
        self.assertIn("Hold", r["recommended_action"])

    def test_confidence_validation(self):
        for value in [True, float("inf"), -0.1, 1.1, "high"]:
            p = self.payload()
            p["detections"][0]["confidence"] = value
            with self.assertRaises(ValueError):
                summarize_inspection(p)

    def test_severstal_not_fused(self):
        p = self.payload()
        p["segmentation"] = {"total_defect_pixels": 999, "mask": "NOT-GC10"}
        r = summarize_inspection(p)
        self.assertNotIn("999", str(r))
        p["taxonomy"] = "Severstal"
        with self.assertRaises(ValueError):
            summarize_inspection(p)

    def test_null_area_derived_from_bbox(self):
        p = self.payload("4_shuiban", None)
        p["detections"][0]["bbox_xyxy"] = [0, 0, 10, 10]
        self.assertEqual(summarize_inspection(p)["evidence"][0]["area_px"], 100)


if __name__ == "__main__":
    unittest.main(verbosity=2)
