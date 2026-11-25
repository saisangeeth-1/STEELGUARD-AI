"""HTTP dispatch coverage. Isolate persistence before importing the real server."""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP / "backend"))
import db


class IntelligenceApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.original_db = db.DEFAULT_DB_PATH
        db.DEFAULT_DB_PATH = Path(cls.temp.name) / "test.db"
        from server import SteelGuardRequestHandler
        cls.handler = SteelGuardRequestHandler

    @classmethod
    def tearDownClass(cls):
        db.DEFAULT_DB_PATH = cls.original_db
        cls.temp.cleanup()

    def dispatch(self, raw):
        h = self.handler.__new__(self.handler)
        h.path = "/api/intelligence/summarize"
        h.headers = {"Content-Length": str(len(raw))}
        h.rfile = io.BytesIO(raw)
        h.wfile = io.BytesIO()
        statuses = []
        h.send_response = lambda code: statuses.append(code)
        h.send_header = lambda *args: None
        h.end_headers = lambda: None
        h.do_POST()
        return statuses[0], json.loads(h.wfile.getvalue())

    def test_invalid_json_and_inspections(self):
        for raw in [b"{", b"[]", b"null", b"{}", b'{"source":"AI_INFERENCE","status":"ERROR","detections":[]}']:
            with self.subTest(raw=raw):
                code, body = self.dispatch(raw)
                self.assertEqual(code, 400)
                self.assertIn("error", body)

    def test_empty_completed_inference(self):
        code, body = self.dispatch(json.dumps({"source": "AI_INFERENCE", "status": "SUCCESS", "detections": []}).encode())
        self.assertEqual(code, 200)
        self.assertEqual(body["evidence"], [])
        self.assertTrue(any("defect-free" in w for w in body["warnings"]))

    def test_multiple_defects_and_no_history_write(self):
        p = {"source": "AI_INFERENCE", "status": "SUCCESS", "filename": "test.jpg",
             "detections": [{"raw_label": "1_chongkong", "area_px": 100},
                            {"raw_label": "7_yiwu", "bbox_xyxy": [0, 0, 100, 100]}]}
        code, body = self.dispatch(json.dumps(p).encode())
        self.assertEqual(code, 200)
        self.assertEqual(body["assessment"]["status"], "REJECT")
        self.assertEqual(len(body["evidence"]), 2)
        self.assertFalse(body["llm_used"])
        with db.ManagedConnection() as conn:
            for table in ["inspections", "segmentation_inspections"]:
                self.assertEqual(conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
