"""Advisory API tests with private SQLite/config; no real Llama inference."""
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
import urllib.request

APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP / "backend"))
import db
from advisory_service import AdvisoryService
from inspection_intelligence import summarize_inspection


class AdvisoryApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.original_db = db.DEFAULT_DB_PATH
        db.DEFAULT_DB_PATH = Path(cls.temp.name) / "advisory.db"
        import model_manager
        cls.original_config = model_manager.CONFIG_FILE
        model_manager.CONFIG_FILE = Path(cls.temp.name) / "model_config.json"
        import server
        cls.server = server

    @classmethod
    def tearDownClass(cls):
        import model_manager
        model_manager.CONFIG_FILE = cls.original_config
        db.DEFAULT_DB_PATH = cls.original_db
        cls.temp.cleanup()

    def dispatch(self, payload=None, *, raw=None, length=None, method="POST"):
        if raw is None:
            raw = json.dumps(payload).encode()
        h = self.server.SteelGuardRequestHandler.__new__(self.server.SteelGuardRequestHandler)
        h.path = "/api/intelligence/advisory" + ("/status" if method == "GET" else "")
        h.headers = {"Content-Length": str(len(raw) if length is None else length)}
        h.rfile = io.BytesIO(raw)
        h.wfile = io.BytesIO()
        codes = []
        h.send_response = lambda code: codes.append(code)
        h.send_header = lambda *args: None
        h.end_headers = lambda: None
        with patch.object(self.server, "advisory_service", AdvisoryService()):
            getattr(h, "do_" + method)()
        return codes[0], json.loads(h.wfile.getvalue())

    def test_valid_summary_returns_labeled_fallback_without_writes(self):
        s = summarize_inspection({"source":"AI_INFERENCE", "status":"SUCCESS", "detections":[{"raw_label":"1_chongkong","area_px":50}]})
        code, body = self.dispatch({"summary":s})
        self.assertEqual(code, 200)
        self.assertEqual(body["source"], "RULE_BASED_FALLBACK")
        self.assertFalse(body["llm_used"])
        self.assertEqual(body["quality_verdict"], "REJECT")
        self.assertEqual(body["measured_facts"], s["evidence"])
        with db.ManagedConnection() as conn:
            for table in ("inspections", "segmentation_inspections"):
                self.assertEqual(conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0], 0)

    def test_empty_summary_is_qualified(self):
        s = summarize_inspection({"source":"AI_INFERENCE", "status":"SUCCESS", "detections":[]})
        code, body = self.dispatch({"summary":s})
        self.assertEqual(code, 200)
        self.assertEqual(body["reason_code"], "EMPTY_EVIDENCE")

    def test_invalid_summary_or_runtime_options_rejected(self):
        for payload in ([], {}, {"summary":{}}, {"summary":{}, "runner":"untrusted"}, {"image_base64":"pixels"}):
            with self.subTest(payload=payload):
                self.assertEqual(self.dispatch(payload)[0], 400)

    def test_request_limits_and_invalid_encoding(self):
        self.assertEqual(self.dispatch(raw=b"", length=131073)[0], 413)
        for raw, length in [(b"{", None), (b"\xff", None), (b"", -1), (b"", "bad")]:
            self.assertEqual(self.dispatch(raw=raw, length=length)[0], 400)

    def test_status_does_not_claim_inference(self):
        code, body = self.dispatch(method="GET")
        self.assertEqual(code, 200)
        self.assertFalse(body["configured"])
        self.assertFalse(body["configuration_validated"])
        self.assertFalse(body["live_deployment_attested"])

    def test_loopback_http_round_trip(self):
        s = summarize_inspection({"source":"AI_INFERENCE", "status":"SUCCESS", "detections":[{"raw_label":"1_chongkong","area_px":50}]})
        httpd = self.server.HTTPServer(("127.0.0.1", 0), self.server.SteelGuardRequestHandler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        with patch.object(self.server, "advisory_service", AdvisoryService()):
            thread.start()
            try:
                req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_port}/api/intelligence/advisory",
                    data=json.dumps({"summary":s}).encode(), headers={"Content-Type":"application/json"})
                with urllib.request.urlopen(req, timeout=5) as res:
                    self.assertEqual(res.status, 200)
                    self.assertEqual(json.load(res)["source"], "RULE_BASED_FALLBACK")
            finally:
                httpd.shutdown(); httpd.server_close(); thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
