"""
STEELGUARD AI - Backend Unit Test Suite
Verifies API routes, static asset delivery, and truthful telemetry.
"""

import sys
import io
import json
from pathlib import Path

# Add backend directory to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "backend"))

from server import SteelGuardRequestHandler, get_real_system_telemetry, GC10_CLASSES, STATIC_DIR


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


def dispatch_test_route(path: str):
    raw = f"GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n".encode("utf-8")
    req = MockSocketRequest(raw)
    SteelGuardRequestHandler(req, ("127.0.0.1", 1234), None)
    output = req.wfile.getvalue().decode("utf-8", errors="replace")
    header_part, _, body = output.partition("\r\n\r\n")
    status_line = header_part.split("\r\n")[0]
    return status_line, body


def run_tests():
    print("=" * 60)
    print("STEELGUARD AI - STAGE 2 VERIFICATION TEST SUITE")
    print("=" * 60)

    # 1. Telemetry Verification
    telemetry = get_real_system_telemetry()
    print("[1/5] Host Telemetry Truthfulness:")
    print(f"      Architecture: {telemetry['host']['architecture']}")
    print(f"      Vision Status: {telemetry['models']['vision']['status']}")
    print(f"      LLM Status: {telemetry['models']['llm']['status']}")
    print(f"      Camera Status: {telemetry['camera']['status']}")
    v_status = telemetry["models"]["vision"]["status"]
    assert v_status in ("READY", "NOT_LOADED"), f"Vision status must be READY or NOT_LOADED, got {v_status}"
    if v_status == "READY":
        assert telemetry["models"]["vision"]["checkpoint_path"] is not None
        assert telemetry["models"]["vision"]["runtime_ready"] is True
    else:
        assert telemetry["models"]["vision"]["checkpoint_path"] is None
        assert telemetry["models"]["vision"]["runtime_ready"] is False
    assert telemetry["models"]["llm"]["status"] == "NOT_DETECTED", "LLM status must be NOT_DETECTED"
    assert telemetry["camera"]["status"] == "OFFLINE", "Camera status must be OFFLINE"
    print("      -> PASSED")

    # 2. GC10 Catalog Verification
    print("[2/5] GC10-DET Defect Catalog:")
    print(f"      Total Classes: {len(GC10_CLASSES)}")
    assert len(GC10_CLASSES) == 10
    assert GC10_CLASSES[0]["code"] == "1_chongkong"
    print("      -> PASSED")

    # 3. Static Assets Check
    print("[3/5] Static File Verification:")
    assets = [
        STATIC_DIR / "index.html",
        STATIC_DIR / "css" / "tokens.css",
        STATIC_DIR / "css" / "layout.css",
        STATIC_DIR / "css" / "components.css",
        STATIC_DIR / "js" / "app.js",
    ]
    for asset in assets:
        assert asset.is_file(), f"Missing file: {asset}"
        print(f"      Found: {asset.name} ({asset.stat().st_size} bytes)")
    print("      -> PASSED")

    # 4. API Endpoints Dispatch
    print("[4/5] In-Memory API Endpoint Dispatch:")
    status, body = dispatch_test_route("/api/health")
    assert "200" in status
    health_json = json.loads(body)
    assert health_json["models"]["vision"]["status"] in ("READY", "NOT_LOADED")
    print("      /api/health -> 200 OK")

    status, body = dispatch_test_route("/api/overview")
    assert "200" in status
    ov_real = json.loads(body)
    assert ov_real["is_demo"] is False
    assert ov_real["metrics"]["total_inspections"] == 0
    print("      /api/overview (Real mode) -> 200 OK (0 inspections, no fake data)")

    status, body = dispatch_test_route("/api/overview?demo=true")
    assert "200" in status
    ov_demo = json.loads(body)
    assert ov_demo["is_demo"] is True
    print("      /api/overview?demo=true (Demo mode) -> 200 OK (Cleanly flagged DEMO DATA)")

    # 5. Root HTML Dispatch
    print("[5/5] Static HTML Dispatch:")
    status, body = dispatch_test_route("/index.html")
    assert "200" in status
    assert "STEELGUARD" in body
    print(f"      /index.html -> 200 OK ({len(body)} chars)")

    print("=" * 60)
    print("ALL STAGE 2 TESTS PASSED SUCCESSFULLY.")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
