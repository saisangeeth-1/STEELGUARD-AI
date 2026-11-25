"""
STEELGUARD AI - Part 3: Live Dashboard Acceptance and Browser Screenshot Verification
Starts server on localhost:8080, validates endpoints, uses Headless Edge to capture
visual evidence of Overview, Inspect, and Strip Segmentation views.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import urllib.request

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
BACKEND_DIR = APP_DIR / "backend"
SCREENSHOTS_DIR = PROJECT_ROOT / "runs" / "segmentation" / "screenshots"
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR = Path("C:/Users/SAI SANGEETH/.gemini/antigravity/brain/0d680f3a-f2b2-43d8-bcc2-2fe97b51246c")

sys.path.insert(0, str(BACKEND_DIR))
import server

EDGE_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

def run_browser_acceptance():
    print("=" * 70)
    print(" STEELGUARD AI — LIVE DASHBOARD ACCEPTANCE & BROWSER TESTING")
    print("=" * 70)

    # 1. Start Server on port 8080
    print("\n[Step 1] Starting SteelGuard Server on 127.0.0.1:8080...")
    httpd = server.HTTPServer(('127.0.0.1', 8080), server.SteelGuardRequestHandler)
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()
    time.sleep(1.0)
    print("Server running and accepting connections.")

    try:
        # 2. Test HTTP Endpoints
        print("\n[Step 2] Testing Core API Endpoints...")
        # Health
        res = urllib.request.urlopen("http://127.0.0.1:8080/api/health")
        health = json.loads(res.read().decode())
        print(f"  /api/health -> 200 OK | Host: {health['host']['architecture']} | Vision: {health['models']['vision']['status']} | Segmentation: {health['models']['segmentation']['status']}")

        # Segmentation Samples
        res = urllib.request.urlopen("http://127.0.0.1:8080/api/segmentation/samples?class=all&limit=5")
        samples_data = json.loads(res.read().decode())
        print(f"  /api/segmentation/samples -> 200 OK | Returned: {samples_data['count']} samples")
        sample_id = samples_data['samples'][0]['sample_id']

        # Segmentation Inference
        post_data = json.dumps({"sample_id": sample_id, "save_to_db": True, "operator_notes": "Live Acceptance Test"}).encode("utf-8")
        req = urllib.request.Request("http://127.0.0.1:8080/api/segmentation/run", data=post_data, headers={"Content-Type": "application/json"})
        res = urllib.request.urlopen(req)
        seg_result = json.loads(res.read().decode())
        print(f"  /api/segmentation/run -> 200 OK | Workpiece: {seg_result['filename']} | Defect: {seg_result['has_defect']} ({seg_result['total_defect_pixels']} px) | Latency: {seg_result['inference_time_ms']} ms")

        # Segmentation Inspections
        res = urllib.request.urlopen("http://127.0.0.1:8080/api/segmentation/inspections?limit=5")
        insp_history = json.loads(res.read().decode())
        print(f"  /api/segmentation/inspections -> 200 OK | Total logged: {insp_history['total']}")

        # 3. Headless Edge Browser Rendering & Screenshots
        print("\n[Step 3] Rendering Live Dashboard Views in Headless Microsoft Edge...")
        if os.path.exists(EDGE_PATH):
            views_to_capture = [
                ("dashboard_overview.png", "http://127.0.0.1:8080/index.html#overview", "Overview Command Center", 0),
                ("dashboard_inspect.png", "http://127.0.0.1:8080/index.html#inspect", "YOLO Inspection Workspace", 0),
                ("dashboard_segmentation.png", "http://127.0.0.1:8080/index.html#segmentation", "Strip Segmentation Console", 0),
                ("dashboard_segmentation_inferred.png", "http://127.0.0.1:8080/index.html?autorun=true#segmentation", "Strip Segmentation with Inferred Overlay", 3000),
                ("dashboard_history.png", "http://127.0.0.1:8080/index.html#history", "Inspection Traceability & History", 0)
            ]

            for item in views_to_capture:
                fname, url, label, delay_budget = item
                out_path = SCREENSHOTS_DIR / fname
                cmd = [
                    EDGE_PATH,
                    "--headless=new",
                    "--disable-gpu",
                    "--hide-scrollbars",
                    "--window-size=1600,1050",
                ]
                if delay_budget > 0:
                    cmd.append(f"--virtual-time-budget={delay_budget}")
                cmd.extend([
                    f"--screenshot={out_path}",
                    url
                ])
                print(f"  Capturing {label} -> {fname}...")
                subprocess.run(cmd, check=True, timeout=25)
                assert out_path.is_file(), f"Screenshot failed for {fname}"
                print(f"    Saved: {out_path} ({out_path.stat().st_size:,} bytes)")

                # Also copy to artifacts directory
                if ARTIFACTS_DIR.is_dir():
                    shutil.copy2(out_path, ARTIFACTS_DIR / fname)
                    print(f"    Copied to artifact dir: {ARTIFACTS_DIR / fname}")

            print("\nAll 4 live dashboard views rendered and captured successfully!")
        else:
            print(f"Edge executable not found at {EDGE_PATH}")

    finally:
        print("\nShutting down acceptance test server...")
        httpd.shutdown()
        print("Server closed cleanly.")

if __name__ == "__main__":
    run_browser_acceptance()
