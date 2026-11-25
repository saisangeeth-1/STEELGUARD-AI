"""Real Edge headless smoke test: sample -> YOLO -> summary -> rendered assistant.

Uses a loopback-only test route to drive the unchanged dashboard. Temporary DB/config;
no history save, training, external browser dependency, or production server required.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "steelguard_app/backend"))

HARNESS = r"""
<script>
document.addEventListener('DOMContentLoaded', async () => {
  const waitFor = async predicate => {
    for (let i=0; i<600; i++) {
      if (predicate()) return;
      await new Promise(resolve => setTimeout(resolve, 50));
    }
    throw new Error('Timed out waiting for dashboard state');
  };
  try {
    await loadVerifiedSample('img_01_3402617700_01009');
    await waitFor(() => state.currentImage && state.dataMode === 'GROUND_TRUTH');
    await handleRunAIInference();
    await waitFor(() => document.getElementById('intelligence-content').dataset.status === 'ready');
    navigateTo('assistant');
    const host = document.getElementById('intelligence-content');
    for (const expected of ['Quality verdict:', 'Policy severity', 'Policy:', 'SHA-256', 'Provenance', 'Recommended operator action', 'U-Net segmentation is separate']) {
      if (!host.textContent.includes(expected)) throw new Error('Missing rendered evidence: ' + expected);
    }
    if (state.dataMode !== 'AI_INFERENCE' || !state.detections.length) throw new Error('YOLO did not populate detection state');
    for (const detection of state.detections) {
      if (!host.textContent.includes(detection.raw_label)) throw new Error('Missing actual detection class');
    }
    if (document.getElementById('btn-save-inspection').disabled) throw new Error('Save action unavailable');
    document.body.dataset.intelligenceTest = 'passed';
  } catch(error) {
    document.body.dataset.intelligenceTest = 'failed';
    document.body.dataset.intelligenceError = error.message;
  }
});
</script>
"""


def main():
    os.environ["YOLO_OFFLINE"] = "True"
    os.environ["ULTRALYTICS_AUTOUPDATE"] = "0"
    output = ROOT / "runs/phase5b_verification"
    output.mkdir(parents=True, exist_ok=True)
    edge = Path(os.environ.get("PROGRAMFILES(X86)", "C:/Program Files (x86)")) / "Microsoft/Edge/Application/msedge.exe"
    if not edge.is_file():
        raise RuntimeError("Microsoft Edge is required for this headless browser smoke test")
    with tempfile.TemporaryDirectory(prefix="steelguard-browser-") as temp:
        import db
        db.DEFAULT_DB_PATH = Path(temp) / "browser.db"
        import model_manager
        model_manager.CONFIG_FILE = Path(temp) / "model_config.json"
        import server

        class TestHandler(server.SteelGuardRequestHandler):
            def do_GET(self):
                if self.path == "/__test/intelligence":
                    page = (ROOT / "steelguard_app/static/index.html").read_text(encoding="utf-8")
                    # Absolute script URLs keep the test-only path from changing asset resolution.
                    page = page.replace('<head>', '<head><base href="/">')
                    self._set_headers(200, "text/html; charset=utf-8")
                    self.wfile.write((page + HARNESS).encode("utf-8"))
                else:
                    super().do_GET()

        httpd = server.HTTPServer(("127.0.0.1", 0), TestHandler)
        worker = threading.Thread(target=httpd.serve_forever, daemon=True)
        worker.start()
        try:
            proc = subprocess.run([
                str(edge), "--headless", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
                "--disable-background-networking", f"--user-data-dir={Path(temp) / 'edge-profile'}",
                "--window-size=1440,1100", "--virtual-time-budget=45000", "--dump-dom",
                f"--screenshot={output / 'intelligence_dashboard.png'}",
                f"http://127.0.0.1:{httpd.server_port}/__test/intelligence"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=90)
            (output / "browser_dom.html").write_bytes(proc.stdout)
            (output / "browser_stderr.log").write_bytes(proc.stderr)
            html = proc.stdout.decode("utf-8", errors="replace")
            if proc.returncode != 0 or 'data-intelligence-test="passed"' not in html:
                raise AssertionError(f"Browser smoke failed (exit {proc.returncode}); inspect browser_dom.html and browser_stderr.log")
            with db.ManagedConnection() as conn:
                for table in ("inspections", "segmentation_inspections"):
                    assert conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0
            (output / "browser_report.json").write_text(json.dumps({
                "passed": True, "browser": "Microsoft Edge headless", "real_yolo": True,
                "history_writes": 0, "pi_verified": False
            }, indent=2), encoding="utf-8")
            print("PASS: real YOLO -> summary endpoint -> operator dashboard; no history writes")
        finally:
            httpd.shutdown()
            httpd.server_close()
            worker.join(timeout=5)


if __name__ == "__main__":
    main()
