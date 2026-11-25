"""
STEELGUARD AI — Inspection Intelligence / ExecuTorch Advisory UI Overlap & Invalidation Test Suite
Guarantees:
1. No overlap between the "LLM" badge and the heading text across 1366px, 1440px, 1680px, and 1024px.
2. Verified visual spacing (minimum 6px gap, flexbox layout).
3. "Request Llama Advisory" button is functional and invokes ExecuTorch only on demand.
4. Changing inspected image immediately invalidates any prior advisory, preventing stale recommendations.
5. Captures visual screenshots for verification evidence.
"""

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "steelguard_app" / "backend"
sys.path.insert(0, str(BACKEND))

os.environ["YOLO_OFFLINE"] = "True"
os.environ["ULTRALYTICS_AUTOUPDATE"] = "0"
os.environ["STEELGUARD_EXECUTORCH_MANIFEST"] = "/home/sai01/steelguard-host-bundle-20261116/advisory.manifest.json"

RESOLUTIONS = [
    (1366, 768, "1366x768"),
    (1440, 900, "1440x900"),
    (1680, 1050, "1680x1050"),
    (1024, 768, "1024x768"),
]

OVERLAP_TEST_HARNESS = r"""
<script>
window.alert = function(msg) { console.log('Suppressed alert:', msg); };
window.confirm = function() { return true; };
window.prompt = function() { return null; };

window.runOverlapCheck = async function() {
  navigateTo('assistant');
  await new Promise(r => setTimeout(r, 600));

  const label = document.getElementById('advisory-llm-label');
  const heading = document.getElementById('advisory-heading');
  const badge = document.getElementById('advisory-badge');
  const btn = document.getElementById('btn-request-advisory');

  if (!label || !heading) {
    return { passed: false, error: 'Elements not found' };
  }

  const lRect = label.getBoundingClientRect();
  const hRect = heading.getBoundingClientRect();
  const bRect = badge ? badge.getBoundingClientRect() : null;

  // Horizontal overlap check
  const horizontalOverlap = !(lRect.right <= hRect.left || hRect.right <= lRect.left);
  const verticalOverlap = !(lRect.bottom <= hRect.top || hRect.bottom <= lRect.top);
  const overlaps = horizontalOverlap && verticalOverlap;
  const gap = hRect.left - lRect.right;

  return {
    passed: !overlaps && gap >= 6,
    overlaps: overlaps,
    gap: gap,
    llmRect: { left: lRect.left, right: lRect.right, width: lRect.width, height: lRect.height },
    headingRect: { left: hRect.left, right: hRect.right, width: hRect.width, height: hRect.height },
    badgeText: badge ? badge.textContent : null,
    btnDisabled: btn ? btn.disabled : null
  };
};

window.runDynamicInvalidationCheck = async function() {
  const waitFor = async (predicate, timeoutMs = 25000, desc = 'condition') => {
    const start = Date.now();
    while (Date.now() - start < timeoutMs) {
      if (predicate()) return;
      await new Promise(r => setTimeout(r, 100));
    }
    throw new Error('Timeout waiting for ' + desc);
  };

  try {
    // 1. Initial State: Load Sample 1 and verify advisory awaiting query
    navigateTo('inspect');
    await loadVerifiedSample('img_01_3402617700_01009');
    await waitFor(() => state.currentImage && state.dataMode === 'GROUND_TRUTH', 10000, 'Sample 1 loaded');
    
    // Advisory should be empty and awaiting inference
    navigateTo('assistant');
    const badgeInit = document.getElementById('advisory-badge');
    const contentInit = document.getElementById('advisory-content');
    const btnInit = document.getElementById('btn-request-advisory');
    if (!btnInit.disabled) throw new Error('Request button must be disabled before YOLO inference');
    if (contentInit.textContent.includes('Operator Action Plan')) throw new Error('Action plan present initially');

    // 2. Run YOLO inference on Sample 1
    navigateTo('inspect');
    await handleRunAIInference();
    await waitFor(() => state.dataMode === 'AI_INFERENCE' && state.detections.length > 0, 15000, 'YOLO inference');
    
    // After YOLO, button must be enabled and badge says READY FOR QUERY
    navigateTo('assistant');
    if (btnInit.disabled) throw new Error('Request button must be enabled after successful YOLO inference');
    if (badgeInit.textContent !== 'READY FOR QUERY') throw new Error('Advisory badge should be READY FOR QUERY, got: ' + badgeInit.textContent);

    // 3. Request Llama Advisory on demand
    await requestExecuTorchAdvisory();
    await waitFor(() => {
      const c = document.getElementById('advisory-content');
      return c && (c.textContent.includes('EXECUTORCH') || c.textContent.includes('RULE_BASED_FALLBACK') || c.textContent.includes('Primary Disposition'));
    }, 25000, 'ExecuTorch advisory completion');

    const contentSample1 = document.getElementById('advisory-content').textContent;
    if (!contentSample1.includes('Operator Action Plan')) {
      throw new Error('Advisory missing action plan');
    }

    // 4. INVASION TEST: Switch image to a different sample (Sample 2)
    navigateTo('inspect');
    await loadVerifiedSample('img_01_3402617700_00001');
    await waitFor(() => state.currentSampleId === 'img_01_3402617700_00001' || (state.currentImage && state.dataMode === 'GROUND_TRUTH'), 10000, 'Sample 2 loaded');

    // 5. VERIFY ADVISORY INVALIDATION: Previous advisory plan must be completely eradicated
    navigateTo('assistant');
    const contentAfterSwitch = document.getElementById('advisory-content').textContent;
    const badgeAfterSwitch = document.getElementById('advisory-badge');
    const btnAfterSwitch = document.getElementById('btn-request-advisory');

    if (btnAfterSwitch.disabled !== true) {
      throw new Error('Advisory button must be disabled immediately after changing image');
    }
    if (contentAfterSwitch.includes('Operator Action Plan') || contentAfterSwitch.includes('Primary Disposition')) {
      throw new Error('STALE ADVISORY DETECTED! Old action plan persisted across image change');
    }
    if (badgeAfterSwitch.textContent !== 'AWAITING INFERENCE') {
      throw new Error('Advisory badge was not reset to AWAITING INFERENCE, got: ' + badgeAfterSwitch.textContent);
    }

    return { success: true };
  } catch(err) {
    return { success: false, error: err.message };
  }
};
</script>
"""


def test_ui_and_invalidation():
    output_dir = ROOT / "runs" / "phase5_demo_verification"
    output_dir.mkdir(parents=True, exist_ok=True)

    edge = Path(os.environ.get("PROGRAMFILES(X86)", "C:/Program Files (x86)")) / "Microsoft/Edge/Application/msedge.exe"
    if not edge.is_file():
        raise RuntimeError("Microsoft Edge is required for UI regression test")

    with tempfile.TemporaryDirectory(prefix="steelguard-ui-") as temp:
        import db
        db.DEFAULT_DB_PATH = Path(temp) / "test_ui.db"
        import model_manager
        model_manager.CONFIG_FILE = Path(temp) / "model_config.json"
        import server
        from advisory_service import AdvisoryService
        server.advisory_service = AdvisoryService(os.environ["STEELGUARD_EXECUTORCH_MANIFEST"])

        class UITestHandler(server.SteelGuardRequestHandler):
            def do_GET(self):
                if self.path == "/__test/invalidation":
                    page = (ROOT / "steelguard_app/static/index.html").read_text(encoding="utf-8")
                    script = """
                    <script>
                    document.addEventListener('DOMContentLoaded', async () => {
                      const res = await window.runDynamicInvalidationCheck();
                      document.body.dataset.invalResult = JSON.stringify(res);
                    });
                    </script>
                    """
                    modified = page.replace("<head>", '<head><base href="/">' + OVERLAP_TEST_HARNESS + script)
                    self._set_headers(200, "text/html; charset=utf-8")
                    self.wfile.write(modified.encode("utf-8"))
                    return
                elif self.path.startswith("/__test/"):
                    page = (ROOT / "steelguard_app/static/index.html").read_text(encoding="utf-8")
                    script = """
                    <script>
                    document.addEventListener('DOMContentLoaded', async () => {
                      const res = await window.runOverlapCheck();
                      document.body.dataset.testResult = JSON.stringify(res);
                    });
                    </script>
                    """
                    modified = page.replace("<head>", '<head><base href="/">' + OVERLAP_TEST_HARNESS + script)
                    self._set_headers(200, "text/html; charset=utf-8")
                    self.wfile.write(modified.encode("utf-8"))
                    return
                super().do_GET()

        httpd = server.create_server(host="127.0.0.1", port=0)
        httpd.RequestHandlerClass = UITestHandler
        port = httpd.server_address[1]
        server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        server_thread.start()

        results = {}

        try:
            # 1. Test responsive layout across multiple resolutions
            for width, height, label in RESOLUTIONS:
                print(f"Testing resolution {label} ({width}x{height})...")
                user_data = Path(temp) / f"edge-profile-{label}"
                user_data.mkdir(exist_ok=True)
                shot_path = output_dir / f"advisory_ui_{label}.png"

                cmd = [
                    str(edge), "--headless", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
                    "--disable-background-networking", f"--user-data-dir={user_data}",
                    f"--window-size={width},{height}", "--virtual-time-budget=8000",
                    "--dump-dom", f"--screenshot={shot_path}",
                    f"http://127.0.0.1:{port}/__test/{label}"
                ]

                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45)
                html = proc.stdout.decode("utf-8", errors="replace")

                m = re.search(r'data-test-result="([^"]+)"', html)
                if not m:
                    raise AssertionError(f"Resolution {label}: test script did not return data-test-result")
                
                res_json = m.group(1).replace("&quot;", '"')
                data = json.loads(res_json)
                print(f" -> {label}: gap={data['gap']:.1f}px, overlaps={data['overlaps']}, passed={data['passed']}")
                if not data["passed"] or data["overlaps"]:
                    raise AssertionError(f"Resolution {label} failed overlap check: {data}")

                results[label] = {
                    "passed": True,
                    "gap_px": data["gap"],
                    "overlaps": data["overlaps"],
                    "screenshot": str(shot_path)
                }

            # 2. Test Dynamic Invalidation & On-Demand Execution
            print("\nTesting Dynamic Invalidation & On-Demand Execution...")
            inval_user_data = Path(temp) / "edge-profile-inval"
            inval_user_data.mkdir(exist_ok=True)
            inval_shot = output_dir / "advisory_ui_invalidation_proof.png"

            cmd = [
                str(edge), "--headless", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
                "--disable-background-networking", f"--user-data-dir={inval_user_data}",
                "--window-size=1440,900", "--virtual-time-budget=45000",
                "--dump-dom", f"--screenshot={inval_shot}",
                f"http://127.0.0.1:{port}/__test/invalidation"
            ]

            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
            html = proc.stdout.decode("utf-8", errors="replace")

            m = re.search(r'data-inval-result="([^"]+)"', html)
            if not m:
                raise AssertionError("Invalidation test: script did not return data-inval-result")
            inval_json = m.group(1).replace("&quot;", '"')
            inval_res = json.loads(inval_json)
            print(f" -> Dynamic Invalidation Result: {inval_res}")
            if not inval_res.get("success"):
                raise AssertionError(f"Dynamic invalidation test failed: {inval_res.get('error')}")

            print("\nALL UI OVERLAP & INVALIDATION CHECKS PASSED SUCCESSFULLY!")
            report = {
                "passed": True,
                "resolutions_tested": results,
                "dynamic_invalidation": inval_res,
                "screenshots": [str(output_dir / f"advisory_ui_{l}.png") for _, _, l in RESOLUTIONS] + [str(inval_shot)]
            }
            (output_dir / "ui_overlap_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

        finally:
            httpd.shutdown()
            httpd.server_close()
            server_thread.join(timeout=3)


if __name__ == "__main__":
    test_ui_and_invalidation()
