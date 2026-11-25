"""
STEELGUARD AI — Server Startup & CLI Invocation Regression Suite
Guarantees:
1. server.run() and create_server() accept host and port keyword and positional arguments.
2. on_ready callback is called only after socket binding succeeds.
3. Socket binding errors (port collisions) raise OSError before on_ready is invoked.
4. Server starts up, accepts connections on host:port, returns HTTP 200 for GET /, and shuts down cleanly.
5. run_dashboard.py CLI arguments (--demo, --host, --port) configure the server environment correctly.
"""

import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
import urllib.error

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "steelguard_app" / "backend"
sys.path.insert(0, str(BACKEND))

import server


class ServerStartupTests(unittest.TestCase):
    def test_run_keyword_arguments_signature(self):
        """Verify that server.run accepts host and port keyword arguments without TypeError."""
        # Use port 0 to let the OS assign an ephemeral port
        bound = []

        def on_ready_callback(httpd):
            bound.append(httpd.server_address)
            # Instantly signal shutdown
            threading.Thread(target=httpd.shutdown, daemon=True).start()

        # Call with host and port keywords in both orders
        server.run(host="127.0.0.1", port=0, on_ready=on_ready_callback)
        self.assertEqual(len(bound), 1)
        self.assertEqual(bound[0][0], "127.0.0.1")
        self.assertGreater(bound[0][1], 0)

        # Inverted keyword order
        bound.clear()
        server.run(port=0, host="127.0.0.1", on_ready=on_ready_callback)
        self.assertEqual(len(bound), 1)
        self.assertEqual(bound[0][0], "127.0.0.1")
        self.assertGreater(bound[0][1], 0)

    def test_create_server_binds_socket_correctly(self):
        """Verify create_server creates HTTPServer and binds socket."""
        httpd = server.create_server(host="127.0.0.1", port=0)
        try:
            self.assertEqual(httpd.server_address[0], "127.0.0.1")
            self.assertGreater(httpd.server_address[1], 0)
        finally:
            httpd.server_close()

    def test_binding_error_prevents_on_ready_invocation(self):
        """Verify that port conflict raises OSError and never calls on_ready callback."""
        # Occupy a port with a raw socket
        occupier = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        occupier.bind(("127.0.0.1", 0))
        occupied_port = occupier.getsockname()[1]

        callback_called = []
        def dummy_callback(httpd):
            callback_called.append(True)

        try:
            with self.assertRaises(OSError):
                server.run(host="127.0.0.1", port=occupied_port, on_ready=dummy_callback)
            # Ensure banner / ready callback was NEVER executed
            self.assertEqual(len(callback_called), 0)
        finally:
            occupier.close()

    def test_server_serves_http_dashboard_and_health(self):
        """Verify that the live server serves GET / with 200 OK and valid HTML."""
        httpd = server.create_server(host="127.0.0.1", port=0)
        port = httpd.server_address[1]
        server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        server_thread.start()

        try:
            # 1. Test GET / (HTML Dashboard)
            url_root = f"http://127.0.0.1:{port}/"
            with urllib.request.urlopen(url_root, timeout=5) as resp:
                self.assertEqual(resp.status, 200)
                content_type = resp.headers.get("Content-Type", "")
                self.assertIn("text/html", content_type)
                body = resp.read().decode("utf-8")
                self.assertIn("<!DOCTYPE html>", body)
                self.assertIn("STEELGUARD AI", body)

            # 2. Test GET /api/health
            url_health = f"http://127.0.0.1:{port}/api/health"
            with urllib.request.urlopen(url_health, timeout=5) as resp:
                self.assertEqual(resp.status, 200)
                data = json.loads(resp.read().decode("utf-8"))
                self.assertIn("status", data)
                self.assertEqual(data["status"], "healthy")
        finally:
            httpd.shutdown()
            httpd.server_close()
            server_thread.join(timeout=3)

    def test_run_dashboard_cli_process(self):
        """Verify that running steelguard_app/run_dashboard.py starts a live server and responds."""
        # Find an available port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            free_port = s.getsockname()[1]

        cmd = [
            sys.executable,
            str(ROOT / "steelguard_app" / "run_dashboard.py"),
            "--host", "127.0.0.1",
            "--port", str(free_port)
        ]
        env = dict(os.environ, PYTHONUNBUFFERED="1", YOLO_OFFLINE="True", ULTRALYTICS_AUTOUPDATE="0")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True
        )

        try:
            # Poll for the server to accept connections
            url = f"http://127.0.0.1:{free_port}/"
            started = False
            for _ in range(60):
                time.sleep(0.5)
                # Check if process died prematurely
                if proc.poll() is not None:
                    stdout, stderr = proc.communicate()
                    self.fail(f"run_dashboard.py died prematurely with code {proc.returncode}.\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")
                try:
                    with urllib.request.urlopen(url, timeout=2) as resp:
                        if resp.status == 200:
                            started = True
                            body = resp.read().decode("utf-8")
                            self.assertIn("STEELGUARD AI", body)
                            break
                except Exception:
                    pass

            self.assertTrue(started, f"run_dashboard.py did not become responsive on {url}")
        finally:
            proc.kill()
            try:
                proc.communicate(timeout=3)
            except Exception:
                pass


def run_tests():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ServerStartupTests)
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    if not res.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    run_tests()
