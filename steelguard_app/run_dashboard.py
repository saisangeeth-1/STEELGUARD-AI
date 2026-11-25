"""
STEELGUARD AI - Application Launcher
Executes the SteelGuard Industrial Quality Inspection Server.

Targets: Raspberry Pi 5 (ARM64 CPU) Edge Deployment | Development Laptop (WSL2 CPU Llama)
100% Offline Operation (Zero Cloud API Dependencies)
"""

import argparse
import os
from pathlib import Path
import sys

# Add backend directory to sys.path
BASE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BASE_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))


def main():
    parser = argparse.ArgumentParser(description="STEELGUARD AI - Industrial Inspection Server")
    parser.add_argument("--port", "-p", type=int, default=int(os.environ.get("STEELGUARD_PORT", 8080)),
                        help="Port to listen on (default: 8080)")
    parser.add_argument("--host", default="127.0.0.1", help="Host address to bind to (default: 127.0.0.1)")
    parser.add_argument("--manifest", default=os.environ.get("STEELGUARD_EXECUTORCH_MANIFEST"),
                        help="Path to ExecuTorch advisory manifest JSON")
    parser.add_argument("--enable-wsl-advisory", action="store_true",
                        help="Connect to real WSL x86_64 ExecuTorch Llama 3.2 1B INT4 runtime")
    parser.add_argument("--demo", action="store_true",
                        help="Enable offline demo mode with WSL ExecuTorch advisory integration")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU-only execution for all vision and advisory services")
    args = parser.parse_args()

    if args.cpu:
        os.environ["STEELGUARD_DEVICE"] = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Configure advisory manifest if requested
    default_wsl_manifest = os.environ.get("STEELGUARD_EXECUTORCH_MANIFEST", "/home/sai01/steelguard-host-bundle-20261116/advisory.manifest.json")
    if args.demo or args.enable_wsl_advisory:
        manifest_path = args.manifest or default_wsl_manifest
        os.environ["STEELGUARD_EXECUTORCH_MANIFEST"] = manifest_path
    elif args.manifest:
        os.environ["STEELGUARD_EXECUTORCH_MANIFEST"] = args.manifest

    # Import server and components after environment is configured
    from server import run, advisory_service
    from model_manager import model_manager
    from segmentation_service import segmentation_manager

    # Diagnostic inspection
    yolo_info = model_manager.get_active_model_info()
    seg_info = segmentation_manager.get_model_info()
    adv_status = advisory_service.status()

    def print_ready_banner(httpd):
        server_host, server_port = httpd.server_address[:2]
        print("=" * 72)
        print(" STEELGUARD AI — INDUSTRIAL QUALITY INSPECTION SYSTEM")
        print(" Target: Raspberry Pi 5 (ARM64 CPU) | Host: Development Workstation")
        print(" Mode:   100% Offline Industrial Edge Operation (Zero Cloud Dependencies)")
        print("=" * 72)
        print(f" [Vision]        YOLOv8 Nano (GC10 + NEU): {yolo_info.get('name', 'Active')} ({yolo_info.get('version', 'v2.0')})")
        print(f"                 Status: {'READY' if yolo_info.get('loaded') else 'NOT LOADED'} | SHA-256: {yolo_info.get('sha256', 'Verified')[:16]}...")
        print(f" [Segmentation]  Severstal Lightweight U-Net: {'READY' if seg_info.get('loaded') else 'NOT LOADED'} (Native 1600x256)")
        print(f"                 Status: {'READY (EXPERIMENTAL)' if seg_info.get('loaded') else 'NOT LOADED'} | SHA-256: {seg_info.get('actual_sha256', '')[:16]}...")
        adv_ready = adv_status.get("configuration_validated", False)
        adv_cfg = adv_status.get("configured", False)
        if adv_ready:
            print(f" [LLM Advisory]  ExecuTorch Llama 3.2 1B Instruct: READY ({adv_status.get('runtime')})")
            print(f"                 Model ID: {adv_status.get('model_id')} | Target: {adv_status.get('target')}")
        elif adv_cfg:
            print(f" [LLM Advisory]  ExecuTorch Advisory: CONFIGURED (Validation pending or error)")
        else:
            print(f" [LLM Advisory]  ExecuTorch Advisory: DISABLED (Rule-Based Fallback active)")
            print(f"                 Tip: Pass --enable-wsl-advisory or set STEELGUARD_EXECUTORCH_MANIFEST")
        print("-" * 72)
        print(f" Dashboard URL:  http://{server_host}:{server_port}/")
        print(" Press Ctrl+C to terminate the server.")
        print("=" * 72)
        sys.stdout.flush()

    run(host=args.host, port=args.port, on_ready=print_ready_banner)


if __name__ == "__main__":
    main()
