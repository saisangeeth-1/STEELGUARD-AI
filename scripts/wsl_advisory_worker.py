"""Safe local WSL worker for ExecuTorch advisory generation.

Called by Windows development environment or directly in Linux/WSL.
Offline, CPU-only, bounded stdin/stdout, strict fallback on any error.
"""
import argparse
import json
import os
from pathlib import Path
import sys

# Ensure backend modules can be imported
BACKEND_DIR = Path(__file__).resolve().parent.parent / "steelguard_app" / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from advisory_service import AdvisoryService, reply, validate_summary
from executorch_runner import ExecuTorchRunner, RunnerConfig, AdvisoryRuntimeError, strict_json


def handle_status(manifest_path):
    try:
        manifest = Path(manifest_path)
        config = RunnerConfig.load(manifest, allow_host_qualification=True)
        config.verify_unchanged()
        return {
            "runtime": "ExecuTorch llama_main (WSL Host)",
            "configured": True,
            "configuration_validated": True,
            "offline_only": True,
            "target": config.target,
            "model_id": config.model_id,
            "live_deployment_attested": False
        }
    except Exception as exc:
        return {
            "runtime": "ExecuTorch llama_main (WSL Host)",
            "configured": bool(manifest_path),
            "configuration_validated": False,
            "offline_only": True,
            "target": "linux-x86_64",
            "error": str(exc),
            "live_deployment_attested": False
        }


def handle_advise(manifest_path, summary_data):
    try:
        manifest = Path(manifest_path)
        runner = ExecuTorchRunner(allow_host_qualification=True)
        service = AdvisoryService(manifest, runner=runner)
        # Pre-load configuration with allow_host_qualification=True for x86_64 host
        service._config = RunnerConfig.load(manifest, allow_host_qualification=True)
        return service.advise(summary_data)
    except AdvisoryRuntimeError as exc:
        try:
            valid = validate_summary(summary_data)
            return reply(valid, code=exc.code)
        except Exception:
            return {"error": exc.code, "source": "RULE_BASED_FALLBACK", "llm_used": False}
    except Exception as exc:
        try:
            valid = validate_summary(summary_data)
            return reply(valid, code="RESOURCE_ERROR")
        except Exception:
            return {"error": str(exc), "source": "RULE_BASED_FALLBACK", "llm_used": False}


def main():
    parser = argparse.ArgumentParser(description="SteelGuard WSL ExecuTorch Advisory Worker")
    parser.add_argument("--manifest", required=True, help="Absolute Linux path to advisory.manifest.json")
    parser.add_argument("--status", action="store_true", help="Query runtime status")
    args = parser.parse_args()

    if args.status:
        result = handle_status(args.manifest)
        print(json.dumps(result))
        return

    # Read summary from stdin
    try:
        raw_input = sys.stdin.read(131072 + 1024)
        if not raw_input:
            print(json.dumps({"error": "EMPTY_INPUT", "source": "RULE_BASED_FALLBACK", "llm_used": False}))
            return
        summary_data = strict_json(raw_input)
        result = handle_advise(args.manifest, summary_data)
        print(json.dumps(result))
    except Exception as exc:
        print(json.dumps({"error": f"WORKER_ERROR: {str(exc)}", "source": "RULE_BASED_FALLBACK", "llm_used": False}))


if __name__ == "__main__":
    main()
