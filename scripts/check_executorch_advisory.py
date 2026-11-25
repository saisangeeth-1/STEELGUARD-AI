"""Static manifest/integrity preflight only; never runs or downloads a model."""
import argparse
import json
from pathlib import Path
import platform
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "steelguard_app/backend"))
from executorch_runner import AdvisoryRuntimeError, RunnerConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Absolute path to the deployment manifest")
    args = parser.parse_args()
    result = {"host_os": platform.system(), "host_arch": platform.machine(),
              "live_inference_verified": False, "physical_pi_verified": False}
    try:
        config = RunnerConfig.load(args.manifest)
        result.update(static_validation="PASS", model_id=config.model_id,
                      free_artifact_disk_bytes=shutil.disk_usage(config.model.parent).free)
    except AdvisoryRuntimeError as exc:
        result.update(static_validation="FAIL", reason_code=exc.code)
    except (OSError, MemoryError):
        result.update(static_validation="FAIL", reason_code="RESOURCE_ERROR")
    print(json.dumps(result, indent=2))
    return 0 if result["static_validation"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
