"""Run dashboard regression suites in isolated processes and temporary stores.

No pytest dependency. Legacy Phase 4C still creates one SG-SEG-TEST PNG artifact;
the production database and model configuration must remain byte-for-byte unchanged.
"""
import hashlib
from contextlib import closing
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sqlite3
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent
SUITES = {
    "intelligence": ("steelguard_app/test_inspection_intelligence.py", None),
    "intelligence_api": ("steelguard_app/test_intelligence_api.py", None),
    "advisory": ("steelguard_app/test_executorch_advisory.py", None),
    "advisory_api": ("steelguard_app/test_advisory_api.py", None),
    "phase5d_tools": ("steelguard_app/test_phase5d_qualification.py", None),
    "stage4": ("steelguard_app/test_stage4.py", "run_stage4_tests"),
    "phase3": ("steelguard_app/test_stage5_phase3.py", "run_phase3_tests"),
    "phase4c": ("steelguard_app/test_stage6_phase4c.py", "run_all_tests"),
    "segmentation_pipeline": ("scripts/test_severstal_pipeline.py", None),
    "segmentation_training": ("scripts/test_severstal_training.py", None),
    "resources": ("steelguard_app/test_part2_resource_verification.py", "resources"),
    "traceability": ("steelguard_app/test_part4_traceability.py", "run_traceability_tests"),
    "startup": ("steelguard_app/test_startup.py", None),
    "phase5_e2e": ("steelguard_app/test_phase5_e2e_integration.py", None),
    "ui_overlap": ("steelguard_app/test_advisory_ui_overlap.py", "test_ui_and_invalidation"),
}


def child(name):
    path, entry = SUITES[name]
    sys.path[:0] = [str(ROOT / Path(path).parent), str(ROOT / "steelguard_app/backend")]
    with tempfile.TemporaryDirectory(prefix="steelguard-tests-") as temp:
        import db
        db.DEFAULT_DB_PATH = Path(temp) / "history.db"
        if name == "traceability":
            # Existing records are a read-only fixture; execute checks against a private copy.
            with closing(sqlite3.connect((ROOT / "steelguard_app/data/steelguard.db").as_uri() + "?mode=ro", uri=True)) as source:
                with closing(sqlite3.connect(db.DEFAULT_DB_PATH)) as target:
                    source.backup(target)
        if name not in ("segmentation_pipeline", "segmentation_training", "intelligence"):
            import model_manager
            model_manager.CONFIG_FILE = Path(temp) / "model_config.json"
        spec = importlib.util.spec_from_file_location("regression_suite", ROOT / path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if name == "phase3":
            module.TEST_DB_PATH = Path(temp) / "phase3.db"
        if name == "traceability":
            module.DB_PATH = db.DEFAULT_DB_PATH
        if entry == "resources":
            module.verify_checkpoints()
            module.test_inference_benchmarks()
            module.test_cpu_fallback_and_oom_handling()
        elif entry:
            getattr(module, entry)()
        else:
            result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
            if not result.wasSuccessful():
                raise SystemExit(1)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def main():
    os.chdir(ROOT)
    if len(sys.argv) == 3 and sys.argv[1] == "--suite":
        child(sys.argv[2])
        return
    if len(sys.argv) == 3 and sys.argv[1] == "--output":
        output = Path(sys.argv[2]).resolve()
        output.mkdir(parents=True, exist_ok=False)
    elif len(sys.argv) == 1:
        output = ROOT / "runs/phase5d_verification"
        output.mkdir(parents=True, exist_ok=True)
    else:
        raise SystemExit("Usage: run_dashboard_regressions.py [--suite NAME | --output NEW_DIRECTORY]")
    preserved = [ROOT / "steelguard_app/data/steelguard.db", ROOT / "steelguard_app/data/model_config.json",
                 ROOT / "runs/detect/steelguard_gc10n/weights/best.pt",
                 ROOT / "runs/detect/steelguard_gc10_neu_inclusion/weights/best.pt",
                 ROOT / "runs/segmentation/severstal_unet_v1/best_model.pt"]
    before = {str(p.relative_to(ROOT)): digest(p) for p in preserved}
    results = {}
    env = dict(os.environ, PYTHONIOENCODING="utf-8", YOLO_OFFLINE="True", ULTRALYTICS_AUTOUPDATE="0")
    for name in SUITES:
        proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--suite", name],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        (output / f"{name}.log").write_bytes(proc.stdout)
        results[name] = {"exit_code": proc.returncode}
        print(f"{name}: {'PASS' if proc.returncode == 0 else 'FAIL'}", flush=True)
    after = {str(p.relative_to(ROOT)): digest(p) for p in preserved}
    report = {"suites": results, "preserved_before": before, "preserved_after": after,
              "production_files_unchanged": before == after,
              "scope": "Windows development host; no Raspberry Pi validation"}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if before != after or any(v["exit_code"] for v in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
