"""Portable qualification tooling tests. Synthetic fixtures; no Llama inference."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "steelguard_app"), str(ROOT / "steelguard_app/backend"), str(ROOT / "scripts")]
import test_executorch_advisory as fixtures
from executorch_runner import AdvisoryRuntimeError, ExecuTorchRunner, RunnerConfig, run_bounded
from phase5d_toolchain import external, export_command, write_new, runtime_libraries, stage_export
from phase5d_bundle import make_kit, relocate
from phase5d_qualify import qualify, stats_record
from audit_executorch_contract import audit


class QualificationToolsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): fixtures.AdvisoryTests.setUpClass()

    def setUp(self):
        self.fixture = fixtures.AdvisoryTests("test_disabled_fallback")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_exact_patch_version_and_mismatch(self):
        f = self.fixture
        f.data.update(runtime_version="1.5.1", export_version="1.5.1"); f.write()
        self.assertEqual(RunnerConfig.load(f.manifest).target, "linux-aarch64")
        f.data["export_version"] = "1.5"; f.write()
        with self.assertRaises(AdvisoryRuntimeError): RunnerConfig.load(f.manifest)

    def test_x86_qualification_does_not_enable_production(self):
        f = self.fixture
        content = bytearray(f.runner.read_bytes()); content[18:20] = (62).to_bytes(2, "little")
        f.runner.write_bytes(content)
        f.data["target"] = "linux-x86_64"
        f.data["artifacts"]["runner"]["sha256"] = hashlib.sha256(content).hexdigest(); f.write()
        with self.assertRaises(AdvisoryRuntimeError): RunnerConfig.load(f.manifest)
        config = RunnerConfig.load(f.manifest, allow_host_qualification=True)
        with patch("executorch_runner.platform.system", return_value="Linux"), patch("executorch_runner.platform.machine", return_value="x86_64"):
            with self.assertRaises(AdvisoryRuntimeError): ExecuTorchRunner().generate(config, "prompt")
            with patch("executorch_runner.run_bounded", return_value=b"synthetic test only") as boundary:
                ExecuTorchRunner(allow_host_qualification=True).generate(config, "prompt")
                boundary.assert_called_once()

    def test_missing_artifacts_never_claim_generation(self):
        report = qualify(self.fixture.root / "missing.json")
        self.assertFalse(report["runtime_generation_verified"])
        self.assertFalse(report["physical_pi_inference_verified"])
        self.assertFalse(report["physical_deployment_qualified"])
        self.assertEqual(report["blocked_reason_code"], "MISSING_ARTIFACT")

    def test_network_isolation_guard(self):
        with patch("phase5d_qualify.environment", return_value={"loopback_only_namespace_observed":False}):
            report = qualify(self.fixture.manifest, require_network_isolation=True)
        self.assertEqual(report["blocked_reason_code"], "NETWORK_ISOLATION_REQUIRED")
        self.assertFalse(report["runtime_generation_verified"])

    def test_export_requires_checkpoint_and_quotes_hydra_paths(self):
        root = self.fixture.root
        lock = json.loads((ROOT / "deployment/phase5d/toolchain.lock.json").read_text())
        with self.assertRaises(ValueError): export_command(root, root / "absent", root / "out", lock)
        original = root / "snapshot $(literal)" / "original"; original.mkdir(parents=True)
        for name in ("consolidated.00.pth", "params.json", "tokenizer.model"): (original / name).write_bytes(b"synthetic")
        args = export_command(root, original.parent, root / "out", lock)
        value = next(a.split("=", 1)[1] for a in args if a.startswith("+base.checkpoint="))
        self.assertEqual(json.loads(value), str(original / "consolidated.00.pth"))
        self.assertIn('+export.output_name="steelguard_llama32_1b_int4"', args)
        self.assertNotIn("--max-seq-len", args)

    def test_output_and_source_locations_preserved(self):
        with self.assertRaises(ValueError): external(ROOT / "weights")
        path = self.fixture.root / "receipt.json"
        write_new(path, {"synthetic":True})
        with self.assertRaises(FileExistsError): write_new(path, {"overwrite":True})

    def test_source_contract_audit_is_static_not_inference(self):
        result = audit(ROOT / "runs/phase5d_qualification/references")
        self.assertEqual(result["undefined_flags"], [])
        self.assertFalse(result["real_runner_executed"])
        self.assertEqual(result["source_version"], "1.5.1")

    def test_model_free_kit_is_deterministic_and_contains_no_weights(self):
        first, second = self.fixture.root / "first.zip", self.fixture.root / "second.zip"
        self.assertEqual(make_kit(first), make_kit(second))
        with zipfile.ZipFile(first) as archive:
            self.assertTrue(all(not n.endswith((".pth", ".pte", ".safetensors")) for n in archive.namelist()))
            self.assertIn("deployment/phase5d/toolchain.lock.json", archive.namelist())

    def test_bundle_relocation_rejects_escaping_paths(self):
        root = self.fixture.root
        (root / "checksums.json").write_text(json.dumps({"../not-inside":"0" * 64}))
        with self.assertRaises(ValueError): relocate(root)

    def test_statistics_record_is_not_inference_attestation(self):
        result = stats_record(b'text\nPyTorchObserver {"generated_tokens":10}\n')
        self.assertEqual(result["generated_tokens"], 10)
        with self.assertRaises(ValueError): stats_record(b"no record")

    def test_process_metrics_have_real_wall_time_without_fake_rss(self):
        metrics = {}
        raw = run_bounded([sys.executable, "-c", "print('synthetic child')"], timeout_seconds=3,
                          max_output_bytes=1024, env=None, cwd=self.fixture.root, metrics=metrics)
        self.assertIn(b"synthetic child", raw)
        self.assertGreater(metrics["wall_seconds"], 0)
        self.assertIsInstance(metrics["process_id"], int)
        if sys.platform == "win32": self.assertIsNone(metrics["peak_sampled_rss_kib"])

    def test_runtime_dependencies_keep_required_loader_names(self):
        root = self.fixture.root
        library = root / "libtest.so.1"; library.write_bytes(b"synthetic library")
        def output(argv, **kwargs):
            return " (NEEDED) Shared library: [libtest.so.1]\n (NEEDED) Shared library: [libc.so.6]\n" if Path(argv[-1]) == self.fixture.runner else ""
        with patch("phase5d_toolchain.subprocess.check_output", side_effect=output):
            found, system = runtime_libraries(self.fixture.runner, root)
        self.assertEqual(found["libtest.so.1"], library.resolve())
        self.assertEqual(system, ["libc.so.6"])

    def test_staging_does_not_copy_original_weights_or_overwrite(self):
        root = self.fixture.root
        native, arm = root / "native", root / "arm"
        (native / "receipts").mkdir(parents=True); (arm / "receipts").mkdir(parents=True)
        (native / "model-source").mkdir()
        pte = native / "synthetic.pte"; pte.write_bytes(b"synthetic exported model, not real")
        tokenizer = native / "tokenizer.model"; tokenizer.write_bytes(b"synthetic token data")
        (native / "original-checkpoint.pth").write_bytes(b"must not copy")
        for name in ("LICENSE.txt", "USE_POLICY.md"): (native / "model-source" / name).write_text("synthetic notice")
        (native / "receipts/download.json").write_text("{}")
        receipt = {key:{"path":str(path), "sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
                   for key, path in (("pte", pte), ("tokenizer", tokenizer))}
        (native / "receipts/export.json").write_text(json.dumps(receipt))
        stage_export(native, arm)
        self.assertEqual(list(arm.rglob("*.pth")), [])
        self.assertEqual((arm / "export/steelguard_llama32_1b_int4.pte").read_bytes(), pte.read_bytes())
        with self.assertRaises(ValueError): stage_export(native, arm)


if __name__ == "__main__": unittest.main(verbosity=2)
