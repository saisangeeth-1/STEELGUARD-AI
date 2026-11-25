"""Read-only advisory plans grounded in recomputed Inspection Intelligence summaries."""
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import threading

from executorch_runner import AdvisoryRuntimeError, ExecuTorchRunner, RunnerConfig, strict_json
from inspection_intelligence import GC10_NAMES, summarize_inspection

MAX_SUMMARY_BYTES = 131072
MAX_FINDINGS = 32
_GENERATION_LOCK = threading.Lock()
ACTION_TEXT = {
    "HOLD_FOR_DISPOSITION": "Hold the workpiece for operator disposition under the configured rejection policy.",
    "OPERATOR_REVIEW": "Request operator review under the applicable acceptance policy.",
    "CHECK_IMAGE_COVERAGE": "Check image coverage, lighting and focus before disposition.",
    "CHECK_CALIBRATION": "Use a calibrated measurement procedure if physical dimensions are required.",
    "CONFIRM_ACCEPTANCE_POLICY": "Confirm applicable acceptance policy before releasing the workpiece."
}


def validate_summary(summary):
    """Recompute rules, evidence, counts and warnings; reject tampered/extra fields.

    This validates consistency, not image authenticity or provenance attestation.
    No raw pixels, notes, prompts, arbitrary provider options or filesystem paths accepted.
    """
    try:
        encoded = json.dumps(summary, allow_nan=False, ensure_ascii=True)
        if len(encoded.encode("ascii")) > MAX_SUMMARY_BYTES or not isinstance(summary, dict):
            raise ValueError()
        if not isinstance(summary.get("evidence"), list) or len(summary["evidence"]) > 10000:
            raise ValueError()
        detections = []
        for row in summary["evidence"]:
            if not isinstance(row, dict):
                raise ValueError()
            detections.append({key: row[key] for key in ("raw_label", "area_px", "bbox_xyxy", "confidence")})
        provenance = summary.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError()
        canonical = summarize_inspection(dict(provenance, source=summary.get("source"), status="SUCCESS",
                                             detections=detections, taxonomy="GC10-DET"))
        if summary != canonical:
            raise ValueError()
        return canonical
    except (KeyError, TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError("Expected an unmodified, consistent Inspection Intelligence v1.1 summary") from exc


def actions_for(verdict):
    actions = ["OPERATOR_REVIEW", "CHECK_IMAGE_COVERAGE", "CHECK_CALIBRATION"]
    if verdict == "REJECT":
        actions.insert(0, "HOLD_FOR_DISPOSITION")
    elif verdict == "PASS":
        actions.append("CONFIRM_ACCEPTANCE_POLICY")
    return actions


ACTION_SELECTION_TOKENS = 2
ACTION_SELECTION_INSTRUCTION = "Answer with the letter of the best next action and nothing else."


def generate_action(runner, config, prompt, **kwargs):
    return runner.generate(replace(config, max_new_tokens=ACTION_SELECTION_TOKENS), prompt, **kwargs)


def build_prompt(summary, config):
    if len(summary["evidence"]) > MAX_FINDINGS:
        raise AdvisoryRuntimeError("INPUT_LIMIT")
    facts = []
    for row in summary["evidence"]:
        # Arbitrary labels and provenance strings NEVER enter the model context.
        label = row["raw_label"] if row["raw_label"] in GC10_NAMES else "UNMAPPED"
        facts.append({"id": row["index"], "class": label, "severity": row["severity"],
                      "box_px": row["bbox_xyxy"], "box_area_px2": row["area_px"],
                      "model_confidence": row["confidence"]})
    verdict = summary["assessment"]["status"]
    data = {"source": summary["source"], "immutable_verdict": verdict, "findings": facts,
            "warning_codes": ["UNVERIFIED_PROVENANCE", "UNCALIBRATED_PIXEL_AREA", "HEURISTIC_POLICY"],
            "allowed_action_codes": actions_for(verdict)}
    if summary["source"] == "GROUND_TRUTH":
        data["warning_codes"].append("ANNOTATIONS_NOT_PREDICTIONS")
    if any(row["class"] == "UNMAPPED" for row in facts):
        data["warning_codes"].append("UNMAPPED_CLASS")
    system = "You are a steel inspection assistant. Answer the multiple-choice question. The inspection verdict is fixed."
    options = "\n".join(f"{letter}. {ACTION_TEXT[code]}" for letter, code in zip("ABCDE", actions_for(verdict)))
    question = ("\nThe choices apply to this inspection only. Choosing an action acknowledges the listed limitations.\n"
                + options + "\n" + ACTION_SELECTION_INSTRUCTION)
    prompt = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + system +
              "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" +
              json.dumps(data, ensure_ascii=True, separators=(",", ":"), allow_nan=False) + question +
              "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
    if len(prompt.encode("ascii")) > config.max_input_bytes:
        raise AdvisoryRuntimeError("INPUT_LIMIT")
    return prompt


def parse_plan(output, prompt, summary, config):
    try:
        text = output.decode("utf-8", errors="strict")
        if text.startswith(prompt):
            text = text[len(prompt):]
        # Upstream llama_main emits token text, then a PyTorchObserver JSON report.
        match = re.search(r"\nPyTorchObserver ([^\r\n]+)\s*$", text)
        if not match:
            raise ValueError()
        stats = strict_json(match.group(1))
        for key, limit in (("generated_tokens", config.max_new_tokens), ("prompt_tokens", config.context_tokens)):
            if type(stats.get(key)) is not int or not 0 <= stats[key] <= limit:
                raise ValueError()
        if (stats["prompt_tokens"] < 1
                or stats["prompt_tokens"] + stats["generated_tokens"] + 1 > config.context_tokens
                or stats["prompt_tokens"] > len(prompt.encode("ascii")) + 128):
            raise ValueError()
        body = text[:match.start()].strip()
        for marker in ("<|eot_id|>", "<|end_of_text|>"):
            body = body.removesuffix(marker).rstrip()
        verdict = summary["assessment"]["status"]
        ids_expected = [r["index"] for r in summary["evidence"]]
        choice = re.fullmatch(r"([A-E])\.?", body)
        if choice and ACTION_SELECTION_INSTRUCTION in prompt:
            options = dict(zip("ABCDE", actions_for(verdict)))
            if stats["generated_tokens"] >= ACTION_SELECTION_TOKENS or choice[1] not in options:
                raise ValueError()
            plan = {"verdict": verdict, "finding_ids": ids_expected,
                    "action_codes": [options[choice[1]]], "acknowledge_warnings": True}
        else:
            if stats["generated_tokens"] < 1:
                raise ValueError()
            plan = strict_json(body)
        if not isinstance(plan, dict) or set(plan) != {"verdict", "finding_ids", "action_codes", "acknowledge_warnings"}:
            raise ValueError()
        if plan["verdict"] != summary["assessment"]["status"] or plan["acknowledge_warnings"] is not True:
            raise ValueError()
        ids = plan["finding_ids"]
        if not isinstance(ids, list) or any(type(i) is not int for i in ids) or ids != ids_expected:
            raise ValueError()
        codes = plan["action_codes"]
        if (not isinstance(codes, list) or not codes or any(not isinstance(c, str) for c in codes)
                or len(codes) != len(set(codes)) or not set(codes) <= set(actions_for(plan["verdict"]))):
            raise ValueError()
        return plan
    except (UnicodeError, ValueError, TypeError, KeyError, AttributeError, RecursionError) as exc:
        raise AdvisoryRuntimeError("MALFORMED_OUTPUT") from exc


def reply(summary, *, code, plan=None, model_id=None):
    generated = plan is not None
    codes = plan["action_codes"] if generated else []
    # The mandatory policy action is always first, regardless of model selection.
    recommendations = list(dict.fromkeys([summary["recommended_action"]] + [ACTION_TEXT[c] for c in codes]))
    return {
        "schema_version": "1.0", "source": "EXECUTORCH" if generated else "RULE_BASED_FALLBACK",
        "llm_used": generated, "reason_code": code, "model_id": model_id if generated else None,
        "advisory_kind": "constrained_model_selection" if generated else "deterministic_explanation",
        "quality_verdict": summary["assessment"]["status"],
        "inspection_source": summary["source"],
        "explanation": summary["assessment"]["summary"],
        "measured_facts": deepcopy(summary["evidence"]),
        "recommendations": recommendations,
        "warnings": deepcopy(summary["warnings"]) + [
            "Advisory recommendations require operator confirmation; they are not engineering measurements.",
            "Evidence consistency has been checked; image provenance is not authenticated."
        ],
        "policy_version": summary["policy_version"], "segmentation_fused": False,
        "facts_source": "DETERMINISTIC_INSPECTION_INTELLIGENCE",
        "live_deployment_attested": False
    }


class AdvisoryService:
    def __init__(self, manifest_path=None, runner=None):
        self.manifest_path = manifest_path
        self.runner = runner or ExecuTorchRunner()
        self._config = None

    def _is_wsl_mode(self):
        if platform.system() != "Windows" or not self.manifest_path:
            return False
        if type(self.runner) is not ExecuTorchRunner:
            return False
        path_str = str(self.manifest_path)
        return path_str.startswith("/") or path_str.startswith("wsl:") or path_str.lower().startswith("\\\\wsl")

    def _advise_wsl(self, summary):
        distro = os.environ.get("STEELGUARD_WSL_DISTRO", "Ubuntu-24.04")
        wsl_python = os.environ.get("STEELGUARD_WSL_PYTHON", "/home/sai01/steelguard-build/.venv/bin/python")
        manifest = str(self.manifest_path).removeprefix("wsl:")
        root = Path(__file__).resolve().parents[2]
        wsl_worker = f"/mnt/{root.drive[0].lower()}/{root.as_posix()[3:]}/scripts/wsl_advisory_worker.py"
        cmd = ["wsl", "-d", distro, "--", wsl_python, wsl_worker, "--manifest", manifest]
        payload = json.dumps(summary)
        timeout = 60
        try:
            proc = subprocess.run(cmd, input=payload, text=True, capture_output=True, timeout=timeout)
            if proc.returncode != 0:
                return reply(summary, code="PROCESS_EXIT")
            output = proc.stdout.strip()
            if not output:
                return reply(summary, code="MALFORMED_OUTPUT")
            res = json.loads(output)
            if not isinstance(res, dict) or "source" not in res:
                return reply(summary, code="MALFORMED_OUTPUT")
            return res
        except subprocess.TimeoutExpired:
            return reply(summary, code="TIMEOUT")
        except (OSError, ValueError):
            return reply(summary, code="RESOURCE_ERROR")

    def _status_wsl(self):
        distro = os.environ.get("STEELGUARD_WSL_DISTRO", "Ubuntu-24.04")
        wsl_python = os.environ.get("STEELGUARD_WSL_PYTHON", "/home/sai01/steelguard-build/.venv/bin/python")
        manifest = str(self.manifest_path).removeprefix("wsl:")
        root = Path(__file__).resolve().parents[2]
        wsl_worker = f"/mnt/{root.drive[0].lower()}/{root.as_posix()[3:]}/scripts/wsl_advisory_worker.py"
        cmd = ["wsl", "-d", distro, "--", wsl_python, wsl_worker, "--manifest", manifest, "--status"]
        try:
            proc = subprocess.run(cmd, text=True, capture_output=True, timeout=10)
            if proc.returncode == 0 and proc.stdout.strip():
                return json.loads(proc.stdout.strip())
        except Exception:
            pass
        return {"runtime": "ExecuTorch llama_main (WSL Host)", "configured": True,
                "configuration_validated": False, "offline_only": True,
                "target": "linux-x86_64", "live_deployment_attested": False}

    def advise(self, submitted_summary):
        summary = validate_summary(submitted_summary)
        if not summary["evidence"]:
            return reply(summary, code="EMPTY_EVIDENCE")
        if not self.manifest_path:
            return reply(summary, code="NOT_CONFIGURED")
        if not _GENERATION_LOCK.acquire(blocking=False):
            return reply(summary, code="BUSY")
        try:
            if self._is_wsl_mode():
                return self._advise_wsl(summary)
            if self._config is None:
                self._config = RunnerConfig.load(self.manifest_path)
            config = self._config
            config.verify_unchanged()
            prompt = build_prompt(summary, config)
            output = generate_action(self.runner, config, prompt)
            if not isinstance(output, bytes) or len(output) > config.max_output_bytes:
                raise AdvisoryRuntimeError("OUTPUT_LIMIT")
            plan = parse_plan(output, prompt, summary, config)
            return reply(summary, code="OK", plan=plan, model_id=config.model_id)
        except AdvisoryRuntimeError as exc:
            return reply(summary, code=exc.code)
        except (OSError, MemoryError):
            return reply(summary, code="RESOURCE_ERROR")
        finally:
            _GENERATION_LOCK.release()

    def status(self):
        if self._is_wsl_mode():
            return self._status_wsl()
        valid = False
        if self._config is not None:
            try:
                self._config.verify_unchanged()
                valid = True
            except (AdvisoryRuntimeError, OSError):
                pass
        return {"runtime": "ExecuTorch llama_main", "configured": bool(self.manifest_path),
                "configuration_validated": valid,
                "offline_only": True, "target": "linux-aarch64-cpu",
                "live_deployment_attested": False}


# Explicit administrator configuration only. Requests cannot select runners or artifacts.
advisory_service = AdvisoryService(os.environ.get("STEELGUARD_EXECUTORCH_MANIFEST"))
