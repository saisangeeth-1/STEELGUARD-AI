"""Bounded native ExecuTorch llama_main CLI transport (no Python ET dependency).

Supported production target: Linux AArch64, ExecuTorch release/1.5 CLI.
This module does not download models, invoke a shell, or choose another provider.
"""
import base64
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import queue
import re
import signal
import subprocess
import tempfile
import threading
import time


class AdvisoryRuntimeError(Exception):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def strict_json(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    def invalid_constant(value):
        raise ValueError("Nonfinite JSON constant")
    return json.loads(text, object_pairs_hook=unique, parse_constant=invalid_constant)


def local_file(value):
    if not isinstance(value, str) or not value or "\x00" in value or "://" in value or value.startswith(("\\\\", "//")):
        raise AdvisoryRuntimeError("INVALID_ARTIFACT_PATH")
    path = Path(value)
    if not path.is_absolute():
        raise AdvisoryRuntimeError("INVALID_ARTIFACT_PATH")
    try:
        path = path.resolve(strict=True)
        if not path.is_file() or path.stat().st_size == 0:
            raise OSError()
        return path
    except (OSError, RuntimeError, UnicodeError) as exc:
        raise AdvisoryRuntimeError("MISSING_ARTIFACT") from exc


def stamp(path):
    s = path.stat()
    return s.st_size, s.st_mtime_ns, s.st_ino


def check_hash(path, expected):
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise AdvisoryRuntimeError("INVALID_MANIFEST")
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    if h.hexdigest() != expected:
        raise AdvisoryRuntimeError("ARTIFACT_HASH_MISMATCH")


def check_tokenizer(path):
    """Llama 3.2 tiktoken mergeable-rank file, not GGUF or SentencePiece."""
    if path.stat().st_size > 16 * 1024 * 1024:
        raise AdvisoryRuntimeError("INVALID_TOKENIZER")
    count = 0
    tokens = set()
    try:
        with path.open("rb") as source:
            for count, line in enumerate(source, 1):
                token, rank = line.split()
                decoded = base64.b64decode(token, validate=True)
                if int(rank) != count - 1 or not decoded or decoded in tokens or count > 128000:
                    raise ValueError()
                tokens.add(decoded)
        if count != 128000:
            raise ValueError()
    except (ValueError, OSError) as exc:
        raise AdvisoryRuntimeError("INVALID_TOKENIZER") from exc


def integer(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise AdvisoryRuntimeError("INVALID_LIMITS")
    return value


@dataclass(frozen=True)
class RunnerConfig:
    model: Path
    tokenizer: Path
    runner: Path
    libraries: tuple
    model_id: str
    context_tokens: int
    max_input_bytes: int
    max_new_tokens: int
    timeout_seconds: int
    max_output_bytes: int
    cpu_threads: int
    fingerprints: tuple
    target: str = "linux-aarch64"

    @classmethod
    def load(cls, manifest_path, *, allow_host_qualification=False):
        manifest = local_file(str(manifest_path))
        if manifest.stat().st_size > 32768:
            raise AdvisoryRuntimeError("INVALID_MANIFEST")
        try:
            data = strict_json(manifest.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError) as exc:
            raise AdvisoryRuntimeError("INVALID_MANIFEST") from exc
        required = {"schema_version", "runtime", "runtime_version", "runtime_commit", "export_version", "export_commit",
                    "model_id", "model_family", "tokenizer_format", "backend", "target", "offline_only",
                    "artifacts", "context_tokens", "limits"}
        if not isinstance(data, dict) or set(data) != required:
            raise AdvisoryRuntimeError("INVALID_MANIFEST")
        expected = {"schema_version": 1, "runtime": "executorch-llama-main",
                    "model_family": "llama3_2", "tokenizer_format": "tiktoken",
                    "backend": "xnnpack", "offline_only": True}
        if any(type(data[k]) is not type(v) or data[k] != v for k, v in expected.items()):
            raise AdvisoryRuntimeError("UNSUPPORTED_CONFIGURATION")
        targets = ("linux-aarch64", "linux-x86_64") if allow_host_qualification else ("linux-aarch64",)
        if (data["target"] not in targets or data["runtime_version"] not in ("1.5", "1.5.1")
                or data["export_version"] != data["runtime_version"]):
            raise AdvisoryRuntimeError("UNSUPPORTED_CONFIGURATION")
        if data["model_id"] not in ("meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"):
            raise AdvisoryRuntimeError("UNSUPPORTED_MODEL")
        if not isinstance(data["export_commit"], str) or not re.fullmatch("[0-9a-f]{40}", data["export_commit"]):
            raise AdvisoryRuntimeError("INVALID_MANIFEST")
        if data["runtime_commit"] != data["export_commit"]:
            raise AdvisoryRuntimeError("UNSUPPORTED_CONFIGURATION")
        artifacts = data["artifacts"]
        if not isinstance(artifacts, dict) or set(artifacts) != {"model", "tokenizer", "runner", "libraries"}:
            raise AdvisoryRuntimeError("INVALID_MANIFEST")
        if not isinstance(artifacts["libraries"], list) or len(artifacts["libraries"]) > 16:
            raise AdvisoryRuntimeError("INVALID_MANIFEST")
        entries = [artifacts[k] for k in ("model", "tokenizer", "runner")] + artifacts["libraries"]
        paths, fingerprints = [], [(manifest, stamp(manifest))]
        size_caps = [8 * 1024**3, 16 * 1024**2, 128 * 1024**2] + [256 * 1024**2] * len(artifacts["libraries"])
        for entry, size_cap in zip(entries, size_caps):
            if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
                raise AdvisoryRuntimeError("INVALID_MANIFEST")
            path = local_file(entry["path"])
            before = stamp(path)
            if before[0] > size_cap:
                raise AdvisoryRuntimeError("ARTIFACT_SIZE_LIMIT")
            check_hash(path, entry["sha256"])
            if stamp(path) != before:
                raise AdvisoryRuntimeError("ARTIFACT_CHANGED")
            paths.append(path)
            fingerprints.append((path, before))
        model, tokenizer, runner, *libraries = paths
        if model.suffix != ".pte" or runner.name != "llama_main" or len(set(paths)) != len(paths):
            raise AdvisoryRuntimeError("INVALID_ARTIFACT_CONFIGURATION")
        # Reject scripts, PE executables, x86 binaries and renamed alternative runtimes.
        with runner.open("rb") as binary:
            header = binary.read(20)
        machine = 183 if data["target"] == "linux-aarch64" else 62
        if len(header) != 20 or header[:6] != b"\x7fELF\x02\x01" or int.from_bytes(header[18:20], "little") != machine:
            raise AdvisoryRuntimeError("INVALID_RUNNER")
        if platform.system() == "Linux" and not os.access(runner, os.X_OK):
            raise AdvisoryRuntimeError("INVALID_RUNNER")
        check_tokenizer(tokenizer)
        limits = data["limits"]
        keys = {"max_input_bytes", "max_new_tokens", "timeout_seconds", "max_output_bytes", "cpu_threads"}
        if not isinstance(limits, dict) or set(limits) != keys:
            raise AdvisoryRuntimeError("INVALID_LIMITS")
        context = integer(data["context_tokens"], 2048, 32768)
        inp = integer(limits["max_input_bytes"], 2048, 16000)
        new = integer(limits["max_new_tokens"], 64, 512)
        timeout = integer(limits["timeout_seconds"], 1, 60)
        out = integer(limits["max_output_bytes"], 4096, 65536)
        threads = integer(limits["cpu_threads"], 1, 4)
        # ASCII prompt byte count is a conservative Llama byte-BPE token bound.
        if inp + new + 128 > context or out < inp + 2048:
            raise AdvisoryRuntimeError("INVALID_LIMITS")
        return cls(model, tokenizer, runner, tuple(libraries), data["model_id"], context,
                   inp, new, timeout, out, threads, tuple(fingerprints), data["target"])

    def verify_unchanged(self):
        try:
            if any(stamp(path) != previous for path, previous in self.fingerprints):
                raise AdvisoryRuntimeError("ARTIFACT_CHANGED")
        except OSError as exc:
            raise AdvisoryRuntimeError("MISSING_ARTIFACT") from exc


def run_bounded(argv, *, timeout_seconds, max_output_bytes, env, cwd, metrics=None):
    """Stream both pipes through a bounded queue; kill the POSIX group on any failure.

    This transport is also tested using synthetic Python subprocesses. Those tests
    establish containment behavior, not real ExecuTorch inference.
    """
    started = time.monotonic()
    if metrics is not None:
        metrics.update(wall_seconds=None, peak_sampled_rss_kib=None, observed_rss_high_water_kib=None, process_id=None)
    try:
        proc = subprocess.Popen(argv, shell=False, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, cwd=cwd, env=env,
                                start_new_session=(os.name == "posix"))
    except (OSError, MemoryError) as exc:
        raise AdvisoryRuntimeError("PROCESS_START_FAILED") from exc
    events = queue.Queue(maxsize=8)
    stop = threading.Event()

    def pump(channel, pipe):
        try:
            while not stop.is_set():
                chunk = os.read(pipe.fileno(), 1024)
                event = (channel, chunk if chunk else None)
                while not stop.is_set():
                    try:
                        events.put(event, timeout=0.05)
                        break
                    except queue.Full:
                        continue
                if not chunk:
                    break
        except OSError:
            while not stop.is_set():
                try:
                    events.put((channel, None), timeout=0.05)
                    break
                except queue.Full:
                    continue

    workers = [threading.Thread(target=pump, args=(channel, pipe), daemon=True)
               for channel, pipe in (("out", proc.stdout), ("err", proc.stderr))]
    for worker in workers:
        worker.start()
    deadline = time.monotonic() + timeout_seconds
    total, ended, output = 0, set(), bytearray()
    try:
        while len(ended) < 2 or proc.poll() is None:
            if metrics is not None:
                metrics["process_id"] = proc.pid
                try:
                    status = Path(f"/proc/{proc.pid}/status").read_text()
                    match = re.search(r"^VmRSS:\s+(\d+)\s+kB", status, re.MULTILINE)
                    if match:
                        metrics["peak_sampled_rss_kib"] = max(metrics["peak_sampled_rss_kib"] or 0, int(match.group(1)))
                    high_water = re.search(r"^VmHWM:\s+(\d+)\s+kB", status, re.MULTILINE)
                    if high_water:
                        metrics["observed_rss_high_water_kib"] = max(
                            metrics["observed_rss_high_water_kib"] or 0, int(high_water.group(1)))
                except OSError:
                    pass
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AdvisoryRuntimeError("TIMEOUT")
            try:
                channel, chunk = events.get(timeout=min(0.05, remaining))
            except queue.Empty:
                continue
            if chunk is None:
                ended.add(channel)
            else:
                total += len(chunk)
                if total > max_output_bytes:
                    raise AdvisoryRuntimeError("OUTPUT_LIMIT")
                if channel == "out":
                    output.extend(chunk)
        if proc.returncode != 0:
            raise AdvisoryRuntimeError("PROCESS_EXIT")
        return bytes(output)
    finally:
        stop.set()
        # Kill descendants holding pipes too, even if the direct child already exited.
        if os.name == "posix":
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        elif proc.poll() is None:
            proc.kill()
        proc.wait(timeout=2)
        for worker in workers:
            worker.join(timeout=1)
        proc.stdout.close()
        proc.stderr.close()
        if metrics is not None:
            metrics["wall_seconds"] = time.monotonic() - started


class ExecuTorchRunner:
    def __init__(self, *, allow_host_qualification=False):
        self.allow_host_qualification = allow_host_qualification

    def generate(self, config, prompt, *, metrics=None):
        machine = platform.machine().lower()
        host_matches = machine in ("aarch64", "arm64") if config.target == "linux-aarch64" else machine in ("x86_64", "amd64")
        if (platform.system() != "Linux" or not host_matches
                or (config.target != "linux-aarch64" and not self.allow_host_qualification)):
            raise AdvisoryRuntimeError("UNSUPPORTED_HOST")
        config.verify_unchanged()
        if len(prompt.encode("ascii")) > config.max_input_bytes:
            raise AdvisoryRuntimeError("INPUT_LIMIT")
        env = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
               "OMP_NUM_THREADS": str(config.cpu_threads), "CUDA_VISIBLE_DEVICES": "",
               "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        # No inherited LD_PRELOAD, proxy, credentials, Python or provider environment.
        if config.libraries:
            env["LD_LIBRARY_PATH"] = ":".join(sorted({str(p.parent) for p in config.libraries}))
        argv = [str(config.runner), f"--model_path={config.model}", f"--tokenizer_path={config.tokenizer}",
                f"--max_new_tokens={config.max_new_tokens}", "--temperature=0", "--num_bos=0", "--num_eos=0",
                f"--cpu_threads={config.cpu_threads}", "--warmup=false", "--ignore_eos=false", f"--prompt={prompt}"]
        with tempfile.TemporaryDirectory(prefix="steelguard-advisory-") as temp:
            return run_bounded(argv, timeout_seconds=config.timeout_seconds,
                               max_output_bytes=config.max_output_bytes, env=env, cwd=temp, metrics=metrics)
