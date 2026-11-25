# Phase 5C: offline ExecuTorch advisory deployment

Phase 5D update: see [qualification report/runbook](PHASE5D_QUALIFICATION.md) for exact source/model
pins, the export/build/bundle helpers, real offline WSL inference measurements and the ARM64
package transfer procedure. Physical Pi inference is pending. The pinned source is 1.5.1; the adapter
now accepts matching 1.5.1 export/runtime versions. Explicit Linux x86 qualification does not
enable x86 manifests through the production HTTP service.

The original Phase 5C Windows tests exercised the adapter. Phase 5D then produced the PTE,
verified real x86-64 WSL generation and cross-built the ARM64 package. The Pi still needs a
runtime and coexistence test. The official PS5 specification remains unavailable.

## Interface and separation of facts from recommendations

The existing vision services, summary API, dashboard and history schema are unchanged.
The advisory service accepts only the complete v1.1 JSON result returned by
POST /api/intelligence/summarize. It reconstructs the deterministic summary from its evidence
and rejects mismatched verdicts, severity, counts, warnings, policy, source or extra fields.
This is consistency validation, not proof that an image or submitted result is authentic.

POST /api/intelligence/advisory accepts exactly:

```json
{"summary": "REPLACE THIS STRING WITH THE COMPLETE SUMMARY OBJECT"}
```

The example string is deliberately not valid input. Pass the actual object, without
editing its fields. Raw images, notes, custom prompts, paths and provider settings are rejected.
Request size is limited to 128 KiB. Invalid requests return 400; oversized requests return 413.
Valid requests return 200 with either EXECUTORCH or RULE_BASED_FALLBACK as source, an explicit
llm_used flag and reason_code. Runtime errors do not erase inspection results or write history.

Every response preserves quality_verdict and measured_facts from deterministic evidence;
inspection_source distinguishes AI_INFERENCE from GROUND_TRUTH.
explanation is the deterministic rule explanation. recommendations are operator guidance,
never measurements. facts_source explicitly identifies the deterministic layer.
live_deployment_attested remains false; a successful process is not a hardware acceptance test.

GET /api/intelligence/advisory/status reports configuration state without loading a model or
claiming live readiness. configuration_validated means static manifest validation has previously
succeeded and tracked files have not changed; it is not an inference or deployment attestation.
The dashboard continues displaying its existing deterministic summary and does not automatically
call this potentially slower advisory endpoint.

## How the adapter validates the model's action

The prompt uses Llama 3 instruction delimiters and fixed system rules. It sends only bounded
numeric evidence, known GC10 class codes, fixed severity names, warning codes and the existing
verdict. Unknown class labels become UNMAPPED. Filenames, free-form rule prose, raw labels
outside the known catalog, model names, images and masks never enter the prompt.

The qualified 1B model returns one offered letter (optionally followed by a period). The runtime
caps generation at two tokens. The adapter maps the letter to an allowed recommendation and
constructs the plan with the existing rule verdict and finding IDs. An accepted choice acknowledges
the listed limitations. This is the resulting plan shape:

```json
{
  "verdict": "REJECT",
  "finding_ids": [1],
  "action_codes": ["HOLD_FOR_DISPOSITION", "CHECK_IMAGE_COVERAGE"],
  "acknowledge_warnings": true
}
```

All finding IDs must match the supplied evidence in order. Actions come from the verdict-specific
allowlist. Unoffered letters and malformed responses cause fallback. The existing strict JSON
parser also checks complete plan responses against deterministic facts. The live 1B test showed
that the longer JSON prompt produced malformed plans, while the bounded choices passed all three
offline trials of the qualification fixture.

Python renders factual explanations and operator actions from approved templates. Only the
selection of additional actions comes from Llama. No unrestricted generated prose is exposed.
The mandatory deterministic policy action remains first even if Llama omits it. This prevents
generated dimensions, confidence or severity from entering engineering evidence. U-Net evidence
remains separate; this adapter makes no claim of image/coordinate compatibility for fusion.

## Supported deployment contract

Use Raspberry Pi OS 64-bit on Pi 5, Linux AArch64, CPU-only XNNPACK. Begin qualification with
meta-llama/Llama-3.2-1B-Instruct. The 3B Instruct identity is also accepted, but memory and
latency feasibility are unverified. Other architectures, runtimes and model families fail closed.

The [official ExecuTorch Pi guide](https://docs.pytorch.org/executorch/stable/raspberry_pi_llama_tutorial.html)
describes Linux export/build and ARM64 deployment of a .pte, tokenizer and Llama executable.
Keep this tooling in an isolated Linux environment, separate from the Windows CUDA venv.
Pin the exact source commit and matching release/1.5 exporter/runtime, then follow that
release's setup and export instructions. Export a self-contained quantized XNNPACK .pte
with the intended context capacity. External model-data files are not supported by this adapter.

The pinned [native CLI source](https://github.com/pytorch/executorch/blob/release/1.5/examples/models/llama/main.cpp)
defines the model/tokenizer, generated-token, prompt and CPU-thread options used here.
Generation is greedy, with no warmup. The
[upstream runner interface](https://github.com/pytorch/executorch/blob/release/1.5/extension/llm/runner/irunner.h)
enables prompt echo by default; the adapter removes only an exact echo. It validates the
[upstream statistics record](https://github.com/pytorch/executorch/blob/release/1.5/extension/llm/runner/stats.h)
after the generated plan. Other CLI versions/output layouts require a reviewed adapter update,
not silent compatibility guesses. The runtime itself must validate the exported graph on target.

Acquire the original matching Llama tiktoken tokenizer.model, not GGUF, SentencePiece or an
Ollama model. The static validator requires the 128,000 ranked base-token entries used by
this supported Llama family. It checks rank structure and uniqueness, not semantic model
compatibility. Preserve the licensed model source and tokenizer identity in deployment records.

The [Meta model page](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) documents gated
access and the Llama 3.2 Community License. Obtain authorized access and review the license,
use policy, attribution and redistribution requirements before downloading or shipping artifacts.
Store license/notice files alongside deployment artifacts. Model licensing is not granted by
this repository. Do not download third-party runtime substitutes.

## Manifest and startup

Copy deployment/executorch_advisory.example.json to a separately managed local manifest.
The example contains invalid placeholders and is never loaded automatically. Set actual absolute
local paths, SHA256 hashes, exact export commit, exported context capacity and tested limits.
runtime_commit must match export_commit; release versions must be equal supported 1.5 or 1.5.1.
The Phase 5D candidate pins 1.5.1 to an exact immutable source commit.
Include all non-system runtime libraries and their hashes. A statically linked build may use
an empty libraries list. Verify target GLIBC and C++ library compatibility; prefer rebuilding
against the Pi's sysroot instead of changing system libraries for an incompatible binary.

Native artifacts and manifest must be administrator-controlled and read-only to the service.
Hashes provide integrity against the trusted manifest, not a publisher signature or proof of
model architecture. Static checks cannot prove tokenizer/PTE pairing, CPU graph lowering or
absence of hidden native behavior. Pin the source/build provenance and inspect runtime linkage.
The runner must be a Linux AArch64 ELF named llama_main, not a script or renamed Windows binary.
Artifact replacement after validation causes fallback; restart after an intentional deployment.
Static size ceilings are 8 GiB per .pte, 16 MiB tokenizer, 128 MiB runner and 256 MiB per
listed runtime library. Initial integrity hashing is streamed and adds cold-request latency
outside the generation timeout; subsequent requests check file fingerprints. Administrative
artifact replacement or concurrent modification is unsupported; hashes are not re-read on
every inference if file fingerprints are unchanged.

On the Pi, run static preflight (this never launches inference):

```bash
python3 scripts/check_executorch_advisory.py --manifest /opt/steelguard/advisory.json
```

Then explicitly enable the service in its launch environment:

```bash
export STEELGUARD_EXECUTORCH_MANIFEST=/opt/steelguard/advisory.json
python3 steelguard_app/run_dashboard.py
```

No manifest means deterministic fallback. There is no environment option for a cloud URL,
arbitrary command, alternate provider, shell wrapper or client-selected executable.
Existing model_config.json is not modified. The adapter only starts the configured native runner.

## Bounds and failure behavior

Example limits: 6,000 prompt bytes, 20 seconds, 64 KiB combined stdout and stderr, and two CPU
threads. The advisory action protocol uses a two-token generation cap. General runner manifests
allow up to 16,000 input bytes, 512 new tokens,
60 seconds, 64 KiB combined output and four threads. A prompt may contain at most 32 findings.
Excessive evidence falls back, without silently discarding findings.

The complete prompt is ASCII; its byte length is a conservative byte-BPE token bound.
max_input_bytes + max_new_tokens + 128 must fit the declared exported context_tokens.
The CLI separately limits generated tokens; the returned token-count statistics are checked.
Export metadata must still be verified against the actual .pte on target.

One generation runs at a time per server process, with no waiting queue. Both output pipes
are drained concurrently through a bounded queue. Timeout or excessive output kills the POSIX
process group and reaps the direct child. stdin is closed, shell execution is disabled, argv
entries are separate, and the process receives a minimal environment without inherited proxies,
credentials or LD_PRELOAD. CUDA is hidden. Dependencies must be local; no HTTP inference code,
model downloads, subprocess provider discovery or runtime fallback provider exists.

Reason codes distinguish NOT_CONFIGURED, EMPTY_EVIDENCE, BUSY, missing/invalid artifacts,
hash mismatch, changed artifacts, unsupported host/configuration, input/output limits,
timeout, process exit/start failure, malformed output and resource errors. Fallback is always
explicit RULE_BASED_FALLBACK with llm_used false. Failed runtime output and stderr are never
returned as fabricated Llama advice or leaked in API errors.

The adapter enforces time/input/output/concurrency limits, not a hard native RSS ceiling.
Use OS cgroup limits and qualification measurements to protect vision memory. The current
stdlib HTTPServer handles requests synchronously, so one advisory call can delay other
requests up to its configured bound. The dashboard does not invoke it automatically.
Before enabling frequent operator advisory calls, measure this impact and decide whether a
separate bounded worker service is required; do not assume concurrent-server safety for the
existing model services.

Offline-only configuration prevents alternate providers; environment flags are not a network
sandbox for native code. For deployment assurance, run with outbound network denied by the OS
and verify operation with network disconnected. Use least privilege, a read-only artifact mount,
and a writable private temp directory. A systemd unit can restrict address families to AF_UNIX
and AF_INET/AF_INET6 as needed for the local dashboard while firewall rules restrict outbound
traffic. Test isolation together with camera and loopback access; no service policy was installed
or tested in this Windows session.

## Acceptance and remaining deployment work

1. Confirm PS5 specification, Pi RAM/OS, disk capacity, model access and license records.
2. Export authorized quantized 1B Instruct weights and matching tokenizer using isolated Linux
   tooling; build matching ARM64 CPU runner/libraries and fill the manifest with real hashes.
3. Run static preflight, native CLI help/load checks and real adapter requests on the Pi.
   Verify output layout, JSON success/fallback rate and actual prompt/context bounds.
4. Measure cold startup, peak RSS, temperature/throttling, token latency and concurrent vision
   budget. Test malformed output, forced process termination, restart and disconnected network.
5. Record an explicit hardware acceptance report before claiming PS5 deployment compliance.

Automated tests cover synthetic configuration fixtures, mocked generation contracts and real
Python subprocess containment. These are not ExecuTorch inference. Existing YOLO/U-Net GPU and
CPU fallback tests remain Windows evidence, not Pi benchmarks. The next concrete deployment
task is to produce the real self-contained .pte/tokenizer/ARM64 runner bundle and qualify it
on physical Pi 5; the adapter implementation does not supply those missing artifacts.

Development verification on 2026-11-15: 30 new adapter/process tests, 6 new API tests and
all 11 isolated regression suites passed. The existing 15 frontend scenarios and real Edge
YOLO/dashboard smoke passed. Production history/configuration and three checkpoint hashes
were preserved. Reports are in runs/phase5c_verification/. These results do not include any
real ExecuTorch model execution or physical Pi test.
