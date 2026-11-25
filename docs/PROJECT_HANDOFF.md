# STEELGUARD AI continuation handoff

## Antigravity handoff on 2026-11-17

Phase 5D is qualified for WSL x86-64 CPU. The 768.8 MiB INT4 export ran three times offline
and passed action validation. The closeout run exercised the real advisory service and verified
both successful advice and a one-second rule-based timeout fallback with the same `REJECT` verdict.
The cross-built Pi 5 ARM64 package is `/home/sai01/steelguard-pi5-bundle-20261116-v2` in WSL.
All 26 indexed files passed the closeout checksum check. The artifact paths, full hashes,
timings, and transfer commands are in [Phase 5D qualification](PHASE5D_QUALIFICATION.md).

All 12 isolated regression suites passed at closeout, including the advisory HTTP API
and qualification tooling. The report records matching production SQLite, configuration
and vision checkpoint hashes. [Regression report](../runs/phase5d_verification/closeout-20261117/report.json)

### UI Polish & Lifecycle Invalidation Verification (2026-11-17)
- **ExecuTorch Advisory Heading Overlap Fix**:
  Resolved visual overlap between the cyan "LLM" badge and the heading "EXECUTORCH LOCAL ADVISORY — LLAMA 3.2 1B INSTRUCT".
  Replaced rigid 14px square icon sizing with dedicated semantic inline-flex tags (`.advisory-card-header`, `.advisory-card-title`, `.advisory-tag-llm`, `.advisory-heading-text`) in `steelguard_app/static/index.html` and `steelguard_app/static/css/components.css`.
  Validated layout with headless Edge across multiple resolutions: 1366x768 (12px gap, 0 overlap), 1440x900 (12px gap, 0 overlap), 1680x1050 (12px gap, 0 overlap), and 1024x768 (8px gap, 0 overlap).
- **On-Demand Advisory & Lifecycle Invalidation**:
  Updated `steelguard_app/static/js/app.js` with `invalidateAdvisory(reason)`. Real ExecuTorch advisory is invoked strictly on operator demand via "Request Llama Advisory".
  Switching inspected samples, loading dataset samples, or initiating new inference immediately purges stale advisory action plans, resets the advisory status badge to `AWAITING INFERENCE`, and disables the request button until fresh YOLO detection completes.
- **Regression Suite**:
  Created `steelguard_app/test_advisory_ui_overlap.py` and registered as suite #15 in `scripts/run_dashboard_regressions.py`. All 15 suites passed cleanly with 0 production file modifications. Output screenshots and report recorded in `runs/phase5_demo_verification/`.

Antigravity should transfer that exact package to the physical 4 GB Pi 5, verify the GLIBC and
GLIBCXX requirements and relocate its manifest, then run the offline and timeout probes. Next,
measure RAM, swap, temperature, throttling and inspection latency with the camera, vision,
dashboard and advisory service running together. Keep advice in rule-based mode until the target
checks pass. Record physical results separately and obtain the official PS5 specification before
signing off. The sections below retain the project history and earlier test evidence.

Verified 2026-11-15 in D:\INDUSTRIAL-QC-AI. No Git metadata or existing AGENTS.md was
found. No official PS5 specification, Llama weights/tokenizer, or ExecuTorch .pte artifact
was found outside the Python environment. User handoff requires offline Pi 5 ARM64 CPU,
Llama + ExecuTorch, camera, dashboard and traceability. Ollama is unauthorized despite
the older phase4c_final_acceptance_report.md recommending it.

## Current state
The working dashboard is stdlib HTTPServer, not the claimed FastAPI implementation.
FastAPI and ExecuTorch are not installed in the dedicated Python 3.12.10 environment;
torch, ultralytics, OpenCV and NumPy are available. pytest and playwright are absent.
YOLO and U-Net successfully loaded and performed real GPU inference in regression tests.
SQLite persistence and separate model taxonomies are implemented. Camera and LLM telemetry
are placeholders. Phase 5B now connects successful YOLO results to the deterministic summary
endpoint and renders them in the Inspection Intelligence panel.
No physical Pi or ARM64 test was performed. No claim of full project completion is warranted.

## Code map
- train_yolo.py / val_yolo.py / infer_yolo.py: detection lifecycle.
- scripts/prepare_yolo_dataset.py, prepare_neu_inclusion_dataset.py: detection data preparation.
- scripts/train_severstal_unet.py, evaluate_severstal_unet.py,
  visualize_severstal_predictions.py: segmentation lifecycle.
- scripts/severstal_dataset.py, severstal_rle.py, segmentation_losses.py, unet_model.py:
  data decoding, training and architecture. Backend contains its own U-Net definition.
- steelguard_app/backend/model_manager.py: model catalog, integrity checks, YOLO inference.
- steelguard_app/backend/segmentation_service.py: U-Net inference, masks and sample browsing.
- steelguard_app/backend/quality_engine.py: GC10 quality rules.
- steelguard_app/backend/server.py: API routing, telemetry, static serving.
- steelguard_app/backend/db.py: SQLite inspection and segmentation stores and exports.
- steelguard_app/static/: dashboard; steelguard_app/run_dashboard.py: launch entrypoint.
- runs/project_audit/, runs/experiments/, runs/segmentation/: historical reports.
- No operational Pi deployment scripts were located.

## Checkpoints (SHA256 freshly verified)
| File | SHA256 |
| --- | --- |
| runs/detect/steelguard_gc10n/weights/best.pt | ffacba4cc1a781d6fd023baf04ccb135890cbd4edf9d78cdab5bb89cbab08ee6 |
| runs/detect/steelguard_gc10_neu_inclusion/weights/best.pt | 41ce4fd1099db9bce7e93b734bcfba40810fa365f1e58a82c30da8a24a3f8e2c |
| runs/segmentation/severstal_unet_v1/best_model.pt | 10666e11ecf088489fc2544d85e50cce0108b93c5cb888974f30d886029104e3 |

Historical metrics confirmed in saved artifacts, not recomputed on the full test sets:
candidate mAP50 0.641442 (baseline reported 0.6063), inclusion AP50 0.305396
(baseline reported 0.1498), U-Net defect macro IoU 0.227437 / Dice 0.347713.
Sources: runs/experiments/neu_inclusion_training/experiment_metrics.json and
runs/segmentation/severstal_unet_v1/test_evaluation_summary.json.
The latter reports 65.35% clean-image false positives and crack IoU 0.17252;
segmentation remains experimental. GPU latency must not be presented as Pi performance.

## Actual verification this session
- scripts/test_severstal_pipeline.py: 13/13 passed.
- scripts/test_severstal_training.py: 10/10 passed (includes synthetic training steps,
  not retraining either production model).
- test_stage5_phase3.py: 10/10 passed.
- test_stage6_phase4c.py: 10/10 passed.
- test_stage4.py: 8/8 passed, invoking Stage 3 (7 checks) and Stage 2 internally.
  Captured Python exit code 0; log: regression_stage4.log.
- test_inspection_intelligence.py: 8/8 passed after final edits; backend py_compile passed.
- Phase 3 rerun after rule-engine edits: 10/10 passed (regression_phase3_final.log).
- In-memory HTTP dispatch: summary returns 200 with QR-01 REJECT; invalid list returns 400.
- Browser screenshots, resource verification, full held-out evaluations and physical Pi
  execution were not run. The historical 48-test count uses different suite grouping;
  avoid summing nested legacy checks as independent unit tests.

Windows sandbox denied temporary file access for some legacy suites; rerunning with
approval succeeded. Phase 3/4C used temporary database and model configuration overrides.
Log: regression_phase3_4c.log. Phase 4C wrote a test PNG under data/segmentation_artifacts;
API smoke check created runs/intelligence_api_test.db. Production checkpoints were untouched.
Final Phase 3 rerun also created runs/test_model_config.json with an isolated test configuration.

## Changes in this continuation
- Added root AGENTS.md and this handoff.
- Added backend/inspection_intelligence.py and test_inspection_intelligence.py.
- Added POST /api/intelligence/summarize, accepting existing inference-result-shaped JSON:
  source AI_INFERENCE, status SUCCESS, explicit detections list. GROUND_TRUTH is separately
  labeled. Missing/failed inference, malformed geometry and nonfinite areas are rejected.
- Deterministic result contains class counts, GC10 assessment, policy version, warnings
  and operator action. Unknown classes cannot silently pass alongside a passing known class.
  It does not fuse Severstal into GC10, call an LLM or persist history. Submitted evidence
  is not authenticated; do not treat this as provenance verification.
- Fixed area-only fluid detection crash (eager evaluation of dict.get default) and sorted
  set-derived explanations to make rule output reproducible.

## Next increments and deployment strategy
1. Phase 5B dashboard integration is complete (details below). Persist policy version/evidence
   linkage if summaries later enter history; summaries currently do not write history.
2. Obtain PS5 document. Keep the stdlib server unless a demonstrated need warrants migration.
3. Use an isolated Linux export/build environment, preserving Windows CUDA venv. Pin an
   ExecuTorch release and compatible export dependencies; start by evaluating Llama 3.2 1B
   Instruct, subject to license/access and measured Pi memory. Model choice is provisional.
4. Export a quantized .pte for XNNPACK CPU, build the corresponding ARM64 llama runner,
   and record model/tokenizer/binary hashes and versions in a deployment manifest.
5. Add a bounded local runner adapter (no shell or network, timeout, serialized generation,
   disabled when artifacts absent). Keep generated prose advisory and separate from the
   immutable deterministic verdict. Missing runtime must return a truthful fallback.
6. Transfer artifacts for offline execution; measure startup, peak RSS, token latency and
   concurrent vision budget on actual Pi 5. Verify no CUDA dependency on target.
7. Add Pi camera capture with lifecycle/error handling; test disconnected camera and recovery.
8. Complete FastAPI alignment if required, end-to-end traceability, restart/offline tests,
   service packaging and operator documentation. No raw-data or checkpoint replacement.

Primary deployment reference consulted:
[Official ExecuTorch Raspberry Pi guide](https://docs.pytorch.org/executorch/stable/raspberry_pi_llama_tutorial.html).
It documents Linux-host export, ARM64 cross-compilation, .pte deployment and a local Llama
runner. Recheck release-specific commands before building; these steps were not executed.

## Phase 5B — operator dashboard integration (2026-11-15)

Completed without FastAPI migration, model retraining, checkpoint changes or raw-data writes.
After a successful POST /api/inference/run, app.js forwards the actual YOLO result to
POST /api/intelligence/summarize. The existing Explain Inspection button and sidebar open
the Inspection Intelligence panel. No user action is needed to generate a summary.

Contract version 1.1 adds per-detection evidence (GC10 type, policy severity, confidence,
bounding box and pixel area), scalar image/model provenance, and verdict-specific operator
actions. Severity derives from configured rules, not confidence or caller-supplied criticality.
Existing Phase 5A fields remain available. Confidence is validated; null/missing area can be
derived from a valid box. Explicit non-GC10 taxonomy is rejected. Ground truth remains supported
by the endpoint but the dashboard summarizes only successful AI_INFERENCE results.

The panel uses text nodes for submitted strings, shows policy and provenance limitations,
qualifies empty detections, and states that Llama/ExecuTorch is unconfigured. It never fuses
Severstal results. A strict request allowlist excludes masks/generated prose; a summary marked
LLM-generated or fused is rejected by the renderer. There is no claim that pixel area is a
physical measurement, or that client-submitted evidence has authenticated provenance.

State handling covers loading, malformed responses, HTTP/network errors, a 15-second request
timeout, new images, reruns and out-of-order results. Summary failures preserve YOLO results
and the save action. Loading a new sample now clears old upload bytes and metadata; late image
and inference responses are discarded. Successful reruns receive a new inspection ID. Saved
YOLO records now retain the actual model name/version and inference latency rather than the
old generic Unloaded/null metadata. Existing database schema and history records are preserved.

Files changed:
- backend/inspection_intelligence.py: evidence/provenance contract and validation.
- static/js/app.js: YOLO summary hook, image generation guards, save metadata.
- static/js/intelligence.js (new): isolated controller and text-only renderer.
- static/index.html, static/css/components.css: operator panel and responsive evidence table.
- test_inspection_intelligence.py: extended from 8 to 12 tests.
- test_intelligence_api.py (new): 3 API dispatch tests, including no-history-write assertions.
- test_intelligence_frontend.mjs (new): 15 DOM/controller and real app-handler scenarios.
- test_intelligence_browser.py (new): actual Edge + HTTP + YOLO + rendered panel smoke.
- scripts/run_dashboard_regressions.py (new): isolated legacy-suite runner with hash preservation.
- Root AGENTS.md and this document.

Validation artifacts are under runs/phase5b_verification/. The frontend tests cover invalid,
empty and multiple defects; API/network failure; stale responses; safe string rendering;
source separation; model metadata on save; and refusal to display generated/fused contracts.
The real Edge run uses the existing held-out sample and actual YOLO predictions (not a mocked
defect class), with an empty temporary history store and no save requests. Browser DOM, JSON
result and intelligence_dashboard.png are retained. The screenshot was visually inspected.

The regression runner includes the existing Stage 4/3/2 suites, Phase 3/4C vision suites,
Severstal pipeline/training mechanics, resource/CPU/OOM checks and traceability against a
read-only copy of existing history. It keeps production DB/config unchanged and compares
their hashes plus all three checkpoints before/after. Phase 4C still leaves a new test PNG
in its legacy segmentation artifact directory. Browser and Node tests are separate commands.
See report.json, frontend_report.json and browser_report.json for final results.

Final Phase 5B results: intelligence 12/12; API 3/3; frontend 15/15; real browser smoke PASS;
all nine suites in the isolated regression runner exited 0. Stage 4 includes nested legacy
checks; those are not counted again as independent unit tests. Production-file preservation
checks passed. The legacy screenshot acceptance script with external Antigravity paths was
not executed; the new self-contained Edge test verifies this dashboard integration instead.

The next implementation identified at Phase 5B was a disabled-by-default local ExecuTorch advisory adapter with
a manifest containing Llama model identity, matching tokenizer, .pte and ARM64 runner hashes,
and the pinned export/runtime version. Enforce no shell execution, bounded input/output,
generation timeout and one active request. Missing artifacts must yield a deterministic-only
fallback. Keep returned generated prose in a separate advisory field; it cannot replace the
rule verdict, boxes, confidence, pixel area or model provenance. Test process failure/timeout
and absent/incorrect artifacts on Windows first, then prepare quantized Llama 3.2 1B Instruct
with isolated Linux tooling and validate XNNPACK CPU execution on actual Pi 5. No ARM64/Pi
performance or deployment compliance is established by this Windows work.

## Phase 5C — bounded ExecuTorch advisory adapter (2026-11-15)

Implementation and local API are complete. **Live ExecuTorch inference and physical Pi
deployment remain unverified.** No .pte, matching licensed tokenizer or real ARM64 runner was
available. No model/runtime downloads, checkpoint changes, training, raw-data writes, database
schema changes, dashboard changes or FastAPI migration were made. The official PS5 document
remains missing; do not assert full specification compliance.

New files:
- steelguard_app/backend/executorch_runner.py: strict administrator manifest validation,
  streamed SHA256 checks, Llama 3.2 tokenizer-rank validation, AArch64 ELF runner checks,
  bounded stdout/stderr, timeout, child-process cleanup and real llama_main CLI invocation.
- steelguard_app/backend/advisory_service.py: summary recomputation/validation, safe prompt,
  constrained plan parser, read-only replies and deterministic fallback.
- steelguard_app/test_executorch_advisory.py and test_advisory_api.py: automated adapter,
  subprocess and API regression coverage, explicitly distinguishing mocks from inference.
- deployment/executorch_advisory.example.json: inactive manifest with invalid placeholders.
- scripts/check_executorch_advisory.py: static preflight with truthful unverified flags.
- docs/EXECUTORCH_ADVISORY_DEPLOYMENT.md: deployment, bounds, licensing and acceptance guide.

Updated server.py adds POST /api/intelligence/advisory and GET /api/intelligence/advisory/status,
with advisory body-size and malformed-JSON/header handling. The existing summary, vision and
history routes remain in place. scripts/run_dashboard_regressions.py now includes advisory
suites and reports under runs/phase5c_verification/. Root AGENTS.md includes commands/constraints.

Only an unmodified v1.1 summary object is accepted, wrapped in {"summary": object}. The service
recomputes evidence and rules before invoking any runner. Raw images, custom prompts, runner
paths, provider options and tampered verdicts/severity/warnings are rejected. Evidence consistency
does not attest image provenance. The original dashboard remains deterministic and does not
automatically invoke the advisory API.

The supported production transport is ExecuTorch release/1.5 native llama_main on Linux ARM64,
CPU-only XNNPACK, with explicit absolute .pte/tokenizer/runner/library paths and hashes. Only
Llama 3.2 1B/3B Instruct identities are accepted. Shell execution and alternate providers are
not supported. Configuration is disabled unless STEELGUARD_EXECUTORCH_MANIFEST is set.
Static checks and hashes are not a proof of graph architecture, tokenizer pairing or native
binary authenticity. Files must come from a trusted licensed export/build process.

The Llama instruction prompt contains only known class codes, numeric facts, severity enums,
warning codes and the immutable verdict. Unknown/untrusted labels and provenance strings are
excluded. The model returns finding references plus approved action codes; arbitrary generated
prose, dimensions or new facts are rejected. Python renders facts and actions from templates.
The mandatory policy action and complete warnings remain visible regardless of model selection.
Responses identify EXECUTORCH versus RULE_BASED_FALLBACK, llm_used, reason_code, facts_source,
and live_deployment_attested=false. They never replace deterministic inspection verdicts/history.

Bounds include 128 KiB request/summary, at most 32 findings in the prompt, configured prompt
byte/context and generated-token budgets, combined stdout/stderr byte ceiling, execution timeout,
and one generation per server process. Failed/missing runtime, changed artifacts, malformed
plans, resource failures and excessive output yield explicit deterministic fallback. No silent
provider substitution or Llama-like fabricated text is returned.

The stdlib server remains synchronous: an explicitly requested advisory call can delay other
requests up to its configured bound. There is no hard native RSS cap in Python; qualify Pi
memory and OS cgroup/network policy before deployment. Native offline flags are not a network
sandbox. The documented process-group containment must be exercised on Linux Pi hardware;
Windows subprocess tests establish only the behavior exercised on this development host.

Final Phase 5C verification:
- 30 adapter/configuration/prompt/parser/subprocess tests passed. Successful model-output
  cases use mocks; Python subprocess containment cases use real child processes, not Llama.
- 6 advisory API tests passed, including an actual loopback HTTP round trip and no history writes.
- All 11 isolated regression suites exited 0, including existing vision/history/resource suites.
- Existing 15 frontend scenarios passed. The real Edge + YOLO + summary dashboard smoke passed;
  it does not exercise ExecuTorch. Browser report is copied into the Phase 5C output folder.
- Changed Python files compiled. Static preflight correctly rejects the inactive placeholder
  manifest with INVALID_MANIFEST and live/physical verification flags false.
- Final hashes confirm production history/config and all three checkpoints remain unchanged.
  Results: runs/phase5c_verification/report.json, validation_summary.json, advisory.log,
  advisory_api.log, frontend_report.json, browser_report.json and example_preflight.json.
No runtime, model or licensing grant is implied by these automated results.

Next deployment task: acquire authorized Llama 3.2 1B Instruct weights and matching tokenizer,
export a self-contained quantized XNNPACK .pte on an isolated Linux host, build/pin the matching
ARM64 runner/libraries, fill the manifest, then perform real adapter inference and resource,
offline, restart and vision-coexistence acceptance on Pi 5. Follow the linked official tutorial;
record license/access requirements and available host/target disk before acquiring weights.

## Phase 5D — source-qualified toolchain preparation (2026-11-15)

**Real ExecuTorch inference and Pi deployment are still unverified.** Auth-check for the manually
gated Meta model returned 401, with no existing local HF credential. The Windows host initially
reported WSL absent; later inventory reported WSL has no installed distributions. No Docker or
Linux export environment was available. The user elected to set up WSL and authenticate locally.
Only those access/host actions are pending; no token was solicited or recorded. No model weights,
tokenizer, native binary or large dependencies were downloaded. Free D: disk was 123,987,197,952
bytes; target storage/RAM remain unknown. No original PS5 document was found.

Pinned public candidate: meta-llama/Llama-3.2-1B-Instruct revision
9213176726f574b556790deb65791e0c5aa438b6 and ExecuTorch source
985c1ceccc8bb8b6a32294f71aeb5d2f299562f7 (version.txt 1.5.1). Upstream installer specifies
torch 2.14.0 / torchao 0.18.0 and a torchtune commit. These are source-derived candidate pins,
not an installed/qualified environment. Public LFS model/tokenizer hashes are recorded as
published metadata, not local artifact verification. Full Linux pip freeze is still pending.

The static CLI audit found no undefined adapter flags. It verified prompt echo and observer
statistics framing in the locked source. Two configuration mismatches were identified: the
branch now reports 1.5.1, and the tutorial's SpinQuant recipe requires prequantized weights
with 2048 context. Adapter now accepts matching 1.5.1 versions; the separate original-checkpoint
PTQ candidate explicitly exports 8192 context with 512-token prefill chunks. PTQ plan reliability
and actual graph loading are unverified. Production HTTP runtime stays Linux ARM64-only; only
the qualification tools can explicitly enable a matching Linux x86_64 native runner.

New implementation:
- deployment/phase5d/: immutable source/model lock, upstream core package constraints and
  reviewed original-checkpoint XNNPACK PTQ config.
- scripts/phase5d_toolchain.py: guarded external source/artifact workspace, pinned checkout,
  isolated Linux bootstrap, gated authorized download, offline export, native/cross CPU build,
  verified export staging, artifact/provenance/build receipts. Does not overwrite outputs.
- scripts/phase5d_bundle.py: model-free reproducible ZIP kit; actual artifact package requiring
  matching receipts/hashes; checksum-verified absolute manifest relocation. Original checkpoint
  and credentials are excluded. Real artifact bundle cannot yet be produced.
- scripts/phase5d_qualify.py: actual native process generation/plan qualification, network
  namespace observation, output hashes/raw evidence, wall time, sampled Linux RSS and thermal
  readings, optional one-second timeout probe. Missing artifacts produce explicit blocked reports.
- scripts/audit_executorch_contract.py: repeatable source/flag/hash/framing audit.
- steelguard_app/test_phase5d_qualification.py: portable synthetic-fixture/process tests, not Llama.
- docs/PHASE5D_QUALIFICATION.md: candidate identity, audit, source hashes, access constraints,
  RAM/storage estimates, export/build/transfer/offline commands, stability and recovery plan.

Updated executorch_runner.py adds backward-compatible explicit version support, a separate
host-qualification opt-in and optional process timing/sample-RSS observations. Existing APIs,
dashboard, quality policy, database/config and checkpoints were preserved. Regression runner
now includes the Phase 5D suite and reports under runs/phase5d_verification/.

All feasible host-side verification and evidence are recorded there and under
runs/phase5d_qualification/. Source snapshots are public upstream source plus its LICENSE,
not weights. Remaining execution: finish WSL/Linux + local licensed access, then invoke the
documented checkout/bootstrap/download/export/build steps, generate a real versioned artifact
bundle and run native host/Pi qualification. Do not call the portable ZIP a model deployment or
claim latency/RSS/thermal/PS5 results from synthetic test subprocesses.

## Phase 5D execution on WSL on 2026-11-16

The Ubuntu 24.04 WSL workspace completed the authorized download, original-checkpoint INT4
PTQ export and real offline generation with the pinned ExecuTorch 1.5.1 native CPU runner.
The 806,179,840-byte self-contained PTE and the exact original tokenizer share the checksums in
the export and package receipts. The first JSON prompt produced tokens but failed strict plan
validation. The bounded two-token action selection passed three runs of the same inspection
fixture and the one-second timeout test. The adapter carries the deterministic verdict, IDs and
warnings into the plan. The model selects the recommended action from the offered choices.

The validated WSL runs had median end-to-end wall time 2.41 s, median model load 1.543 s,
median first token after load 0.729 s and highest observed runner VmHWM 1,348,680 KiB. Longer
decode probes on the same PTE and runner measured 32.37–36.58 tokens/s. See the current
`runs/phase5d_qualification/qualification_report.json` for scope and raw evidence references.

The Pi 5 ARM64 CPU bundle is `/home/sai01/steelguard-pi5-bundle-20261116-v2`, built with
XNNPACK and KleidiAI and checked on WSL for hashes, ELF architecture, and manifest structure.
The runbook documents the measured ARM64 loader and symbol requirements and upload/relocation
steps. Physical Pi generation, memory, camera and vision coexistence, and PS5 acceptance are
outstanding. See `docs/PHASE5D_QUALIFICATION.md` for target commands.

## Phase 5 Laptop Hackathon-Ready Integration & Verification on 2026-11-17

### 1. Architectural Integration
- **Safe Local WSL ExecuTorch Bridge:** Connected the real ExecuTorch 1.5.1 CPU runtime (`steelguard_llama32_1b_int4.pte`, SHA256 `d7d509a4...`) hosted in WSL2 Ubuntu 24.04 (`/home/sai01/steelguard-host-bundle-20261116/`) to the Windows dashboard via `AdvisoryService` and `scripts/wsl_advisory_worker.py`.
- **Zero Cloud / Zero Unauthorized Substitutions:** Fully compliant with offline constraints. Ollama, llama.cpp, and external cloud APIs are strictly excluded.
- **Fail-Safe Fallback Contract:** If WSL is unavailable, manifest is missing, or execution exceeds 5.0 seconds, the system seamlessly returns `RULE_BASED_FALLBACK` with reason code `NOT_CONFIGURED` or `TIMEOUT`, preserving the immutable deterministic quality verdict.
- **Dual-Model Vision Taxonomy Isolation:**
  - YOLOv8n (GC10-DET + NEU Inclusion) detects bounding boxes and triggers deterministic GC10 QR rules.
  - Severstal U-Net performs pixel-level surface defect segmentation and logs to `segmentation_inspections`.
  - The models remain strictly independent; U-Net evidence is never fed into YOLO or the deterministic QR rule engine.

### 2. Operator UI Enhancements & Offline Demo Mode
- **1-Click Offline Demo Presets:** Added quick-action presets to `#view-inspect` (`Punching Hole [REJECT]`, `Inclusion [REJECT]`, `Minor Rolled Scale [PASS]`, `Clean Steel [PASS]`) allowing instant single-click demo execution without manual file hunting.
- **ExecuTorch Local Advisory Card:** Implemented dedicated UI component in `#view-assistant` displaying runtime telemetry (`EXECUTORCH`, `meta-llama/Llama-3.2-1B-Instruct`), bounded action plan (`Operator Action Plan`), rule grounding citations, and anti-hallucination disclaimers.
- **Live System Telemetry:** Topbar indicators accurately reflect YOLO readiness (`v2.0 READY`) and local ExecuTorch LLM status (`READY` when manifest configured, `NOT_DETECTED` when offline fallback).
- **Dashboard Launcher Options:** `steelguard_app/run_dashboard.py` now supports `--demo` and `--enable-wsl-advisory` flags, printing a clean startup diagnostic banner.

### 3. Verification & Evidence
- **Automated Master Regression Runner (`scripts/run_dashboard_regressions.py`):**
  - All **14 isolated regression test suites** passed with Exit Code 0:
    1. `intelligence`: PASS
    2. `intelligence_api`: PASS
    3. `advisory`: PASS
    4. `advisory_api`: PASS
    5. `phase5d_tools`: PASS
    6. `stage4`: PASS (nested Stage 3/2)
    7. `phase3`: PASS
    8. `phase4c`: PASS
    9. `segmentation_pipeline`: PASS
    10. `segmentation_training`: PASS
    11. `resources`: PASS (CPU fallback / OOM)
    12. `traceability`: PASS
    13. `startup`: PASS (`server.run(host, port)` signature & port binding verification)
    14. `phase5_e2e`: PASS (full YOLO + WSL ExecuTorch + U-Net + SQLite end-to-end)
  - Zero state pollution: `steelguard.db` and `model_config.json` SHA-256 hashes remained byte-for-byte identical before and after the test run.
- **Real Headless Microsoft Edge Demonstration (`steelguard_app/test_browser_demo_e2e.py`):**
  - Fully automated headless browser test exercised the complete inspection workflow: Sample load -> Real YOLO inference -> Deterministic summary generation -> Real WSL ExecuTorch advisory generation -> SQLite persistence.
  - High-resolution visual proof captured at `runs/phase5_demo_verification/phase5_e2e_demo.png` (137.6 KB).
  - DOM snapshot and execution log archived under `runs/phase5_demo_verification/`.

### 4. Target Pi 5 4GB Qualification Runbook
- Authored `docs/PI5_4GB_DEPLOYMENT_GUIDE.md` detailing:
  - Exact target hardware profile (Raspberry Pi 5 4GB RAM, BCM2712 AArch64).
  - Strict resource budget (~2,210 MB peak memory, leaving 1,886 MB / 46.0% available RAM headroom).
  - Bundle transfer, dynamic linker verification (`ldd`), manifest relocation (`phase5d_bundle.py relocate`), and unshared network namespace offline qualification commands.
  - Transparent statement that physical Raspberry Pi 5 hardware testing remains pending until the board is received.

### 5. Quickstart: Running the Offline Hackathon Demo
To launch the complete demo on the laptop:
```powershell
# Open PowerShell in workspace root:
cd D:\INDUSTRIAL-QC-AI
.\.venv\Scripts\python.exe steelguard_app/run_dashboard.py --demo
```
Open browser at: `http://127.0.0.1:8080/`

**Demo Presentation Script:**
1. **Overview Dashboard:** Point out system telemetry (Vision: READY, LLM: READY).
2. **Inspect Console:** Click `Punching Hole (Critical)` preset. Live YOLO detects `3_yueyawan` and `2_hanfeng`.
3. **Deterministic Quality Engine:** Show that the verdict is immediately evaluated as `QR-01 REJECT` based strictly on GC10 deterministic rules.
4. **Local ExecuTorch Advisory:** Click "Ask Assistant". Live ExecuTorch Llama 3.2 1B Instruct running locally on CPU generates a bounded `SCRAP_IMMEDIATE` disposition plan in ~2.4s.
5. **Traceability:** Click "Save Inspection". Switch to **History** tab to inspect the immutable SQLite audit record.
6. **Strip Segmentation:** Switch to **Strip Segmentation** tab to showcase pixel-level U-Net defect masks on Severstal strip samples.

### 6. Prioritized Remaining Hackathon Requirements
| Priority | Task Description | Dependencies / Prerequisites |
| :---: | :--- | :--- |
| **P1** | Acquire & Unbox physical Raspberry Pi 5 (4GB RAM) with Active Cooler | Physical hardware delivery |
| **P2** | Flash Raspberry Pi OS 64-bit (Debian 13 Trixie) & verify GLIBC 2.38+ | Pi 5 board |
| **P3** | Transfer ARM64 bundle (`steelguard-pi5-bundle-20261116-v2`) & relocate manifest | Local Wi-Fi / Ethernet |
| **P4** | Execute `phase5d_qualify.py` under `sudo unshare --net` on Pi 5 | Pi 5 terminal access |
| **P5** | Connect physical Pi Camera / CSI ribbon cable & implement live capture worker | Physical camera module |
| **P6** | Obtain and audit official PS5 competition specification document | Hackathon organizers |
