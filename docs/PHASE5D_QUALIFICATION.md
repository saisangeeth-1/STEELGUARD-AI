# Phase 5D qualification report and reproducible runbook

## Completed WSL qualification on 2026-11-16

Real, network-isolated Llama 3.2 1B Instruct CPU generation passed on Ubuntu 24.04 x86-64
with ExecuTorch `1.5.1+985c1ce`. Three runs of the same synthetic inspection fixture passed
the adapter's action and verdict validation. A separate one-second timeout trial returned
`TIMEOUT`. Physical Pi generation and deployment acceptance are pending.

| Host measurement | Observed result |
| --- | --- |
| Self-contained INT4 `.pte` | 806,179,840 bytes (768.8 MiB) |
| SHA-256 | `d7d509a435d3ece32bbe349b5042d2198cd9ea53cc73f5c28c024fd90fbac5f3` |
| Model load time | 1.515–1.985 seconds, median 1.543 seconds |
| First token after model load | 0.716–0.739 seconds, median 0.729 seconds |
| End-to-end wall time per bounded selection | 2.39–2.86 seconds, median 2.41 seconds |
| Highest observed kernel runner VmHWM | 1,348,680 KiB (1.29 GiB) |
| Sustained decode probe | 32.37–36.58 tokens/sec, median 33.01 across three longer runs |

Each bounded selection generated two tokens. The sustained decode probe generated 256 tokens
per run including the first token, using the same PTE and native runner. Its generated JSON
failed application validation. The shorter validated protocol lets the model select one code
from approved choices and assembles the plan with deterministic verdict and evidence IDs.
Observed runner VmHWM excludes other processes and can miss a later peak. The WSL page cache
was not flushed between runs.

Evidence under `runs/phase5d_qualification/` includes `wsl-action-qualification-20261116.json`,
`wsl-offline-verified-20261116.json` and the WSL download, export, environment, and build receipts.
The first namespace probe was rejected because the inherited WSL sysfs mount showed the parent
interfaces. The live qualification uses the active namespace's kernel interface list and
requires only `lo` before launching the runner.

Ubuntu 24.04 on WSL2 ran as `sai01` with Python 3.12.3 in
`/home/sai01/steelguard-build/.venv`. The guest reported 7.4 GiB RAM and 2 GiB swap.
Its virtual filesystem had 954 GiB available and the Windows system drive had 102 GiB free.
CMake 3.28.3, GCC 13.3, Torch 2.14.0+cpu and TorchAO 0.18.0 were verified. The ARM64 GCC
cross compiler linked an AArch64 probe. An unprivileged user and network namespace exposed
only loopback in the probe.

Authorized download completed for `meta-llama/Llama-3.2-1B-Instruct` at the locked revision.
The original checkpoint and tokenizer matched both upstream SHA-256 pins. The verified download
receipt is `runs/phase5d_qualification/wsl-download-20261116.json`. The downloaded model geometry
implies 512 MiB for an FP32 KV cache at context length 8192. This is cache-size arithmetic.

The pinned Release 1.5.1 checkout now fetches CPU submodules directly, including their recursive
dependencies at the upstream gitlinks. `--resume-checkout` validates the existing superproject
before completing its CPU submodules. Bootstrap uses the upstream `--minimal` option, explicit
CPU/XNNPACK wheel build flags, and an exporter import check. The original PTQ path passed with
the pinned wheel dependencies. The export passes a filename stem because the pinned exporter treats
names ending in `.pte` as complete paths. `deployment/phase5d/pi5-ubuntu-noble-aarch64.cmake`
selects the Ubuntu cross compiler and Pi 5 CPU flags. Target loader compatibility still needs the
finished runner’s symbol requirements and the target’s installed libraries.

The cross-built Pi 5 package is `/home/sai01/steelguard-pi5-bundle-20261116-v2` in Ubuntu.
It includes the same verified PTE and tokenizer plus `llama_main` and `libllama_runner.so`
for AArch64, notices, receipts, manifests and checksums. The ARM64 executables require the
`/lib/ld-linux-aarch64.so.1` loader and symbols through GLIBC 2.38 and GLIBCXX 3.4.32.
Use Raspberry Pi OS 64-bit based on Debian Trixie and verify library resolution with
`ldd` on the Pi before enabling the adapter. [Raspberry Pi OS releases](https://www.raspberrypi.com/software/operating-systems/)

Transfer the bundle to a fresh directory on the Pi. Exclude the WSL-generated absolute-path
`advisory.manifest.json` and regenerate it from the checked template on the Pi:

```bash
# WSL, with PI_HOST set to the user's own Pi SSH destination.
rsync -a --exclude=advisory.manifest.json /home/sai01/steelguard-pi5-bundle-20261116-v2/ "$PI_HOST:steelguard-pi5-bundle/"
# Pi, in the transferred directory.
cd ~/steelguard-pi5-bundle
python3 scripts/phase5d_bundle.py relocate --destination "$PWD"
LD_LIBRARY_PATH="$PWD/artifacts" ldd artifacts/llama_main
sudo unshare --net python3 scripts/phase5d_qualify.py --manifest "$PWD/advisory.manifest.json" --output pi5-offline.json --runs 3 --require-network-isolation --probe-timeout
```

### Antigravity handoff for Pi 5 with 4 GB RAM

The WSL closeout on 2026-11-17 rehashed all 26 indexed files and checked the ARM64 manifest.
The ARM64 package is 820,393,307 bytes and includes the model, tokenizer, runner, shared
library, adapter sources, qualification scripts, manifests, receipts and license notices.
The WSL-host and ARM64 bundles have identical PTE and tokenizer hashes.

| Artifact | WSL path | SHA-256 |
| --- | --- | --- |
| Exported PTE | `/home/sai01/steelguard-build/export/steelguard_llama32_1b_int4.pte` | `d7d509a435d3ece32bbe349b5042d2198cd9ea53cc73f5c28c024fd90fbac5f3` |
| Original tokenizer | `/home/sai01/steelguard-build/model-source/original/tokenizer.model` | `82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55` |
| x86-64 runner | `/home/sai01/steelguard-host-bundle-20261116/artifacts/llama_main` | `57fe5d8e9dc067c1f98ef13ddbe69814080176c48bf4138e8655eeaa06a44ecb` |
| x86-64 library | `/home/sai01/steelguard-host-bundle-20261116/artifacts/libllama_runner.so` | `e81c9c972a21a12eca54388861e034bf1591f18864c95fadd48b755dca670705` |
| ARM64 runner | `/home/sai01/steelguard-pi5-bundle-20261116-v2/artifacts/llama_main` | `a742c0f02a502035087fb64b8bb9d7594047938ba91bac9d3ee323f87b069c84` |
| ARM64 library | `/home/sai01/steelguard-pi5-bundle-20261116-v2/artifacts/libllama_runner.so` | `406d97812d48fa9df88064393f999aff20ffe22e46d36793ea8140ca5aa13929` |

The same closeout exercised `AdvisoryService` with the real x86-64 runner through a host-only
configuration in a loopback-only network namespace. The normal call returned `EXECUTORCH` and
`llm_used: true`. With a one-second timeout the same call returned `RULE_BASED_FALLBACK`,
`TIMEOUT`, and `llm_used: false`. Both kept the fixture's `REJECT` verdict. The complete
responses are in `runs/phase5d_qualification/wsl-closeout-20261117.json`.

1. Record the physical Pi model, its 4 GB RAM variant, `uname -m`, `/etc/os-release`,
   `free -h`, and storage headroom. Use `aarch64`, check the loader and all dependencies with
   `ldd`, and confirm GLIBC 2.38 and GLIBCXX 3.4.32 symbols are available. Transfer the existing
   bundle with the commands above and validate the relocated manifest before running inference.
2. Run the three isolated trials above and retain `pi5-offline.json`. Require verified generation,
   accepted actions, the deterministic verdict, and the expected timeout probe. Record model
   load, first token, wall time, and observed VmHWM on the Pi. The 1.29 GiB WSL observation
   does not establish memory headroom on a 4 GB Pi. Collect system `MemAvailable` and swap use
   alongside the runner measurements with `free -h` and `vmstat 1`.
3. Test advice on real Pi inspection summaries with `STEELGUARD_EXECUTORCH_MANIFEST` set to
   the relocated manifest when starting `python3 steelguard_app/run_dashboard.py` from the
   installed project. Submit the verified summary as `{"summary": ...}` to
   `POST /api/intelligence/advisory`. Compare its verdict, IDs, and measured facts with the
   original summary. Record behavior for timeout and unavailable runtime in a staging launch.
4. Repeat with the Pi camera and both vision services under the intended workload. Measure
   available RAM, swapping, OOM events, inspection latency, temperature and throttling for at
   least 30 minutes. Check restart, offline operation, and SQLite history. The 4 GB memory
   budget, Pi throughput, CPU thermals, camera, and service coexistence remain unverified.
5. If the system swaps persistently, runs out of memory, or delays inspections, unset the
   advisory manifest and restart to retain rule-based advice. Use the observed failure to decide
   whether a smaller-context export is needed. Record the target acceptance results separately
   from the WSL report and leave PS5 acceptance pending the official specification.

The entries below document the earlier preparation and include the historical blocker state.

Date: 2026-11-15. State: **ACCESS/HOST BLOCKED; HOST-SIDE PREPARATION IMPLEMENTED**.
This report is not evidence of real Llama inference. No checkpoint/PTE/tokenizer/native runner
has been downloaded, exported, built or run in this session. No Pi is connected or identified.

## Verified blockers and available resources

- Windows 11 AMD64; dedicated Python 3.12.10 CUDA environment preserved.
- WSL executable exists, but Windows reports WSL is not installed. No Docker, CMake, Ninja,
  HF CLI or ExecuTorch/export Python packages were found on PATH/in the dedicated environment.
- No existing HF environment/local-cache credential was found. Official model auth-check
  returned HTTP 401. The model API identifies access as manually gated.
- Free disk at inspection: C: 109,323,726,848 bytes; D: 123,987,197,952 bytes. These are host
  disk observations, not target capacity or export-memory measurements.
- Windows reports 16,334,508,032 bytes physical RAM. WSL's allocation and export peak RSS
  are unmeasured; export/build memory pressure must be assessed before acquiring more artifacts.
- User elected to set up WSL and authenticate locally. No token was requested in chat or logged.
  Model access/license acceptance and a working Linux host remain the specific pending actions.
- Original PS5 document remains absent; neither adapter tests nor this preparation prove PS5 compliance.

## Immutable qualification candidate

| Component | Exact identity / status |
| --- | --- |
| Model | meta-llama/Llama-3.2-1B-Instruct |
| Model revision | 9213176726f574b556790deb65791e0c5aa438b6 |
| ExecuTorch source | 985c1ceccc8bb8b6a32294f71aeb5d2f299562f7 |
| Source version.txt | 1.5.1, from current release/1.5 branch |
| Upstream installer requirements | torch 2.14.0; torchao 0.18.0 |
| Upstream torchtune pin | 3c1872ace149f03ef4ffec765d3f0a5fd0399497 |
| Python export environment | isolated 3.12; actual Linux patch/build not yet available |
| Quantization candidate | original checkpoint, 8da4w PTQ group 128; embedding 4-bit group 32 |
| Backend / context | CPU XNNPACK; prefill chunks 512, total context 8192 |
| Fully resolved environment | NOT YET RESOLVED/INSTALLED; bootstrap records pip freeze |
| Live compatibility | UNVERIFIED; these are source-derived candidate pins, not a tested installation |

Machine-readable pins and reviewed config are in deployment/phase5d/. The source commit pins
its submodules as well. Core constraints prevent accidental replacement of the two upstream
package requirements. Other transitive packages are resolved in the isolated Linux install and
must be recorded/frozen before claiming fully reproducible, compatible export tooling.

The [official Pi workflow](https://docs.pytorch.org/executorch/stable/raspberry_pi_llama_tutorial.html)
is the deployment reference. The immutable
[model README](https://github.com/pytorch/executorch/blob/985c1ceccc8bb8b6a32294f71aeb5d2f299562f7/examples/models/llama/README.md)
distinguishes original and prequantized checkpoints; the pinned configs/schema and CMake presets
were inspected. Public source snapshots, URLs and SHA256 values are retained under
runs/phase5d_qualification/references/. Those snapshots are upstream BSD-licensed source,
not model weights; LICENSE is retained alongside them.

## Adapter contract audit and corrections

The actual adapter invokes a native llama_main executable with separate argv entries for
model_path, tokenizer_path, max_new_tokens, temperature, num_bos, num_eos, cpu_threads, warmup,
ignore_eos and prompt. It uses greedy generation, closed stdin and no shell. The source audit
compares every flag with DEFINE declarations in the locked main.cpp. No undefined flag was found.

The runner echoes the input by default, emits generated token text and then a PyTorchObserver
JSON statistics record. The parser removes only the exact prompt echo, checks counts and validates
the constrained JSON plan. This framing is source-verified, not yet observed from a real model.
Generation statistics exclude the separately emitted prefill token in this source; the adapter's
context check reserves one extra token.

Corrections/important mismatches:

1. The moving release/1.5 reference currently contains version 1.5.1. The adapter now accepts
   equal 1.5.1 runtime/export version pairs as well as its existing equal 1.5 pairs. The new lock
   and generated manifests use the exact 1.5.1 source commit above.
2. The tutorial's SpinQuant preset expects prequantized checkpoint tensors and uses 2048-token
   context. It must not be blindly applied to an original BF16 checkpoint or labeled as an 8192
   context export. This candidate uses a separate source-reviewed PTQ recipe instead. PTQ can
   lose model quality; advisory format/behavior must be evaluated, with fallback on failure.
3. Production remains Linux ARM64-only. Linux x86_64 manifests/runners are accepted only by
   explicit qualification-tool options, not by the HTTP advisory service. This permits real
   WSL CPU qualification later without claiming Pi execution or weakening production target checks.
4. The prompt's BOS/EOS/header delimiters match the public model metadata. The application
   supplies its own fixed system instructions, omitting the model card template's date boilerplate
   and tools. Exact tokenization/plan reliability still require the authorized tokenizer and model.

Audit command:

```powershell
.\.venv\Scripts\python.exe scripts/audit_executorch_contract.py --references runs/phase5d_qualification/references --output NEW_AUDIT.json
```

## Model access, source artifacts and expected footprint

Review/accept the [Meta model terms and access requirements](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
and authenticate locally on the Linux build host. Preserve LICENSE.txt and USE_POLICY.md.
Do not paste tokens, commit restricted files, use an alternate redistributed model to evade
the gate, or infer a licensing grant from this repository.

The public immutable repository metadata reports these source hashes/sizes. They are **not
locally verified download hashes**; download tooling compares authorized files to these pins.

| File | Bytes | Published LFS SHA256 |
| --- | ---: | --- |
| original/consolidated.00.pth | 2471677246 | fc17d497df5e4175b3a8acb4f5865b26f7fc1b009b25bef814b95fde10e8a1f3 |
| original/tokenizer.model | 2183982 | 82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55 |
| original/params.json | 220 | SHA256 pending authorized download |

Public parameter count is 1,235,814,400. Uniform BF16 payload arithmetic is about 2.30 GiB;
uniform four-bit arithmetic is about 0.575 GiB, **not** a prediction of this mixed-precision PTE
or total RSS. Headers, nonquantized tensors, quantization scales, KV cache and activations add
memory. The actual PTE size, model RAM, latency and JSON success rate are all unmeasured.

Compute FP32 KV cache planning bytes from the downloaded params as
2 × layers × KV heads × head dimension × context × 4. Do not assume values from a different
model. The export receipt records actual params, and the runtime must validate the declared
context. Start qualification with the 1B model and conservative thread count; do not promise
fit on a specific Pi RAM variant. Export/build scripts require a 30 GiB free-disk planning reserve,
which is not a guaranteed bound on source/build/venv growth. Measure host RSS before scaling.

## WSL/Linux preparation and export

Finish WSL/Ubuntu installation through Windows; this can require administrator changes/reboot
and is left to the user. On Linux install the build prerequisites from the official guide
(compiler/binutils, Git, CMake 3.26 through 3.x, Python 3.12 venv tooling). Use the Linux filesystem
for build/source/artifacts, outside /mnt/d/INDUSTRIAL-QC-AI. The Windows CUDA venv is not reused.

Example Linux session, after prerequisites are available:

```bash
python3 -m venv "$HOME/steelguard-export-env"
source "$HOME/steelguard-export-env/bin/activate"
cd /mnt/d/INDUSTRIAL-QC-AI
python scripts/phase5d_toolchain.py checkout --workspace "$HOME/steelguard-llama-native"
python scripts/phase5d_toolchain.py bootstrap --workspace "$HOME/steelguard-llama-native"
hf auth login
python scripts/phase5d_toolchain.py download --workspace "$HOME/steelguard-llama-native" --license-accepted
python scripts/phase5d_toolchain.py export --workspace "$HOME/steelguard-llama-native"
python scripts/phase5d_toolchain.py build --workspace "$HOME/steelguard-llama-native"
python scripts/phase5d_bundle.py package --workspace "$HOME/steelguard-llama-native" --destination "$HOME/steelguard-host-bundle-v1"
python scripts/phase5d_bundle.py relocate --destination "$HOME/steelguard-host-bundle-v1"
```

These Linux steps are implemented but **not executed** on this Windows host. Checkout is detached
at the lock commit; source dirtiness/version mismatches fail. Bootstrap operates only in a
separate Python 3.12 venv. Download requests only listed original checkpoint/tokenizer/config
and notice files at the locked revision. No random-weight export is allowed: explicit nonempty
checkpoint, params and tokenizer files plus the download receipt are mandatory. Output and
receipt files are exclusive-created; choose a new version/workspace rather than overwriting.
Export runs with HF offline flags and CUDA hidden, after all source artifacts have been acquired.

The export config keys and Hydra overrides come from the pinned schema. The helper checks for
exactly one self-contained .pte and rejects external .ptd files. Build uses source CMake CPU
presets; source commit, compiler/CMake details, environment, recipe and resulting hashes must
be retained in external receipts. A source-only review cannot prove these commands finish on
a clean Linux system; any build failure must be resolved and recorded before qualification.

## Real native host and Pi qualification

The model-free qualification kit contains only our adapter/scripts/docs/pins. It contains no
weights, tokenizer, compiled runner or credentials. The artifact-package command requires real
export/build receipts and verifies their artifact hashes and locked source identity; it does
not accept an existing directory as a destination. It copies only the exported model, matching
tokenizer, runtime files, notices and receipts, never the original training-format checkpoint.

After relocation, qualify the native Linux bundle first. For stronger offline evidence, create
a fresh network namespace (requires the appropriate Linux privilege) containing only loopback:

```bash
sudo unshare --net "$HOME/steelguard-export-env/bin/python" scripts/phase5d_qualify.py --manifest "$HOME/steelguard-host-bundle-v1/advisory.manifest.json" --output host-offline-v1.json --runs 3 --require-network-isolation --probe-timeout
```

No network namespace was created here. Without that privilege, omit the require flag for a
preliminary local run, and report offline isolation unverified. HF offline flags alone are not
proof of network denial. The qualifier records network interfaces, actual stdout/statistics,
output hashes, plan acceptance, process wall time and sampled Linux child VmRSS. It distinguishes
successful token generation from a valid adapter plan. A timeout probe attempts a real run
with one-second budget and reports the observed result, without assuming it must time out.

## ARM64 deployment bundle

Use a separate checkout/workspace for cross and native builds. Either build natively on Pi
using the same source CPU presets, or follow the official cross-compilation workflow with a
reviewed AArch64 toolchain and a sysroot matching the Pi's GLIBC/C++ libraries. Do not fix
ABI errors by mixing Debian unstable libraries into an existing Pi deployment.

The build helper accepts --cross-toolchain ABSOLUTE_CMAKE_TOOLCHAIN_FILE; it records the toolchain
hash and target. The official script documents architecture flags for Pi 5, but its downloadable
toolchain/environment is not silently installed here. Export can occur on x86 Linux; transfer
the exported model/tokenizer and their receipts unchanged to the ARM workspace, then build the
matching ARM64 runner and package a fresh version. Keep each receipt's paths valid in that
workspace with the stage-export command; no original weights need to go to the Pi:

```bash
python scripts/phase5d_toolchain.py checkout --workspace "$HOME/steelguard-llama-arm"
python scripts/phase5d_toolchain.py stage-export --workspace "$HOME/steelguard-llama-arm" --from-workspace "$HOME/steelguard-llama-native"
python scripts/phase5d_toolchain.py build --workspace "$HOME/steelguard-llama-arm" --cross-toolchain /absolute/path/to/reviewed-pi-sysroot-toolchain.cmake
python scripts/phase5d_bundle.py package --workspace "$HOME/steelguard-llama-arm" --destination "$HOME/steelguard-pi-bundle-v1"
```

Staging copies only the PTE/tokenizer/notices and adjusts verified receipt paths; it does not
copy the original checkpoint. Build receipts verify ELF machine and actual CPU/XNNPACK CMake
cache flags, and collect build-local NEEDED libraries with their required loader filenames.
System NEEDED libraries are recorded for the target ABI check rather than copied from the host.

Transfer the versioned ARM64 bundle directory with scp/rsync over the operator's chosen secure
transport. Run relocation on Pi, which checks checksums, native ELF/tokenizer structure and
absolute paths before publishing a manifest. On target verify runtime library resolution with
ldd on the trusted runner, ensure executable permissions, then run qualification with the same
offline command (adjust paths). The qualifier detects /proc/device-tree/model; a Linux ARM64
host alone is not labeled as Pi 5. Checksums are integrity evidence, not publisher authentication.

Set STEELGUARD_EXECUTORCH_MANIFEST to the ARM64 manifest only after real plan acceptance. Start
the existing dashboard as documented in the deployment guide; test an actual summary through
the advisory API. No server migration or production configuration edit is needed. Unset the
environment variable and restart to return to deterministic-only operation.

## Thermal, memory, stability and recovery acceptance

Record Pi OS/version, RAM, cooling, power supply and storage. Run cold-start trials and repeated
separate-process trials at one and two CPU threads, then alongside vision. Use --runs up to 20
per report and separate report filenames for additional batches. Thermal sysfs before/after,
token counts and wall/sample-RSS measurements are emitted only when actually available. Sampled
VmRSS may miss transient peaks and excludes other processes; also record OS/cgroup peak memory.

Exercise at least a sustained 30-minute workload, camera activity, network disconnected,
runtime termination, storage exhaustion and restart. Monitor throttling/frequency and temperature
using Pi OS tooling; record real observations and acceptance limits agreed for this project.
Do not convert development GPU latency or upstream phone measurements into claimed Pi results.

If the model cannot fit or coexist safely, disable advisory and keep deterministic vision,
reduce context with a **new export and matching manifest limits**, reduce prompt/output budgets,
and confirm 1B is selected. Maintain a tested memory reserve for vision; avoid uncontrolled swap
thrashing. Use OS memory/network restrictions and measured budgets before production enablement.
The current single-threaded HTTP server can be occupied for up to the configured advisory timeout.

## Windows regression closeout on 2026-11-17

All 12 isolated suites passed, including 31 advisory tests, six advisory API tests and 13 Phase
5D tooling tests. The report records matching before-and-after checksums for the three vision
checkpoints, production SQLite database and model configuration.
[Regression report](../runs/phase5d_verification/closeout-20261117/report.json)

The machine-readable qualification report indexes the WSL measurements, ARM64 package checks
and remaining physical acceptance work.
[Qualification evidence](../runs/phase5d_qualification/qualification_report.json)

## Earlier Windows verification

- 13 Phase 5D tooling tests passed, including version/architecture separation, artifact/path
  guards, source-contract audit, deterministic model-free packaging and receipt staging.
- All 12 regression suites passed. Existing advisory coverage remains 30 adapter tests and
  6 API tests; nested Stage 4/3/2 checks are not counted again as independent unit tests.
- All 15 frontend scenarios and the real Edge/YOLO/deterministic-summary dashboard smoke passed.
  That browser test is vision/dashboard evidence, not ExecuTorch inference.
- Changed Python files compiled. Production history/configuration and all three vision
  checkpoint SHA256 values were unchanged after testing.
- A deliberately missing-model qualification invocation wrote blocked_inference.json with
  no run records and every real-inference/Pi flag false. The ZIP kit has no model artifacts.

Evidence directories: runs/phase5d_qualification/ and runs/phase5d_verification/. The portable
kit is tooling, not a ready-to-run model deployment. No simulated timings are reported as Llama
latency, RSS or thermal measurements.
