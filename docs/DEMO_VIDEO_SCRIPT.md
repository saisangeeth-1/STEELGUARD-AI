# STEELGUARD AI — Official Hackathon Demonstration Video Script & Shot List

**Bharat AI-SoC Hackathon | Problem Statement 5: Real-Time Industrial Defect Detection & Quality-Control Assistant**  
**Target Video Duration:** Approximately 5:00 minutes (300 seconds)  
**Resolution:** 1080p (1920x1080) at 30/60 FPS  
**Demonstration Target:** 100% Offline CPU-Only Execution  

---

## Shot List & Demonstration Timeline

```
+------------+-------------------------------------------------------------+----------+
| Timestamp  | Scene Title                                                 | Duration |
+------------+-------------------------------------------------------------+----------+
| 0:00 - 0:45| Scene 1: Introduction, Problem Statement & Architecture     | 0:45     |
| 0:45 - 1:45| Scene 2: Real YOLO Detection & Deterministic Quality Rules  | 1:00     |
| 1:45 - 2:40| Scene 3: Dual-Model U-Net Pixel-Level Boundary Segmentation | 0:55     |
| 2:40 - 3:45| Scene 4: ExecuTorch Llama 3.2 1B Advisory & Safe Fallback   | 1:05     |
| 3:45 - 4:30| Scene 5: Local SQLite Traceability & Audit Logging          | 0:45     |
| 4:30 - 5:00| Scene 6: Pi 5 4GB Budget, Disclosure & Conclusion           | 0:30     |
+------------+-------------------------------------------------------------+----------+
```

---

## Detailed Script & Visual Instructions

### Scene 1: Introduction, Problem Statement & Hardware Disclosure (0:00 – 0:45)

**Visuals On Screen:**
1. Title card: **STEELGUARD AI: Offline Edge-Native Defect Detection & Quality-Control Assistant**.
2. Split view: Left shows industrial steel strip rolling line; right shows edge SoC target diagram (Raspberry Pi 5, 4 GB RAM, ARM64 CPU).
3. Text overlay banner: *"Verified Development Host (CPU-Only) | Physical Raspberry Pi 5 Deployment Pending Hardware Delivery"*.

**Spoken Narration (Presenter):**
> *"Welcome to our demonstration of STEELGUARD AI, engineered for Problem Statement 5 of the Bharat AI-SoC Hackathon.*  
> *In high-speed steel strip rolling lines, millimetres of edge cracks or embedded inclusions cause catastrophic strip breakage, halting multi-million dollar rolling mills.*  
> *Traditional inspection systems require massive cloud servers or high-power GPUs. STEELGUARD AI is different: it is engineered to run 100% offline on a low-cost, 4 GB edge System-on-Chip like the Raspberry Pi 5.*  
> *Today, we are demonstrating live, authentic CPU-only inference on our development workstation, with CUDA explicitly disabled, while our pre-compiled ARM64 bundle awaits physical board arrival."*

---

### Scene 2: Real YOLO Detection & Deterministic Quality Rules (0:45 – 1:45)

**Visuals On Screen:**
1. Browser navigates to `http://127.0.0.1:8080/`. The dark industrial dashboard loads instantly.
2. Presenter selects Sample 1: `img_01_3402617700_01009.jpg` (Held-out GC10-DET test sample).
3. Clicks **Run AI Inference**.
4. Screen zooms into the viewer: Real-time bounding box renders on the crescent gap defect.
5. Latency badge displays: `~25 - 34 ms (CPU)`.
6. Inspection Intelligence panel highlights: Status turns bright red: **REJECT**, citing rule `QR-01: Critical Perforation / Edge Gap Defect`.
7. Presenter selects Sample 2: `img_02_4402623300_00034.jpg` (Inclusion Defect from our NEU-augmented candidate model). Model detects Inclusion with high confidence.

**Spoken Narration (Presenter):**
> *"Here is the STEELGUARD operator dashboard running locally over loopback. Let's load a held-out test sample containing a severe edge crescent gap.*  
> *Running inference on CPU takes just 34 milliseconds—that is roughly 30 frames per second on standard hardware.*  
> *Notice our core safety principle: the vision model detects bounding boxes, but the PASS/REJECT verdict is calculated by our Deterministic Quality Engine enforcing immutable metallurgical rules.*  
> *Because a crescent gap causes strip tears under mill rolling tension, Rule QR-01 triggers an immediate, uncompromised REJECT.*  
> *Next, we test an inclusion sample. Our fine-tuned YOLOv8 Nano model increased Inclusion AP@50 from 14.98% to 30.54%, successfully capturing foreign inclusions."*

---

### Scene 3: Dual-Model U-Net Pixel-Level Boundary Segmentation (1:45 – 2:40)

**Visuals On Screen:**
1. Switch to the **Segmentation Studio** tab in the dashboard.
2. Load Severstal test strip sample: `0007a71bf.jpg` (Native 1600 x 256 aspect ratio).
3. Click **Run U-Net Segmentation**.
4. Mask toggle overlay: Blue/cyan highlight displays the precise pixel contour of longitudinal surface scratches.
5. Telemetry panel shows: `Defect Coverage: 2.91%`, `Inference Latency: 218 ms (CPU)`.

**Spoken Narration (Presenter):**
> *"For complex surface anomalies, bounding boxes alone are insufficient. STEELGUARD AI features an independent, dual-model Lightweight U-Net trained on the Severstal dataset.*  
> *Here, on a native 1600-by-256 steel strip, the U-Net computes pixel-level defect contours in just 218 milliseconds on CPU.*  
> *It detects longitudinal surface scratches covering 2.91% of the surface.*  
> *Importantly, the Severstal taxonomy is kept completely isolated from the YOLO GC10 taxonomy, ensuring modularity and preventing interference with verified rejection rules."*

---

### Scene 4: ExecuTorch Llama 3.2 1B Advisory & Safe Fallback (2:40 – 3:45)

**Visuals On Screen:**
1. Return to the **Inspection Intelligence** panel.
2. In the advisory card, click **Request Llama Advisory**.
3. Status badge updates from `READY FOR QUERY` to `EXECUTORCH ACTIVE`.
4. In ~3.5 seconds, the action plan populates:
   - Primary Disposition: `Hold the workpiece for operator disposition under the configured rejection policy.`
   - Secondary Action: `Use a calibrated measurement procedure if physical dimensions are required.`
5. Presenter points out the guardrails: The verdict remains locked at **REJECT**.
6. Presenter loads a clean sample (`000789191.jpg`): The previous advisory is immediately invalidated and wiped from the DOM.
7. Presenter shows terminal with simulated timeout: Demonstrates immediate graceful fallback to **`RULE_BASED_FALLBACK`** in < 1 ms.

**Spoken Narration (Presenter):**
> *"Now let's examine the local AI assistant. We do not use cloud APIs, Ollama, or unconstrained chat.*  
> *When the operator clicks 'Request Llama Advisory', our backend communicates with a local INT4 quantized Llama 3.2 1B Instruct model running natively under ExecuTorch 1.5.1.*  
> *By quantizing the weights to 4-bit, we reduced model size from 2.5 GB to 768 megabytes, fitting comfortably into edge RAM.*  
> *Notice two critical safety features: first, the LLM cannot hallucinate or overturn the REJECT verdict; second, it chooses from pre-authorized metallurgical action codes.*  
> *If the model ever times out or crashes, our system instantly falls back to deterministic rule explanations in less than 1 millisecond, guaranteeing zero downtime on the production line."*

---

### Scene 5: Local SQLite Traceability & Compliance Logging (3:45 – 4:30)

**Visuals On Screen:**
1. Switch to the **Inspection History** tab.
2. Table shows live persisted rows: Inspection ID, Timestamp, Sample ID, Defect Count, Status (REJECT/PASS), Bounding Box Area, Latency.
3. Presenter clicks on a record: Modal opens showing detailed defect instances, coordinates, and triggered rules.
4. Presenter clicks **Export CSV** and **Export JSON**: Files download instantly.
5. Brief cut to terminal: Querying `sqlite3 steelguard_app/data/steelguard.db` confirms relational integrity and zero external dependencies.

**Spoken Narration (Presenter):**
> *"Every industrial inspection requires strict regulatory traceability.*  
> *All detections, bounding box coordinates, and triggered quality rules are logged transactionally into a local SQLite database.*  
> *Operators can filter by status, search historical coils, inspect triggered rules, and export full JSON and CSV audit trails for factory MES systems without any internet connection."*

---

### Scene 6: Target SoC Budget, Disclosure & Conclusion (4:30 – 5:00)

**Visuals On Screen:**
1. Slide showing **Raspberry Pi 5 4GB RAM Headroom Analysis**:
   - Total RAM: 4,096 MB
   - OS & HTTP Server: ~560 MB
   - Vision & Segmentation: ~300 MB
   - ExecuTorch Llama INT4: ~1,350 MB peak
   - **Guaranteed Free Headroom: 1,886 MB (46.0% Free)**.
2. Summary bullets:
   - 100% Offline Edge Operation.
   - Dual-Model Vision + Deterministic Rules + ExecuTorch INT4.
   - 15/15 Regression Suites Passing.
   - Complete GitHub-ready repository with one-click CPU launcher.
3. Closing title card with team credits.

**Spoken Narration (Presenter):**
> *"To ensure edge safety on the Raspberry Pi 5, our total combined peak memory is 2.2 gigabytes, leaving nearly 1.9 gigabytes—or 46%—of free RAM headroom to prevent any out-of-memory crashes.*  
> *In summary, STEELGUARD AI combines real-time computer vision, deterministic quality rules, and true local LLM reasoning into an open, auditable edge system.*  
> *Thank you to the Bharat AI-SoC Hackathon committee."*
"""
    (DOCS_DIR / "DEMO_VIDEO_SCRIPT.md").write_text(md, encoding="utf-8")
    print(f" -> Generated Demo Video Script: {DOCS_DIR / 'DEMO_VIDEO_SCRIPT.md'}")
    # Sync to github_release and release
    shutil.copy2(DOCS_DIR / "DEMO_VIDEO_SCRIPT.md", ROOT / "github_release" / "docs" / "DEMO_VIDEO_SCRIPT.md")
    shutil.copy2(DOCS_DIR / "DEMO_VIDEO_SCRIPT.md", ROOT / "release" / "steelguard_portable_cpu" / "docs" / "DEMO_VIDEO_SCRIPT.md")


if __name__ == "__main__":
    build_markdown_report()
    html_path = build_html_report("")
    export_pdf_report(html_path)
    # Also generate video script
