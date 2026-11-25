/**

 * STEELGUARD AI - Industrial Inspection Command Center (Stage 4)

 * Client Application Controller:

 * - Verified GC10-DET sample browser and Pascal VOC XML ground-truth rendering.

 * - Persistent SQLite inspection storage and traceability history.

 * - Formal Industrial Inspection Reports with PDF/Print, CSV, and JSON exports.

 * - Operator sign-off, notes persistence, and bi-directional canvas inspection viewer.

 */



// Application State

const state = {

  activePage: "overview",

  isDemoMode: false,

  systemHealth: null,

  currentImage: null,

  imageMeta: null,

  dataMode: "IDLE", // "GROUND_TRUTH" | "USER_UPLOAD" | "IDLE"

  currentSampleId: null,

  currentInspectionId: null,

  detections: [],

  selectedDetectionId: null,

  assessment: null,

  inspectionGeneration: 0,

  samplesCache: [],

  isSaving: false,

  activeReport: null

};



// Global interactive viewer instances

let steelViewer = null;

const intelligence = InspectionIntelligence.createController({ document, fetch: (...args) => fetch(...args) });



function invalidateAdvisory(reason) {
  state.lastAdvisoryResult = null;
  const advBtn = document.getElementById("btn-request-advisory");
  const advContent = document.getElementById("advisory-content");
  const advBadge = document.getElementById("advisory-badge");

  if (advBtn) {
    advBtn.disabled = !state.lastInferenceData;
    advBtn.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-right: 4px;"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg><span>Request Llama Advisory</span>`;
  }
  if (advBadge) {
    advBadge.textContent = state.lastInferenceData ? "READY FOR QUERY" : "AWAITING INFERENCE";
    advBadge.className = "badge-verdict pending";
  }
  if (advContent) {
    advContent.dataset.status = state.lastInferenceData ? "ready" : "empty";
    const defaultMsg = state.lastInferenceData 
      ? `YOLO inspection complete (${state.detections ? state.detections.length : 0} defect(s) detected). Click <strong>Request Llama Advisory</strong> to generate grounded operator actions.`
      : "Run a YOLO inspection first to generate the grounded ExecuTorch advisory plan.";
    const msg = reason || defaultMsg;
    advContent.innerHTML = `<p style="color: var(--text-muted);">${msg}</p>`;
  }
}

function invalidateIntelligence(message) {
  state.inspectionGeneration += 1;
  state.lastInferenceData = null;
  intelligence.reset(message);
  invalidateAdvisory(message || "Run a YOLO inspection first to generate the grounded ExecuTorch advisory plan.");
  return state.inspectionGeneration;
}



// DOM Content Loaded Initializer

document.addEventListener("DOMContentLoaded", () => {

  initNavigation();

  fetchSystemHealth();

  initOverviewPage();

  initInspectWorkspace();

  initSegmentationView();

  initHistoryPage();

  initResultsPage();

  initSettingsPage();

  intelligence.reset();

});



/* ==========================================================

   1. NAVIGATION & ROUTING

   ========================================================== */

function initNavigation() {

  const navItems = document.querySelectorAll(".nav-item");

  navItems.forEach(item => {

    item.addEventListener("click", (e) => {

      e.preventDefault();

      const page = item.getAttribute("data-page");

      navigateTo(page);

    });

  });



  // Support direct deep-linking via URL hash (e.g. #segmentation, #inspect)

  const hash = window.location.hash.replace("#", "").trim();

  if (hash) {

    navigateTo(hash);

  }

}



function navigateTo(pageId) {

  state.activePage = pageId;



  // Update sidebar active classes

  document.querySelectorAll(".nav-item").forEach(item => {

    if (item.getAttribute("data-page") === pageId) {

      item.classList.add("active");

    } else {

      item.classList.remove("active");

    }

  });



  // Update topbar title and subtitle

  const pageMeta = {

    overview: { title: "Overview", subtitle: "Inspection Command Center" },

    inspect: { title: "Inspect", subtitle: "Steel Inspection Workspace" },

    segmentation: { title: "Strip Segmentation", subtitle: "Severstal U-Net Pixel-Level Inspection (Experimental)" },

    results: { title: "Inspection Results", subtitle: "Inspection Reports" },

    history: { title: "History & Traceability", subtitle: "Inspection Records" },

    analytics: { title: "Quality Analytics", subtitle: "Quality Intelligence" },

    assistant: { title: "Inspection Intelligence", subtitle: "Deterministic GC10 Evidence" },

    health: { title: "Model & Edge Health", subtitle: "System Diagnostics" },

    settings: { title: "Settings", subtitle: "System Configuration" }

  };

  const meta = pageMeta[pageId] || { title: "Inspection Console", subtitle: "Operational System" };

  const titleElem = document.getElementById("topbar-page-name");

  const subtitleElem = document.getElementById("topbar-page-subtitle");

  if (titleElem) {

    titleElem.textContent = meta.title;

  }

  if (subtitleElem) {

    subtitleElem.textContent = meta.subtitle;

  }



  // Hide all views and show active

  document.querySelectorAll(".page-view").forEach(view => {

    view.classList.remove("active");

  });

  const activeView = document.getElementById(`view-${pageId}`);

  if (activeView) {

    activeView.classList.add("active");

  }



  // View specific activations

  if (pageId === "inspect" && steelViewer) {

    setTimeout(() => steelViewer.handleResize(), 50);

  } else if (pageId === "segmentation") {

    if (!segState.initialized) {

      initSegmentationView();

    } else {

      renderSegmentationCanvas();

    }

  } else if (pageId === "history") {

    loadHistoryData();

  } else if (pageId === "overview") {

    loadOverviewData(state.isDemoMode);

  } else if (pageId === "health") {

    loadHealthTelemetry();

  } else if (pageId === "settings") {

    loadSettingsData();

  }

}



/* ==========================================================

   2. SYSTEM HEALTH & TELEMETRY

   ========================================================== */

async function fetchSystemHealth() {

  try {

    const res = await fetch("/api/health");

    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const health = await res.json();

    state.systemHealth = health;



    // Truthful host architecture

    const archElem = document.getElementById("sidebar-host-arch");

    if (archElem) {

      archElem.textContent = health.host.architecture || "AMD64";

      if (health.host.is_arm) {

        archElem.style.color = "var(--cyan-primary)";

      }

    }



    // Vision model status

    const visionElem = document.getElementById("topbar-vision-status");

    if (visionElem) {

      const isReady = health.models.vision.status === "READY";

      const ver = health.models.vision.version || "ONLINE";

      visionElem.textContent = isReady ? ver : "NOT LOADED";

      if (isReady) {

        visionElem.parentElement.classList.remove("offline");

        visionElem.parentElement.classList.add("ready");

        visionElem.parentElement.title = `${health.models.vision.name} (${ver}) - ${health.models.vision.device_name || 'Active'}`;

      } else {

        visionElem.parentElement.classList.remove("ready");

        visionElem.parentElement.classList.add("offline");

      }

    }



    // Segmentation U-Net status

    const segElem = document.getElementById("topbar-seg-status");

    if (segElem && health.models && health.models.segmentation) {

      const seg = health.models.segmentation;

      const isSegReady = seg.status && seg.status.includes("READY");

      const segVer = seg.version || "U-NET v1.0";

      segElem.textContent = isSegReady ? segVer : "NOT LOADED";

      if (isSegReady) {

        segElem.parentElement.classList.remove("offline");

        segElem.parentElement.classList.add("ready");

        segElem.parentElement.title = `${seg.name} (${segVer}) - ${seg.device_name || 'Active'}`;

      } else {

        segElem.parentElement.classList.remove("ready");

        segElem.parentElement.classList.add("offline");

      }

    }



    // LLM status & ExecuTorch Advisory Badge

    const llamaElem = document.getElementById("topbar-llama-status");

    if (llamaElem && health.models && health.models.llm) {

      const llm = health.models.llm;

      const isLlmReady = llm.runtime_ready;

      llamaElem.textContent = isLlmReady ? "EXECUTORCH READY" : (llm.status || "NOT DETECTED");

      if (isLlmReady) {

        llamaElem.parentElement.classList.remove("warning", "offline");

        llamaElem.parentElement.classList.add("ready");

        llamaElem.parentElement.title = `${llm.name} - ${llm.runtime} (${llm.target || 'CPU'})`;

      }



      const advBadge = document.getElementById("advisory-badge");

      if (advBadge) {

        if (isLlmReady) {

          advBadge.textContent = "EXECUTORCH ACTIVE";

          advBadge.className = "badge-verdict pass";

        } else if (llm.checkpoint_path) {

          advBadge.textContent = "CONFIGURED (FALLBACK)";

          advBadge.className = "badge-verdict warning";

        } else {

          advBadge.textContent = "RULE-BASED FALLBACK";

          advBadge.className = "badge-verdict pending";

        }

      }

    }



    // Camera status

    const camElem = document.getElementById("topbar-camera-status");

    if (camElem) {

      camElem.textContent = health.camera.status;

    }

  } catch (err) {

    console.warn("Unable to fetch telemetry:", err);

  }

}



function loadHealthTelemetry() {

  const dumpElem = document.getElementById("health-telemetry-dump");

  if (!dumpElem) return;

  fetch("/api/health")

    .then(res => res.json())

    .then(data => {

      dumpElem.textContent = JSON.stringify(data, null, 2);

    })

    .catch(err => {

      dumpElem.textContent = `Error fetching telemetry: ${err.message}`;

    });

}



document.getElementById("btn-refresh-health")?.addEventListener("click", loadHealthTelemetry);



/* ==========================================================

   3. PAGE 1: OVERVIEW CONTROLLER

   ========================================================== */

function initOverviewPage() {

  const toggleBtn = document.getElementById("btn-overview-demo-toggle");

  const bannerToggle = document.getElementById("btn-toggle-demo");



  const toggleHandler = () => {

    state.isDemoMode = !state.isDemoMode;

    loadOverviewData(state.isDemoMode);

  };



  toggleBtn?.addEventListener("click", toggleHandler);

  bannerToggle?.addEventListener("click", toggleHandler);



  loadOverviewData(false);

}



async function loadOverviewData(useDemo) {

  try {

    const res = await fetch(`/api/overview?demo=${useDemo ? "true" : "false"}`);

    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const data = await res.json();



    const demoBanner = document.getElementById("demo-banner-box");

    const toggleBtn = document.getElementById("btn-overview-demo-toggle");

    if (demoBanner) {

      demoBanner.style.display = data.is_demo ? "flex" : "none";

    }

    if (toggleBtn) {

      toggleBtn.textContent = data.is_demo ? "Switch to Real Feed" : "View Demo Feed";

    }



    // Update KPI cards

    document.getElementById("kpi-total-inspections").textContent = data.metrics.total_inspections;

    document.getElementById("kpi-defects-detected").textContent = data.metrics.defects_detected;

    document.getElementById("kpi-requiring-review").textContent = data.metrics.requiring_review;

    document.getElementById("kpi-pass-rate").textContent = data.metrics.pass_rate_pct != null 

      ? `${data.metrics.pass_rate_pct}%` 

      : "--%";



    const subText = document.getElementById("kpi-total-sub");

    if (subText) {

      subText.textContent = data.is_demo ? "Demonstration aggregation" : "Local SQLite records";

    }



    // Update Recent Inspections Table

    const tbody = document.getElementById("overview-recent-table-body");

    if (tbody) {

      if (!data.recent_inspections || data.recent_inspections.length === 0) {

        tbody.innerHTML = `

          <tr>

            <td colspan="7" style="text-align: center; padding: var(--space-6); color: var(--text-muted);">

              No inspections recorded in local database yet.<br>

              Switch to <strong>Inspect</strong> to run an image inspection or toggle Demo Feed for UI layout verification.

            </td>

          </tr>

        `;

      } else {

        tbody.innerHTML = data.recent_inspections.map(row => {

          const badgeClass = row.assessment.toLowerCase().replace(/\s+/g, '-');

          return `

            <tr>

              <td style="color: var(--cyan-primary); font-weight: 600; cursor: pointer;" onclick="openInspectionReport('${row.id}')">${row.id}</td>

              <td>${row.timestamp}</td>

              <td style="color: var(--text-primary);">${row.filename}</td>

              <td><span style="font-weight: 700; color: ${row.defect_count > 0 ? 'var(--status-reject)' : 'var(--status-pass)'}">${row.defect_count}</span></td>

              <td>${row.top_defect}</td>

              <td><span class="badge-verdict ${badgeClass}">${row.assessment}</span></td>

              <td><span style="font-size: 10px; color: var(--text-muted);">${row.status}</span></td>

            </tr>

          `;

        }).join("");

      }

    }



    // Update Defect Distribution List

    const distList = document.getElementById("overview-defect-list");

    if (distList && data.defect_distribution) {

      distList.innerHTML = Object.entries(data.defect_distribution).map(([name, count]) => {

        return `

          <div class="defect-dist-row">

            <span class="defect-dist-name">

              <span style="display:inline-block; width: 8px; height: 8px; border-radius: 2px; background: var(--cyan-primary);"></span>

              ${name}

            </span>

            <span class="defect-dist-count">${count}</span>

          </div>

        `;

      }).join("");

    }

  } catch (err) {

    console.error("Failed to load overview data:", err);

  }

}



/* ==========================================================

   4. PAGE 2: INSPECT WORKSPACE CONTROLLER

   ========================================================== */

function initInspectWorkspace() {

  // Initialize Canvas Viewer

  steelViewer = new SteelViewer("inspection-canvas", "viewer-container");

  steelViewer.onSelectDetection = (detId) => {

    highlightDetectionInList(detId);

  };



  const dropzone = document.getElementById("inspect-dropzone");

  const fileInput = document.getElementById("file-input");

  const browseBtn = document.getElementById("btn-browse-file");

  const clearBtn = document.getElementById("btn-clear-image");

  const resetBtn = document.getElementById("btn-reset-inspection");

  const runInspectionBtn = document.getElementById("btn-run-inspection");

  const saveInspectionBtn = document.getElementById("btn-save-inspection");



  // Sample Browser Controls

  const categorySelect = document.getElementById("sample-category-select");

  const fileSelect = document.getElementById("sample-file-select");

  const loadSampleBtn = document.getElementById("btn-load-sample");



  // Load Initial Verified Samples into Dropdown

  fetchSampleList("");



  categorySelect?.addEventListener("change", (e) => {

    fetchSampleList(e.target.value);

  });



  loadSampleBtn?.addEventListener("click", () => {

    const selectedId = fileSelect?.value;

    if (selectedId) {

      loadVerifiedSample(selectedId);

    }

  });



  fileSelect?.addEventListener("change", (e) => {

    if (e.target.value) {

      loadVerifiedSample(e.target.value);

    }

  });



  // Local File Upload Handlers

  browseBtn?.addEventListener("click", () => fileInput?.click());

  dropzone?.addEventListener("click", (e) => {

    if (e.target !== browseBtn) fileInput?.click();

  });



  dropzone?.addEventListener("dragover", (e) => {

    e.preventDefault();

    dropzone.classList.add("dragover");

  });



  dropzone?.addEventListener("dragleave", () => {

    dropzone.classList.remove("dragover");

  });



  dropzone?.addEventListener("drop", (e) => {

    e.preventDefault();

    dropzone.classList.remove("dragover");

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {

      handleUserUploadedFile(e.dataTransfer.files[0]);

    }

  });



  fileInput?.addEventListener("change", (e) => {

    if (e.target.files && e.target.files[0]) {

      handleUserUploadedFile(e.target.files[0]);

    }

  });



  clearBtn?.addEventListener("click", clearLoadedInspection);

  resetBtn?.addEventListener("click", clearLoadedInspection);



  // Run Inspection & Save Actions

  const runAiBtn = document.getElementById("btn-run-ai");

  runAiBtn?.addEventListener("click", handleRunAIInference);

  runInspectionBtn?.addEventListener("click", handleRunInspection);

  saveInspectionBtn?.addEventListener("click", saveCurrentInspection);



  // Viewer Toolbar Controls

  document.getElementById("btn-view-fit")?.addEventListener("click", () => steelViewer.fitToScreen());

  document.getElementById("btn-view-100")?.addEventListener("click", () => steelViewer.setScale(1.0));

  document.getElementById("btn-zoom-in")?.addEventListener("click", () => steelViewer.zoomBy(0.25));

  document.getElementById("btn-zoom-out")?.addEventListener("click", () => steelViewer.zoomBy(-0.25));



  // Toggle bounding boxes & raw mode

  document.getElementById("btn-toggle-bbox")?.addEventListener("click", (e) => {

    steelViewer.setShowBBoxes(!steelViewer.showBBoxes);

    e.target.textContent = steelViewer.showBBoxes ? "BBoxes [ON]" : "BBoxes [OFF]";

    e.target.classList.toggle("active", steelViewer.showBBoxes);

  });



  document.getElementById("btn-toggle-annotated")?.addEventListener("click", (e) => {

    steelViewer.setRawMode(!steelViewer.rawMode);

    e.target.textContent = steelViewer.rawMode ? "Annotated View" : "Raw Mode";

    e.target.classList.toggle("active", steelViewer.rawMode);

  });



  // Explain with AI button hook

  document.getElementById("btn-explain-ai")?.addEventListener("click", () => {

    navigateTo("assistant");

    if (state.lastInferenceData) {

      requestExecuTorchAdvisory();

    }

  });



  // Offline Hackathon Demo Presets (Real Local Inference)

  document.getElementById("btn-demo-critical")?.addEventListener("click", async () => {

    await loadVerifiedSample("img_01_3402617700_01009");

    await handleRunAIInference();

  });



  document.getElementById("btn-demo-inclusion")?.addEventListener("click", async () => {

    await loadVerifiedSample("img_01_425391700_00025");

    await handleRunAIInference();

  });



  document.getElementById("btn-demo-minor")?.addEventListener("click", async () => {

    await loadVerifiedSample("img_01_424995500_00255");

    await handleRunAIInference();

  });



  document.getElementById("btn-demo-clean")?.addEventListener("click", () => {

    clearLoadedInspection();

    setAssessmentStatus("PASS", "Zero surface anomalies detected. Steel workpiece released for processing.", ["QR-00"]);

    navigateTo("inspect");

  });



  // Manual request advisory button in Assistant page

  document.getElementById("btn-request-advisory")?.addEventListener("click", () => {

    requestExecuTorchAdvisory();

  });

}



/* ----------------------------------------------------------

   Sample Browser API Integration

   ---------------------------------------------------------- */

async function fetchSampleList(defectClass) {

  const fileSelect = document.getElementById("sample-file-select");

  if (!fileSelect) return;



  fileSelect.innerHTML = `<option value="">Loading verified samples...</option>`;



  try {

    const url = defectClass ? `/api/samples?class=${encodeURIComponent(defectClass)}&limit=80` : `/api/samples?limit=80`;

    const res = await fetch(url);

    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const data = await res.json();



    state.samplesCache = data.samples;



    if (!data.samples || data.samples.length === 0) {

      fileSelect.innerHTML = `<option value="">No samples found for category</option>`;

      return;

    }



    fileSelect.innerHTML = data.samples.map(s => {

      const classStr = s.classes.join(", ");

      return `<option value="${s.id}">${s.filename} (${s.defect_count} defect${s.defect_count > 1 ? 's' : ''}: ${classStr})</option>`;

    }).join("");



  } catch (err) {

    fileSelect.innerHTML = `<option value="">Failed to load samples: ${err.message}</option>`;

  }

}



async function loadVerifiedSample(sampleId) {

  clearLoadedInspection();

  const generation = invalidateIntelligence("Dataset annotations selected. Run YOLO to obtain a model inspection summary.");

  state.uploadedBase64 = null;

  try {

    // 1. Fetch Pascal VOC Ground Truth Annotation

    const annoRes = await fetch(`/api/samples/annotation?id=${encodeURIComponent(sampleId)}`);

    if (!annoRes.ok) throw new Error(`Failed to load annotation: HTTP ${annoRes.status}`);

    const annoData = await annoRes.json();

    if (generation !== state.inspectionGeneration) return;



    // 2. Fetch Image with Promise awaiting completion
    const img = new Image();
    await new Promise((resolve, reject) => {
      img.onload = () => {
        if (generation !== state.inspectionGeneration) {
          resolve();
          return;
        }
        state.currentImage = img;
        state.currentSampleId = sampleId;
        state.currentInspectionId = annoData.inspection_id;
        state.dataMode = "GROUND_TRUTH";
        state.detections = annoData.detections;
        state.assessment = annoData.assessment;

        state.imageMeta = {
          filename: annoData.filename,
          width: annoData.image_width,
          height: annoData.image_height,
          filesizeKb: 512,
          mode: "PASCAL VOC GROUND TRUTH"
        };

        updateImageMetadata();

        // Show canvas and load into viewer
        const emptyState = document.getElementById("viewer-empty");
        const canvas = document.getElementById("inspection-canvas");
        if (emptyState) emptyState.style.display = "none";
        if (canvas) canvas.style.display = "block";

        steelViewer.setImage(img, annoData.detections);

        // Render Detection Results in Right Panel
        renderDetectionResults(annoData.detections, "GROUND_TRUTH");

        // Update Bottom Assessment Panel
        setAssessmentStatus(
          annoData.assessment.status,
          annoData.assessment.summary,
          annoData.assessment.rule_ids
        );

        // Enable Action Buttons
        const runAiBtn = document.getElementById("btn-run-ai");
        const runBtn = document.getElementById("btn-run-inspection");
        const saveBtn = document.getElementById("btn-save-inspection");
        if (runAiBtn) {
          runAiBtn.disabled = false;
          document.getElementById("btn-run-ai-text").textContent = "RUN LOCAL YOLO INFERENCE";
        }
        if (runBtn) {
          runBtn.disabled = false;
          document.getElementById("btn-run-inspection-text").textContent = "EVALUATE QUALITY RULES";
        }
        if (saveBtn) {
          saveBtn.disabled = false;
          document.getElementById("btn-save-inspection-text").textContent = "SAVE TO DATABASE";
        }
        resolve();
      };
      img.onerror = () => reject(new Error("Failed to load sample image"));
      img.src = `/api/samples/image?id=${encodeURIComponent(sampleId)}`;
    });



  } catch (err) {

    if (generation !== state.inspectionGeneration) return;

    alert(`Error loading verified sample: ${err.message}`);

  }

}



function handleUserUploadedFile(file) {

  if (!file.type.startsWith("image/")) {

    alert("Please select a valid image file (JPG, PNG, BMP).");

    return;

  }



  clearLoadedInspection();

  const generation = invalidateIntelligence();

  const reader = new FileReader();

  reader.onload = (e) => {

    if (generation !== state.inspectionGeneration) return;

    const img = new Image();

    img.onload = () => {

      if (generation !== state.inspectionGeneration) return;

      state.currentImage = img;

      state.uploadedBase64 = e.target.result;

      state.currentSampleId = null;

      state.currentInspectionId = `SG-${Date.now()}`;

      state.dataMode = "USER_UPLOAD";

      state.detections = [];

      state.assessment = null;



      state.imageMeta = {

        filename: file.name,

        width: img.width,

        height: img.height,

        filesizeKb: Math.round(file.size / 1024),

        mode: "USER UPLOAD (UNANNOTATED)"

      };



      updateImageMetadata();



      const emptyState = document.getElementById("viewer-empty");

      const canvas = document.getElementById("inspection-canvas");

      if (emptyState) emptyState.style.display = "none";

      if (canvas) canvas.style.display = "block";



      steelViewer.setImage(img, []);



      // Right panel: awaiting inference

      const modeBanner = document.getElementById("right-panel-mode-banner");

      if (modeBanner) {

        modeBanner.style.display = "block";

        modeBanner.className = "mode-banner-ai";

        modeBanner.textContent = "AI INFERENCE REQUIRED";

      }



      const list = document.getElementById("inspect-detection-list");

      if (list) {

        list.innerHTML = `

          <div style="text-align: center; padding: var(--space-6); color: var(--text-muted); font-size: var(--text-xs);">

            Custom steel image loaded.<br>Click <strong>RUN LOCAL YOLO INFERENCE</strong> to query vision model.

          </div>

        `;

      }

      document.getElementById("detection-count-badge").textContent = "PENDING";



      // Bottom panel: pending

      setAssessmentStatus(

        "ASSESSMENT PENDING",

        "Awaiting vision model defect localization. Quality assessment cannot evaluate uninspected workpiece.",

        []

      );



      const runAiBtn = document.getElementById("btn-run-ai");

      const runBtn = document.getElementById("btn-run-inspection");

      const saveBtn = document.getElementById("btn-save-inspection");

      if (runAiBtn) {

        runAiBtn.disabled = false;

        document.getElementById("btn-run-ai-text").textContent = "RUN LOCAL YOLO INFERENCE";

      }

      if (runBtn) {

        runBtn.disabled = true;

      }

      if (saveBtn) {

        saveBtn.disabled = true;

      }

    };

    img.src = e.target.result;

  };

  reader.readAsDataURL(file);

}



async function handleRunAIInference() {

  if (!state.currentImage) {

    alert("Please select a verified sample or upload a steel image first.");

    return;

  }



  const payload = {};

  if (state.uploadedBase64) {

    payload.image_base64 = state.uploadedBase64;

    payload.filename = state.imageMeta?.filename;

  } else if (state.currentSampleId) {

    payload.sample_id = state.currentSampleId;

    payload.filename = state.imageMeta?.filename;

  } else {

    alert("No valid image data available for inference.");

    return;

  }



  const confInput = document.getElementById("input-conf-thresh");

  const iouInput = document.getElementById("input-iou-thresh");

  if (confInput) payload.confidence_threshold = parseFloat(confInput.value);

  if (iouInput) payload.iou_threshold = parseFloat(iouInput.value);



  const runAiBtn = document.getElementById("btn-run-ai");

  const runAiText = document.getElementById("btn-run-ai-text");

  if (runAiBtn) runAiBtn.disabled = true;

  if (runAiText) runAiText.textContent = "INFERRING (YOLOv8)...";

  const generation = invalidateIntelligence("YOLO inference in progress. A summary will follow successful inference.");



  try {

    const res = await fetch("/api/inference/run", {

      method: "POST",

      headers: { "Content-Type": "application/json" },

      body: JSON.stringify(payload)

    });

    const data = await res.json();

    if (generation !== state.inspectionGeneration) return;



    if (!res.ok || data.status !== "SUCCESS" || !Array.isArray(data.detections)) {

      throw new Error(data.message || data.error || `HTTP ${res.status}`);

    }



    state.dataMode = "AI_INFERENCE";

    state.currentInspectionId = data.inspection_id || `SG-AI-${Date.now()}`;

    state.detections = data.detections || [];

    state.assessment = data.assessment;

    state.activeModelName = data.model_name;

    state.activeModelVersion = data.model_version;

    state.inferenceTimeMs = data.inference_time_ms;



    steelViewer.setImage(state.currentImage, state.detections);

    renderDetectionResults(state.detections, "AI_INFERENCE");



    if (data.assessment) {

      setAssessmentStatus(

        data.assessment.status,

        data.assessment.summary,

        data.assessment.rule_ids

      );

    }



    if (state.imageMeta) {

      state.imageMeta.mode = `AI INFERENCE: ${data.model_version} (${data.inference_time_ms} ms, ${data.total_defects} defects)`;

      updateImageMetadata();

    }



    const saveBtn = document.getElementById("btn-save-inspection");

    const saveText = document.getElementById("btn-save-inspection-text");

    if (saveBtn) {

      saveBtn.disabled = false;

      saveBtn.style.color = "var(--cyan-primary)";

      saveBtn.style.borderColor = "var(--cyan-muted)";

    }

    if (saveText) saveText.textContent = "SAVE TO DATABASE";

    // Cache last successful inference for grounded advisory and enable on-demand query
    state.lastInferenceData = data;
    invalidateAdvisory();



    // Non-blocking: an advisory failure cannot discard a successful vision result.

    void intelligence.summarize(data);



  } catch (err) {

    if (generation !== state.inspectionGeneration) return;

    intelligence.reset("Inspection summary unavailable because YOLO inference failed. Run inference again.");

    alert(`[INFERENCE FAILED]\n${err.message}`);

  } finally {

    if (generation === state.inspectionGeneration) {

      if (runAiBtn) runAiBtn.disabled = false;

      if (runAiText) runAiText.textContent = "RUN LOCAL YOLO INFERENCE";

    }

  }

}



function handleRunInspection() {

  if (!state.currentImage) return;



  if (state.dataMode === "GROUND_TRUTH") {

    if (state.assessment) {

      setAssessmentStatus(

        state.assessment.status,

        state.assessment.summary,

        state.assessment.rule_ids

      );

      alert(`[EVALUATION COMPLETE]\nMode: Pascal VOC Ground Truth\nApplied Rules: ${state.assessment.rule_ids.join(", ")}\nFinal Verdict: ${state.assessment.status}\n\nNote: Current display reflects verified GC10-DET dataset ground truth.`);

    }

  } else if (state.dataMode === "USER_UPLOAD" || state.dataMode === "AI_INFERENCE") {

    handleRunAIInference();

  }

}



/* ----------------------------------------------------------

   Save Inspection to SQLite (TASK 2)

   ---------------------------------------------------------- */

async function saveCurrentInspection() {

  if (!state.currentImage || state.isSaving) return;



  const saveBtn = document.getElementById("btn-save-inspection");

  const saveText = document.getElementById("btn-save-inspection-text");



  state.isSaving = true;

  if (saveText) saveText.textContent = "SAVING...";

  if (saveBtn) saveBtn.disabled = true;



  const payload = {

    inspection_id: state.currentInspectionId || `SG-${Date.now()}`,

    timestamp: new Date().toISOString(),

    filename: state.imageMeta.filename,

    image_reference: state.currentSampleId || state.imageMeta.filename,

    source: state.dataMode === "GROUND_TRUTH" ? "GROUND_TRUTH" : "AI_INFERENCE",

    image_width: state.imageMeta.width,

    image_height: state.imageMeta.height,

    detections: state.detections,

    assessment: state.assessment || {

      status: "ASSESSMENT PENDING",

      summary: "Workpiece saved without evaluation.",

      rule_ids: [],

      rules_applied: []

    },

    model_name: state.dataMode === "GROUND_TRUTH" ? "Pascal VOC Certified Annotation" : state.activeModelName,

    model_version: state.dataMode === "GROUND_TRUTH" ? "GC10-DET Ground Truth" : state.activeModelVersion,

    inference_time_ms: state.dataMode === "GROUND_TRUTH" ? null : state.inferenceTimeMs,

    operator_notes: "Saved from inspection console workspace.",

    review_status: "UNREVIEWED"

  };



  try {

    const res = await fetch("/api/inspections", {

      method: "POST",

      headers: { "Content-Type": "application/json" },

      body: JSON.stringify(payload)

    });

    const result = await res.json();



    if (result.success) {

      if (saveText) saveText.textContent = "✓ SAVED TO DATABASE";

      if (saveBtn) {

        saveBtn.disabled = true;

        saveBtn.style.color = "var(--status-pass)";

        saveBtn.style.borderColor = "var(--status-pass)";

      }

      alert(`[PERSISTENCE SUCCESS]\n${result.message}\nInspection ID: ${result.inspection_id}`);

    } else if (result.already_exists) {

      if (saveText) saveText.textContent = "ALREADY RECORDED";

      alert(`[RECORD EXISTS]\n${result.message}`);

    } else {

      throw new Error(result.error || "Save operation failed.");

    }

  } catch (err) {

    alert(`Failed to save inspection: ${err.message}`);

    if (saveText) saveText.textContent = "SAVE TO DATABASE";

    if (saveBtn) saveBtn.disabled = false;

  } finally {

    state.isSaving = false;

  }

}



function renderDetectionResults(detections, mode) {

  const modeBanner = document.getElementById("right-panel-mode-banner");

  if (modeBanner) {

    modeBanner.style.display = "block";

    if (mode === "GROUND_TRUTH") {

      modeBanner.className = "mode-banner-gt";

      modeBanner.textContent = "PASCAL VOC GROUND TRUTH (CERTIFIED)";

    } else {

      modeBanner.className = "mode-banner-ai";

      modeBanner.textContent = "AI MODEL INFERENCE";

    }

  }



  const countBadge = document.getElementById("detection-count-badge");

  if (countBadge) {

    countBadge.textContent = `${detections.length} FOUND`;

    countBadge.className = detections.length > 0 ? "badge-verdict review" : "badge-verdict pass";

  }



  const list = document.getElementById("inspect-detection-list");

  if (!list) return;



  if (!detections || detections.length === 0) {

    list.innerHTML = `

      <div style="text-align: center; padding: var(--space-6); color: var(--status-pass); font-size: var(--text-xs); font-family: var(--font-mono);">

        ✓ No surface defects found.<br>Clean steel coil surface.

      </div>

    `;

    return;

  }



  list.innerHTML = detections.map((det) => {

    const isSelected = det.detection_id === state.selectedDetectionId;

    const [xmin, ymin, xmax, ymax] = det.bbox_xyxy;

    const [bw, bh] = det.bbox_wh;



    return `

      <div class="detection-card ${isSelected ? 'selected' : ''}" data-det-id="${det.detection_id}" id="det-card-${det.detection_id}">

        <div class="detection-header">

          <span class="detection-class" style="color: ${det.color || 'var(--text-primary)'}">

            ${det.detection_id}: ${det.class_name}

          </span>

          <span class="badge-verdict ${det.criticality.toLowerCase()}">${det.criticality}</span>

        </div>

        <div class="detection-coords">

          BBox: [${xmin}, ${ymin}, ${xmax}, ${ymax}]

        </div>

        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 4px;">

          <span style="font-size: 10px; color: var(--text-muted); font-family: var(--font-mono);">

            Size: ${bw} × ${bh} px (${det.area_px} px²)

          </span>

          <span style="font-size: 10px; font-family: var(--font-mono); color: var(--cyan-primary); font-weight: 600;">

            ${det.confidence != null ? `Conf: ${(det.confidence * 100).toFixed(1)}%` : ''}

          </span>

          <button class="btn-locate" data-locate-id="${det.detection_id}">Locate Defect</button>

        </div>

      </div>

    `;

  }).join("");



  list.querySelectorAll(".detection-card").forEach(card => {

    card.addEventListener("click", (e) => {

      if (e.target.classList.contains("btn-locate")) return;

      const detId = card.getAttribute("data-det-id");

      selectDetection(detId);

    });

  });



  list.querySelectorAll(".btn-locate").forEach(btn => {

    btn.addEventListener("click", (e) => {

      e.stopPropagation();

      const detId = btn.getAttribute("data-locate-id");

      const det = detections.find(d => d.detection_id === detId);

      if (det) {

        selectDetection(detId);

        steelViewer.zoomToBox(det.bbox_xyxy);

      }

    });

  });

}



function selectDetection(detId) {

  state.selectedDetectionId = detId;

  steelViewer.setSelectedId(detId);

  highlightDetectionInList(detId);

}



function highlightDetectionInList(detId) {

  document.querySelectorAll(".detection-card").forEach(c => {

    c.classList.remove("selected");

  });

  if (detId) {

    const card = document.getElementById(`det-card-${detId}`);

    if (card) {

      card.classList.add("selected");

      card.scrollIntoView({ behavior: "smooth", block: "nearest" });

    }

  }

}



function updateImageMetadata() {

  if (!state.imageMeta) return;

  document.getElementById("meta-filename").textContent = state.imageMeta.filename;

  document.getElementById("meta-data-mode").textContent = state.imageMeta.mode || "--";

  document.getElementById("meta-dimensions").textContent = `${state.imageMeta.width} × ${state.imageMeta.height} px`;

  const mp = ((state.imageMeta.width * state.imageMeta.height) / 1000000).toFixed(2);

  document.getElementById("meta-resolution").textContent = `${mp} MP`;

  document.getElementById("meta-filesize").textContent = `${state.imageMeta.filesizeKb} KB`;

}



function clearLoadedInspection() {

  invalidateIntelligence();

  state.uploadedBase64 = null;

  state.activeModelName = null;

  state.activeModelVersion = null;

  state.inferenceTimeMs = null;

  state.currentImage = null;

  state.imageMeta = null;

  state.dataMode = "IDLE";

  state.currentSampleId = null;

  state.currentInspectionId = null;

  state.detections = [];

  state.selectedDetectionId = null;

  state.assessment = null;

  state.lastInferenceData = null;
  invalidateAdvisory("Run a YOLO inspection first to generate the grounded ExecuTorch advisory plan.");



  document.getElementById("meta-filename").textContent = "None selected";

  document.getElementById("meta-data-mode").textContent = "--";

  document.getElementById("meta-dimensions").textContent = "-- × --";

  document.getElementById("meta-resolution").textContent = "-- MP";

  document.getElementById("meta-filesize").textContent = "-- KB";



  const emptyState = document.getElementById("viewer-empty");

  const canvas = document.getElementById("inspection-canvas");

  if (emptyState) emptyState.style.display = "flex";

  if (canvas) canvas.style.display = "none";



  const modeBanner = document.getElementById("right-panel-mode-banner");

  if (modeBanner) modeBanner.style.display = "none";



  const runAiBtn = document.getElementById("btn-run-ai");

  if (runAiBtn) {

    runAiBtn.disabled = true;

    document.getElementById("btn-run-ai-text").textContent = "RUN LOCAL YOLO INFERENCE";

  }



  document.getElementById("btn-run-inspection").disabled = true;

  document.getElementById("btn-run-inspection-text").textContent = "EVALUATE QUALITY RULES";



  const saveBtn = document.getElementById("btn-save-inspection");

  if (saveBtn) {

    saveBtn.disabled = true;

    saveBtn.style.color = "var(--cyan-primary)";

    saveBtn.style.borderColor = "var(--cyan-muted)";

    document.getElementById("btn-save-inspection-text").textContent = "SAVE TO DATABASE";

  }



  const list = document.getElementById("inspect-detection-list");

  if (list) {

    list.innerHTML = `

      <div style="text-align: center; padding: var(--space-6); color: var(--text-muted); font-size: var(--text-xs);">

        Awaiting inspection execution.<br>Detected bounding boxes and confidence metrics will appear here.

      </div>

    `;

  }

  document.getElementById("detection-count-badge").textContent = "0 FOUND";

  document.getElementById("detection-count-badge").className = "badge-verdict pending";



  setAssessmentStatus("PENDING", "No inspection run on current workpiece. Quality assessment pending input.", []);

}



function setAssessmentStatus(verdict, explanation, rules = []) {

  const badge = document.getElementById("verdict-badge");

  const expl = document.getElementById("verdict-explanation");

  const rulesElem = document.getElementById("assessment-rules-applied");



  const vClass = verdict.toLowerCase().replace(/\s+/g, '-');

  if (badge) {

    badge.className = `verdict-box ${vClass}`;

    badge.textContent = verdict;

  }

  if (expl) expl.textContent = explanation;

  if (rulesElem) {

    rulesElem.textContent = rules.length > 0 ? `Applied Rules: ${rules.join(", ")}` : "Applied Rules: None";

  }

}



/* ==========================================================

   5. PAGE 4: HISTORY & TRACEABILITY (TASK 3)

   ========================================================== */

function initHistoryPage() {

  const searchInput = document.getElementById("history-search-input");

  const statusFilter = document.getElementById("history-filter-status");

  const sourceFilter = document.getElementById("history-filter-source");

  const refreshBtn = document.getElementById("btn-history-refresh");

  const exportCsvBtn = document.getElementById("btn-history-export-csv");



  let searchTimeout = null;

  searchInput?.addEventListener("input", () => {

    clearTimeout(searchTimeout);

    searchTimeout = setTimeout(loadHistoryData, 300);

  });



  statusFilter?.addEventListener("change", loadHistoryData);

  sourceFilter?.addEventListener("change", loadHistoryData);

  refreshBtn?.addEventListener("click", loadHistoryData);



  exportCsvBtn?.addEventListener("click", () => {

    const search = searchInput?.value || "";

    const status = statusFilter?.value || "ALL";

    const source = sourceFilter?.value || "ALL";

    window.location.href = `/api/inspections/export/csv?search=${encodeURIComponent(search)}&status=${encodeURIComponent(status)}&source=${encodeURIComponent(source)}`;

  });

}



async function loadHistoryData() {

  const search = document.getElementById("history-search-input")?.value || "";

  const status = document.getElementById("history-filter-status")?.value || "ALL";

  const source = document.getElementById("history-filter-source")?.value || "ALL";



  const tbody = document.getElementById("history-table-body");

  if (!tbody) return;



  try {

    const url = `/api/inspections?search=${encodeURIComponent(search)}&status=${encodeURIComponent(status)}&source=${encodeURIComponent(source)}&limit=100`;

    const res = await fetch(url);

    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const data = await res.json();



    if (!data.inspections || data.inspections.length === 0) {

      tbody.innerHTML = `

        <tr>

          <td colspan="8" style="text-align: center; padding: var(--space-8); color: var(--text-muted);">

            No inspections found matching search criteria.<br>

            Inspect a steel sample and click <strong>SAVE TO DATABASE</strong> to create an entry.

          </td>

        </tr>

      `;

      return;

    }



    tbody.innerHTML = data.inspections.map(insp => {

      const vClass = insp.assessment_status.toLowerCase().replace(/\s+/g, '-');

      const isGt = insp.source === "GROUND_TRUTH";

      return `

        <tr>

          <td style="color: var(--cyan-primary); font-weight: 600; font-family: var(--font-mono);">${insp.id}</td>

          <td>${insp.timestamp.replace("T", " ").replace("Z", "")}</td>

          <td style="color: var(--text-primary);">${insp.filename}</td>

          <td>

            <span class="badge-verdict ${isGt ? 'review' : 'pending'}" style="font-size: 9px;">

              ${isGt ? 'GROUND TRUTH' : 'AI INFERENCE'}

            </span>

          </td>

          <td>

            <span style="font-weight: 700; color: ${insp.total_defects > 0 ? 'var(--status-reject)' : 'var(--status-pass)'}">

              ${insp.total_defects}

            </span>

          </td>

          <td><span class="badge-verdict ${vClass}">${insp.assessment_status}</span></td>

          <td><span style="font-size: 10px; color: var(--text-secondary);">${insp.review_status}</span></td>

          <td style="white-space: nowrap;">

            <button class="btn btn-secondary" style="padding: 2px 6px; font-size: 10px;" onclick="openInspectionReport('${insp.id}')">View Report</button>

            <button class="btn btn-secondary" style="padding: 2px 6px; font-size: 10px; margin-left: 4px;" onclick="reloadIntoInspect('${insp.image_reference}')">Inspect</button>

          </td>

        </tr>

      `;

    }).join("");



  } catch (err) {

    tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: var(--status-reject); padding: var(--space-4);">Failed to load history: ${err.message}</td></tr>`;

  }

}



/* ==========================================================

   6. PAGE 3: INSPECTION RESULTS / REPORT (TASK 4)

   ========================================================== */

function initResultsPage() {

  document.getElementById("btn-print-report")?.addEventListener("click", () => window.print());

  

  document.getElementById("btn-export-report-json")?.addEventListener("click", () => {

    if (!state.activeReport) {

      alert("No inspection report loaded.");

      return;

    }

    window.location.href = `/api/inspections/export/json?id=${encodeURIComponent(state.activeReport.id)}`;

  });



  document.getElementById("btn-save-report-notes")?.addEventListener("click", async () => {

    if (!state.activeReport) return;

    const notes = document.getElementById("report-notes-textarea")?.value || "";

    const status = document.getElementById("report-review-status-select")?.value || "UNREVIEWED";



    try {

      const res = await fetch("/api/inspections/notes", {

        method: "POST",

        headers: { "Content-Type": "application/json" },

        body: JSON.stringify({

          inspection_id: state.activeReport.id,

          operator_notes: notes,

          review_status: status

        })

      });

      const data = await res.json();

      if (data.success) {

        alert("Sign-off notes and review status updated successfully.");

        state.activeReport.operator_notes = notes;

        state.activeReport.review_status = status;

      }

    } catch (err) {

      alert(`Failed to save sign-off: ${err.message}`);

    }

  });

}



async function openInspectionReport(inspId) {

  try {

    const res = await fetch(`/api/inspections?id=${encodeURIComponent(inspId)}`);

    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const record = await res.json();



    state.activeReport = record;

    renderInspectionReport(record);

    navigateTo("results");

  } catch (err) {

    alert(`Unable to open inspection report: ${err.message}`);

  }

}



function renderInspectionReport(insp) {

  document.getElementById("report-insp-id").textContent = insp.id;

  document.getElementById("report-timestamp").textContent = insp.timestamp;

  document.getElementById("report-filename").textContent = insp.filename;



  const isGt = insp.source === "GROUND_TRUTH";

  const sourceBadge = document.getElementById("report-source-badge");

  if (sourceBadge) {

    sourceBadge.className = isGt ? "badge-verdict review" : "badge-verdict pass";

    sourceBadge.textContent = isGt ? "PASCAL VOC GROUND TRUTH" : "AI MODEL INFERENCE";

  }



  // Verdict box

  const vClass = insp.assessment_status.toLowerCase().replace(/\s+/g, '-');

  const verdictBox = document.getElementById("report-verdict-box");

  if (verdictBox) {

    verdictBox.className = `verdict-box ${vClass}`;

    verdictBox.textContent = insp.assessment_status;

  }

  document.getElementById("report-verdict-summary").textContent = insp.assessment_summary;



  // Rules list

  const rulesList = document.getElementById("report-rules-list");

  if (rulesList) {

    if (insp.rules_applied && insp.rules_applied.length > 0) {

      rulesList.innerHTML = insp.rules_applied.map(r => `

        <div style="background: rgba(0,0,0,0.2); padding: 4px 6px; border-radius: 2px;">

          <span style="color: var(--cyan-primary); font-weight: bold;">[${r.rule_id}] ${r.rule_title}</span>: 

          <span>${r.reason}</span>

        </div>

      `).join("");

    } else {

      rulesList.innerHTML = `<span style="color: var(--text-muted);">None applied</span>`;

    }

  }



  // Defect table

  const tbody = document.getElementById("report-defect-table-body");

  const countBadge = document.getElementById("report-defects-count-badge");

  if (countBadge) {

    countBadge.textContent = `${insp.total_defects} DEFECT${insp.total_defects !== 1 ? 'S' : ''}`;

    countBadge.className = insp.total_defects > 0 ? "badge-verdict review" : "badge-verdict pass";

  }



  if (tbody) {

    if (!insp.detections || insp.detections.length === 0) {

      tbody.innerHTML = `<tr><td colspan="7" style="text-align:center; padding: var(--space-4); color: var(--status-pass);">✓ Prime Steel Strip Surface: Zero Defect Instances</td></tr>`;

    } else {

      tbody.innerHTML = insp.detections.map(d => `

        <tr>

          <td style="color: var(--cyan-primary); font-weight: 600;">${d.detection_id}</td>

          <td style="color: var(--text-primary); font-weight: 500;">${d.class_name}</td>

          <td><span class="badge-verdict ${d.criticality.toLowerCase()}">${d.criticality}</span></td>

          <td>${(d.confidence * 100).toFixed(0)}%</td>

          <td>[${d.xmin}, ${d.ymin}, ${d.xmax}, ${d.ymax}]</td>

          <td>${d.width} × ${d.height} px</td>

          <td>${d.area_px} px²</td>

        </tr>

      `).join("");

    }

  }



  // Operator notes and review status

  const notesArea = document.getElementById("report-notes-textarea");

  const reviewSelect = document.getElementById("report-review-status-select");

  if (notesArea) notesArea.value = insp.operator_notes || "";

  if (reviewSelect) reviewSelect.value = insp.review_status || "UNREVIEWED";



  // Draw annotated image on report canvas

  renderReportCanvas(insp);

}



function renderReportCanvas(insp) {

  const canvas = document.getElementById("report-canvas");

  const container = document.getElementById("report-canvas-container");

  if (!canvas || !container) return;



  const cw = container.clientWidth || 700;

  const ch = container.clientHeight || 380;

  canvas.width = cw;

  canvas.height = ch;

  const ctx = canvas.getContext("2d");



  ctx.fillStyle = "#03060a";

  ctx.fillRect(0, 0, cw, ch);



  const img = new Image();

  img.onload = () => {

    const scale = Math.min((cw - 20) / img.width, (ch - 20) / img.height);

    const sw = Math.round(img.width * scale);

    const sh = Math.round(img.height * scale);

    const ox = Math.round((cw - sw) / 2);

    const oy = Math.round((ch - sh) / 2);



    ctx.drawImage(img, ox, oy, sw, sh);

    ctx.strokeStyle = "#1e2c4a";

    ctx.strokeRect(ox, oy, sw, sh);



    // Draw defect bounding boxes

    if (insp.detections) {

      insp.detections.forEach(d => {

        const [xmin, ymin, xmax, ymax] = d.bbox_xyxy;

        const bx = ox + xmin * scale;

        const by = oy + ymin * scale;

        const bw = (xmax - xmin) * scale;

        const bh = (ymax - ymin) * scale;



        ctx.strokeStyle = "#ef4444";

        ctx.lineWidth = 2;

        ctx.strokeRect(bx, by, bw, bh);



        // Defect label

        ctx.fillStyle = "#ef4444";

        ctx.fillRect(bx, by - 16, Math.max(70, d.class_name.length * 8 + 20), 16);

        ctx.fillStyle = "#000000";

        ctx.font = "bold 10px monospace";

        ctx.fillText(d.class_name, bx + 4, by - 4);

      });

    }

  };



  img.src = `/api/samples/image?id=${encodeURIComponent(insp.image_reference || insp.filename.replace(/\.[^/.]+$/, ""))}`;

}



function reloadIntoInspect(sampleId) {

  if (sampleId) {

    loadVerifiedSample(sampleId);

    navigateTo("inspect");

  }

}



/* ==========================================================

   8. PAGE 8: SETTINGS & MODEL CONFIGURATION CONTROLLER

   ========================================================== */

function initSettingsPage() {

  const switchBtn = document.getElementById("btn-switch-model");

  const modelSelect = document.getElementById("settings-model-select");

  const confInput = document.getElementById("input-conf-thresh");

  const iouInput = document.getElementById("input-iou-thresh");

  const confLbl = document.getElementById("lbl-conf-val");

  const iouLbl = document.getElementById("lbl-iou-val");



  confInput?.addEventListener("input", (e) => {

    if (confLbl) confLbl.textContent = e.target.value;

  });

  iouInput?.addEventListener("input", (e) => {

    if (iouLbl) iouLbl.textContent = e.target.value;

  });



  switchBtn?.addEventListener("click", async () => {

    const selectedKey = modelSelect?.value;

    if (!selectedKey) return;

    switchBtn.disabled = true;

    switchBtn.textContent = "Activating Model...";



    try {

      const res = await fetch("/api/models/switch", {

        method: "POST",

        headers: { "Content-Type": "application/json" },

        body: JSON.stringify({ model_key: selectedKey })

      });

      const data = await res.json();

      if (!res.ok || data.status === "ERROR") {

        throw new Error(data.error || "Failed to switch model");

      }

      alert(`[MODEL ACTIVATED SUCCESSFULLY]\n${data.message}`);

      loadSettingsData();

      fetchSystemHealth();

    } catch (err) {

      alert(`[SWITCH FAILED]\n${err.message}`);

    } finally {

      switchBtn.disabled = false;

      switchBtn.textContent = "Switch Active Model";

    }

  });



  loadSettingsData();

}



async function loadSettingsData() {

  try {

    const res = await fetch("/api/models/status");

    if (!res.ok) return;

    const data = await res.json();

    const v = data.vision;

    if (!v) return;



    const nameElem = document.getElementById("settings-meta-name");

    const verElem = document.getElementById("settings-meta-version");

    const hashElem = document.getElementById("settings-meta-hash");

    const devElem = document.getElementById("settings-meta-device");

    const descElem = document.getElementById("settings-meta-desc");

    const badge = document.getElementById("settings-active-badge");

    const modelSelect = document.getElementById("settings-model-select");



    if (nameElem) nameElem.textContent = v.model_name;

    if (verElem) verElem.textContent = v.version;

    if (hashElem) hashElem.textContent = v.sha256;

    if (devElem) devElem.textContent = `${v.device_name || 'Host'} (${v.device || 'CPU'})`;

    if (modelSelect && v.model_id) modelSelect.value = v.model_id;



    if (descElem && data.catalog && data.catalog.models) {

      const activeObj = data.catalog.models.find(m => m.key === v.model_id);

      if (activeObj) descElem.textContent = activeObj.description;

    }



    if (badge) {

      badge.textContent = `ACTIVE: ${v.version ? v.version.toUpperCase() : 'ONLINE'}`;

      badge.className = v.model_id === "candidate" ? "badge-verdict pass" : "badge-verdict review";

    }

  } catch (err) {

    console.warn("Error loading settings data:", err);

  }

}



/* ==========================================================

   8. STRIP SEGMENTATION CONTROLLER (PHASE 4C)

   Severstal Lightweight U-Net Native 1600x256 Pixel Inspection

   Decoupled from GC10 YOLO Detector

   ========================================================== */



const segState = {

  initialized: false,

  samples: [],

  currentSampleId: null,

  currentImageBase64: null,

  baseImage: null,

  predColorImage: null,

  gtColorImage: null,

  predMaskData: null,

  gtMaskData: null,

  viewMode: "overlay",    // "overlay" | "sbs" | "predonly"

  scaleMode: "fit",       // "fit" | "native"

  opacity: 0.65,

  showBase: true,

  showPred: true,

  showGt: true,

  showClasses: { 1: true, 2: true, 3: true, 4: true },

  isRunning: false

};



function initSegmentationView() {

  if (segState.initialized) return;

  segState.initialized = true;



  // Filter dropdown change

  const filterSelect = document.getElementById("seg-filter-select");

  if (filterSelect) {

    filterSelect.addEventListener("change", () => {

      loadSegmentationSamples(filterSelect.value);

    });

  }



  // Sample select change

  const sampleSelect = document.getElementById("seg-sample-select");

  if (sampleSelect) {

    sampleSelect.addEventListener("change", () => {

      const sampleId = sampleSelect.value;

      if (sampleId) {

        loadSegmentationSample(sampleId);

      }

    });

  }



  // File upload dropzone & button

  const dropzone = document.getElementById("seg-dropzone");

  const fileInput = document.getElementById("seg-file-input");

  const browseBtn = document.getElementById("btn-seg-browse");



  if (browseBtn && fileInput) {

    browseBtn.addEventListener("click", () => fileInput.click());

  }



  if (fileInput) {

    fileInput.addEventListener("change", (e) => {

      const file = e.target.files[0];

      if (file) handleSegFile(file);

    });

  }



  if (dropzone) {

    dropzone.addEventListener("dragover", (e) => {

      e.preventDefault();

      dropzone.classList.add("dragover");

    });

    dropzone.addEventListener("dragleave", () => {

      dropzone.classList.remove("dragover");

    });

    dropzone.addEventListener("drop", (e) => {

      e.preventDefault();

      dropzone.classList.remove("dragover");

      if (e.dataTransfer.files.length > 0) {

        handleSegFile(e.dataTransfer.files[0]);

      }

    });

  }



  // Run Inference button

  const runBtn = document.getElementById("btn-seg-run");

  if (runBtn) {

    runBtn.addEventListener("click", runSegmentationInference);

  }



  // Opacity Slider

  const opacitySlider = document.getElementById("seg-opacity-slider");

  const opacityVal = document.getElementById("seg-opacity-val");

  if (opacitySlider) {

    opacitySlider.addEventListener("input", () => {

      segState.opacity = parseInt(opacitySlider.value, 10) / 100.0;

      if (opacityVal) opacityVal.textContent = `${opacitySlider.value}%`;

      renderSegmentationCanvas();

    });

  }



  // View Mode buttons

  const btnOverlay = document.getElementById("btn-view-mode-overlay");

  const btnSbs = document.getElementById("btn-view-mode-sbs");

  const btnPredOnly = document.getElementById("btn-view-mode-predonly");



  if (btnOverlay && btnSbs && btnPredOnly) {

    btnOverlay.addEventListener("click", () => {

      segState.viewMode = "overlay";

      btnOverlay.classList.add("active");

      btnSbs.classList.remove("active");

      btnPredOnly.classList.remove("active");

      const sbsContainer = document.getElementById("seg-sbs-container");

      if (sbsContainer) sbsContainer.style.display = "none";

      renderSegmentationCanvas();

    });

    btnSbs.addEventListener("click", () => {

      segState.viewMode = "sbs";

      btnSbs.classList.add("active");

      btnOverlay.classList.remove("active");

      btnPredOnly.classList.remove("active");

      const sbsContainer = document.getElementById("seg-sbs-container");

      if (sbsContainer) sbsContainer.style.display = "flex";

      renderSegmentationCanvas();

    });

    btnPredOnly.addEventListener("click", () => {

      segState.viewMode = "predonly";

      btnPredOnly.classList.add("active");

      btnOverlay.classList.remove("active");

      btnSbs.classList.remove("active");

      const sbsContainer = document.getElementById("seg-sbs-container");

      if (sbsContainer) sbsContainer.style.display = "none";

      renderSegmentationCanvas();

    });

  }



  // Toggle Scale (Fit vs Native 1:1)

  const toggleScaleBtn = document.getElementById("btn-seg-toggle-scale");

  if (toggleScaleBtn) {

    toggleScaleBtn.addEventListener("click", () => {

      const canvas = document.getElementById("seg-canvas");

      const gtCanvas = document.getElementById("seg-gt-canvas");

      if (segState.scaleMode === "fit") {

        segState.scaleMode = "native";

        toggleScaleBtn.textContent = "1:1 Native Resolution";

        toggleScaleBtn.classList.add("btn-primary");

        toggleScaleBtn.classList.remove("btn-secondary");

        if (canvas) {

          canvas.style.maxWidth = "none";

          canvas.style.width = "1600px";

          canvas.style.height = "256px";

        }

        if (gtCanvas) {

          gtCanvas.style.maxWidth = "none";

          gtCanvas.style.width = "1600px";

          gtCanvas.style.height = "256px";

        }

      } else {

        segState.scaleMode = "fit";

        toggleScaleBtn.textContent = "Fit Width";

        toggleScaleBtn.classList.remove("btn-primary");

        toggleScaleBtn.classList.add("btn-secondary");

        if (canvas) {

          canvas.style.maxWidth = "100%";

          canvas.style.width = "";

          canvas.style.height = "auto";

        }

        if (gtCanvas) {

          gtCanvas.style.maxWidth = "100%";

          gtCanvas.style.width = "";

          gtCanvas.style.height = "auto";

        }

      }

    });

  }



  // Layer Checkboxes

  const layerBase = document.getElementById("seg-layer-base");

  const layerPred = document.getElementById("seg-layer-pred");

  const layerGt = document.getElementById("seg-layer-gt");

  if (layerBase) layerBase.addEventListener("change", () => { segState.showBase = layerBase.checked; renderSegmentationCanvas(); });

  if (layerPred) layerPred.addEventListener("change", () => { segState.showPred = layerPred.checked; renderSegmentationCanvas(); });

  if (layerGt) layerGt.addEventListener("change", () => { segState.showGt = layerGt.checked; renderSegmentationCanvas(); });



  // Class Filter Checkboxes

  [1, 2, 3, 4].forEach(clsId => {

    const el = document.getElementById(`seg-cls-${clsId}`);

    if (el) {

      el.addEventListener("change", () => {

        segState.showClasses[clsId] = el.checked;

        renderSegmentationCanvas();

      });

    }

  });



  // Initial load of samples

  loadSegmentationSamples("all");

}



async function loadSegmentationSamples(filter = "all") {

  const sampleSelect = document.getElementById("seg-sample-select");

  if (!sampleSelect) return;



  sampleSelect.innerHTML = '<option value="">Loading test samples...</option>';



  try {

    const res = await fetch(`/api/segmentation/samples?class=${encodeURIComponent(filter)}&limit=60`);

    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const data = await res.json();

    segState.samples = data.samples || [];



    if (segState.samples.length === 0) {

      sampleSelect.innerHTML = '<option value="">No samples found for filter</option>';

      return;

    }



    sampleSelect.innerHTML = '<option value="">-- Select Severstal Test Strip --</option>' +

      segState.samples.map(s => {

        const desc = s.defect_combination === "clean" ? "Clean / Prime" : `Defects: ${s.defect_combination} (${s.total_defect_pixels.toLocaleString()} px)`;

        return `<option value="${s.sample_id}">${s.sample_id} [${desc}]</option>`;

      }).join("");



    // Automatically select the first sample with defect for an instant preview if none loaded

    const firstDefective = segState.samples.find(s => s.has_defect) || segState.samples[0];

    if (firstDefective && !segState.currentSampleId) {

      sampleSelect.value = firstDefective.sample_id;

      loadSegmentationSample(firstDefective.sample_id);

    }

  } catch (err) {

    console.error("Failed to load segmentation samples:", err);

    sampleSelect.innerHTML = '<option value="">Failed to load samples</option>';

  }

}



async function loadSegmentationSample(sampleId) {

  segState.currentSampleId = sampleId;

  segState.currentImageBase64 = null;

  segState.predColorImage = null;

  segState.predMaskData = null;



  const metaFilename = document.getElementById("seg-meta-filename");

  const metaGt = document.getElementById("seg-meta-gt");

  const statusText = document.getElementById("seg-status-text");

  const placeholder = document.getElementById("seg-canvas-placeholder");



  if (metaFilename) metaFilename.textContent = sampleId;

  if (statusText) statusText.textContent = "READY FOR INFERENCE";

  if (placeholder) placeholder.style.display = "none";



  // Reset metrics

  resetSegMetricsTable();



  // Load Base Image

  const imgUrl = `/api/segmentation/samples/image?id=${encodeURIComponent(sampleId)}`;

  const img = new Image();

  img.crossOrigin = "anonymous";

  img.onload = () => {

    segState.baseImage = img;

    renderSegmentationCanvas();

    if (window.location.search.includes("autorun=true") && !segState.predColorImage && !segState.isRunning) {

      setTimeout(runSegmentationInference, 300);

    }

  };

  img.src = imgUrl;



  // Load Ground Truth Mask

  try {

    const res = await fetch(`/api/segmentation/samples/ground_truth?id=${encodeURIComponent(sampleId)}`);

    if (res.ok) {

      const gtData = await res.json();

      segState.gtMaskData = gtData;

      if (metaGt) {

        metaGt.textContent = gtData.has_defect ? `${gtData.active_classes.join(", ")} (${gtData.total_defect_pixels.toLocaleString()} px)` : "Clean Surface";

        metaGt.style.color = gtData.has_defect ? "var(--status-review)" : "var(--status-pass)";

      }



      // Populate GT Comparison Box

      const gtBox = document.getElementById("seg-gt-comparison-box");

      const gtTotPx = document.getElementById("seg-gt-tot-px");

      const gtTotPct = document.getElementById("seg-gt-tot-pct");

      const gtClasses = document.getElementById("seg-gt-classes");



      if (gtBox) gtBox.style.display = "block";

      if (gtTotPx) gtTotPx.textContent = `${gtData.total_defect_pixels.toLocaleString()} px`;

      if (gtTotPct) gtTotPct.textContent = `${gtData.defect_percentage.toFixed(3)}%`;

      if (gtClasses) gtClasses.textContent = gtData.active_classes.length > 0 ? gtData.active_classes.join(", ") : "None (Prime Steel)";



      // Load GT colored mask image

      if (gtData.color_mask_base64) {

        const gtImg = new Image();

        gtImg.onload = () => {

          segState.gtColorImage = gtImg;

          renderSegmentationCanvas();

        };

        gtImg.src = `data:image/png;base64,${gtData.color_mask_base64}`;

      }

    } else {

      segState.gtMaskData = null;

      segState.gtColorImage = null;

      if (metaGt) metaGt.textContent = "Not available";

      const gtBox = document.getElementById("seg-gt-comparison-box");

      if (gtBox) gtBox.style.display = "none";

    }

  } catch (err) {

    console.warn("Failed to load ground truth for sample:", err);

  }

}



function handleSegFile(file) {

  segState.currentSampleId = null;

  segState.predColorImage = null;

  segState.gtColorImage = null;

  segState.predMaskData = null;

  segState.gtMaskData = null;



  const metaFilename = document.getElementById("seg-meta-filename");

  const metaGt = document.getElementById("seg-meta-gt");

  const statusText = document.getElementById("seg-status-text");

  const placeholder = document.getElementById("seg-canvas-placeholder");

  const gtBox = document.getElementById("seg-gt-comparison-box");



  if (metaFilename) metaFilename.textContent = file.name;

  if (metaGt) metaGt.textContent = "Custom File (No GT)";

  if (statusText) statusText.textContent = "CUSTOM WORKPIECE READY";

  if (placeholder) placeholder.style.display = "none";

  if (gtBox) gtBox.style.display = "none";



  resetSegMetricsTable();



  const reader = new FileReader();

  reader.onload = (e) => {

    segState.currentImageBase64 = e.target.result;

    const img = new Image();

    img.onload = () => {

      segState.baseImage = img;

      renderSegmentationCanvas();

    };

    img.src = e.target.result;

  };

  reader.readAsDataURL(file);

}



async function runSegmentationInference() {

  if (segState.isRunning) return;

  if (!segState.currentSampleId && !segState.currentImageBase64) {

    alert("Please select a Severstal sample or upload a strip image first.");

    return;

  }



  const runBtn = document.getElementById("btn-seg-run");

  const statusText = document.getElementById("seg-status-text");

  const latencyText = document.getElementById("seg-latency-text");

  const deviceText = document.getElementById("seg-device-text");

  const saveDbCheck = document.getElementById("seg-save-db-check");



  segState.isRunning = true;

  if (runBtn) {

    runBtn.disabled = true;

    runBtn.innerHTML = `

      <span class="spinner-sm" style="display:inline-block; width:12px; height:12px; border:2px solid #fff; border-top-color:transparent; border-radius:50%; animation:spin 0.6s linear infinite; margin-right:6px;"></span>

      <span>INFERRING SEGMENTATION...</span>

    `;

  }

  if (statusText) statusText.textContent = "EXECUTING U-NET INFERENCE...";



  const payload = {

    save_to_db: saveDbCheck ? saveDbCheck.checked : true,

    operator_notes: "Severstal U-Net dashboard inspection"

  };



  if (segState.currentSampleId) {

    payload.sample_id = segState.currentSampleId;

  } else {

    payload.image_base64 = segState.currentImageBase64;

    payload.filename = document.getElementById("seg-meta-filename")?.textContent || "uploaded_strip.png";

  }



  try {

    const res = await fetch("/api/segmentation/run", {

      method: "POST",

      headers: { "Content-Type": "application/json" },

      body: JSON.stringify(payload)

    });



    const data = await res.json();

    if (!res.ok || data.status === "ERROR") {

      throw new Error(data.message || `HTTP ${res.status}`);

    }



    segState.predMaskData = data;



    // Update UI telemetry

    if (statusText) statusText.textContent = "INSPECTION COMPLETE";

    if (latencyText) latencyText.textContent = `${data.inference_time_ms} ms (${(1000.0 / data.inference_time_ms).toFixed(1)} FPS)`;

    if (deviceText) deviceText.textContent = data.device_used.toUpperCase();



    // Update Verdict Card

    const verdictCard = document.getElementById("seg-verdict-card");

    const verdictTitle = document.getElementById("seg-verdict-title");

    const verdictSub = document.getElementById("seg-verdict-sub");



    if (data.has_defect) {

      if (verdictTitle) {

        verdictTitle.textContent = "DEFECT DETECTED";

        verdictTitle.style.color = "var(--status-reject)";

      }

      if (verdictSub) {

        verdictSub.textContent = `Classes: ${data.active_classes.join(", ")} | Total Defect: ${data.total_defect_pixels.toLocaleString()} px (${data.defect_percentage.toFixed(3)}%)`;

      }

      if (verdictCard) verdictCard.style.borderColor = "var(--status-reject)";

    } else {

      if (verdictTitle) {

        verdictTitle.textContent = "PRIME SURFACE / PASS";

        verdictTitle.style.color = "var(--status-pass)";

      }

      if (verdictSub) {

        verdictSub.textContent = `Zero defect pixels identified. Surface meets quality specification.`;

      }

      if (verdictCard) verdictCard.style.borderColor = "var(--status-pass)";

    }



    // Update 4-Class Table

    const breakdown = data.class_breakdown || [];

    const cMap = {};

    breakdown.forEach(c => { cMap[c.class_id] = c; });



    [1, 2, 3, 4].forEach(clsId => {

      const c = cMap[clsId] || { pixel_count: 0, area_percentage: 0 };

      const pxEl = document.getElementById(`seg-val-c${clsId}-px`);

      const pctEl = document.getElementById(`seg-val-c${clsId}-pct`);

      if (pxEl) pxEl.textContent = c.pixel_count.toLocaleString();

      if (pctEl) pctEl.textContent = `${c.area_percentage.toFixed(3)}%`;

    });



    const totPxEl = document.getElementById("seg-val-tot-px");

    const totPctEl = document.getElementById("seg-val-tot-pct");

    if (totPxEl) totPxEl.textContent = data.total_defect_pixels.toLocaleString();

    if (totPctEl) totPctEl.textContent = `${data.defect_percentage.toFixed(3)}%`;



    // Load predicted color mask

    if (data.color_mask_base64) {

      const predImg = new Image();

      predImg.onload = () => {

        segState.predColorImage = predImg;

        renderSegmentationCanvas();

      };

      predImg.src = `data:image/png;base64,${data.color_mask_base64}`;

    }



  } catch (err) {

    console.error("Segmentation inference error:", err);

    if (statusText) statusText.textContent = `ERROR: ${err.message}`;

    alert(`Segmentation inference failed: ${err.message}`);

  } finally {

    segState.isRunning = false;

    if (runBtn) {

      runBtn.disabled = false;

      runBtn.innerHTML = `

        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">

          <polygon points="5 3 19 12 5 21 5 3"></polygon>

        </svg>

        <span>RUN U-NET SEGMENTATION</span>

      `;

    }

  }

}



function resetSegMetricsTable() {

  [1, 2, 3, 4].forEach(clsId => {

    const pxEl = document.getElementById(`seg-val-c${clsId}-px`);

    const pctEl = document.getElementById(`seg-val-c${clsId}-pct`);

    if (pxEl) pxEl.textContent = "0";

    if (pctEl) pctEl.textContent = "0.00%";

  });

  const totPxEl = document.getElementById("seg-val-tot-px");

  const totPctEl = document.getElementById("seg-val-tot-pct");

  if (totPxEl) totPxEl.textContent = "0";

  if (totPctEl) totPctEl.textContent = "0.00%";



  const verdictTitle = document.getElementById("seg-verdict-title");

  const verdictSub = document.getElementById("seg-verdict-sub");

  const verdictCard = document.getElementById("seg-verdict-card");

  if (verdictTitle) {

    verdictTitle.textContent = "--";

    verdictTitle.style.color = "var(--text-secondary)";

  }

  if (verdictSub) verdictSub.textContent = "Awaiting inference execution";

  if (verdictCard) verdictCard.style.borderColor = "var(--border-subtle)";

}



function renderSegmentationCanvas() {

  const canvas = document.getElementById("seg-canvas");

  if (!canvas) return;

  const ctx = canvas.getContext("2d");



  // Clear canvas

  ctx.clearRect(0, 0, canvas.width, canvas.height);



  // If in side-by-side mode, also render GT canvas

  const gtCanvas = document.getElementById("seg-gt-canvas");

  let gtCtx = null;

  if (gtCanvas && segState.viewMode === "sbs") {

    gtCtx = gtCanvas.getContext("2d");

    gtCtx.clearRect(0, 0, gtCanvas.width, gtCanvas.height);

  }



  // 1. Draw Base Steel Strip Image

  if (segState.showBase && segState.baseImage) {

    if (segState.viewMode !== "predonly") {

      ctx.drawImage(segState.baseImage, 0, 0, canvas.width, canvas.height);

    }

    if (gtCtx && segState.baseImage) {

      gtCtx.drawImage(segState.baseImage, 0, 0, gtCanvas.width, gtCanvas.height);

    }

  } else if (segState.viewMode === "predonly") {

    ctx.fillStyle = "#0a0f1d";

    ctx.fillRect(0, 0, canvas.width, canvas.height);

  }



  // 2. Draw Predicted Mask on Main Canvas

  if (segState.showPred && segState.predColorImage) {

    ctx.save();

    ctx.globalAlpha = segState.viewMode === "predonly" ? 1.0 : segState.opacity;



    const allChecked = segState.showClasses[1] && segState.showClasses[2] && segState.showClasses[3] && segState.showClasses[4];

    if (allChecked) {

      ctx.drawImage(segState.predColorImage, 0, 0, canvas.width, canvas.height);

    } else {

      // Filter out unchecked classes using offscreen canvas

      const offCanvas = document.createElement("canvas");

      offCanvas.width = canvas.width;

      offCanvas.height = canvas.height;

      const offCtx = offCanvas.getContext("2d");

      offCtx.drawImage(segState.predColorImage, 0, 0, canvas.width, canvas.height);

      const imgData = offCtx.getImageData(0, 0, canvas.width, canvas.height);

      const d = imgData.data;



      for (let i = 0; i < d.length; i += 4) {

        const r = d[i], g = d[i+1], b = d[i+2];

        if (r === 0 && g === 0 && b === 0) continue;



        let cls = 0;

        if (r > 200 && g > 150 && b < 100) cls = 1;      // C1 Amber

        else if (r < 50 && g > 150 && b > 180) cls = 2;  // C2 Cyan

        else if (r > 200 && g < 100 && b < 100) cls = 3; // C3 Red

        else if (r < 100 && g > 150 && b < 120) cls = 4; // C4 Green



        if (cls > 0 && !segState.showClasses[cls]) {

          d[i+3] = 0; // Transparent

        }

      }

      offCtx.putImageData(imgData, 0, 0);

      ctx.drawImage(offCanvas, 0, 0);

    }

    ctx.restore();

  }



  // 3. Ground Truth Handling

  if (segState.viewMode === "sbs" && gtCtx && segState.showGt && segState.gtColorImage) {

    gtCtx.save();

    gtCtx.globalAlpha = segState.opacity;

    gtCtx.drawImage(segState.gtColorImage, 0, 0, gtCanvas.width, gtCanvas.height);

    gtCtx.restore();

  } else if (segState.viewMode === "overlay" && segState.showGt && segState.gtColorImage && !segState.predColorImage) {

    ctx.save();

    ctx.globalAlpha = segState.opacity;

    ctx.drawImage(segState.gtColorImage, 0, 0, canvas.width, canvas.height);

    ctx.restore();

  }

}





/* ----------------------------------------------------------

   ExecuTorch Bounded Advisory Integration (Phase 5)

   ---------------------------------------------------------- */

async function requestExecuTorchAdvisory() {

  const advContent = document.getElementById("advisory-content");

  const advBtn = document.getElementById("btn-request-advisory");

  const advBadge = document.getElementById("advisory-badge");



  if (!state.lastInferenceData) {

    if (advContent) advContent.innerHTML = `<p style="color: var(--text-muted);">Run a YOLO inspection first to generate the grounded ExecuTorch advisory plan.</p>`;

    return;

  }



  if (advBtn) {
    advBtn.disabled = true;
    advBtn.innerHTML = `<span>Querying ExecuTorch...</span>`;
  }
  if (advBadge) {
    advBadge.textContent = "INFERRING ADVISORY...";
    advBadge.className = "badge-verdict pending";
  }
  if (advContent) {
    advContent.dataset.status = "loading";
    advContent.innerHTML = `
      <div style="padding: var(--space-4); text-align: center; color: var(--cyan-primary);">
        <p style="font-weight: 600;">Executing bounded local inference with Llama 3.2 1B Instruct on CPU...</p>
        <p style="font-size: var(--text-xs); color: var(--text-muted); margin-top: 4px;">Zero cloud transmission · Deterministic ground truth · XNNPACK CPU</p>
      </div>`;
  }



  try {

    const summaryPayload = {};

    ["source", "status", "detections", "filename", "model_name", "model_version", "model_key", "sha256", "image_width", "image_height"].forEach(k => {

      if (state.lastInferenceData[k] !== undefined) summaryPayload[k] = state.lastInferenceData[k];

    });



    const sumRes = await fetch("/api/intelligence/summarize", {

      method: "POST",

      headers: { "Content-Type": "application/json" },

      body: JSON.stringify(summaryPayload)

    });

    if (!sumRes.ok) throw new Error("Could not compute deterministic summary for advisory grounding.");

    const canonicalSummary = await sumRes.json();



    const advRes = await fetch("/api/intelligence/advisory", {

      method: "POST",

      headers: { "Content-Type": "application/json" },

      body: JSON.stringify({ summary: canonicalSummary })

    });

    if (!advRes.ok) throw new Error(`Advisory HTTP error: ${advRes.status}`);

    const advisory = await advRes.json();



    renderAdvisoryResult(advisory);



  } catch (err) {
    if (advBadge) {
      advBadge.textContent = "QUERY FAILED";
      advBadge.className = "badge-verdict reject";
    }
    if (advContent) {
      advContent.dataset.status = "error";
      advContent.innerHTML = `<div style="padding: var(--space-3); color: var(--status-fail);"><p><strong>Advisory Error:</strong> ${err.message}</p></div>`;
    }
  } finally {

    if (advBtn) {

      advBtn.disabled = false;

      advBtn.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-right: 4px;"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg><span>Request Llama Advisory</span>`;

    }

  }

}



function renderAdvisoryResult(adv) {

  const advContent = document.getElementById("advisory-content");

  const advBadge = document.getElementById("advisory-badge");

  if (!advContent) return;



  const isLlm = adv.llm_used === true;



  if (advBadge) {

    if (isLlm) {

      advBadge.textContent = "EXECUTORCH ACTIVE";

      advBadge.className = "badge-verdict pass";

    } else {

      advBadge.textContent = `FALLBACK (${adv.reason_code || 'RULE-BASED'})`;

      advBadge.className = "badge-verdict warning";

    }

  }



  let recsHtml = (adv.recommendations || []).map((r, i) => `

    <li style="margin-bottom: 6px; padding: 8px 12px; background: ${i === 0 ? 'rgba(0, 210, 255, 0.08)' : 'var(--bg-panel)'}; border-left: 3px solid ${i === 0 ? 'var(--cyan-primary)' : 'var(--border-subtle)'}; border-radius: 2px;">

      <span style="font-weight: 600; color: ${i === 0 ? 'var(--cyan-primary)' : 'var(--text-primary)'};">${i === 0 ? '★ Primary Disposition: ' : '• '}</span>${r}

    </li>`).join("");



  let warningsHtml = (adv.warnings || []).map(w => `<li style="font-size: 11px; color: var(--text-muted); margin-bottom: 3px;">⚠ ${w}</li>`).join("");



  advContent.innerHTML = `

    <div style="display: flex; flex-direction: column; gap: var(--space-3);">

      <!-- Status pill banner -->

      <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 12px; background: var(--bg-panel); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle); flex-wrap: wrap; gap: 8px;">

        <div>

          <span style="font-size: 11px; color: var(--text-muted);">Advisory Source: </span>

          <span style="font-weight: 700; color: ${isLlm ? 'var(--status-pass)' : 'var(--status-warn)'};">${adv.source}</span>

          ${adv.model_id ? `<span style="font-size: 11px; color: var(--text-muted); margin-left: 8px;">(${adv.model_id})</span>` : ''}

        </div>

        <div>

          <span style="font-size: 11px; color: var(--text-muted);">Quality Verdict: </span>

          <span style="font-weight: 700; color: ${adv.quality_verdict === 'REJECT' ? 'var(--status-fail)' : 'var(--status-pass)'};">${adv.quality_verdict}</span>

          <span style="font-size: 10px; color: var(--text-muted); margin-left: 4px;">(Immutable)</span>

        </div>

      </div>



      <!-- Action Plan Recommendations -->

      <div>

        <h4 style="font-size: 12px; text-transform: uppercase; color: var(--text-secondary); margin-bottom: 6px;">Operator Action Plan (Bounded Model Selection)</h4>

        <ul style="list-style: none; padding: 0; margin: 0;">

          ${recsHtml}

        </ul>

      </div>



      <!-- Grounded Facts -->

      <div style="font-size: 11px; color: var(--text-secondary); background: var(--bg-panel); padding: 8px; border-radius: var(--radius-sm);">

        <strong>Rule Grounding:</strong> ${adv.explanation || 'Evaluated against GC10 Quality Rules.'}

      </div>



      <!-- Containment Disclaimers -->

      <div>

        <h5 style="font-size: 10px; text-transform: uppercase; color: var(--text-muted); margin-bottom: 4px;">Anti-Hallucination Guardrails & Containment</h5>

        <ul style="list-style: none; padding: 0; margin: 0;">

          ${warningsHtml}

        </ul>

      </div>

    </div>

  `;

}

