/* Deterministic GC10 summaries only. No access to segmentation state or history writes. */
(function (root) {
  "use strict";
  function createController({ document, fetch: request }) {
    let generation = 0;
    const host = () => document.getElementById("intelligence-content");
    function text(parent, tag, value, className) {
      const node = document.createElement(tag);
      node.textContent = String(value);
      if (className) node.className = className;
      parent.appendChild(node);
      return node;
    }
    function message(status, value) {
      const target = host();
      if (!target) return;
      target.replaceChildren();
      target.dataset.status = status;
      text(target, "p", value);
    }
    function reset(value = "Run YOLO on a steel image to view its inspection summary.") {
      generation += 1;
      message("empty", value);
    }
    function render(summary) {
      if (!summary || summary.engine !== "deterministic" || summary.llm_used !== false ||
          summary.segmentation_fused !== false || summary.taxonomy !== "GC10-DET" ||
          !summary.assessment || !Array.isArray(summary.evidence) || !Array.isArray(summary.warnings)) {
        throw new Error("Unsupported summary contract");
      }
      const target = host();
      if (!target) return;
      target.replaceChildren();
      target.dataset.status = "ready";
      text(target, "h3", `Quality verdict: ${summary.assessment.status}`);
      text(target, "p", `Source: ${summary.source} | GC10-DET | Policy: ${summary.policy_version}`);
      const identity = summary.provenance || {};
      text(target, "p", `Image: ${identity.filename || "not supplied"} | Model: ${identity.model_name || "not supplied"} ${identity.model_version || ""}`);
      text(target, "p", `Image geometry: ${identity.image_width ?? "unknown"} × ${identity.image_height ?? "unknown"} px | Checkpoint SHA-256: ${identity.sha256 || "not supplied"}`, "intelligence-provenance");
      text(target, "p", "Severity is configured rule policy. Confidence is model output, not severity. Areas are bounding-box pixels, not physical defect area.");
      text(target, "h4", `Detection evidence (${summary.total_defects})`);
      if (!summary.evidence.length) text(target, "p", "No detections reported. This does not establish a defect-free surface.");
      else {
        const wrapper = document.createElement("div");
        wrapper.className = "intelligence-table-scroll";
        target.appendChild(wrapper);
        const table = document.createElement("table");
        table.className = "industrial-table";
        wrapper.appendChild(table);
        const head = document.createElement("thead"); table.appendChild(head);
        const row = document.createElement("tr"); head.appendChild(row);
        ["Defect type", "Policy severity", "Confidence", "Box [x1, y1, x2, y2] px", "Box area px²"].forEach(v => text(row, "th", v));
        const body = document.createElement("tbody"); table.appendChild(body);
        summary.evidence.forEach(d => {
          const tr = document.createElement("tr"); body.appendChild(tr);
          [ `${d.defect_type} (${d.raw_label})`, d.severity,
            d.confidence == null ? "Not supplied" : `${(d.confidence * 100).toFixed(1)}%`,
            d.bbox_xyxy ? d.bbox_xyxy.join(", ") : "Not supplied", d.area_px
          ].forEach(v => text(tr, "td", v));
        });
      }
      text(target, "h4", "Rule evidence");
      (summary.assessment.rules_applied || []).forEach(rule => text(target, "p", `${rule.rule_id} · ${rule.verdict}: ${rule.reason}`));
      text(target, "h4", "Provenance and limitations");
      const warnings = document.createElement("ul"); target.appendChild(warnings);
      summary.warnings.forEach(warning => text(warnings, "li", warning));
      text(target, "p", "U-Net segmentation is separate. Image provenance and coordinate compatibility have not been verified for fusion.");
      text(target, "h4", "Recommended operator action");
      text(target, "p", summary.recommended_action);
    }
    async function summarize(result) {
      const ticket = ++generation;
      // Only a successful YOLO result can enter this dashboard panel.
      if (!result || result.status !== "SUCCESS" || result.source !== "AI_INFERENCE" || !Array.isArray(result.detections)) {
        message("error", "Inspection summary unavailable: a successful YOLO result is required.");
        return;
      }
      // Explicit allowlist: never transmit U-Net masks, generated text or history state.
      const payload = {};
      ["source", "status", "detections", "filename", "model_name", "model_version", "model_key", "sha256", "image_width", "image_height"].forEach(key => {
        if (result[key] !== undefined) payload[key] = result[key];
      });
      message("loading", "Summarizing current YOLO evidence…");
      const abort = new AbortController();
      const timer = setTimeout(() => abort.abort(), 15000);
      try {
        const response = await request("/api/intelligence/summarize", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload), signal: abort.signal
        });
        const summary = await response.json();
        if (ticket !== generation) return;
        if (!response.ok) throw new Error(summary.error || `HTTP ${response.status}`);
        render(summary);
      } catch (error) {
        if (ticket === generation) message("error", `Inspection summary unavailable: ${error.message}. Detection results remain available in Inspect.`);
      } finally {
        clearTimeout(timer);
      }
    }
    return { reset, summarize };
  }
  root.InspectionIntelligence = { createController };
})(globalThis);
