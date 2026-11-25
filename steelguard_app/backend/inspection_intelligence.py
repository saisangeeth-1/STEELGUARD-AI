"""Portable deterministic evidence summary. No model, storage or network side effects."""
from collections import Counter
from copy import deepcopy
import math

from quality_engine import assess_quality, CONFIGURED_RULES

GC10_NAMES = dict(zip(
    ("1_chongkong", "2_hanfeng", "3_yueyawan", "4_shuiban", "5_youban",
     "6_siban", "7_yiwu", "8_yahen", "9_zhehen", "10_yaozhe"),
    ("Punching Hole", "Welding Line", "Crescent Gap", "Water Spot", "Oil Spot",
     "Silk Spot", "Inclusion / Foreign Object", "Indentation / Rolled-in Scale", "Crease", "Waist Folding")
))
GC10_NAMES["10_yaozhed"] = "Waist Folding"


def summarize_inspection(payload):
    """Summarize GC10 evidence; never treat missing inference as a clean inspection.

    Input is an existing YOLO result (source, status, detections). This endpoint
    summarizes submitted evidence; it does not authenticate inference provenance.
    Severstal segmentation is deliberately not fused into the GC10 verdict.
    """
    if not isinstance(payload, dict):
        raise ValueError("Expected an inspection object")
    source = payload.get("source")
    if source not in ("AI_INFERENCE", "GROUND_TRUTH"):
        raise ValueError("source must be AI_INFERENCE or GROUND_TRUTH")
    if payload.get("taxonomy", "GC10-DET") != "GC10-DET":
        raise ValueError("Only GC10-DET detection evidence is supported")
    detections = payload.get("detections")
    if not isinstance(detections, list) or len(detections) > 10000:
        raise ValueError("detections must be an explicit list of at most 10000 instances")
    if source == "AI_INFERENCE" and payload.get("status") != "SUCCESS":
        raise ValueError("Successful inference is required")
    evidence = deepcopy(detections)
    for d in evidence:
        if not isinstance(d, dict):
            raise ValueError("Each detection must be an object")
        label = d.get("raw_label", d.get("class_name"))
        if not isinstance(label, str) or not label.strip():
            raise ValueError("Each detection requires a class label")
        d["raw_label"] = label.strip()
        box = d.get("bbox_xyxy")
        if box is not None:
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                raise ValueError("bbox_xyxy must contain four coordinates")
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in box):
                raise ValueError("Bounding box coordinates must be finite and nonnegative")
            if box[2] < box[0] or box[3] < box[1]:
                raise ValueError("Bounding box coordinates must be ordered")
        area = d.get("area_px")
        if area is not None and (isinstance(area, bool) or not isinstance(area, (int, float)) or not math.isfinite(area) or area < 0):
            raise ValueError("area_px must be finite and nonnegative")
        if area is None and box is None:
            raise ValueError("Measured area or bounding box is required")
        confidence = d.get("confidence")
        if confidence is not None and (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
                                       or not math.isfinite(confidence) or not 0 <= confidence <= 1):
            raise ValueError("confidence must be finite and between zero and one")
        if area is None:
            d["area_px"] = (box[2] - box[0]) * (box[3] - box[1])
    assessment = assess_quality(evidence)
    known = {label for rule in CONFIGURED_RULES for label in rule["defect_classes"]}
    unknown = sorted({d["raw_label"] for d in evidence} - known)
    # Legacy rules can omit unknown classes when a known class also triggers.
    if unknown and assessment["status"] == "PASS":
        assessment["status"] = "REVIEW REQUIRED"
    if unknown and "QR-GEN" not in assessment["rule_ids"]:
        reason = "Unmapped classes require operator review: " + ", ".join(unknown)
        assessment["rule_ids"].append("QR-GEN")
        assessment["rules_applied"].append({"rule_id": "QR-GEN", "title": "Unmapped evidence",
                                            "verdict": "REVIEW REQUIRED", "reason": reason,
                                            "defects": unknown})
        assessment["summary"] += " | " + reason
    warnings = [
        "Engineering policy only; acceptance thresholds are not a certified steel standard.",
        "Pixel areas are not physical measurements; camera calibration is required.",
        "Submitted evidence provenance is not independently verified."
    ]
    if not evidence:
        warnings.append("No detections does not establish a defect-free surface.")
    if source == "GROUND_TRUTH":
        warnings.append("Dataset annotations, not model predictions.")
    if unknown:
        warnings.append("Unmapped classes require operator review: " + ", ".join(unknown))
    severity_by_label = {label: rule["severity"] for rule in CONFIGURED_RULES for label in rule["defect_classes"]}
    evidence_rows = [{
        "index": index + 1, "raw_label": d["raw_label"],
        "defect_type": GC10_NAMES.get(d["raw_label"], d["raw_label"]),
        "severity": severity_by_label.get(d["raw_label"], "UNCLASSIFIED"),
        "bbox_xyxy": d.get("bbox_xyxy"), "area_px": d["area_px"],
        "confidence": d.get("confidence")
    } for index, d in enumerate(evidence)]
    actions = {
        "REJECT": "Hold the workpiece for operator disposition under the configured rejection policy.",
        "REVIEW REQUIRED": "Route the workpiece to operator review before disposition.",
        "PASS": "Confirm image coverage and applicable acceptance policy before releasing the workpiece."
    }
    # Only report scalar identity metadata, never masks or caller-provided prose.
    provenance = {key: payload[key] for key in
                  ("filename", "model_name", "model_version", "model_key", "sha256", "image_width", "image_height")
                  if key in payload and isinstance(payload[key], (str, int, float))
                  and not isinstance(payload[key], bool)
                  and (not isinstance(payload[key], float) or math.isfinite(payload[key]))}
    return {
        "schema_version": "1.1", "engine": "deterministic", "policy_version": "gc10-qr-v1",
        "source": source, "taxonomy": "GC10-DET", "total_defects": len(evidence),
        "class_counts": dict(sorted(Counter(d["raw_label"] for d in evidence).items())),
        "assessment": assessment, "warnings": warnings,
        "evidence": evidence_rows, "provenance": provenance,
        "segmentation_fused": False, "llm_used": False,
        "recommended_action": actions[assessment["status"]]
    }
