"""
STEELGUARD AI - Quality Assessment Engine
Deterministic, rule-based quality evaluation for industrial steel strip inspection.
Explicitly decouples defect detection from industrial acceptance thresholds.
"""

from typing import List, Dict, Any, Tuple

# Pre-configured Industrial Steel Surface Quality Acceptance Rules
CONFIGURED_RULES = [
    {
        "id": "QR-01",
        "title": "Structural Breach Protection",
        "defect_classes": ["1_chongkong", "3_yueyawan"],
        "default_verdict": "REJECT",
        "severity": "CRITICAL",
        "description": "Perforations (punching holes) or strip edge tears (crescent gaps) severely compromise coil tensile stability and trigger immediate rejection to prevent mill line breaks."
    },
    {
        "id": "QR-02",
        "title": "Joint & Buckling Anomaly",
        "defect_classes": ["2_hanfeng", "10_yaozhe", "10_yaozhed"],
        "default_verdict": "REVIEW REQUIRED",
        "severity": "HIGH",
        "description": "Welding seams or transverse strip waist folding require mandatory operator review for shear trimming prior to customer shipment."
    },
    {
        "id": "QR-03",
        "title": "Inclusion & Embedded Particle Tolerance",
        "defect_classes": ["7_yiwu", "8_yahen"],
        "default_verdict": "REVIEW REQUIRED",
        "area_threshold_px": 5000,
        "reject_on_large_area": True,
        "severity": "HIGH",
        "description": "Foreign inclusions and rolled-in scale exceeding 5,000 px² induce stress concentration, requiring rejection; smaller instances require metallurgical review."
    },
    {
        "id": "QR-04",
        "title": "Surface Finish & Coating Integrity",
        "defect_classes": ["6_siban", "9_zhehen"],
        "default_verdict": "REVIEW REQUIRED",
        "severity": "MEDIUM",
        "description": "Silk spots and longitudinal creases degrade coating adhesion and cosmetic appearance, requiring downstream routing assessment."
    },
    {
        "id": "QR-05",
        "title": "Superficial Fluid Contamination",
        "defect_classes": ["4_shuiban", "5_youban"],
        "default_verdict": "PASS",
        "area_review_threshold_px": 30000,
        "severity": "LOW",
        "description": "Water spots and minor oil spots typically burn off during annealing; flagged for review only if extensive continuous pooling occurs."
    }
]


def assess_quality(detections: List[Dict[str, Any]], rules_enabled: bool = True) -> Dict[str, Any]:
    """
    Evaluates detected defects against validated industrial steel rules.

    Args:
        detections: List of detection or ground-truth annotation dictionaries.
                    Each dict must contain 'raw_label' or 'class_name' and optionally 'bbox_xyxy' / 'area_px'.
        rules_enabled: Boolean flag. If False, returns 'ASSESSMENT PENDING'.

    Returns:
        Structured quality decision containing status, applied rule IDs, and detailed rationale.
    """
    if not rules_enabled:
        return {
            "status": "ASSESSMENT PENDING",
            "rule_ids": [],
            "summary": "Rule engine inactive or unconfigured. Quality assessment pending evaluation.",
            "rules_applied": []
        }

    # Case 0: Clean workpiece with no defects
    if not detections or len(detections) == 0:
        return {
            "status": "PASS",
            "rule_ids": ["QR-00"],
            "summary": "Zero surface defects detected. Workpiece conforms to prime steel strip quality standard.",
            "rules_applied": [
                {
                    "rule_id": "QR-00",
                    "title": "Prime Coil Acceptance",
                    "verdict": "PASS",
                    "reason": "No defect instances identified across inspected strip surface."
                }
            ]
        }

    applied_rules = []
    verdicts = []

    # Map defect classes in current inspection
    defect_classes = set()
    for d in detections:
        label = d.get("raw_label", d.get("class_name", "")).strip()
        defect_classes.add(label)

    # Rule QR-01: Critical Structural Breach (Punching Hole / Crescent Gap)
    crit_defects = defect_classes.intersection({"1_chongkong", "3_yueyawan"})
    if crit_defects:
        defect_names = ", ".join(sorted(crit_defects))
        applied_rules.append({
            "rule_id": "QR-01",
            "title": "Structural Breach Protection",
            "verdict": "REJECT",
            "defects": sorted(crit_defects),
            "reason": f"Critical perforation / edge gap defect ({defect_names}) detected. Risk of coil breakage under rolling tension."
        })
        verdicts.append("REJECT")

    # Rule QR-02: Joint & Buckling Anomaly (Weld Seam / Waist Folding)
    weld_fold = defect_classes.intersection({"2_hanfeng", "10_yaozhe", "10_yaozhed"})
    if weld_fold:
        count = sum(1 for d in detections if d.get("raw_label", d.get("class_name")) in weld_fold)
        defect_names = ", ".join(sorted(weld_fold))
        # If multiple waist folds occur, reject; otherwise review required
        verdict = "REJECT" if ("10_yaozhe" in weld_fold or "10_yaozhed" in weld_fold) and count > 2 else "REVIEW REQUIRED"
        applied_rules.append({
            "rule_id": "QR-02",
            "title": "Joint & Buckling Anomaly",
            "verdict": verdict,
            "defects": sorted(weld_fold),
            "reason": f"Discontinuity defect ({defect_names}, count: {count}) requires station operator verification for crop shear."
        })
        verdicts.append(verdict)

    # Rule QR-03: Inclusions & Rolled Scale
    inclusions = [d for d in detections if d.get("raw_label", d.get("class_name")) in {"7_yiwu", "8_yahen"}]
    if inclusions:
        max_area = 0
        for d in inclusions:
            area = d.get("area_px")
            if area is None and "bbox_xyxy" in d:
                xmin, ymin, xmax, ymax = d["bbox_xyxy"]
                area = max(0, xmax - xmin) * max(0, ymax - ymin)
            if area and area > max_area:
                max_area = area

        verdict = "REJECT" if max_area >= 5000 else "REVIEW REQUIRED"
        applied_rules.append({
            "rule_id": "QR-03",
            "title": "Inclusion & Scale Tolerance",
            "verdict": verdict,
            "defects": ["7_yiwu" if any(d.get("raw_label", d.get("class_name")) == "7_yiwu" for d in inclusions) else "8_yahen"],
            "reason": f"Metallurgical foreign particles/scale detected (max area: {int(max_area)} px²). Threshold: 5,000 px²."
        })
        verdicts.append(verdict)

    # Rule QR-04: Surface Finish (Silk Spots / Creases)
    finish_defects = defect_classes.intersection({"6_siban", "9_zhehen"})
    if finish_defects:
        applied_rules.append({
            "rule_id": "QR-04",
            "title": "Surface Finish Integrity",
            "verdict": "REVIEW REQUIRED",
            "defects": sorted(finish_defects),
            "reason": "Silk spots or creases identified. May require regrading or secondary finishing evaluation."
        })
        verdicts.append("REVIEW REQUIRED")

    # Rule QR-05: Superficial Fluid Staining (Water / Oil spots)
    fluid_defects = [d for d in detections if d.get("raw_label", d.get("class_name")) in {"4_shuiban", "5_youban"}]
    if fluid_defects and not (crit_defects or weld_fold or inclusions or finish_defects):
        total_fluid_area = sum(
            d["area_px"] if d.get("area_px") is not None else max(0, d["bbox_xyxy"][2] - d["bbox_xyxy"][0]) * max(0, d["bbox_xyxy"][3] - d["bbox_xyxy"][1])
            for d in fluid_defects if "bbox_xyxy" in d or "area_px" in d
        )
        verdict = "REVIEW REQUIRED" if total_fluid_area > 30000 else "PASS"
        applied_rules.append({
            "rule_id": "QR-05",
            "title": "Fluid Residue Tolerance",
            "verdict": verdict,
            "defects": sorted(defect_classes.intersection({"4_shuiban", "5_youban"})),
            "reason": f"Surface fluid residues ({int(total_fluid_area)} px² total). Conforms to tolerance for downstream thermal processing." if verdict == "PASS" else "Excessive fluid pooling requires wipe-down review."
        })
        verdicts.append(verdict)

    # If unmatched defects exist
    if not applied_rules:
        applied_rules.append({
            "rule_id": "QR-GEN",
            "title": "General Defect Assessment",
            "verdict": "REVIEW REQUIRED",
            "defects": sorted(defect_classes),
            "reason": f"Uncategorized defect instances ({', '.join(sorted(defect_classes))}) flagged for manual engineering review."
        })
        verdicts.append("REVIEW REQUIRED")

    # Final overall verdict determination (Strict precedence: REJECT > REVIEW REQUIRED > PASS)
    if "REJECT" in verdicts:
        overall_status = "REJECT"
    elif "REVIEW REQUIRED" in verdicts:
        overall_status = "REVIEW REQUIRED"
    else:
        overall_status = "PASS"

    rule_ids = [r["rule_id"] for r in applied_rules]

    # Summarize explanation
    reasons = [r["reason"] for r in applied_rules]
    summary_text = " | ".join(reasons)

    return {
        "status": overall_status,
        "rule_ids": rule_ids,
        "summary": summary_text,
        "rules_applied": applied_rules
    }
