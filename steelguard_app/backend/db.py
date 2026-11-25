"""
STEELGUARD AI - Local SQLite Database & Traceability Store
Handles persistent local storage for steel surface inspections, defect instances,
rule-based assessments, operator notes, and export serialization.
Guarantees clean connection teardown and transaction safety.
"""

import csv
import io
import json
import os
from pathlib import Path
import sqlite3
import time
from typing import Dict, List, Optional, Any, Tuple

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "steelguard.db"


class ManagedConnection:
    """Context manager ensuring transactions commit/rollback and connections close cleanly."""

    def __init__(self, db_path: Optional[Path] = None):
        self.target_path = Path(db_path or DEFAULT_DB_PATH).resolve()
        self.target_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn: Optional[sqlite3.Connection] = None

    def __enter__(self) -> sqlite3.Connection:
        self.conn = sqlite3.connect(str(self.target_path), timeout=15.0)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON;")
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            try:
                if exc_type is None:
                    self.conn.commit()
                else:
                    self.conn.rollback()
            finally:
                self.conn.close()


def init_db(db_path: Optional[Path] = None):
    """Initializes tables and indexes."""
    with ManagedConnection(db_path) as conn:
        cursor = conn.cursor()

        # 1. Inspections Master Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS inspections (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                filename TEXT NOT NULL,
                image_reference TEXT,
                source TEXT NOT NULL CHECK(source IN ('GROUND_TRUTH', 'AI_INFERENCE')),
                image_width INTEGER NOT NULL,
                image_height INTEGER NOT NULL,
                total_defects INTEGER NOT NULL DEFAULT 0,
                assessment_status TEXT NOT NULL,
                assessment_summary TEXT NOT NULL,
                applied_rules TEXT,
                model_name TEXT,
                model_version TEXT,
                inference_time_ms REAL,
                operator_notes TEXT DEFAULT '',
                review_status TEXT DEFAULT 'UNREVIEWED' CHECK(review_status IN ('UNREVIEWED', 'CONFIRMED', 'FLAGGED')),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # 2. Defect Instances Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS defect_instances (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                inspection_id TEXT NOT NULL,
                detection_id TEXT NOT NULL,
                class_id INTEGER,
                class_name TEXT NOT NULL,
                raw_label TEXT NOT NULL,
                criticality TEXT NOT NULL,
                confidence REAL NOT NULL,
                is_ground_truth INTEGER NOT NULL DEFAULT 0,
                xmin REAL NOT NULL,
                ymin REAL NOT NULL,
                xmax REAL NOT NULL,
                ymax REAL NOT NULL,
                width REAL NOT NULL,
                height REAL NOT NULL,
                area_px REAL NOT NULL,
                FOREIGN KEY (inspection_id) REFERENCES inspections(id) ON DELETE CASCADE
            );
        """)

        # 3. Quality Rule Triggers Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS quality_rule_triggers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                inspection_id TEXT NOT NULL,
                rule_id TEXT NOT NULL,
                rule_title TEXT NOT NULL,
                verdict TEXT NOT NULL,
                reason TEXT NOT NULL,
                FOREIGN KEY (inspection_id) REFERENCES inspections(id) ON DELETE CASCADE
            );
        """)

        # 4. Independent Segmentation Inspections Table (Phase 4C)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS segmentation_inspections (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                filename TEXT NOT NULL,
                image_reference TEXT,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                checkpoint_hash TEXT NOT NULL,
                image_width INTEGER NOT NULL,
                image_height INTEGER NOT NULL,
                inference_time_ms REAL NOT NULL,
                class_1_pixels INTEGER NOT NULL DEFAULT 0,
                class_2_pixels INTEGER NOT NULL DEFAULT 0,
                class_3_pixels INTEGER NOT NULL DEFAULT 0,
                class_4_pixels INTEGER NOT NULL DEFAULT 0,
                total_defect_pixels INTEGER NOT NULL DEFAULT 0,
                defect_percentage REAL NOT NULL DEFAULT 0.0,
                mask_artifact_path TEXT,
                operator_notes TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Indexes for fast search and traceability lookups
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_inspections_timestamp ON inspections(timestamp);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_inspections_source ON inspections(source);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_inspections_status ON inspections(assessment_status);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_defects_inspection ON defect_instances(inspection_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_defects_class ON defect_instances(class_name);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_seg_timestamp ON segmentation_inspections(timestamp);")


def save_inspection(data: Dict[str, Any], db_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Persists an inspection, its defect instances, and quality rule triggers.
    Prevents duplicate submissions via primary key constraints.
    """
    init_db(db_path)
    insp_id = data.get("inspection_id")
    if not insp_id:
        timestamp_slug = time.strftime("%Y%m%d-%H%M%S")
        insp_id = f"SG-{timestamp_slug}"

    source = data.get("source", "GROUND_TRUTH")
    if source not in ("GROUND_TRUTH", "AI_INFERENCE"):
        source = "GROUND_TRUTH"

    timestamp = data.get("timestamp") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    filename = data.get("filename", "unknown_steel_sample.jpg")
    image_reference = data.get("image_reference", data.get("sample_id", filename))
    width = int(data.get("image_width", 2048))
    height = int(data.get("image_height", 1000))
    
    detections = data.get("detections", [])
    total_defects = len(detections)

    assessment = data.get("assessment", {})
    status = assessment.get("status", "ASSESSMENT PENDING")
    summary = assessment.get("summary", "No assessment evaluation recorded.")
    rule_ids = assessment.get("rule_ids", [])
    applied_rules_str = json.dumps(rule_ids)

    model_name = data.get("model_name")
    model_version = data.get("model_version")
    inference_time = data.get("inference_time_ms")
    operator_notes = data.get("operator_notes", "")
    review_status = data.get("review_status", "UNREVIEWED")

    with ManagedConnection(db_path) as conn:
        cursor = conn.cursor()
        
        # Check for existing record to avoid duplicate submissions
        cursor.execute("SELECT id FROM inspections WHERE id = ?;", (insp_id,))
        if cursor.fetchone():
            return {
                "success": False,
                "already_exists": True,
                "inspection_id": insp_id,
                "message": f"Inspection '{insp_id}' has already been recorded in database."
            }

        # Insert Master Record
        cursor.execute("""
            INSERT INTO inspections (
                id, timestamp, filename, image_reference, source,
                image_width, image_height, total_defects, assessment_status,
                assessment_summary, applied_rules, model_name, model_version,
                inference_time_ms, operator_notes, review_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            insp_id, timestamp, filename, image_reference, source,
            width, height, total_defects, status,
            summary, applied_rules_str, model_name, model_version,
            inference_time, operator_notes, review_status
        ))

        # Insert Defect Instances
        for det in detections:
            det_id = det.get("detection_id", f"D-{time.time()}")
            cls_id = det.get("class_id", 0)
            cls_name = det.get("class_name", "Unknown Defect")
            raw_label = det.get("raw_label", cls_name)
            crit = det.get("criticality", "MEDIUM")
            conf = float(det.get("confidence", 1.0))
            is_gt = 1 if (source == "GROUND_TRUTH" or det.get("is_ground_truth")) else 0

            bbox = det.get("bbox_xyxy", [0, 0, 0, 0])
            xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
            bw = det.get("bbox_wh", [xmax - xmin, ymax - ymin])[0]
            bh = det.get("bbox_wh", [xmax - xmin, ymax - ymin])[1]
            area = det.get("area_px", bw * bh)

            cursor.execute("""
                INSERT INTO defect_instances (
                    inspection_id, detection_id, class_id, class_name, raw_label,
                    criticality, confidence, is_ground_truth, xmin, ymin, xmax, ymax,
                    width, height, area_px
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                insp_id, det_id, cls_id, cls_name, raw_label,
                crit, conf, is_gt, xmin, ymin, xmax, ymax,
                bw, bh, area
            ))

        # Insert Rule Triggers
        rules_applied = assessment.get("rules_applied", [])
        for r in rules_applied:
            cursor.execute("""
                INSERT INTO quality_rule_triggers (
                    inspection_id, rule_id, rule_title, verdict, reason
                ) VALUES (?, ?, ?, ?, ?);
            """, (
                insp_id,
                r.get("rule_id", "QR-GEN"),
                r.get("title", "Quality Rule"),
                r.get("verdict", status),
                r.get("reason", summary)
            ))

    return {
        "success": True,
        "already_exists": False,
        "inspection_id": insp_id,
        "message": f"Successfully persisted inspection '{insp_id}' with {total_defects} defect instances."
    }


def get_inspection(insp_id: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Retrieves full inspection details including defect instances and rule triggers."""
    init_db(db_path)
    with ManagedConnection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM inspections WHERE id = ?;", (insp_id,))
        row = cursor.fetchone()
        if not row:
            return None

        insp = dict(row)
        insp["applied_rules"] = json.loads(insp["applied_rules"]) if insp["applied_rules"] else []

        # Fetch defect instances
        cursor.execute("""
            SELECT detection_id, class_id, class_name, raw_label, criticality,
                   confidence, is_ground_truth, xmin, ymin, xmax, ymax, width, height, area_px
            FROM defect_instances WHERE inspection_id = ? ORDER BY id ASC;
        """, (insp_id,))
        defects = []
        for d in cursor.fetchall():
            d_dict = dict(d)
            d_dict["bbox_xyxy"] = [d_dict["xmin"], d_dict["ymin"], d_dict["xmax"], d_dict["ymax"]]
            d_dict["bbox_wh"] = [d_dict["width"], d_dict["height"]]
            d_dict["is_ground_truth"] = bool(d_dict["is_ground_truth"])
            defects.append(d_dict)
        insp["detections"] = defects

        # Fetch rule triggers
        cursor.execute("""
            SELECT rule_id, rule_title, verdict, reason
            FROM quality_rule_triggers WHERE inspection_id = ? ORDER BY id ASC;
        """, (insp_id,))
        rules = [dict(r) for r in cursor.fetchall()]
        insp["rules_applied"] = rules

        return insp


def list_inspections(
    search: Optional[str] = None,
    status: Optional[str] = None,
    source: Optional[str] = None,
    sort_order: str = "DESC",
    limit: int = 100,
    offset: int = 0,
    db_path: Optional[Path] = None
) -> Dict[str, Any]:
    """Queries inspections with flexible search, filtering, and pagination."""
    init_db(db_path)
    conditions = []
    params = []

    if search:
        s_term = f"%{search.strip()}%"
        conditions.append("(id LIKE ? OR filename LIKE ? OR operator_notes LIKE ?)")
        params.extend([s_term, s_term, s_term])

    if status and status.upper() != "ALL":
        conditions.append("assessment_status = ?")
        params.append(status.upper())

    if source and source.upper() != "ALL":
        conditions.append("source = ?")
        params.append(source.upper())

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    order = "ASC" if sort_order.upper() == "ASC" else "DESC"

    with ManagedConnection(db_path) as conn:
        cursor = conn.cursor()

        # Count total matches
        cursor.execute(f"SELECT COUNT(*) FROM inspections {where_clause};", params)
        total_count = cursor.fetchone()[0]

        # Fetch paginated rows
        query = f"""
            SELECT id, timestamp, filename, image_reference, source,
                   image_width, image_height, total_defects, assessment_status,
                   assessment_summary, applied_rules, model_name, model_version,
                   operator_notes, review_status
            FROM inspections
            {where_clause}
            ORDER BY timestamp {order}
            LIMIT ? OFFSET ?;
        """
        cursor.execute(query, params + [limit, offset])
        rows = []
        for r in cursor.fetchall():
            item = dict(r)
            item["applied_rules"] = json.loads(item["applied_rules"]) if item["applied_rules"] else []
            rows.append(item)

        return {
            "total": total_count,
            "count": len(rows),
            "limit": limit,
            "offset": offset,
            "inspections": rows
        }


def update_inspection_notes(
    insp_id: str,
    notes: str,
    review_status: Optional[str] = None,
    db_path: Optional[Path] = None
) -> bool:
    """Updates operator notes and review status for traceability sign-off."""
    init_db(db_path)
    with ManagedConnection(db_path) as conn:
        cursor = conn.cursor()
        if review_status:
            cursor.execute("""
                UPDATE inspections
                SET operator_notes = ?, review_status = ?
                WHERE id = ?;
            """, (notes, review_status, insp_id))
        else:
            cursor.execute("""
                UPDATE inspections
                SET operator_notes = ?
                WHERE id = ?;
            """, (notes, insp_id))
        return cursor.rowcount > 0


def delete_inspection(insp_id: str, db_path: Optional[Path] = None) -> bool:
    """Deletes an inspection record and cascades to defect instances."""
    init_db(db_path)
    with ManagedConnection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM inspections WHERE id = ?;", (insp_id,))
        return cursor.rowcount > 0


def get_overview_stats(db_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Computes genuine quality KPIs from the SQLite database.
    Separates total inspections, defect instances, inspections containing defects,
    review required, rejects, and passes.
    Does NOT calculate a pass rate if zero valid records exist.
    """
    init_db(db_path)
    with ManagedConnection(db_path) as conn:
        cursor = conn.cursor()

        # 1. Total inspections
        cursor.execute("SELECT COUNT(*) FROM inspections;")
        total_inspections = cursor.fetchone()[0]

        if total_inspections == 0:
            return {
                "total_inspections": 0,
                "defects_detected": 0,
                "inspections_with_defects": 0,
                "requiring_review": 0,
                "rejected": 0,
                "passed": 0,
                "pass_rate_pct": None,
                "status_distribution": {
                    "PASS": 0,
                    "REVIEW REQUIRED": 0,
                    "REJECT": 0,
                    "ASSESSMENT PENDING": 0
                },
                "source_distribution": {
                    "GROUND_TRUTH": 0,
                    "AI_INFERENCE": 0
                },
                "defect_distribution": {}
            }

        # 2. Defect instances and inspections with defects
        cursor.execute("SELECT COUNT(*) FROM defect_instances;")
        total_defects = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM inspections WHERE total_defects > 0;")
        inspections_with_defects = cursor.fetchone()[0]

        # 3. Status breakdown
        cursor.execute("""
            SELECT assessment_status, COUNT(*)
            FROM inspections
            GROUP BY assessment_status;
        """)
        status_counts = {row[0]: row[1] for row in cursor.fetchall()}
        passed = status_counts.get("PASS", 0)
        review = status_counts.get("REVIEW REQUIRED", 0)
        rejected = status_counts.get("REJECT", 0)
        pending = status_counts.get("ASSESSMENT PENDING", 0)

        # 4. Source breakdown
        cursor.execute("""
            SELECT source, COUNT(*)
            FROM inspections
            GROUP BY source;
        """)
        source_counts = {row[0]: row[1] for row in cursor.fetchall()}

        # 5. Defect distribution by class
        cursor.execute("""
            SELECT class_name, COUNT(*)
            FROM defect_instances
            GROUP BY class_name
            ORDER BY COUNT(*) DESC;
        """)
        defect_distribution = {row[0]: row[1] for row in cursor.fetchall()}

        # Pass rate calculation: only considers evaluated inspections (excludes pending)
        evaluated_count = passed + review + rejected
        pass_rate_pct = round((passed / evaluated_count) * 100, 1) if evaluated_count > 0 else None

        return {
            "total_inspections": total_inspections,
            "defects_detected": total_defects,
            "inspections_with_defects": inspections_with_defects,
            "requiring_review": review,
            "rejected": rejected,
            "passed": passed,
            "pass_rate_pct": pass_rate_pct,
            "status_distribution": {
                "PASS": passed,
                "REVIEW REQUIRED": review,
                "REJECT": rejected,
                "ASSESSMENT PENDING": pending
            },
            "source_distribution": source_counts,
            "defect_distribution": defect_distribution
        }


def export_inspections_csv(
    search: Optional[str] = None,
    status: Optional[str] = None,
    source: Optional[str] = None,
    db_path: Optional[Path] = None
) -> str:
    """Exports filtered inspection records to valid CSV string with safe escaping."""
    data = list_inspections(search=search, status=status, source=source, limit=10000, db_path=db_path)
    output = io.StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)

    # Header
    writer.writerow([
        "Inspection ID",
        "Timestamp (UTC)",
        "Filename",
        "Reference",
        "Source",
        "Image Resolution",
        "Defect Count",
        "Assessment Status",
        "Applied Rules",
        "Assessment Summary",
        "Review Status",
        "Operator Notes"
    ])

    for row in data["inspections"]:
        rules_str = ", ".join(row["applied_rules"])
        res_str = f"{row['image_width']}x{row['image_height']}"
        writer.writerow([
            row["id"],
            row["timestamp"],
            row["filename"],
            row["image_reference"],
            row["source"],
            res_str,
            row["total_defects"],
            row["assessment_status"],
            rules_str,
            row["assessment_summary"],
            row["review_status"],
            row["operator_notes"]
        ])

    return output.getvalue()


# ==============================================================================
# Segmentation Traceability API (Phase 4C)
# Completely decoupled from standard YOLO detection records
# ==============================================================================

def save_segmentation_inspection(data: Dict[str, Any], db_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Persists a Severstal U-Net pixel-level segmentation inspection record.
    Mask arrays are stored as PNG files in segmentation_artifacts/ rather than SQLite BLOBs.
    """
    insp_id = data.get("id")
    if not insp_id:
        insp_id = f"SG-SEG-{time.strftime('%Y%m%d-%H%M%S')}-{os.urandom(2).hex()}"

    timestamp = data.get("timestamp") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    filename = data.get("filename", "unknown_strip.png")
    image_ref = data.get("image_reference", filename)
    model_name = data.get("model_name", "Lightweight U-Net (Severstal)")
    model_version = data.get("model_version", "v1.0-severstal-unet")
    checkpoint_hash = data.get("checkpoint_hash", "")
    width = int(data.get("image_width", 1600))
    height = int(data.get("image_height", 256))
    latency = float(data.get("inference_time_ms", 0.0))

    # Pixel breakdowns
    c1 = int(data.get("class_1_pixels", 0))
    c2 = int(data.get("class_2_pixels", 0))
    c3 = int(data.get("class_3_pixels", 0))
    c4 = int(data.get("class_4_pixels", 0))
    tot_defect = int(data.get("total_defect_pixels", c1 + c2 + c3 + c4))
    defect_pct = float(data.get("defect_percentage", (tot_defect / (width * height)) * 100.0))
    notes = data.get("operator_notes", "")

    # Artifact storage for mask if provided
    mask_artifact_path = data.get("mask_artifact_path", "")
    mask_base64 = data.get("mask_png_base64")
    if mask_base64 and not mask_artifact_path:
        try:
            import base64
            artifacts_dir = Path(__file__).resolve().parent.parent / "data" / "segmentation_artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            art_file = artifacts_dir / f"{insp_id}_mask.png"
            with open(art_file, "wb") as f:
                f.write(base64.b64decode(mask_base64))
            mask_artifact_path = str(art_file.relative_to(Path(__file__).resolve().parent.parent)).replace("\\", "/")
        except Exception:
            mask_artifact_path = ""

    with ManagedConnection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO segmentation_inspections (
                id, timestamp, filename, image_reference, model_name, model_version,
                checkpoint_hash, image_width, image_height, inference_time_ms,
                class_1_pixels, class_2_pixels, class_3_pixels, class_4_pixels,
                total_defect_pixels, defect_percentage, mask_artifact_path, operator_notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            insp_id, timestamp, filename, image_ref, model_name, model_version,
            checkpoint_hash, width, height, latency,
            c1, c2, c3, c4,
            tot_defect, defect_pct, mask_artifact_path, notes
        ))

    return {
        "success": True,
        "inspection_id": insp_id,
        "mask_artifact_path": mask_artifact_path
    }


def get_segmentation_inspection(insp_id: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Retrieves a single segmentation inspection record."""
    with ManagedConnection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM segmentation_inspections WHERE id = ?;", (insp_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return dict(row)


def list_segmentation_inspections(limit: int = 50, offset: int = 0, db_path: Optional[Path] = None) -> Dict[str, Any]:
    """Lists recent segmentation inspection records."""
    with ManagedConnection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM segmentation_inspections;")
        total = cursor.fetchone()[0]

        cursor.execute("""
            SELECT * FROM segmentation_inspections
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?;
        """, (limit, offset))
        rows = [dict(r) for r in cursor.fetchall()]

    return {
        "total": total,
        "count": len(rows),
        "inspections": rows
    }
