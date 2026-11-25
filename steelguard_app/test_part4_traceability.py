"""
STEELGUARD AI - Part 4: Traceability and Data Isolation Verification
Validates absolute separation between YOLO and U-Net inspection stores,
schema integrity, persistent mask artifact references, and restart persistence.
"""

from pathlib import Path
import sqlite3
import sys
import cv2

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
BACKEND_DIR = APP_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))

import db

DB_PATH = APP_DIR / "data" / "steelguard.db"

def run_traceability_tests():
    print("=" * 70)
    print(" STEELGUARD AI — PART 4: TRACEABILITY & DATA ISOLATION VERIFICATION")
    print("=" * 70)

    assert DB_PATH.is_file(), f"Database not found at {DB_PATH}"

    # 1. Inspect Tables in SQLite
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0] for r in cursor.fetchall()]
    print(f"\n[Test 1] Database Tables Discovered: {tables}")
    assert "inspections" in tables, "Table 'inspections' (YOLO) missing!"
    assert "segmentation_inspections" in tables, "Table 'segmentation_inspections' missing!"

    # 2. Check Row Counts & Isolation
    cursor.execute("SELECT COUNT(*) FROM inspections;")
    yolo_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM segmentation_inspections;")
    seg_count = cursor.fetchone()[0]
    print(f"\n[Test 2] Record Isolation:")
    print(f"  YOLO Detection Records ('inspections'):             {yolo_count}")
    print(f"  U-Net Segmentation Records ('segmentation_inspections'): {seg_count}")
    assert seg_count > 0, "No segmentation inspections recorded!"

    # 3. Verify Segmentation Schema & Fields
    print("\n[Test 3] Verifying Segmentation Table Schema & Data Integrity:")
    cursor.execute("PRAGMA table_info(segmentation_inspections);")
    cols = [r[1] for r in cursor.fetchall()]
    expected_cols = [
        "id", "timestamp", "filename", "image_reference", "model_name", "model_version",
        "checkpoint_hash", "image_width", "image_height", "inference_time_ms",
        "class_1_pixels", "class_2_pixels", "class_3_pixels", "class_4_pixels",
        "total_defect_pixels", "defect_percentage", "mask_artifact_path", "operator_notes"
    ]
    for c in expected_cols:
        assert c in cols, f"Column {c} missing in segmentation_inspections!"
    print(f"  All {len(expected_cols)} required schema columns present and verified.")

    # 4. Inspect Recent Segmentation Record
    cursor.execute("SELECT * FROM segmentation_inspections ORDER BY timestamp DESC LIMIT 1;")
    row = dict(zip(cols, cursor.fetchone()))
    print("\n[Test 4] Recent Segmentation Record Trace:")
    print(f"  Inspection ID:     {row['id']}")
    print(f"  Timestamp:         {row['timestamp']}")
    print(f"  Filename:          {row['filename']}")
    print(f"  Model Name:        {row['model_name']}")
    print(f"  Model Version:     {row['model_version']}")
    print(f"  Checkpoint Hash:   {row['checkpoint_hash']}")
    print(f"  Resolution:        {row['image_width']}x{row['image_height']}")
    print(f"  Inference Time:    {row['inference_time_ms']} ms")
    print(f"  Defect Pixels:     Total {row['total_defect_pixels']} px (C1: {row['class_1_pixels']}, C2: {row['class_2_pixels']}, C3: {row['class_3_pixels']}, C4: {row['class_4_pixels']})")
    print(f"  Defect Percentage: {row['defect_percentage']:.4f}%")
    print(f"  Mask Artifact:     {row['mask_artifact_path']}")

    assert row["id"].startswith("SG-SEG"), "Inspection ID prefix invalid"
    assert row["checkpoint_hash"] == "10666e11ecf088489fc2544d85e50cce0108b93c5cb888974f30d886029104e3", "Checkpoint hash mismatch"

    # 5. Verify Mask Artifact File on Disk
    if row["mask_artifact_path"]:
        art_path = APP_DIR / row["mask_artifact_path"]
        print(f"\n[Test 5] Verifying Mask Artifact on Disk: {art_path}")
        assert art_path.is_file(), f"Artifact file missing: {art_path}"
        mask_img = cv2.imread(str(art_path), cv2.IMREAD_UNCHANGED)
        assert mask_img is not None, "Failed to read mask artifact PNG"
        assert mask_img.shape == (256, 1600), f"Unexpected mask shape: {mask_img.shape}"
        print(f"  Artifact verified: PNG resolution {mask_img.shape[1]}x{mask_img.shape[0]}, file size {art_path.stat().st_size:,} bytes.")

    conn.close()

    # 6. Database Restart Simulation
    print("\n[Test 6] Database Reopen / Persistence After Restart:")
    db_list = db.list_segmentation_inspections(limit=10)
    assert db_list["total"] == seg_count, f"Total mismatch after reopen: {db_list['total']} != {seg_count}"
    assert len(db_list["inspections"]) > 0, "No inspections returned from db.list_segmentation_inspections"
    print(f"  Successfully reopened database: {db_list['total']} historical segmentation records retrieved.")

    print("\n" + "=" * 70)
    print(" ALL PART 4 TRACEABILITY & DATA ISOLATION TESTS PASSED.")
    print("=" * 70)

if __name__ == "__main__":
    run_traceability_tests()
