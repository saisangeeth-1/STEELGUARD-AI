"""Curate and verify demonstration samples for STEELGUARD AI CPU-only portable release.

Selects held-out test samples covering:
1. Critical Defect (REJECT verdict)
2. Inclusion Defect
3. Minor Defect (REVIEW verdict)
4. Clean Steel Surface (PASS verdict)
5. Severstal Defect Segmentation (Class 3 scratch)

Preserves exact dataset provenance and split verification.
"""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = ROOT / "demo_samples"
DEMO_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES = [
    {
        "id": "img_01_3402617700_01009",
        "category": "critical_defect",
        "title": "Critical Crescent Gap (Edge Defect)",
        "description": "Severe edge perforation risking coil breakage under rolling tension.",
        "dataset": "GC10-DET",
        "split": "test (held-out)",
        "image_src": ROOT / "datasets/gc10_neu_inclusion_experiment/images/test/img_01_3402617700_01009.jpg",
        "label_src": ROOT / "datasets/gc10_neu_inclusion_experiment/labels/test/img_01_3402617700_01009.txt",
        "xml_src": ROOT / "raw/Arcihve/lable/img_01_3402617700_01009.xml",
    },
    {
        "id": "img_02_4402623300_00034",
        "category": "inclusion_defect",
        "title": "Inclusion / Foreign Object Defect",
        "description": "Non-metallic foreign inclusion defect validated by fine-tuned model.",
        "dataset": "GC10-DET",
        "split": "test (held-out)",
        "image_src": ROOT / "datasets/gc10_neu_inclusion_experiment/images/test/img_02_4402623300_00034.jpg",
        "label_src": ROOT / "datasets/gc10_neu_inclusion_experiment/labels/test/img_02_4402623300_00034.txt",
        "xml_src": ROOT / "raw/Arcihve/lable/img_02_4402623300_00034.xml",
    },
    {
        "id": "img_01_425006600_00905",
        "category": "minor_defect",
        "title": "Minor Water Spot (Cosmetic Surface Defect)",
        "description": "Low-criticality cosmetic defect triggering operator review without immediate scrap.",
        "dataset": "GC10-DET",
        "split": "test (held-out)",
        "image_src": ROOT / "datasets/gc10_neu_inclusion_experiment/images/test/img_01_425006600_00905.jpg",
        "label_src": ROOT / "datasets/gc10_neu_inclusion_experiment/labels/test/img_01_425006600_00905.txt",
        "xml_src": ROOT / "raw/Arcihve/lable/img_01_425006600_00905.xml",
    },
    {
        "id": "clean_steel_strip_000789191",
        "category": "clean_sample",
        "title": "Verified Clean Steel Strip Surface",
        "description": "Defect-free steel strip surface with zero anomalies, resulting in PASS verdict.",
        "dataset": "Severstal Steel Strip Defect Detection",
        "split": "test (held-out, test_manifest.csv has_defect=0)",
        "image_src": ROOT / "datasets/severstal_segmentation_experiment/images/000789191.jpg",
        "label_src": None,
        "xml_src": None,
    },
    {
        "id": "severstal_scratch_0007a71bf",
        "category": "segmentation_defect",
        "title": "Severstal Surface Scratch (Pixel-Level Segmentation)",
        "description": "Continuous thin longitudinal scratch localized by dual-model U-Net.",
        "dataset": "Severstal Steel Strip Defect Detection",
        "split": "test (held-out, test_manifest.csv has_defect=1, Class 3)",
        "image_src": ROOT / "datasets/severstal_segmentation_experiment/images/0007a71bf.jpg",
        "mask_src": ROOT / "datasets/severstal_segmentation_experiment/masks/0007a71bf.png",
        "label_src": None,
        "xml_src": None,
    }
]


def main():
    catalog = []
    for s in SAMPLES:
        if not s["image_src"].exists():
            raise FileNotFoundError(f"Missing source image: {s['image_src']}")
        img_dst = DEMO_DIR / s["image_src"].name
        shutil.copy2(s["image_src"], img_dst)
        sha256 = hashlib.sha256(img_dst.read_bytes()).hexdigest()

        entry = {
            "id": s["id"],
            "filename": img_dst.name,
            "category": s["category"],
            "title": s["title"],
            "description": s["description"],
            "dataset": s["dataset"],
            "split": s["split"],
            "sha256": sha256,
            "size_bytes": img_dst.stat().st_size
        }

        if s.get("label_src") and s["label_src"].exists():
            lbl_dst = DEMO_DIR / s["label_src"].name
            shutil.copy2(s["label_src"], lbl_dst)
            entry["yolo_label_file"] = lbl_dst.name
            entry["yolo_labels"] = [l.strip() for l in lbl_dst.read_text(encoding="utf-8").splitlines() if l.strip()]

        if s.get("xml_src") and s["xml_src"].exists():
            xml_dst = DEMO_DIR / s["xml_src"].name
            shutil.copy2(s["xml_src"], xml_dst)
            entry["voc_xml_file"] = xml_dst.name

        if s.get("mask_src") and s["mask_src"].exists():
            mask_dst = DEMO_DIR / s["mask_src"].name
            shutil.copy2(s["mask_src"], mask_dst)
            entry["ground_truth_mask"] = mask_dst.name

        catalog.append(entry)

    manifest_file = DEMO_DIR / "demo_samples_manifest.json"
    manifest_file.write_text(json.dumps({"curated_samples": catalog}, indent=2), encoding="utf-8")
    print(f"Successfully curated {len(catalog)} demonstration samples in {DEMO_DIR}")
    for c in catalog:
        print(f" - [{c['category']}] {c['filename']} ({c['dataset']} - {c['split']})")


if __name__ == "__main__":
    main()
