"""
STEELGUARD AI - Dataset Service
Discovers, indexes, validates, and serves verified GC10-DET sample images and Pascal VOC XML ground-truth annotations.
Enforces strict path security and data integrity.
"""

from collections import Counter, defaultdict
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import xml.etree.ElementTree as ET

from quality_engine import assess_quality

# Raw dataset root (strictly read-only access)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_env_dataset_root = os.environ.get("STEELGUARD_DATASET_ROOT")
if _env_dataset_root:
    DATASET_ROOT = Path(_env_dataset_root).resolve()
elif (PROJECT_ROOT / "raw" / "Arcihve").exists():
    DATASET_ROOT = (PROJECT_ROOT / "raw" / "Arcihve").resolve()
elif (PROJECT_ROOT / "demo_samples").exists():
    DATASET_ROOT = (PROJECT_ROOT / "demo_samples").resolve()
else:
    DATASET_ROOT = (PROJECT_ROOT / "raw" / "Arcihve").resolve()

LABEL_DIR = (DATASET_ROOT / "lable").resolve() if (DATASET_ROOT / "lable").exists() else DATASET_ROOT

# Defect class definitions with color tokens and metadata
CLASS_METADATA = {
    "1_chongkong": {"id": 1, "name": "Punching Hole", "criticality": "CRITICAL", "color": "#ef4444"},
    "2_hanfeng": {"id": 2, "name": "Welding Line", "criticality": "HIGH", "color": "#ec4899"},
    "3_yueyawan": {"id": 3, "name": "Crescent Gap", "criticality": "CRITICAL", "color": "#f97316"},
    "4_shuiban": {"id": 4, "name": "Water Spot", "criticality": "LOW", "color": "#38bdf8"},
    "5_youban": {"id": 5, "name": "Oil Spot", "criticality": "LOW", "color": "#fbbf24"},
    "6_siban": {"id": 6, "name": "Silk Spot", "criticality": "MEDIUM", "color": "#a855f7"},
    "7_yiwu": {"id": 7, "name": "Inclusion / Foreign Object", "criticality": "HIGH", "color": "#e11d48"},
    "8_yahen": {"id": 8, "name": "Indentation / Rolled Scale", "criticality": "HIGH", "color": "#eab308"},
    "9_zhehen": {"id": 9, "name": "Crease", "criticality": "MEDIUM", "color": "#14b8a6"},
    "10_yaozhe": {"id": 10, "name": "Waist Folding", "criticality": "HIGH", "color": "#f43f5e"},
    "10_yaozhed": {"id": 10, "name": "Waist Folding (Variant)", "criticality": "HIGH", "color": "#f43f5e"},
}


class DatasetRegistry:
    """Singleton registry indexing all verified GC10-DET samples."""

    def __init__(self):
        self._samples: Dict[str, Dict[str, Any]] = {}
        self._class_index: Dict[str, List[str]] = defaultdict(list)
        self._initialized = False

    def initialize(self):
        if self._initialized:
            return

        if not DATASET_ROOT.exists() or not LABEL_DIR.exists():
            return

        # 1. Map all image files by filename and detect duplicate basenames
        images_by_name = defaultdict(list)
        for folder in DATASET_ROOT.iterdir():
            if folder.is_dir() and folder.name.isdigit():
                for p in folder.rglob("*"):
                    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                        images_by_name[p.name].append(p)

        # 2. Iterate all Pascal VOC XML files
        xml_files = sorted(LABEL_DIR.glob("*.xml"))
        for xml_file in xml_files:
            stem = xml_file.stem
            matching_images = images_by_name.get(f"{stem}.jpg", [])
            if not matching_images:
                # Try finding any matching image with that stem
                for name, paths in images_by_name.items():
                    if Path(name).stem == stem:
                        matching_images = paths
                        break

            # Exclude if missing or if duplicate basenames exist across numbered folders
            if len(matching_images) != 1:
                continue

            img_path = matching_images[0]

            # 3. Parse XML and validate dimensions and bounding boxes
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()

                size = root.find("size")
                if size is None:
                    continue

                w = int(size.findtext("width", "0"))
                h = int(size.findtext("height", "0"))
                if w <= 0 or h <= 0:
                    continue

                objects = root.findall("object")
                if not objects:
                    continue

                sample_classes = []
                valid_boxes = True

                for obj in objects:
                    raw_name = obj.findtext("name", "").strip()
                    # Discard anomalous 'd' label
                    if not raw_name or raw_name == "d" or raw_name not in CLASS_METADATA:
                        valid_boxes = False
                        break

                    box = obj.find("bndbox")
                    if box is None:
                        valid_boxes = False
                        break

                    xmin = float(box.findtext("xmin", "0"))
                    ymin = float(box.findtext("ymin", "0"))
                    xmax = float(box.findtext("xmax", "0"))
                    ymax = float(box.findtext("ymax", "0"))

                    if not (0 <= xmin < xmax <= w and 0 <= ymin < ymax <= h):
                        valid_boxes = False
                        break

                    sample_classes.append(raw_name)

                if not valid_boxes or not sample_classes:
                    continue

                # Approved sample entry
                unique_classes = sorted(list(set(sample_classes)))
                rel_path = str(img_path.relative_to(DATASET_ROOT)).replace("\\", "/")

                self._samples[stem] = {
                    "id": stem,
                    "filename": img_path.name,
                    "rel_path": rel_path,
                    "full_path": str(img_path),
                    "xml_path": str(xml_file),
                    "width": w,
                    "height": h,
                    "defect_count": len(objects),
                    "classes": unique_classes,
                    "primary_class": unique_classes[0]
                }

                for c in unique_classes:
                    self._class_index[c].append(stem)

            except Exception:
                continue

        self._initialized = True

    def get_sample_meta(self, sample_id: str) -> Optional[Dict[str, Any]]:
        self.initialize()
        return self._samples.get(sample_id)

    def get_sample_image_path(self, sample_id: str) -> Path:
        """
        Retrieves image path safely, strictly validating against the approved registry.
        Guarantees path traversal prevention.
        """
        self.initialize()
        if sample_id not in self._samples:
            raise KeyError(f"Sample '{sample_id}' not found in approved dataset registry.")

        sample = self._samples[sample_id]
        img_path = Path(sample["full_path"]).resolve()

        # Strict security check: ensure resolved path is within DATASET_ROOT
        try:
            img_path.relative_to(DATASET_ROOT)
        except ValueError:
            raise PermissionError("Path traversal violation detected.")

        if not img_path.is_file():
            raise FileNotFoundError(f"Image file does not exist on disk: {img_path}")

        return img_path

    def get_sample_annotation(self, sample_id: str) -> Dict[str, Any]:
        """
        Parses the ground-truth Pascal VOC XML annotation for an approved sample
        and evaluates its rule-based quality assessment.
        """
        self.initialize()
        if sample_id not in self._samples:
            raise KeyError(f"Sample '{sample_id}' not found in approved dataset registry.")

        sample = self._samples[sample_id]
        xml_path = Path(sample["xml_path"]).resolve()

        # Security check
        try:
            xml_path.relative_to(LABEL_DIR)
        except ValueError:
            raise PermissionError("Path traversal violation detected.")

        tree = ET.parse(xml_path)
        root = tree.getroot()

        annotations = []
        det_idx = 1

        for obj in root.findall("object"):
            raw_name = obj.findtext("name", "").strip()
            meta = CLASS_METADATA.get(raw_name, {
                "id": 99,
                "name": raw_name,
                "criticality": "UNKNOWN",
                "color": "#94a3b8"
            })

            box = obj.find("bndbox")
            xmin = round(float(box.findtext("xmin", "0")), 1)
            ymin = round(float(box.findtext("ymin", "0")), 1)
            xmax = round(float(box.findtext("xmax", "0")), 1)
            ymax = round(float(box.findtext("ymax", "0")), 1)

            bw = round(xmax - xmin, 1)
            bh = round(ymax - ymin, 1)
            area = round(bw * bh, 1)

            annotations.append({
                "detection_id": f"GT-{det_idx:02d}",
                "class_id": meta["id"],
                "class_name": meta["name"],
                "raw_label": raw_name,
                "criticality": meta["criticality"],
                "color": meta["color"],
                "confidence": 1.0,  # Explicitly 1.0 for ground truth
                "is_ground_truth": True,
                "data_source": "Pascal VOC Ground Truth",
                "bbox_xyxy": [xmin, ymin, xmax, ymax],
                "bbox_wh": [bw, bh],
                "area_px": area
            })
            det_idx += 1

        # Run quality assessment engine on ground-truth detections
        assessment = assess_quality(annotations, rules_enabled=True)

        return {
            "inspection_id": f"QC-GT-{sample['id'][-8:]}",
            "sample_id": sample["id"],
            "filename": sample["filename"],
            "image_width": sample["width"],
            "image_height": sample["height"],
            "mode": "GROUND_TRUTH",
            "mode_label": "GROUND TRUTH / DATASET ANNOTATIONS",
            "model_name": "None (Pascal VOC XML Ground Truth)",
            "model_version": "GC10-DET Certified",
            "inference_time_ms": None,
            "total_defects": len(annotations),
            "detections": annotations,
            "assessment": assessment
        }

    def list_samples(self, defect_class: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        self.initialize()

        if defect_class and defect_class in self._class_index:
            sample_ids = self._class_index[defect_class][:limit]
        else:
            sample_ids = list(self._samples.keys())[:limit]

        return [
            {
                "id": s_id,
                "filename": self._samples[s_id]["filename"],
                "width": self._samples[s_id]["width"],
                "height": self._samples[s_id]["height"],
                "defect_count": self._samples[s_id]["defect_count"],
                "classes": self._samples[s_id]["classes"],
                "primary_class": self._samples[s_id]["primary_class"]
            }
            for s_id in sample_ids
        ]

    def get_summary(self) -> Dict[str, Any]:
        self.initialize()
        return {
            "total_verified_samples": len(self._samples),
            "class_counts": {k: len(v) for k, v in sorted(self._class_index.items())}
        }


# Global dataset registry instance
registry = DatasetRegistry()
