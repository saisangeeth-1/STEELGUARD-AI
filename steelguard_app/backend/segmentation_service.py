"""
STEELGUARD AI — Independent Segmentation Model Service & Manager (Phase 4C)
Provides isolated management and inference for the Severstal Lightweight U-Net model.
Guarantees complete separation from YOLO detection, independent checkpoint verification,
safe input handling, exact Phase 4B preprocessing, and per-class pixel metrics.
"""

import base64
import csv
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import time
from typing import Dict, List, Optional, Any, Tuple, Union

import cv2
import numpy as np
from PIL import Image
import torch

from unet_model import LightweightUNet

logger = logging.getLogger("steelguard.segmentation_service")

BASE_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = BASE_DIR / "weights"
ARTIFACTS_DIR = BASE_DIR / "data" / "segmentation_artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

PROJECT_ROOT = BASE_DIR.parent
DATASET_EXP_DIR = PROJECT_ROOT / "datasets" / "severstal_segmentation_experiment"
PHASE4B_CHECKPOINT_DEFAULT = PROJECT_ROOT / "runs" / "segmentation" / "severstal_unet_v1" / "best_model.pt"
LOCAL_WEIGHTS_PATH = WEIGHTS_DIR / "severstal_unet_best.pt"

EXPECTED_SHA256 = "10666e11ecf088489fc2544d85e50cce0108b93c5cb888974f30d886029104e3"

# Official Severstal 4-Class Defect Taxonomy (Completely decoupled from GC10 10-Class detection)
SEVERSTAL_CLASSES = [
    {
        "class_id": 0,
        "code": "0_background",
        "name": "Background",
        "display_name": "Background / Prime Steel",
        "criticality": "NONE",
        "color": "#64748b",
        "color_rgb": [100, 116, 139],
        "description": "Defect-free steel strip surface."
    },
    {
        "class_id": 1,
        "code": "1_pitted_surface",
        "name": "Pitted Surface",
        "display_name": "Class 1: Pitted Surface",
        "criticality": "HIGH",
        "color": "#fbbf24",  # Amber/Yellow
        "color_rgb": [251, 191, 36],
        "description": "Surface indentations and pitted depressions from roll debris."
    },
    {
        "class_id": 2,
        "code": "2_cracks_seams",
        "name": "Cracks / Seams",
        "display_name": "Class 2: Cracks / Seams",
        "criticality": "CRITICAL",
        "color": "#06b6d4",  # Cyan/Aqua
        "color_rgb": [6, 182, 212],
        "description": "Longitudinal fissure or seam discontinuity liable to propagate."
    },
    {
        "class_id": 3,
        "code": "3_scratches",
        "name": "Scratches",
        "display_name": "Class 3: Scratches",
        "criticality": "MEDIUM",
        "color": "#ef4444",  # Red
        "color_rgb": [239, 68, 68],
        "description": "Longitudinal mechanical abrasion bands from guide rollers."
    },
    {
        "class_id": 4,
        "code": "4_rolled_scale",
        "name": "Rolled Scale",
        "display_name": "Class 4: Rolled Scale",
        "criticality": "HIGH",
        "color": "#22c55e",  # Green
        "color_rgb": [34, 197, 94],
        "description": "Oxide scale pressed into the sheet surface during rolling."
    }
]

SEVERSTAL_CLASS_MAP = {c["class_id"]: c for c in SEVERSTAL_CLASSES}

# Exact Phase 4B Preprocessing Normalization Constants (ImageNet)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


class SegmentationIntegrityError(Exception):
    """Raised when checkpoint hash or ontology verification fails."""
    pass


class InvalidSegmentationImageError(Exception):
    """Raised when input image is corrupt or unreadable."""
    pass


class SegmentationManager:
    """
    Independent model manager for Severstal U-Net pixel-level segmentation.
    Manages loading, SHA-256 verification, inference, and memory safety without
    touching or altering the active YOLO detector.
    """

    def __init__(self):
        self._model: Optional[LightweightUNet] = None
        self._model_info: Optional[Dict[str, Any]] = None
        self._device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._last_error: Optional[str] = None
        self._init_manager()

    def compute_sha256(self, file_path: Union[str, Path]) -> str:
        """Computes SHA-256 hash of checkpoint."""
        p = Path(file_path)
        if not p.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {p}")
        hasher = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _find_checkpoint(self) -> Path:
        """Locates best_model.pt in local app weights or experiment run directory."""
        if LOCAL_WEIGHTS_PATH.is_file():
            return LOCAL_WEIGHTS_PATH
        if PHASE4B_CHECKPOINT_DEFAULT.is_file():
            return PHASE4B_CHECKPOINT_DEFAULT
        raise FileNotFoundError(
            f"Segmentation checkpoint not found in {LOCAL_WEIGHTS_PATH} or {PHASE4B_CHECKPOINT_DEFAULT}"
        )

    def _init_manager(self):
        """Attempts to discover and load U-Net weights on startup."""
        try:
            chk_path = self._find_checkpoint()
            self.load_model(chk_path)
        except Exception as e:
            self._last_error = str(e)
            logger.warning(f"Segmentation U-Net model not loaded on startup: {e}")

    def load_model(self, checkpoint_path: Optional[Union[str, Path]] = None, force_reload: bool = False) -> Dict[str, Any]:
        """Loads and verifies Lightweight U-Net checkpoint."""
        if self._model is not None and not force_reload:
            return self.get_model_info()

        chk_path = Path(checkpoint_path) if checkpoint_path else self._find_checkpoint()
        if not chk_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found at {chk_path}")

        # 1. Cryptographic SHA-256 Verification
        actual_sha = self.compute_sha256(chk_path)
        if actual_sha != EXPECTED_SHA256:
            raise SegmentationIntegrityError(
                f"Checkpoint SHA-256 mismatch! Expected {EXPECTED_SHA256}, got {actual_sha}"
            )

        # 2. Select Device
        env_dev = os.environ.get("STEELGUARD_DEVICE", "").strip().lower()
        if env_dev == "cpu":
            device_str = "cpu"
        elif self._device == "cpu":
            device_str = "cpu"
        else:
            device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_str)

        # 3. Load Checkpoint State Dict
        chk = torch.load(chk_path, map_location=device)
        config = chk.get("config", {})
        base_channels = config.get("base_channels", 16)
        num_classes = config.get("num_classes", 5)

        # 4. Instantiate Model
        model = LightweightUNet(
            in_channels=3,
            num_classes=num_classes,
            base_channels=base_channels
        ).to(device)

        model.load_state_dict(chk["model_state_dict"])
        model.eval()

        self._model = model
        self._device = device_str
        self._last_error = None

        self._model_info = {
            "name": "Lightweight U-Net (Severstal)",
            "version": "v1.0-severstal-unet",
            "role": "PIXEL_SEGMENTATION_ENGINE",
            "status": "EXPERIMENTAL",
            "loaded": True,
            "weights_path": str(chk_path.resolve()).replace("\\", "/"),
            "expected_sha256": EXPECTED_SHA256,
            "actual_sha256": actual_sha,
            "hash_verified": True,
            "device": device_str,
            "device_name": torch.cuda.get_device_name(0) if device_str.startswith("cuda") else "CPU Host",
            "base_channels": base_channels,
            "num_classes": num_classes,
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "parameters_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 3),
            "input_resolution": config.get("input_resolution", [256, 1600]),
            "selection_metric": config.get("selection_metric", "val_macro_iou_defect"),
            "val_defect_iou": chk.get("val_macro_iou_defect"),
            "best_epoch": chk.get("epoch", 12),
            "classes": SEVERSTAL_CLASSES
        }
        logger.info(f"Loaded Severstal U-Net ({actual_sha[:8]}...) on {device_str}")
        return self._model_info

    def get_model_info(self) -> Dict[str, Any]:
        """Returns metadata and verification status for segmentation model."""
        if self._model_info is not None:
            return dict(self._model_info)
        return {
            "name": "Lightweight U-Net (Severstal)",
            "version": "v1.0-severstal-unet",
            "role": "PIXEL_SEGMENTATION_ENGINE",
            "status": "NOT_LOADED",
            "loaded": False,
            "error": self._last_error,
            "weights_path": str(LOCAL_WEIGHTS_PATH).replace("\\", "/"),
            "expected_sha256": EXPECTED_SHA256,
            "classes": SEVERSTAL_CLASSES
        }

    def decode_input_image(self, image_input: Union[str, bytes, np.ndarray, Path]) -> Tuple[np.ndarray, str]:
        """
        Safely validates and decodes diverse image inputs into an RGB NumPy array.
        """
        filename = "uploaded_strip.png"

        if isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                img_rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            elif image_input.shape[2] == 3:
                img_rgb = image_input.copy()
            else:
                img_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
            return img_rgb, filename

        # File path
        if isinstance(image_input, (str, Path)) and os.path.exists(str(image_input)):
            p = Path(image_input)
            filename = p.name
            img_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise InvalidSegmentationImageError(f"Failed to decode image file: {p}")
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), filename

        # Base64 string
        if isinstance(image_input, str):
            if image_input.startswith("data:image"):
                _, b64data = image_input.split(",", 1)
            else:
                b64data = image_input
            try:
                raw_bytes = base64.b64decode(b64data)
            except Exception as e:
                raise InvalidSegmentationImageError(f"Base64 decoding failed: {e}")
        elif isinstance(image_input, bytes):
            raw_bytes = image_input
        else:
            raise InvalidSegmentationImageError(f"Unsupported image input type: {type(image_input)}")

        try:
            arr = np.frombuffer(raw_bytes, dtype=np.uint8)
            img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise ValueError("cv2.imdecode returned None")
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), filename
        except Exception as e:
            raise InvalidSegmentationImageError(f"Image decode failed: {e}")

    def run_inference(
        self,
        image_input: Union[str, bytes, np.ndarray, Path],
        filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes native 256x1600 segmentation inference using exact Phase 4B preprocessing.
        Returns per-class pixel counts, percentages, and base64 PNG mask.
        """
        if self._model is None:
            self.load_model()

        img_rgb, detected_filename = self.decode_input_image(image_input)
        final_filename = filename or detected_filename

        orig_h, orig_w = img_rgb.shape[:2]
        total_pixels = orig_h * orig_w

        # Preprocessing: exactly ImageNet mean/std normalization
        img_norm = (img_rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        inp_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float()

        env_dev = os.environ.get("STEELGUARD_DEVICE", "").strip().lower()
        if env_dev == "cpu":
            self._device = "cpu"

        device = torch.device(self._device)
        is_cuda = self._device.startswith("cuda")
        if self._model is not None:
            param = next(self._model.parameters(), None)
            if param is not None and param.device != device:
                self._model.to(device)
        inp_tensor = inp_tensor.to(device)

        t0 = time.perf_counter()

        try:
            with torch.inference_mode():
                if is_cuda:
                    with torch.amp.autocast("cuda"):
                        logits = self._model(inp_tensor)
                else:
                    logits = self._model(inp_tensor)

            # Initial multiclass prediction via argmax across 5 output channels
            pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        except torch.cuda.OutOfMemoryError as e:
            logger.warning(f"CUDA OOM in segmentation inference: {e}. Falling back to CPU.")
            torch.cuda.empty_cache()
            self._model.to("cpu")
            self._device = "cpu"
            with torch.inference_mode():
                logits = self._model(inp_tensor.cpu())
            pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        t1 = time.perf_counter()
        inference_time_ms = round((t1 - t0) * 1000.0, 2)

        # Class pixel breakdown
        class_metrics = []
        total_defect_pixels = 0
        active_classes_detected = []

        for cls_id in (0, 1, 2, 3, 4):
            c_info = SEVERSTAL_CLASS_MAP[cls_id]
            px_count = int(np.sum(pred_mask == cls_id))
            area_pct = round((px_count / total_pixels) * 100.0, 4)

            if cls_id > 0:
                total_defect_pixels += px_count
                if px_count > 0:
                    active_classes_detected.append(c_info["name"])

            class_metrics.append({
                "class_id": cls_id,
                "code": c_info["code"],
                "name": c_info["name"],
                "display_name": c_info["display_name"],
                "color": c_info["color"],
                "criticality": c_info["criticality"],
                "pixel_count": px_count,
                "area_percentage": area_pct
            })

        defect_percentage = round((total_defect_pixels / total_pixels) * 100.0, 4)
        has_defect = total_defect_pixels > 0

        # Encode predicted mask as PNG (grayscale uint8 indexed mask)
        _, mask_png_bytes = cv2.imencode(".png", pred_mask)
        mask_base64 = base64.b64encode(mask_png_bytes).decode("ascii")

        # Create colorized overlay preview (RGB)
        color_mask = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        for cls_id in (1, 2, 3, 4):
            c_rgb = SEVERSTAL_CLASS_MAP[cls_id]["color_rgb"]
            color_mask[pred_mask == cls_id] = c_rgb

        _, color_png_bytes = cv2.imencode(".png", cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))
        color_mask_base64 = base64.b64encode(color_png_bytes).decode("ascii")

        return {
            "status": "SUCCESS",
            "model_type": "SEGMENTATION_UNET",
            "model_name": self._model_info["name"],
            "model_version": self._model_info["version"],
            "checkpoint_sha256": EXPECTED_SHA256,
            "filename": final_filename,
            "image_width": orig_w,
            "image_height": orig_h,
            "inference_time_ms": inference_time_ms,
            "device_used": self._device,
            "has_defect": has_defect,
            "total_defect_pixels": total_defect_pixels,
            "defect_percentage": defect_percentage,
            "active_classes": active_classes_detected,
            "class_breakdown": class_metrics,
            "mask_png_base64": mask_base64,
            "color_mask_base64": color_mask_base64,
            "taxonomy": "Severstal 4-Class Industrial Steel Strip Defect Ontology",
            "is_experimental": True
        }


# Global singleton instance
segmentation_manager = SegmentationManager()


def list_test_samples(defect_class: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
    """
    Returns curated list of Severstal test split samples from test_manifest.csv.
    Allows filtering by defect_class: 'all', 'clean', '1', '2', '3', '4', 'multiclass'.
    """
    manifest_path = DATASET_EXP_DIR / "test_manifest.csv"
    if not manifest_path.is_file():
        return {"total_samples": 0, "samples": []}

    samples = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            comb = row["defect_combination"]
            if defect_class and defect_class.lower() != "all":
                target = defect_class.lower()
                if target == "clean" and comb != "clean":
                    continue
                elif target == "multiclass" and ("+" not in comb and ";" not in row.get("defect_classes", "")):
                    continue
                elif target in {"1", "2", "3", "4"} and target not in row.get("defect_classes", "").split(";"):
                    continue

            samples.append({
                "sample_id": row["image_id"],
                "has_defect": row["has_defect"] == "1",
                "defect_combination": comb,
                "defect_classes": row["defect_classes"],
                "total_defect_pixels": int(row["total_defect_pixels"]),
                "defect_pixel_fraction": float(row["defect_pixel_fraction"]),
                "class_1_pixels": int(row["class_1_pixels"]),
                "class_2_pixels": int(row["class_2_pixels"]),
                "class_3_pixels": int(row["class_3_pixels"]),
                "class_4_pixels": int(row["class_4_pixels"]),
                "width": int(row["width"]),
                "height": int(row["height"])
            })
            if len(samples) >= limit:
                break

    return {
        "dataset": "Severstal Steel Strip Defect Segmentation",
        "split": "test",
        "filter": defect_class or "all",
        "count": len(samples),
        "samples": samples
    }


def get_sample_image_path(sample_id: str) -> Path:
    """Safely resolves image path for a Severstal sample, preventing path traversal."""
    safe_name = Path(sample_id).name
    img_path = (DATASET_EXP_DIR / "images" / safe_name).resolve()
    images_base = (DATASET_EXP_DIR / "images").resolve()
    try:
        img_path.relative_to(images_base)
        if img_path.is_file():
            return img_path
    except ValueError:
        pass

    # Check demo_samples fallback
    demo_img = (PROJECT_ROOT / "demo_samples" / safe_name).resolve()
    if demo_img.is_file():
        return demo_img

    raise FileNotFoundError(f"Sample image '{safe_name}' not found.")


def get_sample_mask_path(sample_id: str) -> Path:
    """Safely resolves ground-truth mask path for a Severstal sample."""
    safe_name = Path(sample_id).name
    stem = Path(safe_name).stem
    mask_path = (DATASET_EXP_DIR / "masks" / f"{stem}.png").resolve()
    masks_base = (DATASET_EXP_DIR / "masks").resolve()
    try:
        mask_path.relative_to(masks_base)
        if mask_path.is_file():
            return mask_path
    except ValueError:
        pass

    # Check demo_samples fallback
    demo_mask = (PROJECT_ROOT / "demo_samples" / f"{stem}.png").resolve()
    if demo_mask.is_file():
        return demo_mask

    raise FileNotFoundError(f"Sample ground-truth mask '{stem}.png' not found.")


def get_ground_truth_for_sample(sample_id: str) -> Dict[str, Any]:
    """Loads ground-truth mask for sample and computes class metrics and color mask."""
    mask_path = get_sample_mask_path(sample_id)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"Failed to read mask at {mask_path}")

    h, w = mask.shape[:2]
    total_pixels = h * w
    class_metrics = []
    total_defect_pixels = 0
    active_classes = []

    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id in (0, 1, 2, 3, 4):
        c_info = SEVERSTAL_CLASS_MAP[cls_id]
        px_count = int(np.sum(mask == cls_id))
        area_pct = round((px_count / total_pixels) * 100.0, 4)
        if cls_id > 0:
            total_defect_pixels += px_count
            if px_count > 0:
                active_classes.append(c_info["name"])
            color_mask[mask == cls_id] = c_info["color_rgb"]
        class_metrics.append({
            "class_id": cls_id,
            "name": c_info["name"],
            "color": c_info["color"],
            "pixel_count": px_count,
            "area_percentage": area_pct
        })

    _, mask_png = cv2.imencode(".png", mask)
    _, color_png = cv2.imencode(".png", cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))

    return {
        "sample_id": sample_id,
        "width": w,
        "height": h,
        "has_defect": total_defect_pixels > 0,
        "total_defect_pixels": total_defect_pixels,
        "defect_percentage": round((total_defect_pixels / total_pixels) * 100.0, 4),
        "active_classes": active_classes,
        "class_breakdown": class_metrics,
        "mask_png_base64": base64.b64encode(mask_png).decode("ascii"),
        "color_mask_base64": base64.b64encode(color_png).decode("ascii")
    }

