"""
STEELGUARD AI - Centralized Model Management & Inference Service
Provides safe checkpoint loading, cryptographic SHA-256 verification, model switching/rollback,
local YOLOv8 inference, image validation, and deterministic quality assessment integration.
"""

import os
# Enforce strict offline operation and prevent Ultralytics AutoUpdate
os.environ["YOLO_OFFLINE"] = "True"
os.environ["ULTRALYTICS_AUTOUPDATE"] = "0"
os.environ["YOLO_VERBOSE"] = "False"

import base64
import hashlib
import io
import json
import logging
from pathlib import Path
import time
from typing import Dict, List, Optional, Any, Tuple, Union

import numpy as np
from PIL import Image
import torch

try:
    from ultralytics import YOLO
    import ultralytics.utils.patches as patches
    # Prevent Ultralytics monkey-patch from calling check_requirements('pi-heif')
    patches._pil_plugins_registered = True
    if hasattr(patches, '_image_open'):
        Image.open = patches._image_open
except ImportError:
    YOLO = None

from quality_engine import assess_quality

logger = logging.getLogger("steelguard.model_manager")

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent
WEIGHTS_DIR = BASE_DIR / "weights"
CONFIG_FILE = BASE_DIR / "data" / "model_config.json"
CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)

# 10 Standard GC10-DET Classes with Ontology Metadata
GC10_CLASS_CATALOG = [
    {
        "class_id": 0,
        "defect_number": 1,
        "code": "1_chongkong",
        "name": "Punching Hole",
        "criticality": "CRITICAL",
        "color": "#ef4444",
        "description": "Perforation defect compromising tensile integrity."
    },
    {
        "class_id": 1,
        "defect_number": 2,
        "code": "2_hanfeng",
        "name": "Welding Line",
        "criticality": "HIGH",
        "color": "#ec4899",
        "description": "Transverse weld joint seam requiring shear trimming."
    },
    {
        "class_id": 2,
        "defect_number": 3,
        "code": "3_yueyawan",
        "name": "Crescent Gap",
        "criticality": "CRITICAL",
        "color": "#f97316",
        "description": "Lateral strip edge tear liable to cause line breaks."
    },
    {
        "class_id": 3,
        "defect_number": 4,
        "code": "4_shuiban",
        "name": "Water Spot",
        "criticality": "LOW",
        "color": "#38bdf8",
        "description": "Surface moisture evaporation residue."
    },
    {
        "class_id": 4,
        "defect_number": 5,
        "code": "5_youban",
        "name": "Oil Spot",
        "criticality": "LOW",
        "color": "#fbbf24",
        "description": "Residual rolling lubricant fluid stain."
    },
    {
        "class_id": 5,
        "defect_number": 6,
        "code": "6_siban",
        "name": "Silk Spot",
        "criticality": "MEDIUM",
        "color": "#a855f7",
        "description": "Surface finish streaks affecting appearance/coating."
    },
    {
        "class_id": 6,
        "defect_number": 7,
        "code": "7_yiwu",
        "name": "Inclusion",
        "criticality": "HIGH",
        "color": "#e11d48",
        "description": "Non-metallic metallurgical inclusion or foreign particle."
    },
    {
        "class_id": 7,
        "defect_number": 8,
        "code": "8_yahen",
        "name": "Rolled Pit",
        "criticality": "HIGH",
        "color": "#eab308",
        "description": "Indentation from rolled-in scale or work roll damage."
    },
    {
        "class_id": 8,
        "defect_number": 9,
        "code": "9_zhehen",
        "name": "Crease",
        "criticality": "MEDIUM",
        "color": "#14b8a6",
        "description": "Longitudinal fold from strip tension instability."
    },
    {
        "class_id": 9,
        "defect_number": 10,
        "code": "10_yaozhe",
        "name": "Waist Folding",
        "criticality": "HIGH",
        "color": "#f43f5e",
        "description": "Transverse buckling across strip width."
    }
]

CLASS_MAP_BY_ID = {c["class_id"]: c for c in GC10_CLASS_CATALOG}
CLASS_MAP_BY_CODE = {c["code"]: c for c in GC10_CLASS_CATALOG}
# Handle dataset alias
CLASS_MAP_BY_CODE["10_yaozhed"] = CLASS_MAP_BY_CODE["10_yaozhe"]

# Registered Production & Rollback Models
REGISTERED_MODELS: Dict[str, Dict[str, Any]] = {
    "candidate": {
        "key": "candidate",
        "role": "ACTIVE_PRODUCTION_CANDIDATE",
        "name": "YOLOv8 Nano (NEU-Inclusion Augmented)",
        "version": "v2.0-neu-inclusion",
        "filename": "steelguard_gc10_neu_inclusion_best.pt",
        "expected_sha256": "41ce4fd1099db9bce7e93b734bcfba40810fa365f1e58a82c30da8a24a3f8e2c",
        "original_source": str(PROJECT_ROOT / "runs" / "detect" / "steelguard_gc10_neu_inclusion" / "weights" / "best.pt"),
        "description": "Fine-tuned on GC10-DET augmented with 382 NEU-DET Inclusion samples. Inclusion AP@50 increased to 30.54% (+15.56%), test mAP@50 is 64.14%.",
        "is_rollback": False,
        "experiment_id": "STEELGUARD-EXP-NEU-INCLUSION-001"
    },
    "baseline": {
        "key": "baseline",
        "role": "VERIFIED_ROLLBACK_BASELINE",
        "name": "YOLOv8 Nano (GC10 Baseline)",
        "version": "v1.0-gc10-baseline",
        "filename": "steelguard_gc10n_baseline_best.pt",
        "expected_sha256": "ffacba4cc1a781d6fd023baf04ccb135890cbd4edf9d78cdab5bb89cbab08ee6",
        "original_source": str(PROJECT_ROOT / "runs" / "detect" / "steelguard_gc10n" / "weights" / "best.pt"),
        "description": "Original baseline model trained exclusively on standard GC10-DET dataset (50 epochs). Test mAP@50 is 60.63%, Inclusion AP@50 is 14.98%.",
        "is_rollback": True,
        "experiment_id": "STEELGUARD-BASELINE-GC10N"
    }
}


class ModelIntegrityError(Exception):
    """Raised when checkpoint weights fail SHA-256 verification or class ontology checks."""
    pass


class InvalidImageError(Exception):
    """Raised when an uploaded or selected image cannot be decoded or is corrupt."""
    pass


class ModelManager:
    """
    Centralized manager handling model configuration, verification, switching,
    and local inference execution.
    """

    def __init__(self):
        self._active_model_key: str = "candidate"
        self._active_model: Optional[Any] = None
        self._active_model_info: Optional[Dict[str, Any]] = None
        self._device: str = "auto"
        self._conf_threshold: float = 0.25
        self._iou_threshold: float = 0.45
        self._imgsz: int = 640
        self._last_error: Optional[str] = None
        self._load_config()

    def _load_config(self):
        """Loads configuration from persistent JSON file if present."""
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    self._active_model_key = cfg.get("active_model_key", "candidate")
                    self._conf_threshold = float(cfg.get("confidence_threshold", 0.25))
                    self._iou_threshold = float(cfg.get("iou_threshold", 0.45))
                    self._imgsz = int(cfg.get("imgsz", 640))
                    self._device = cfg.get("device", "auto")
            except Exception as e:
                logger.warning(f"Could not read config file: {e}. Using defaults.")

    def _save_config(self):
        """Persists current configuration to JSON file."""
        try:
            cfg = {
                "active_model_key": self._active_model_key,
                "confidence_threshold": self._conf_threshold,
                "iou_threshold": self._iou_threshold,
                "imgsz": self._imgsz,
                "device": self._device,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save model config: {e}")

    def compute_sha256(self, file_path: Union[str, Path]) -> str:
        """Computes SHA-256 hash of a file."""
        p = Path(file_path)
        if not p.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {p}")
        hasher = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def get_effective_device(self) -> str:
        """Determines effective inference device (CPU forced or CUDA GPU prioritized)."""
        env_dev = os.environ.get("STEELGUARD_DEVICE", "").strip().lower()
        if env_dev:
            return env_dev
        if self._device == "cpu":
            return "cpu"
        if self._device == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        if self._device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda is not available.")
        return self._device

    def load_active_model(self, force_reload: bool = False) -> Any:
        """
        Loads the active model with strict cryptographic and class verification.
        Does NOT silently fallback to another model if loading fails.
        """
        if self._active_model is not None and not force_reload:
            return self._active_model

        if self._active_model_key not in REGISTERED_MODELS:
            err = f"Configured model key '{self._active_model_key}' is not in registered catalog."
            self._last_error = err
            raise ModelIntegrityError(err)

        info = REGISTERED_MODELS[self._active_model_key]
        weights_path = WEIGHTS_DIR / info["filename"]

        if not weights_path.exists():
            # If not in weights/, check if source exists and copy it
            src = Path(info["original_source"])
            if src.exists():
                import shutil
                WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, weights_path)
            else:
                err = f"Checkpoint file missing: {weights_path} (source {src} also missing)"
                self._last_error = err
                raise FileNotFoundError(err)

        # Cryptographic verification
        actual_hash = self.compute_sha256(weights_path)
        expected_hash = info["expected_sha256"]
        if actual_hash != expected_hash:
            err = (
                f"Model integrity violation for {info['filename']}: "
                f"Expected SHA-256 {expected_hash}, got {actual_hash}. "
                f"Model loading aborted to prevent unverified weights execution."
            )
            self._last_error = err
            raise ModelIntegrityError(err)

        if YOLO is None:
            err = "Ultralytics package is not installed."
            self._last_error = err
            raise ImportError(err)

        device = self.get_effective_device()
        try:
            model = YOLO(str(weights_path))
            
            # Verify class names count
            names = model.names
            if len(names) != 10:
                err = f"Model ontology mismatch: Expected 10 GC10 defect classes, got {len(names)}: {names}"
                self._last_error = err
                raise ModelIntegrityError(err)

            # Warmup inference
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            model.predict(source=dummy, device=device, verbose=False, imgsz=640)

            self._active_model = model
            self._active_model_info = {
                **info,
                "weights_path": str(weights_path),
                "actual_sha256": actual_hash,
                "verified": True,
                "device": device,
                "device_name": torch.cuda.get_device_name(0) if "cuda" in device else "CPU Host",
                "loaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "num_classes": len(names),
                "classes": names
            }
            self._last_error = None
            logger.info(f"Loaded active model: {info['name']} on {device}")
            return self._active_model

        except Exception as e:
            self._active_model = None
            self._active_model_info = None
            self._last_error = str(e)
            raise

    def switch_model(self, model_key: str) -> Dict[str, Any]:
        """
        Switches the active model (e.g. candidate <-> baseline rollback).
        """
        if model_key not in REGISTERED_MODELS:
            raise ValueError(f"Unknown model key '{model_key}'. Available: {list(REGISTERED_MODELS.keys())}")

        self._active_model_key = model_key
        self._active_model = None
        self._active_model_info = None
        self._save_config()

        # Load new model immediately to verify
        self.load_active_model(force_reload=True)
        return self.get_active_model_info()

    def get_active_model_info(self) -> Dict[str, Any]:
        """Returns details about the currently active model."""
        if self._active_model is None:
            try:
                self.load_active_model()
            except Exception as e:
                return {
                    "loaded": False,
                    "key": self._active_model_key,
                    "name": REGISTERED_MODELS.get(self._active_model_key, {}).get("name", "Unknown"),
                    "status": "ERROR",
                    "error": str(e),
                    "weights_path": str(WEIGHTS_DIR / REGISTERED_MODELS.get(self._active_model_key, {}).get("filename", "")),
                    "expected_sha256": REGISTERED_MODELS.get(self._active_model_key, {}).get("expected_sha256")
                }

        return {
            "loaded": True,
            "status": "READY",
            **self._active_model_info
        }

    def get_catalog(self) -> Dict[str, Any]:
        """Returns the full catalog of models with their status."""
        active_key = self._active_model_key
        models_list = []
        for key, info in REGISTERED_MODELS.items():
            weights_path = WEIGHTS_DIR / info["filename"]
            exists = weights_path.is_file()
            hash_match = False
            curr_hash = None
            if exists:
                try:
                    curr_hash = self.compute_sha256(weights_path)
                    hash_match = (curr_hash == info["expected_sha256"])
                except Exception:
                    pass

            models_list.append({
                "key": key,
                "name": info["name"],
                "version": info["version"],
                "role": info["role"],
                "is_active": (key == active_key),
                "is_rollback": info["is_rollback"],
                "filename": info["filename"],
                "expected_sha256": info["expected_sha256"],
                "actual_sha256": curr_hash,
                "integrity_verified": hash_match,
                "weights_exist": exists,
                "description": info["description"],
                "experiment_id": info["experiment_id"]
            })

        return {
            "active_model_key": active_key,
            "total_models": len(models_list),
            "models": models_list,
            "device": self.get_effective_device(),
            "confidence_threshold": self._conf_threshold,
            "iou_threshold": self._iou_threshold
        }

    def decode_image_input(self, image_input: Union[str, bytes, Image.Image, np.ndarray]) -> Tuple[Image.Image, str]:
        """
        Decodes and validates various image inputs into a PIL Image.
        Returns (PIL.Image, source_descriptor).
        """
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB"), "PIL_IMAGE"

        if isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                img = Image.fromarray(image_input).convert("RGB")
            elif image_input.ndim == 3:
                img = Image.fromarray(image_input).convert("RGB")
            else:
                raise InvalidImageError(f"Invalid NumPy array shape: {image_input.shape}")
            return img, "NUMPY_ARRAY"

        if isinstance(image_input, bytes):
            if len(image_input) == 0:
                raise InvalidImageError("Image byte stream is empty.")
            try:
                img = Image.open(io.BytesIO(image_input))
                img.verify()  # Verify integrity
                # Reopen because verify closes/invalidates the image object
                img = Image.open(io.BytesIO(image_input)).convert("RGB")
                return img, "RAW_BYTES"
            except Exception as e:
                raise InvalidImageError(f"Failed to decode image bytes: {e}")

        if isinstance(image_input, (str, Path)):
            clean_str = str(image_input).strip()
            # 1. Base64 data URL or raw base64
            if clean_str.startswith("data:image") or len(clean_str) > 500:
                try:
                    payload = clean_str
                    if "base64," in payload:
                        payload = payload.split("base64,")[1]
                    raw_bytes = base64.b64decode(payload)
                    return self.decode_image_input(raw_bytes)[0], "BASE64_PAYLOAD"
                except Exception as e:
                    raise InvalidImageError(f"Failed to decode base64 image data: {e}")

            # 2. File path
            p = Path(clean_str)
            if p.is_file():
                try:
                    with open(p, "rb") as f:
                        return self.decode_image_input(f.read())[0], f"FILE:{p.name}"
                except Exception as e:
                    raise InvalidImageError(f"Failed to open image file {p}: {e}")

            # 3. Check dataset registry if sample ID
            from dataset_service import registry
            try:
                sample_path = registry.get_sample_image_path(clean_str)
                if sample_path and sample_path.is_file():
                    with open(sample_path, "rb") as f:
                        return self.decode_image_input(f.read())[0], f"SAMPLE:{clean_str}"
            except Exception:
                pass

            raise InvalidImageError(f"String input could not be resolved as valid image file, sample ID, or base64: {clean_str[:50]}...")

        raise InvalidImageError(f"Unsupported image input type: {type(image_input)}")

    def run_inference(
        self,
        image_input: Union[str, bytes, Image.Image, np.ndarray],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes local YOLO inference and connects results to deterministic quality assessment.

        Returns structured inference result dictionary with defect instances,
        bounding boxes, confidences, latency, and quality verdict.
        """
        model = self.load_active_model()
        pil_img, source_desc = self.decode_image_input(image_input)

        img_w, img_h = pil_img.size
        if img_w <= 0 or img_h <= 0:
            raise InvalidImageError(f"Invalid image dimensions: {img_w}x{img_h}")

        conf_thresh = conf if conf is not None else self._conf_threshold
        iou_thresh = iou if iou is not None else self._iou_threshold
        device = self.get_effective_device()

        # Synchronize CUDA if GPU is used for accurate latency measurement
        if "cuda" in device and torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        # Run Ultralytics YOLO inference
        results = model.predict(
            source=pil_img,
            conf=conf_thresh,
            iou=iou_thresh,
            imgsz=self._imgsz,
            device=device,
            verbose=False
        )

        if "cuda" in device and torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_s = time.perf_counter() - start_time
        latency_ms = round(elapsed_s * 1000.0, 2)

        # Parse detections
        detections = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            for idx, box in enumerate(boxes):
                cls_id = int(box.cls[0].item())
                conf_val = float(box.conf[0].item())
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = [int(round(coord)) for coord in xyxy]

                # Constrain within image boundaries
                x1 = max(0, min(img_w, x1))
                y1 = max(0, min(img_h, y1))
                x2 = max(0, min(img_w, x2))
                y2 = max(0, min(img_h, y2))
                bw = max(0, x2 - x1)
                bh = max(0, y2 - y1)
                area = bw * bh

                meta = CLASS_MAP_BY_ID.get(cls_id, {
                    "code": f"class_{cls_id}",
                    "name": f"Unknown Defect ({cls_id})",
                    "criticality": "MEDIUM",
                    "color": "#e11d48"
                })

                det_id = f"AI-{idx + 1:03d}"
                detections.append({
                    "detection_id": det_id,
                    "class_id": cls_id,
                    "class_name": meta["name"],
                    "raw_label": meta["code"],
                    "criticality": meta["criticality"],
                    "color": meta["color"],
                    "confidence": round(conf_val, 4),
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "bbox_wh": [bw, bh],
                    "area_px": area,
                    "is_ground_truth": 0
                })

        # Deterministic Quality Assessment (Zero LLM hallucinations)
        assessment = assess_quality(detections, rules_enabled=True)

        return {
            "status": "SUCCESS",
            "model_name": self._active_model_info["name"],
            "model_version": self._active_model_info["version"],
            "model_key": self._active_model_info["key"],
            "checkpoint_filename": self._active_model_info["filename"],
            "sha256": self._active_model_info["expected_sha256"],
            "device": device,
            "inference_time_ms": latency_ms,
            "image_width": img_w,
            "image_height": img_h,
            "filename": filename or source_desc,
            "source": "AI_INFERENCE",
            "total_defects": len(detections),
            "detections": detections,
            "assessment": assessment,
            "confidence_threshold": conf_thresh,
            "iou_threshold": iou_thresh
        }


# Global Singleton Instance
model_manager = ModelManager()
