"""
STEELGUARD AI - Backend Server (Stage 4)
Lightweight, robust Python 3.12 standard library HTTP server and REST API.
Integrates verified GC10-DET dataset browser, deterministic Quality Assessment Engine,
and persistent SQLite traceability database with CSV/JSON exports.
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import os
import platform
import shutil
import sys
import time
import urllib.parse
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "data"
WEIGHTS_DIR = BASE_DIR / "weights"
DATA_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset, Quality Assessment, Storage Services, and Model Management
from quality_engine import assess_quality, CONFIGURED_RULES
from inspection_intelligence import summarize_inspection
from advisory_service import advisory_service
from dataset_service import registry, CLASS_METADATA
from model_manager import model_manager, ModelIntegrityError, InvalidImageError
from segmentation_service import (
    segmentation_manager,
    SEVERSTAL_CLASSES,
    InvalidSegmentationImageError,
    SegmentationIntegrityError,
    list_test_samples,
    get_sample_image_path,
    get_sample_mask_path,
    get_ground_truth_for_sample
)
import db

# Initialize database schema on startup
db.init_db()

# Standard GC10-DET Classes for Reference
GC10_CLASSES = [
    {"id": 1, "code": "1_chongkong", "name": "Punching Hole", "criticality": "CRITICAL"},
    {"id": 2, "code": "2_hanfeng", "name": "Welding Line", "criticality": "HIGH"},
    {"id": 3, "code": "3_yueyawan", "name": "Crescent Gap", "criticality": "CRITICAL"},
    {"id": 4, "code": "4_shuiban", "name": "Water Spot", "criticality": "LOW"},
    {"id": 5, "code": "5_youban", "name": "Oil Spot", "criticality": "LOW"},
    {"id": 6, "code": "6_siban", "name": "Silk Spot", "criticality": "MEDIUM"},
    {"id": 7, "code": "7_yiwu", "name": "Inclusion / Foreign Object", "criticality": "HIGH"},
    {"id": 8, "code": "8_yahen", "name": "Indentation / Rolled-in Scale", "criticality": "HIGH"},
    {"id": 9, "code": "9_zhehen", "name": "Crease", "criticality": "MEDIUM"},
    {"id": 10, "code": "10_yaozhe", "name": "Waist Folding", "criticality": "HIGH"},
]


def check_for_model_weights():
    """Checks active model status via model_manager."""
    info = model_manager.get_active_model_info()
    return {
        "loaded": info.get("loaded", False),
        "path": info.get("weights_path"),
        "name": info.get("name"),
        "version": info.get("version"),
        "key": info.get("key"),
        "sha256": info.get("expected_sha256"),
        "device": info.get("device"),
        "device_name": info.get("device_name")
    }


def get_real_system_telemetry():
    """Returns genuine host telemetry without simulated or fake hardware claims."""
    machine_arch = platform.machine()
    is_arm = "arm" in machine_arch.lower() or "aarch64" in machine_arch.lower()

    disk_total_gb = 0.0
    disk_free_gb = 0.0
    try:
        usage = shutil.disk_usage(DATA_DIR)
        disk_total_gb = round(usage.total / (1024 ** 3), 2)
        disk_free_gb = round(usage.free / (1024 ** 3), 2)
    except Exception:
        pass

    model_info = check_for_model_weights()
    vision_status = "READY" if model_info["loaded"] else "NOT_LOADED"
    seg_info = segmentation_manager.get_model_info()
    seg_status = "READY (EXPERIMENTAL)" if seg_info.get("loaded") else "NOT_LOADED"
    adv_status = advisory_service.status()
    llm_ready = adv_status.get("configuration_validated", False)
    llm_configured = adv_status.get("configured", False)
    if llm_ready:
        llm_status = "READY"
    else:
        llm_status = "NOT_DETECTED"

    return {
        "status": "healthy",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": {
            "os": f"{platform.system()} {platform.release()}",
            "architecture": machine_arch,
            "processor": platform.processor() or "Generic x86_64/ARM",
            "python_version": platform.python_version(),
            "is_arm": is_arm,
            "target_env": "ARM Edge Ready" if is_arm else f"Development Host ({machine_arch})",
        },
        "models": {
            "vision": {
                "name": model_info["name"] or "YOLOv8 Nano (GC10-DET)",
                "status": vision_status,
                "version": model_info.get("version"),
                "model_key": model_info.get("key"),
                "checkpoint_path": model_info["path"],
                "sha256": model_info.get("sha256"),
                "device": model_info.get("device"),
                "device_name": model_info.get("device_name"),
                "supported_classes": len(GC10_CLASSES),
                "runtime_ready": model_info["loaded"],
                "inference_engine": f"Ultralytics YOLOv8 ({model_info.get('device_name', 'PyTorch')})" if model_info["loaded"] else "Pending model weight file"
            },
            "segmentation": {
                "name": seg_info.get("name") or "Lightweight U-Net (Severstal)",
                "status": seg_status,
                "version": seg_info.get("version"),
                "model_key": "severstal-unet-v1",
                "checkpoint_path": seg_info.get("weights_path"),
                "sha256": seg_info.get("actual_sha256") or seg_info.get("expected_sha256"),
                "device": seg_info.get("device"),
                "device_name": seg_info.get("device_name"),
                "supported_classes": len(SEVERSTAL_CLASSES),
                "runtime_ready": seg_info.get("loaded", False),
                "inference_engine": f"PyTorch Lightweight U-Net ({seg_info.get('device_name', 'CUDA')})" if seg_info.get("loaded") else "Pending model checkpoint",
                "is_experimental": True,
                "input_resolution": seg_info.get("input_resolution", [256, 1600])
            },
            "llm": {
                "name": adv_status.get("model_id") or "Llama-3.2-1B-Instruct (ExecuTorch)",
                "status": llm_status,
                "runtime": adv_status.get("runtime", "ExecuTorch Edge Runtime"),
                "checkpoint_path": os.environ.get("STEELGUARD_EXECUTORCH_MANIFEST"),
                "runtime_ready": llm_ready,
                "target": adv_status.get("target", "linux-aarch64-cpu"),
                "offline_only": True
            }
        },
        "camera": {
            "status": "OFFLINE",
            "device_id": None,
            "message": "Direct camera feed offline. Local file ingestion active."
        },
        "storage": {
            "data_directory": str(DATA_DIR),
            "total_gb": disk_total_gb,
            "free_gb": disk_free_gb
        }
    }


class SteelGuardRequestHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler for STEELGUARD AI."""

    def handle(self):
        try:
            super().handle()
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            pass

    def _set_headers(self, status=200, content_type="application/json", extra_headers=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PATCH, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(200)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        try:
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length < 0:
                raise ValueError()
        except ValueError:
            self._set_headers(400)
            self.wfile.write(b'{"error":"Invalid Content-Length"}')
            return
        if path == "/api/intelligence/advisory" and content_length > 131072:
            self._set_headers(413)
            self.wfile.write(b'{"error":"Advisory request exceeds size limit"}')
            return
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        try:
            payload = json.loads(body.decode("utf-8")) if body else {}
        except (ValueError, UnicodeError, RecursionError):
            self._set_headers(400, "application/json")
            self.wfile.write(json.dumps({"error": "Invalid JSON format"}).encode("utf-8"))
            return

        if path == "/api/intelligence/advisory":
            try:
                if not isinstance(payload, dict) or set(payload) != {"summary"}:
                    raise ValueError("Expected only a summary object")
                result = advisory_service.advise(payload["summary"])
            except ValueError as exc:
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": str(exc)}).encode("utf-8"))
                return
            self._set_headers(200)
            self.wfile.write(json.dumps(result, allow_nan=False).encode("utf-8"))
            return

        if path == "/api/intelligence/summarize":
            try:
                result = summarize_inspection(payload)
            except ValueError as exc:
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": str(exc)}).encode("utf-8"))
                return
            self._set_headers(200)
            self.wfile.write(json.dumps(result, indent=2).encode("utf-8"))
            return

        # 1. Quality Assessment Engine
        if path == "/api/quality/assess":
            detections = payload.get("detections", [])
            rules_enabled = payload.get("rules_enabled", True)
            result = assess_quality(detections, rules_enabled=rules_enabled)
            self._set_headers(200, "application/json")
            self.wfile.write(json.dumps(result, indent=2).encode("utf-8"))
            return

        # 2. Save Inspection to SQLite (TASK 2)
        if path == "/api/inspections":
            try:
                res = db.save_inspection(payload)
                status_code = 200 if res["success"] else (409 if res.get("already_exists") else 400)
                self._set_headers(status_code, "application/json")
                self.wfile.write(json.dumps(res, indent=2).encode("utf-8"))
            except Exception as e:
                self._set_headers(500, "application/json")
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        # 3. Update Operator Notes & Review Status
        if path == "/api/inspections/notes":
            insp_id = payload.get("inspection_id")
            notes = payload.get("operator_notes", "")
            review_status = payload.get("review_status")
            if not insp_id:
                self._set_headers(400, "application/json")
                self.wfile.write(json.dumps({"error": "Missing 'inspection_id'"}).encode("utf-8"))
                return
            success = db.update_inspection_notes(insp_id, notes, review_status=review_status)
            self._set_headers(200 if success else 404, "application/json")
            self.wfile.write(json.dumps({"success": success, "inspection_id": insp_id}).encode("utf-8"))
            return

        # 4. Model Switching & Rollback
        if path == "/api/models/switch":
            model_key = payload.get("model_key")
            if not model_key:
                self._set_headers(400, "application/json")
                self.wfile.write(json.dumps({"status": "ERROR", "error": "Missing 'model_key' parameter"}).encode("utf-8"))
                return
            try:
                switched = model_manager.switch_model(model_key)
                self._set_headers(200, "application/json")
                self.wfile.write(json.dumps({
                    "status": "SUCCESS",
                    "message": f"Activated model: {switched['name']} ({switched['version']})",
                    "model": switched
                }, indent=2).encode("utf-8"))
            except Exception as e:
                self._set_headers(400, "application/json")
                self.wfile.write(json.dumps({"status": "ERROR", "error": str(e)}).encode("utf-8"))
            return

        # 5. AI Model Inference Request
        if path == "/api/inference/run":
            model_info = model_manager.get_active_model_info()
            if not model_info.get("loaded"):
                self._set_headers(400, "application/json")
                self.wfile.write(json.dumps({
                    "status": "ERROR",
                    "code": "MODEL_NOT_LOADED",
                    "message": "Vision model checkpoint is not loaded. Real inference requires trained YOLO weights placed in steelguard_app/weights/.",
                    "suggestion": "Inspect verified GC10-DET samples to review Pascal VOC Ground Truth annotations."
                }, indent=2).encode("utf-8"))
                return

            image_input = payload.get("image_base64") or payload.get("sample_id") or payload.get("image_path")
            if not image_input:
                self._set_headers(400, "application/json")
                self.wfile.write(json.dumps({
                    "status": "ERROR",
                    "code": "MISSING_IMAGE",
                    "message": "Missing image input. Provide 'image_base64', 'sample_id', or 'image_path' in request body."
                }, indent=2).encode("utf-8"))
                return

            conf = payload.get("confidence_threshold")
            iou = payload.get("iou_threshold")
            filename = payload.get("filename")

            try:
                result = model_manager.run_inference(image_input, conf=conf, iou=iou, filename=filename)

                # Optional automatic or explicit persistence to SQLite database
                if payload.get("save_to_db"):
                    save_payload = {
                        "source": "AI_INFERENCE",
                        "filename": result["filename"],
                        "image_reference": payload.get("sample_id", result["filename"]),
                        "image_width": result["image_width"],
                        "image_height": result["image_height"],
                        "detections": result["detections"],
                        "assessment": result["assessment"],
                        "model_name": result["model_name"],
                        "model_version": result["model_version"],
                        "inference_time_ms": result["inference_time_ms"],
                        "operator_notes": payload.get("operator_notes", "Automated AI inference inspection."),
                        "review_status": "UNREVIEWED"
                    }
                    db_res = db.save_inspection(save_payload)
                    result["db_saved"] = db_res.get("success", False)
                    result["inspection_id"] = db_res.get("inspection_id")

                self._set_headers(200, "application/json")
                self.wfile.write(json.dumps(result, indent=2).encode("utf-8"))
                return
            except InvalidImageError as e:
                self._set_headers(400, "application/json")
                self.wfile.write(json.dumps({
                    "status": "ERROR",
                    "code": "INVALID_IMAGE",
                    "message": f"Invalid or unreadable image input: {str(e)}"
                }, indent=2).encode("utf-8"))
                return
            except ModelIntegrityError as e:
                self._set_headers(500, "application/json")
                self.wfile.write(json.dumps({
                    "status": "ERROR",
                    "code": "MODEL_INTEGRITY_ERROR",
                    "message": f"Model integrity check failed: {str(e)}"
                }, indent=2).encode("utf-8"))
                return
            except Exception as e:
                self._set_headers(500, "application/json")
                self.wfile.write(json.dumps({
                    "status": "ERROR",
                    "code": "INFERENCE_ERROR",
                    "message": f"Inference execution failed: {str(e)}"
                }, indent=2).encode("utf-8"))
                return

        # 6. Severstal U-Net Pixel Segmentation Inference (Phase 4C)
        if path == "/api/segmentation/run":
            seg_info = segmentation_manager.get_model_info()
            if not seg_info.get("loaded"):
                try:
                    seg_info = segmentation_manager.load_model()
                except Exception as e:
                    self._set_headers(500, "application/json")
                    self.wfile.write(json.dumps({
                        "status": "ERROR",
                        "code": "MODEL_NOT_LOADED",
                        "message": f"Segmentation model checkpoint could not be loaded: {str(e)}"
                    }, indent=2).encode("utf-8"))
                    return

            sample_id = payload.get("sample_id")
            image_base64 = payload.get("image_base64")
            image_path = payload.get("image_path")
            filename = payload.get("filename")

            ground_truth_data = None
            if sample_id:
                try:
                    resolved_img_path = get_sample_image_path(sample_id)
                    image_input = resolved_img_path
                    filename = filename or sample_id
                    try:
                        ground_truth_data = get_ground_truth_for_sample(sample_id)
                    except Exception:
                        ground_truth_data = None
                except Exception as e:
                    self._set_headers(400, "application/json")
                    self.wfile.write(json.dumps({
                        "status": "ERROR",
                        "code": "SAMPLE_NOT_FOUND",
                        "message": str(e)
                    }, indent=2).encode("utf-8"))
                    return
            elif image_base64:
                image_input = image_base64
            elif image_path:
                image_input = image_path
            else:
                self._set_headers(400, "application/json")
                self.wfile.write(json.dumps({
                    "status": "ERROR",
                    "code": "MISSING_IMAGE",
                    "message": "Missing image input. Provide 'sample_id', 'image_base64', or 'image_path'."
                }, indent=2).encode("utf-8"))
                return

            try:
                result = segmentation_manager.run_inference(image_input, filename=filename)
                if ground_truth_data:
                    result["ground_truth"] = ground_truth_data

                # Persist to SQLite if requested
                if payload.get("save_to_db"):
                    c_pixels = {c["class_id"]: c["pixel_count"] for c in result.get("class_breakdown", [])}
                    save_payload = {
                        "filename": result["filename"],
                        "image_reference": sample_id or result["filename"],
                        "model_name": result["model_name"],
                        "model_version": result["model_version"],
                        "checkpoint_hash": result["checkpoint_sha256"],
                        "image_width": result["image_width"],
                        "image_height": result["image_height"],
                        "inference_time_ms": result["inference_time_ms"],
                        "class_1_pixels": c_pixels.get(1, 0),
                        "class_2_pixels": c_pixels.get(2, 0),
                        "class_3_pixels": c_pixels.get(3, 0),
                        "class_4_pixels": c_pixels.get(4, 0),
                        "total_defect_pixels": result["total_defect_pixels"],
                        "defect_percentage": result["defect_percentage"],
                        "mask_png_base64": result.get("mask_png_base64"),
                        "operator_notes": payload.get("operator_notes", "Automated U-Net segmentation inspection.")
                    }
                    db_res = db.save_segmentation_inspection(save_payload)
                    result["db_saved"] = db_res.get("success", False)
                    result["inspection_id"] = db_res.get("inspection_id")

                self._set_headers(200, "application/json")
                self.wfile.write(json.dumps(result, indent=2).encode("utf-8"))
                return

            except InvalidSegmentationImageError as e:
                self._set_headers(400, "application/json")
                self.wfile.write(json.dumps({
                    "status": "ERROR",
                    "code": "INVALID_IMAGE",
                    "message": str(e)
                }, indent=2).encode("utf-8"))
                return
            except SegmentationIntegrityError as e:
                self._set_headers(500, "application/json")
                self.wfile.write(json.dumps({
                    "status": "ERROR",
                    "code": "SEGMENTATION_INTEGRITY_ERROR",
                    "message": str(e)
                }, indent=2).encode("utf-8"))
                return
            except Exception as e:
                self._set_headers(500, "application/json")
                self.wfile.write(json.dumps({
                    "status": "ERROR",
                    "code": "INFERENCE_ERROR",
                    "message": f"Segmentation inference failed: {str(e)}"
                }, indent=2).encode("utf-8"))
                return

        self._set_headers(404, "application/json")
        self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode("utf-8"))

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        if path == "/api/intelligence/advisory/status":
            self._set_headers(200)
            self.wfile.write(json.dumps(advisory_service.status()).encode("utf-8"))
            return
        query = urllib.parse.parse_qs(parsed.query)

        # 1. Telemetry & Host Diagnostics
        if path == "/api/health":
            self._set_headers(200, "application/json")
            payload = get_real_system_telemetry()
            self.wfile.write(json.dumps(payload, indent=2).encode("utf-8"))
            return

        # 2. Defect Classes Catalog
        if path == "/api/classes":
            self._set_headers(200, "application/json")
            self.wfile.write(json.dumps({
                "dataset": "GC10-DET",
                "total_classes": len(GC10_CLASSES),
                "classes": GC10_CLASSES
            }, indent=2).encode("utf-8"))
            return

        # 3. Overview Metrics (TASK 6 - Backed by SQLite in Real Mode)
        if path == "/api/overview":
            self._set_headers(200, "application/json")
            demo_requested = query.get("demo", ["false"])[0].lower() == "true"
            
            if not demo_requested:
                stats = db.get_overview_stats()
                recent = db.list_inspections(limit=10)

                recent_list = []
                for r in recent["inspections"]:
                    top_rule = r["applied_rules"][0] if r["applied_rules"] else "None"
                    recent_list.append({
                        "id": r["id"],
                        "timestamp": r["timestamp"][:19].replace("T", " "),
                        "filename": r["filename"],
                        "defect_count": r["total_defects"],
                        "top_defect": top_rule,
                        "assessment": r["assessment_status"],
                        "source": r["source"],
                        "status": r["review_status"]
                    })

                # Merge default GC10 classes with actual detected counts
                class_dist = {cls["name"]: stats["defect_distribution"].get(cls["name"], 0) for cls in GC10_CLASSES}

                data = {
                    "is_demo": False,
                    "metrics": stats,
                    "recent_inspections": recent_list,
                    "defect_distribution": class_dist
                }
            else:
                data = {
                    "is_demo": True,
                    "notice": "DEMO DATA ONLY - Not genuine inspection measurements",
                    "metrics": {
                        "total_inspections": 142,
                        "defects_detected": 38,
                        "requiring_review": 9,
                        "pass_rate_pct": 73.2,
                    },
                    "defect_distribution": {
                        "Punching Hole": 7,
                        "Welding Line": 9,
                        "Crescent Gap": 4,
                        "Water Spot": 11,
                        "Oil Spot": 15,
                        "Silk Spot": 22,
                        "Inclusion / Foreign Object": 8,
                        "Indentation / Rolled-in Scale": 5,
                        "Crease": 3,
                        "Waist Folding": 2
                    },
                    "recent_inspections": [
                        {
                            "id": "SG-DEMO-001",
                            "timestamp": "2026-10-28 21:15:02",
                            "filename": "coil_425501900_sec17.jpg",
                            "defect_count": 2,
                            "top_defect": "Punching Hole",
                            "assessment": "REJECT",
                            "source": "DEMO",
                            "status": "CONFIRMED"
                        },
                        {
                            "id": "SG-DEMO-002",
                            "timestamp": "2026-10-28 20:44:11",
                            "filename": "coil_425502300_sec18.jpg",
                            "defect_count": 1,
                            "top_defect": "Water Spot",
                            "assessment": "REVIEW REQUIRED",
                            "source": "DEMO",
                            "status": "PENDING"
                        },
                        {
                            "id": "SG-DEMO-003",
                            "timestamp": "2026-10-28 19:30:40",
                            "filename": "coil_425503100_sec17.jpg",
                            "defect_count": 0,
                            "top_defect": "None",
                            "assessment": "PASS",
                            "source": "DEMO",
                            "status": "VERIFIED"
                        }
                    ]
                }
            self.wfile.write(json.dumps(data, indent=2).encode("utf-8"))
            return

        # 4. Model Status & Catalog
        if path == "/api/models/status":
            self._set_headers(200, "application/json")
            model_info = model_manager.get_active_model_info()
            seg_info = segmentation_manager.get_model_info()
            catalog = model_manager.get_catalog()
            status = {
                "vision": {
                    "model_id": model_info.get("key", "yolov8n-gc10"),
                    "model_name": model_info.get("name", "YOLOv8 Nano"),
                    "version": model_info.get("version"),
                    "status": "READY" if model_info.get("loaded") else "NOT_LOADED",
                    "weights_path": model_info.get("weights_path"),
                    "sha256": model_info.get("expected_sha256"),
                    "device": model_info.get("device"),
                    "device_name": model_info.get("device_name"),
                    "target_resolution": [2048, 1000],
                    "guidance": "Model weights verified and active." if model_info.get("loaded") else "Place verified model weights in steelguard_app/weights/."
                },
                "segmentation": {
                    "model_id": "severstal-unet-v1",
                    "model_name": seg_info.get("name", "Lightweight U-Net (Severstal)"),
                    "version": seg_info.get("version"),
                    "status": "READY (EXPERIMENTAL)" if seg_info.get("loaded") else "NOT_LOADED",
                    "weights_path": seg_info.get("weights_path"),
                    "sha256": seg_info.get("actual_sha256") or seg_info.get("expected_sha256"),
                    "device": seg_info.get("device"),
                    "device_name": seg_info.get("device_name"),
                    "target_resolution": [256, 1600],
                    "guidance": "Severstal U-Net model verified and active." if seg_info.get("loaded") else "Place verified model weights in steelguard_app/weights/."
                },
                "catalog": catalog,
                "executorch": {
                    "model_id": "llama-executorch",
                    "status": "NOT_DETECTED",
                    "weights_path": None,
                    "guidance": "ExecuTorch local runtime is offline. AI advisory operates in rule-reference mode."
                }
            }
            self.wfile.write(json.dumps(status, indent=2).encode("utf-8"))
            return

        # 4b. Registered Models Catalog
        if path == "/api/models":
            self._set_headers(200, "application/json")
            catalog = model_manager.get_catalog()
            self.wfile.write(json.dumps(catalog, indent=2).encode("utf-8"))
            return

        # 5. Inspection History & Traceability (TASK 3)
        if path == "/api/inspections":
            insp_id = query.get("id", [None])[0]
            if insp_id:
                record = db.get_inspection(insp_id)
                if not record:
                    self._set_headers(404, "application/json")
                    self.wfile.write(json.dumps({"error": f"Inspection '{insp_id}' not found"}).encode("utf-8"))
                    return
                self._set_headers(200, "application/json")
                self.wfile.write(json.dumps(record, indent=2).encode("utf-8"))
                return

            search = query.get("search", [None])[0]
            status_filter = query.get("status", [None])[0]
            source_filter = query.get("source", [None])[0]
            sort_order = query.get("sort", ["DESC"])[0]
            limit = int(query.get("limit", [100])[0])
            offset = int(query.get("offset", [0])[0])

            records = db.list_inspections(
                search=search,
                status=status_filter,
                source=source_filter,
                sort_order=sort_order,
                limit=limit,
                offset=offset
            )
            self._set_headers(200, "application/json")
            self.wfile.write(json.dumps(records, indent=2).encode("utf-8"))
            return

        # 6. CSV Export (TASK 5)
        if path == "/api/inspections/export/csv":
            search = query.get("search", [None])[0]
            status_filter = query.get("status", [None])[0]
            source_filter = query.get("source", [None])[0]
            csv_data = db.export_inspections_csv(search=search, status=status_filter, source=source_filter)
            headers = {
                "Content-Disposition": 'attachment; filename="steelguard_inspections_export.csv"'
            }
            self._set_headers(200, "text/csv; charset=utf-8", extra_headers=headers)
            self.wfile.write(csv_data.encode("utf-8"))
            return

        # 7. JSON Export (TASK 5)
        if path == "/api/inspections/export/json":
            insp_id = query.get("id", [None])[0]
            if not insp_id:
                self._set_headers(400, "application/json")
                self.wfile.write(json.dumps({"error": "Missing 'id' parameter"}).encode("utf-8"))
                return
            record = db.get_inspection(insp_id)
            if not record:
                self._set_headers(404, "application/json")
                self.wfile.write(json.dumps({"error": f"Inspection '{insp_id}' not found"}).encode("utf-8"))
                return
            headers = {
                "Content-Disposition": f'attachment; filename="inspection_{insp_id}.json"'
            }
            self._set_headers(200, "application/json", extra_headers=headers)
            self.wfile.write(json.dumps(record, indent=2).encode("utf-8"))
            return

        # 8. Verified Dataset Sample Browser (TASK 1)
        if path == "/api/samples":
            self._set_headers(200, "application/json")
            defect_class = query.get("class", [None])[0]
            limit_str = query.get("limit", ["100"])[0]
            try:
                limit = min(200, max(1, int(limit_str)))
            except ValueError:
                limit = 100

            samples = registry.list_samples(defect_class=defect_class, limit=limit)
            summary = registry.get_summary()
            self.wfile.write(json.dumps({
                "total_verified_dataset": summary["total_verified_samples"],
                "class_counts": summary["class_counts"],
                "returned_count": len(samples),
                "samples": samples
            }, indent=2).encode("utf-8"))
            return

        # 9. Sample Image Streaming (Safe against traversal)
        if path == "/api/samples/image":
            sample_id = query.get("id", [None])[0]
            if not sample_id:
                self._set_headers(400, "application/json")
                self.wfile.write(json.dumps({"error": "Missing 'id' parameter"}).encode("utf-8"))
                return
            try:
                img_path = registry.get_sample_image_path(sample_id)
                ext = img_path.suffix.lower()
                mime = "image/jpeg" if ext in {".jpg", ".jpeg"} else "image/png"
                with open(img_path, "rb") as f:
                    content = f.read()
                self._set_headers(200, mime)
                self.wfile.write(content)
            except KeyError:
                self._set_headers(404, "application/json")
                self.wfile.write(json.dumps({"error": f"Sample '{sample_id}' not found in approved registry"}).encode("utf-8"))
            except (PermissionError, ValueError):
                self._set_headers(403, "application/json")
                self.wfile.write(json.dumps({"error": "Path traversal access denied"}).encode("utf-8"))
            except Exception as e:
                self._set_headers(500, "application/json")
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        # 10. Sample Pascal VOC Ground-Truth Annotation
        if path == "/api/samples/annotation":
            sample_id = query.get("id", [None])[0]
            if not sample_id:
                self._set_headers(400, "application/json")
                self.wfile.write(json.dumps({"error": "Missing 'id' parameter"}).encode("utf-8"))
                return
            try:
                anno = registry.get_sample_annotation(sample_id)
                self._set_headers(200, "application/json")
                self.wfile.write(json.dumps(anno, indent=2).encode("utf-8"))
            except KeyError:
                self._set_headers(404, "application/json")
                self.wfile.write(json.dumps({"error": f"Sample '{sample_id}' not found in approved registry"}).encode("utf-8"))
            except Exception as e:
                self._set_headers(500, "application/json")
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        # 11. Quality Assessment Rules Catalog
        if path == "/api/quality/rules":
            self._set_headers(200, "application/json")
            self.wfile.write(json.dumps({
                "rules_active": True,
                "total_rules": len(CONFIGURED_RULES),
                "rules": CONFIGURED_RULES
            }, indent=2).encode("utf-8"))
            return

        # 12. Segmentation Model Info & Status (Phase 4C)
        if path in ("/api/segmentation/status", "/api/segmentation/info"):
            self._set_headers(200, "application/json")
            info = segmentation_manager.get_model_info()
            self.wfile.write(json.dumps(info, indent=2).encode("utf-8"))
            return

        # 13. Segmentation 4-Class Taxonomy
        if path == "/api/segmentation/classes":
            self._set_headers(200, "application/json")
            self.wfile.write(json.dumps({
                "dataset": "Severstal Steel Strip Defect Segmentation",
                "total_classes": len(SEVERSTAL_CLASSES),
                "classes": SEVERSTAL_CLASSES
            }, indent=2).encode("utf-8"))
            return

        # 14. Severstal Test Samples Browser
        if path == "/api/segmentation/samples":
            self._set_headers(200, "application/json")
            defect_class = query.get("class", [None])[0]
            limit_str = query.get("limit", ["50"])[0]
            try:
                limit = min(200, max(1, int(limit_str)))
            except ValueError:
                limit = 50
            samples_data = list_test_samples(defect_class=defect_class, limit=limit)
            self.wfile.write(json.dumps(samples_data, indent=2).encode("utf-8"))
            return

        # 15. Severstal Sample Image Streaming
        if path == "/api/segmentation/samples/image":
            sample_id = query.get("id", [None])[0]
            if not sample_id:
                self._set_headers(400, "application/json")
                self.wfile.write(json.dumps({"error": "Missing 'id' parameter"}).encode("utf-8"))
                return
            try:
                img_path = get_sample_image_path(sample_id)
                ext = img_path.suffix.lower()
                mime = "image/jpeg" if ext in {".jpg", ".jpeg"} else "image/png"
                with open(img_path, "rb") as f:
                    content = f.read()
                self._set_headers(200, mime)
                self.wfile.write(content)
            except FileNotFoundError:
                self._set_headers(404, "application/json")
                self.wfile.write(json.dumps({"error": f"Sample '{sample_id}' not found"}).encode("utf-8"))
            except PermissionError:
                self._set_headers(403, "application/json")
                self.wfile.write(json.dumps({"error": "Path traversal access denied"}).encode("utf-8"))
            except Exception as e:
                self._set_headers(500, "application/json")
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        # 16. Severstal Sample Ground Truth Mask
        if path == "/api/segmentation/samples/ground_truth":
            sample_id = query.get("id", [None])[0]
            if not sample_id:
                self._set_headers(400, "application/json")
                self.wfile.write(json.dumps({"error": "Missing 'id' parameter"}).encode("utf-8"))
                return
            try:
                gt = get_ground_truth_for_sample(sample_id)
                self._set_headers(200, "application/json")
                self.wfile.write(json.dumps(gt, indent=2).encode("utf-8"))
            except FileNotFoundError:
                self._set_headers(404, "application/json")
                self.wfile.write(json.dumps({"error": f"Ground truth for '{sample_id}' not found"}).encode("utf-8"))
            except Exception as e:
                self._set_headers(500, "application/json")
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        # 17. Segmentation Inspection History
        if path == "/api/segmentation/inspections":
            limit = int(query.get("limit", [50])[0])
            offset = int(query.get("offset", [0])[0])
            records = db.list_segmentation_inspections(limit=limit, offset=offset)
            self._set_headers(200, "application/json")
            self.wfile.write(json.dumps(records, indent=2).encode("utf-8"))
            return

        # 18. Segmentation Inspection Detail
        if path == "/api/segmentation/inspections/detail":
            insp_id = query.get("id", [None])[0]
            if not insp_id:
                self._set_headers(400, "application/json")
                self.wfile.write(json.dumps({"error": "Missing 'id' parameter"}).encode("utf-8"))
                return
            record = db.get_segmentation_inspection(insp_id)
            if not record:
                self._set_headers(404, "application/json")
                self.wfile.write(json.dumps({"error": f"Segmentation inspection '{insp_id}' not found"}).encode("utf-8"))
                return
            self._set_headers(200, "application/json")
            self.wfile.write(json.dumps(record, indent=2).encode("utf-8"))
            return

        # Static File Serving
        clean_path = path.lstrip("/")
        if not clean_path or clean_path == "":
            clean_path = "index.html"

        file_path = (STATIC_DIR / clean_path).resolve()

        # Prevent directory traversal
        try:
            file_path.relative_to(STATIC_DIR)
        except ValueError:
            self._set_headers(403, "text/plain")
            self.wfile.write(b"403 Forbidden")
            return

        # Quietly handle favicon requests if file is absent
        if clean_path == "favicon.ico" and not file_path.exists():
            self._set_headers(204, "image/x-icon")
            return

        if not file_path.exists() or not file_path.is_file():
            if "." not in clean_path:
                file_path = STATIC_DIR / "index.html"
            else:
                self._set_headers(404, "text/plain")
                self.wfile.write(b"404 Not Found")
                return

        ext = file_path.suffix.lower()
        mime_types = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".json": "application/json; charset=utf-8",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".svg": "image/svg+xml",
            ".ico": "image/x-icon",
        }
        content_type = mime_types.get(ext, "application/octet-stream")

        try:
            with open(file_path, "rb") as f:
                content = f.read()
            self._set_headers(200, content_type)
            self.wfile.write(content)
        except Exception as e:
            self._set_headers(500, "text/plain")
            self.wfile.write(f"500 Internal Error: {e}".encode("utf-8"))

    def log_message(self, format, *args):
        sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {args[0]} {args[1]} -> {args[2]}\n")


def create_server(host="127.0.0.1", port=8080):
    """Create and bind a standard library HTTPServer instance on host:port."""
    server_address = (host, port)
    return HTTPServer(server_address, SteelGuardRequestHandler)


def run(port=8080, host="127.0.0.1", on_ready=None):
    """Bind and serve standard library HTTPServer on host:port.
    
    Accepts both positional and keyword arguments for host and port.
    Calls on_ready(httpd) after socket binding succeeds before entering serve_forever().
    """
    httpd = create_server(host=host, port=port)
    if on_ready:
        on_ready(httpd)
    else:
        print("=" * 60)
        print(" STEELGUARD AI - Industrial Steel Inspection Server")
        print(f" URL: http://{host}:{port}/")
        print(" Mode: Stage 4 Traceability Store & Formal Inspection Reports")
        print(f" Storage: SQLite Database active at {DATA_DIR / 'steelguard.db'}")
        print("=" * 60)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server cleanly...")
        httpd.server_close()


if __name__ == "__main__":
    port = int(os.environ.get("STEELGUARD_PORT", 8080))
    host = os.environ.get("STEELGUARD_HOST", "127.0.0.1")
    run(port=port, host=host)
