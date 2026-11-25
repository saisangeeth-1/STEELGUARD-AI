/**
 * STEELGUARD AI - Interactive Steel Inspection Canvas Viewer
 * Provides high-precision zooming, panning, and bi-directional bounding-box synchronization.
 */

class SteelViewer {
  constructor(canvasId, containerId) {
    this.canvas = document.getElementById(canvasId);
    this.container = document.getElementById(containerId);
    this.ctx = this.canvas.getContext("2d");

    this.image = null;
    this.annotations = [];
    this.selectedId = null;

    // Viewport transform state
    this.scale = 1.0;
    this.panX = 0;
    this.panY = 0;

    // Display flags
    this.showBBoxes = true;
    this.rawMode = false;

    // Drag / Pan interaction state
    this.isDragging = false;
    this.dragStart = { x: 0, y: 0 };
    this.lastPan = { x: 0, y: 0 };

    // Callback on box click
    this.onSelectDetection = null;

    this._bindEvents();
    window.addEventListener("resize", () => this.handleResize());
  }

  setImage(img, annotations = []) {
    this.image = img;
    this.annotations = annotations;
    this.selectedId = null;
    this.fitToScreen();
  }

  setAnnotations(annotations) {
    this.annotations = annotations;
    this.render();
  }

  setSelectedId(id) {
    this.selectedId = id;
    this.render();
  }

  setShowBBoxes(val) {
    this.showBBoxes = val;
    this.render();
  }

  setRawMode(val) {
    this.rawMode = val;
    this.render();
  }

  fitToScreen() {
    if (!this.image || !this.container) return;

    const cw = this.container.clientWidth;
    const ch = this.container.clientHeight;

    this.canvas.width = cw;
    this.canvas.height = ch;

    const scaleX = (cw - 20) / this.image.width;
    const scaleY = (ch - 20) / this.image.height;
    this.scale = Math.min(scaleX, scaleY, 1.0);

    // Center image
    this.panX = Math.round((cw - this.image.width * this.scale) / 2);
    this.panY = Math.round((ch - this.image.height * this.scale) / 2);

    this.render();
    this._updateZoomText();
  }

  setScale(scale, originX = null, originY = null) {
    if (!this.image) return;
    const prevScale = this.scale;
    this.scale = Math.max(0.1, Math.min(scale, 8.0));

    if (originX !== null && originY !== null) {
      // Zoom centered on specified point
      const mouseX = originX - this.panX;
      const mouseY = originY - this.panY;
      this.panX -= (mouseX / prevScale) * (this.scale - prevScale);
      this.panY -= (mouseY / prevScale) * (this.scale - prevScale);
    } else {
      // Zoom centered on canvas center
      const cx = this.canvas.width / 2 - this.panX;
      const cy = this.canvas.height / 2 - this.panY;
      this.panX -= (cx / prevScale) * (this.scale - prevScale);
      this.panY -= (cy / prevScale) * (this.scale - prevScale);
    }

    this.render();
    this._updateZoomText();
  }

  zoomBy(delta, originX = null, originY = null) {
    this.setScale(this.scale + delta, originX, originY);
  }

  zoomToBox(bbox_xyxy) {
    if (!this.image || !bbox_xyxy) return;
    const [xmin, ymin, xmax, ymax] = bbox_xyxy;
    const boxW = Math.max(20, xmax - xmin);
    const boxH = Math.max(20, ymax - ymin);
    const boxCenterX = xmin + boxW / 2;
    const boxCenterY = ymin + boxH / 2;

    const cw = this.canvas.width;
    const ch = this.canvas.height;

    // Calculate scale to focus nicely on the defect
    const targetScale = Math.min(cw / (boxW * 3.5), ch / (boxH * 3.5), 2.5);
    this.scale = Math.max(0.5, targetScale);

    this.panX = Math.round(cw / 2 - boxCenterX * this.scale);
    this.panY = Math.round(ch / 2 - boxCenterY * this.scale);

    this.render();
    this._updateZoomText();
  }

  render() {
    if (!this.canvas || !this.ctx) return;
    const ctx = this.ctx;
    const cw = this.canvas.width;
    const ch = this.canvas.height;

    ctx.clearRect(0, 0, cw, ch);

    if (!this.image) return;

    // Draw background texture grid
    ctx.fillStyle = "#03060a";
    ctx.fillRect(0, 0, cw, ch);

    // Save context for transform
    ctx.save();

    // Draw Steel Image
    const imgW = Math.round(this.image.width * this.scale);
    const imgH = Math.round(this.image.height * this.scale);
    ctx.drawImage(this.image, this.panX, this.panY, imgW, imgH);

    // Subtle image border
    ctx.strokeStyle = "#1e2c4a";
    ctx.lineWidth = 1;
    ctx.strokeRect(this.panX, this.panY, imgW, imgH);

    // Draw Bounding Boxes if enabled
    if (this.showBBoxes && !this.rawMode && this.annotations && this.annotations.length > 0) {
      this.annotations.forEach((det) => {
        const isSelected = det.detection_id === this.selectedId;
        const [xmin, ymin, xmax, ymax] = det.bbox_xyxy;

        const bx = this.panX + xmin * this.scale;
        const by = this.panY + ymin * this.scale;
        const bw = (xmax - xmin) * this.scale;
        const bh = (ymax - ymin) * this.scale;

        const color = isSelected ? "#00d2ff" : (det.color || "#ef4444");

        // Box border
        ctx.strokeStyle = color;
        ctx.lineWidth = isSelected ? 3 : 2;
        ctx.strokeRect(bx, by, bw, bh);

        // Highlight glow / corners if selected
        if (isSelected) {
          ctx.fillStyle = "rgba(0, 210, 255, 0.15)";
          ctx.fillRect(bx, by, bw, bh);

          // Corner brackets
          const cornerLen = Math.min(10, bw / 2, bh / 2);
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth = 2;
          ctx.beginPath();
          // Top-left
          ctx.moveTo(bx, by + cornerLen); ctx.lineTo(bx, by); ctx.lineTo(bx + cornerLen, by);
          // Top-right
          ctx.moveTo(bx + bw - cornerLen, by); ctx.lineTo(bx + bw, by); ctx.lineTo(bx + bw, by + cornerLen);
          // Bottom-left
          ctx.moveTo(bx, by + bh - cornerLen); ctx.lineTo(bx, by + bh); ctx.lineTo(bx + cornerLen, by + bh);
          // Bottom-right
          ctx.moveTo(bx + bw - cornerLen, by + bh); ctx.lineTo(bx + bw, by + bh); ctx.lineTo(bx + bw, by + bh - cornerLen);
          ctx.stroke();
        }

        // Label pill
        const labelText = `${det.class_name}`;
        ctx.font = "bold 11px monospace";
        const textMetrics = ctx.measureText(labelText);
        const pillW = textMetrics.width + 12;
        const pillH = 18;

        const pillY = by > 20 ? by - pillH : by + bh;

        ctx.fillStyle = isSelected ? "#00d2ff" : color;
        ctx.fillRect(bx, pillY, pillW, pillH);

        ctx.fillStyle = "#000000";
        ctx.fillText(labelText, bx + 6, pillY + 13);
      });
    }

    ctx.restore();
  }

  _bindEvents() {
    const canvas = this.canvas;

    // Mouse Wheel Zoom
    canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      const delta = e.deltaY < 0 ? 0.15 : -0.15;
      this.zoomBy(delta, mouseX, mouseY);
    }, { passive: false });

    // Drag to Pan
    canvas.addEventListener("mousedown", (e) => {
      if (e.button === 0) { // Left click
        this.isDragging = true;
        this.dragStart = { x: e.clientX, y: e.clientY };
        this.lastPan = { x: this.panX, y: this.panY };
        canvas.style.cursor = "grabbing";
      }
    });

    window.addEventListener("mousemove", (e) => {
      if (!this.isDragging) return;
      const dx = e.clientX - this.dragStart.x;
      const dy = e.clientY - this.dragStart.y;
      this.panX = this.lastPan.x + dx;
      this.panY = this.lastPan.y + dy;
      this.render();
    });

    window.addEventListener("mouseup", (e) => {
      if (this.isDragging) {
        this.isDragging = false;
        canvas.style.cursor = "crosshair";

        // Check if it was a click (not a drag) to detect bounding box selection
        const dist = Math.hypot(e.clientX - this.dragStart.x, e.clientY - this.dragStart.y);
        if (dist < 4) {
          const rect = canvas.getBoundingClientRect();
          const clickX = e.clientX - rect.left;
          const clickY = e.clientY - rect.top;
          this._handleCanvasClick(clickX, clickY);
        }
      }
    });
  }

  _handleCanvasClick(canvasX, canvasY) {
    if (!this.annotations || this.annotations.length === 0) return;

    // Convert canvas click to image coordinates
    const imgX = (canvasX - this.panX) / this.scale;
    const imgY = (canvasY - this.panY) / this.scale;

    // Find clicked detection
    let clickedDet = null;
    for (let i = this.annotations.length - 1; i >= 0; i--) {
      const d = this.annotations[i];
      const [xmin, ymin, xmax, ymax] = d.bbox_xyxy;
      if (imgX >= xmin && imgX <= xmax && imgY >= ymin && imgY <= ymax) {
        clickedDet = d;
        break;
      }
    }

    if (clickedDet) {
      this.setSelectedId(clickedDet.detection_id);
      if (this.onSelectDetection) {
        this.onSelectDetection(clickedDet.detection_id);
      }
    } else {
      this.setSelectedId(null);
      if (this.onSelectDetection) {
        this.onSelectDetection(null);
      }
    }
  }

  _updateZoomText() {
    const zoomText = document.getElementById("zoom-level-text");
    if (zoomText) {
      zoomText.textContent = `${Math.round(this.scale * 100)}%`;
    }
  }

  handleResize() {
    if (!this.container || !this.canvas) return;
    const cw = this.container.clientWidth;
    const ch = this.container.clientHeight;
    this.canvas.width = cw;
    this.canvas.height = ch;
    this.render();
  }
}
