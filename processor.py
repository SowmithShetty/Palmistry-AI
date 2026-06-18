"""
processor.py — Image preprocessing & deep learning-based edge detection for palm line isolation.

This module replaces the traditional Canny edge detection pipeline with a neural network
running Holistically-Nested Edge Detection (HED) locally in real-time.

It handles:
  1. Grayscale conversion.
  2. Contrast enhancement via CLAHE.
  3. ImageNet mean subtraction and forward pass through the Caffe HED model.
  4. Rescaling and thresholding the edge map to feed the palmistry analyzer.
"""

import os
import sys
import cv2
import numpy as np

# ─────────────────────────────────────────────
# HED Model Configuration & Auto-Download URLs
# ─────────────────────────────────────────────
_PROTO_PATH = os.path.join(os.path.dirname(__file__), "deploy.prototxt")
_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "hed_pretrained_bsds.caffemodel")

_PROTO_URL = "https://raw.githubusercontent.com/ashukid/hed-edge-detector/master/deploy.prototxt"
_WEIGHTS_URL = "https://github.com/ashukid/hed-edge-detector/raw/master/hed_pretrained_bsds.caffemodel"


def _ensure_hed_models() -> None:
    """Download the HED model definition and weights if they don't already exist."""
    if os.path.isfile(_PROTO_PATH) and os.path.isfile(_WEIGHTS_PATH):
        return

    import urllib.request
    
    if not os.path.isfile(_PROTO_PATH):
        print("[INFO] HED prototxt config not found. Downloading...")
        try:
            urllib.request.urlretrieve(_PROTO_URL, _PROTO_PATH)
            print(f"[INFO] HED prototxt saved to: {_PROTO_PATH}")
        except Exception as e:
            print(f"[ERROR] Failed to download HED prototxt: {e}")
            sys.exit(1)

    if not os.path.isfile(_WEIGHTS_PATH):
        print("[INFO] HED pretrained weights not found. Downloading (~29 MB)...")
        try:
            urllib.request.urlretrieve(_WEIGHTS_URL, _WEIGHTS_PATH)
            print(f"[INFO] HED weights saved to: {_WEIGHTS_PATH}")
        except Exception as e:
            print(f"[ERROR] Failed to download HED weights: {e}")
            sys.exit(1)


# ─────────────────────────────────────────────
# Custom HED Crop Layer for OpenCV DNN
# ─────────────────────────────────────────────
class CropLayer(object):
    """
    Custom crop layer required because the standard Caffe Crop layer 
    is not parsed automatically by OpenCV's Caffe importer.
    """
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]
        
        self.ystart = int((inputShape[2] - targetShape[2]) / 2)
        self.xstart = int((inputShape[3] - targetShape[3]) / 2)
        self.yend = self.ystart + height
        self.xend = self.xstart + width
        
        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:, :, self.ystart:self.yend, self.xstart:self.xend]]


# Register the Crop layer with OpenCV's DNN module
try:
    cv2.dnn_registerLayer('Crop', CropLayer)
except Exception:
    # If the module is reloaded, it might already be registered
    pass


class PalmProcessor:
    """
    Multi-stage image processing pipeline using deep learning-based HED
    (Holistically-Nested Edge Detection) for palm line extraction.
    """

    def __init__(
        self,
        clahe_clip_limit: float = 2.5,
        clahe_tile_size: tuple[int, int] = (8, 8),
    ):
        # Auto-download HED models if they are missing
        _ensure_hed_models()

        self._clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_tile_size,
        )
        
        # Load the HED neural network using OpenCV DNN
        self.net = cv2.dnn.readNetFromCaffe(_PROTO_PATH, _WEIGHTS_PATH)
        
        # HED is typically trained on 256x256 or 500x500. 256x256 runs faster on CPU.
        self.dnn_width = 256
        self.dnn_height = 256

    def process(self, roi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run HED deep learning edge detection on a palm ROI image.

        Parameters
        ----------
        roi : np.ndarray
            The cropped palm region in BGR colour (from tracker.get_palm_roi).

        Returns
        -------
        (grayscale, clahe_enhanced, edges)
            - grayscale: raw grayscale image
            - clahe_enhanced: after CLAHE enhancement
            - edges: final thresholded binary HED edge map
        """
        # Step 1: Grayscale conversion
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Step 2: CLAHE (Contrast Enhancement)
        enhanced = self._clahe.apply(gray)
        
        # Step 3: Deep learning HED edge detection
        # HED requires a 3-channel BGR input
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        h, w = roi.shape[:2]
        
        # Preprocess the image to construct an input blob
        blob = cv2.dnn.blobFromImage(
            enhanced_bgr,
            scalefactor=1.0,
            size=(self.dnn_width, self.dnn_height),
            mean=(104.00698793, 116.66876762, 122.67891434),  # ImageNet RGB means
            swapRB=False,
            crop=False
        )
        
        self.net.setInput(blob)
        hed_output = self.net.forward()
        
        # Rescale the output (squeezed to single channel)
        hed_edges = np.squeeze(hed_output)
        
        # Resize HED output back to the original ROI size
        hed_edges = cv2.resize(hed_edges, (w, h))
        
        # Convert back to CV_8U format [0, 255]
        hed_edges = (hed_edges * 255.0).astype(np.uint8)
        
        # Apply a binary threshold to get a clean binary edge map
        # HED outputs probabilities; thresholding at 100/255 maps directly to main lines
        _, binary_edges = cv2.threshold(hed_edges, 100, 255, cv2.THRESH_BINARY)

        return gray, enhanced, binary_edges

    def destroy_windows(self) -> None:
        """No-op, as no trackbar windows are created."""
        pass
