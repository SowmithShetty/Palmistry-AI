# Palmistry AI 🖐️✨

A real-time desktop application that reads your palm using computer vision and AI. It detects your hand via a webcam, extracts the palm region, isolates the major palm lines (Heart, Head, Life), and generates a live palmistry reading.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange?logo=google&logoColor=white)

---

## 🎯 Features

- **Real-time hand tracking** — 21-point landmark detection via MediaPipe
- **Palm ROI extraction** — Intelligent cropping that isolates the palm, excluding fingers
- **Advanced image processing** — CLAHE contrast enhancement + Canny edge detection
- **Live palmistry reading** — Heuristic analysis of Heart, Head, and Life lines
- **Interactive controls** — Real-time Canny threshold adjustment via trackbars
- **Debounced overlay** — Smooth, flicker-free reading panel on the live feed

## 📸 How It Works

```
Webcam Feed → Hand Detection → Palm Cropping → CLAHE + Canny → Line Analysis → Reading Overlay
```

| Window | Description |
|--------|-------------|
| **Palmistry AI** | Main feed with landmarks, bounding box, and reading panel |
| **Palm ROI** | Cropped palm region in colour |
| **Palm Lines (Edges)** | Binary edge map showing detected palm lines |
| **Edge Controls** | Trackbars to tune Canny thresholds in real-time |

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** — [Download here](https://www.python.org/downloads/)
- **Webcam** — Built-in or external USB camera
- **Git** — [Download here](https://git-scm.com/)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/SowmithShetty/Palmistry-AI.git
cd Palmistry-AI

# 2. (Recommended) Create a virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python main.py
```

> **Note:** The MediaPipe hand landmarker model (~7.8 MB) will be **auto-downloaded** on first run if not already present.

### Controls

| Key / Control | Action |
|---------------|--------|
| `q` | Quit the application |
| Canny Low trackbar | Adjust lower edge detection threshold (0–300) |
| Canny High trackbar | Adjust upper edge detection threshold (0–500) |

---

## 🗂️ Project Structure

```
Palmistry-AI/
├── main.py              # App loop, UI overlays, window management
├── tracker.py           # MediaPipe HandLandmarker wrapper + ROI extraction
├── processor.py         # CLAHE → Gaussian blur → Canny edge pipeline
├── analyzer.py          # Zone-based heuristic engine + rules lookup
├── rules.json           # Palmistry trait definitions (Heart, Head, Life)
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## 🧠 How the Reading Works

The palm ROI is divided into **three horizontal zones**, each mapped to a major palm line:

| Zone | Line | Governs |
|------|------|---------|
| Top third | **Heart Line** | Emotions, relationships, emotional intelligence |
| Middle third | **Head Line** | Intellect, decision-making, mental clarity |
| Bottom third | **Life Line** | Vitality, life changes, physical well-being |

For each zone, the analyzer measures:
- **Edge density** — How prominent the lines are (strong vs faint)
- **Edge extent** — How far the lines span horizontally (long vs short)

These measurements are mapped to personality traits defined in `rules.json`.

---

## ⚙️ Tech Stack

| Technology | Purpose |
|------------|---------|
| [OpenCV](https://opencv.org/) | Image processing, webcam capture, UI display |
| [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide) | Hand detection & landmark tracking |
| [NumPy](https://numpy.org/) | Array operations for edge analysis |
| Python `json` | Palmistry rules storage |

## 🛠️ Customization

### Adjust Palmistry Rules
Edit `rules.json` to change the personality traits and descriptions for each line classification.

### Tune Detection Parameters
In `tracker.py`, you can adjust:
- `detection_confidence` — How confident the detector must be (default: 0.7)
- `roi_padding` — How much padding around the palm bounding box (default: 0.15)

### Tune Image Processing
In `processor.py`, you can adjust:
- `clahe_clip_limit` — Contrast enhancement strength (default: 2.5)
- `blur_kernel_size` — Noise reduction level (default: 5)

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

---

*Built with ❤️ using Python, OpenCV, and MediaPipe*