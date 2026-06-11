<div align="center">

```
████████╗██████╗ ██╗   ██╗███████╗███████╗██╗ ██████╗ ██╗  ██╗████████╗
╚══██╔══╝██╔══██╗██║   ██║██╔════╝██╔════╝██║██╔════╝ ██║  ██║╚══██╔══╝
   ██║   ██████╔╝██║   ██║█████╗  ███████╗██║██║  ███╗███████║   ██║   
   ██║   ██╔══██╗██║   ██║██╔══╝  ╚════██║██║██║   ██║██╔══██║   ██║   
   ██║   ██║  ██║╚██████╔╝███████╗███████║██║╚██████╔╝██║  ██║   ██║   
   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝  
```

### **AI-Powered Deepfake & Synthetic Video Detection**
*Forensic-grade authenticity verdicts. Built for the age of misinformation.*

<br>

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-Web_App-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![Roboflow](https://img.shields.io/badge/Roboflow-AI_Inference-purple?style=for-the-badge)](https://roboflow.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-Video_Processing-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![License](https://img.shields.io/badge/License-Open_Source-22c55e?style=for-the-badge)](LICENSE)

</div>

---

## ◈ What Is TrueSight?

> *"Seeing through the synthetic."*

TrueSight AI is a forensic video analysis platform that detects AI-generated deepfakes using multi-frame inference. Upload a video or drop in a YouTube link — TrueSight extracts strategic frames, runs each through a trained Roboflow model, and delivers a verdict backed by confidence scores, risk classification, and a downloadable PDF forensic report.

Built for a **Cyber Hackathon** targeting the growing threat of AI-generated media in fraud, identity attacks, and disinformation campaigns.

---

## ◈ Capabilities

<table>
<tr>
<td width="50%">

### 🎥 Input Modes
- Upload video files directly from your device
- Analyze any **YouTube URL** via `yt-dlp`

### 🔍 Scan Depths
| Mode | Frames | Best For |
|------|--------|----------|
| **Quick Scan** | 3 frames | Fast screening |
| **Deep Scan** | 7 frames | Standard review |
| **Ultra Scan** | 10 frames | Full forensic analysis |

</td>
<td width="50%">

### 🧠 Detection Intelligence
- Frame-by-frame **AI vs Real** classification
- Per-frame confidence scoring
- **Low-Confidence Override** — if ≥2 frames score below 60%, the entire video is flagged as AI-generated

### 📄 Forensic PDF Report Includes
- Unique case ID & metadata
- Frame-by-frame visual evidence
- Risk level: `Minimal` → `Moderate` → `High` → `Critical`
- Recommended actions
- Temporal & audio analysis notes
- Explainable AI evidence section

</td>
</tr>
</table>

---

## ◈ Tech Stack

```
┌─────────────────────────────────────────────────────────┐
│                     TRUESIGHT STACK                     │
├──────────────────┬──────────────────────────────────────┤
│  Backend         │  Python · Flask · Waitress (WSGI)    │
│  AI Inference    │  Roboflow SDK · custom-workflow-2    │
│  Video Engine    │  OpenCV (headless) · yt-dlp          │
│  Report Gen      │  ReportLab (PDF)                     │
│  Frontend        │  HTML · CSS · JavaScript (SPA)       │
│  Config          │  python-dotenv                       │
└──────────────────┴──────────────────────────────────────┘
```

---

## ◈ How It Works

```
  [ Video File / YouTube URL ]
            │
            ▼
  ┌─────────────────────┐
  │   Frame Extraction  │  ← OpenCV pulls N frames at strategic timestamps
  └────────┬────────────┘
           │
           ▼
  ┌─────────────────────┐
  │   AI Inference      │  ← Each frame → Roboflow serverless endpoint
  └────────┬────────────┘
           │
           ▼
  ┌──────────────────────────────────────────────────┐
  │   Verdict Logic                                  │
  │   ≥2 frames < 60% confidence?  →  FLAG AS AI    │
  │   Otherwise?  →  Highest-confidence prediction  │
  └────────┬─────────────────────────────────────────┘
           │
           ▼
  ┌─────────────────────┐
  │   PDF Report        │  ← Forensic-grade export with full evidence chain
  └─────────────────────┘
```

---

## ◈ Project Structure

```
truesight/
├── app.py                  ← Flask application & inference logic
├── requirements.txt        ← Python dependencies
├── runtime.txt             ← Python version pin
├── .env.example            ← Environment variable template
├── AI.jpg                  ← Sample AI-generated reference image
├── REAL.jpg                ← Sample authentic reference image
│
├── templates/
│   └── index.html          ← Single-page web interface
│
└── static/
    ├── frames/             ← Auto-created: extracted video frames
    └── reports/            ← Auto-created: generated PDF reports
```

---

## ◈ Getting Started

### 1 — Clone

```bash
git clone https://github.com/Vikasthangavel/true_sight_AI.git
cd true_sight_AI
```

### 2 — Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### 4 — Set your API key

Create a `.env` file in the project root:

```env
ROBOFLOW_API_KEY=your_roboflow_api_key_here
```

> ⚠️ Never commit your `.env` file — it is already listed in `.gitignore`.
>
> Get a free API key at [roboflow.com](https://roboflow.com)

### 5 — Run

```bash
python app.py
```

App starts at **`http://localhost:5000`**

---

## ◈ Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `ROBOFLOW_API_KEY` | Roboflow API key for model inference | ✅ Yes |

---

## ◈ Dependencies

```
flask                     # Web framework
opencv-python-headless    # Video frame extraction
yt-dlp                    # YouTube video downloading
inference-sdk             # Roboflow inference client
reportlab                 # PDF report generation
python-dotenv             # Environment config
waitress                  # Production WSGI server
gunicorn                  # Alternative WSGI server
requests                  # HTTP client
```

---

## ◈ Disclaimer

TrueSight AI is a forensic **aid** tool — not a sole arbiter of authenticity. Results are probabilistic. Always combine AI analysis with human judgment and independent verification before drawing legal or editorial conclusions.

---

<div align="center">

**Built for the [Cyber Hackathon] — fighting synthetic media, one frame at a time.**

*Made with intent for a safer digital world.*

</div>