# 🛡️ TrueSight AI — Video Authenticity Detector

<p align="center">
  <img src="AI.jpg" alt="TrueSight AI Banner" width="600"/>
</p>

> **TrueSight AI** is an AI-powered deepfake and synthetic video detection platform built for the modern era of misinformation. Powered by Roboflow's inference engine, it analyzes video frames with multi-mode scanning to deliver a forensic-grade authenticity verdict — complete with a downloadable PDF report.

---

## 🚀 Features

- 🎥 **Video Upload & YouTube URL Analysis** — Analyze videos directly from your device or via YouTube links
- 🔍 **Multi-Mode Scanning**
  - **Quick Scan** — 3 strategic frames (fast)
  - **Deep Scan** — 7 frames (thorough)
  - **Ultra Scan** — 10 frames (most comprehensive)
- 🤖 **AI vs Real Classification** — Frame-by-frame detection using Roboflow's custom workflow
- 📊 **Confidence Scoring** — Per-frame confidence scores with aggregate verdict logic
- 🧠 **Low-Confidence Override Rule** — Flags content as AI-generated if ≥2 frames score below 60%
- 📄 **PDF Forensic Report Generation** — Downloadable professional-grade report with:
  - Case metadata & report ID
  - Executive summary
  - Frame-by-frame visual evidence
  - Risk classification (Minimal → Critical)
  - Recommended actions
  - Temporal & audio analysis notes
  - Explainable AI evidence
- 🌐 **Web Interface** — Clean, single-page Flask web app with a modern dark UI

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python, Flask, Waitress (WSGI) |
| **AI Inference** | Roboflow Inference SDK (`inference-sdk`) |
| **Video Processing** | OpenCV (`opencv-python-headless`), yt-dlp |
| **PDF Generation** | ReportLab |
| **Frontend** | HTML, CSS, JavaScript (single-page) |
| **Config** | python-dotenv |

---

## 📁 Project Structure

```
detetor/
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
├── runtime.txt             # Python version pin
├── .env.example            # Environment variable template (see below)
├── AI.jpg                  # Sample AI-generated image asset
├── REAL.jpg                # Sample real image asset
├── templates/
│   └── index.html          # Main web UI
├── static/
│   ├── frames/             # Extracted video frames (auto-created)
│   └── reports/            # Generated PDF reports (auto-created)
└── uploads/                # Temporary video uploads (auto-created)
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/Vikasthangavel/true_sight_AI.git
cd true_sight_AI
```

### 2. Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root (see `.env.example`):

```env
ROBOFLOW_API_KEY=your_roboflow_api_key_here
```

> ⚠️ **Never commit your `.env` file.** It is listed in `.gitignore`.

### 5. Run the application

```bash
python app.py
```

The app will start on `http://localhost:5000` by default.

---

## 🔑 Environment Variables

| Variable | Description | Required |
|---|---|---|
| `ROBOFLOW_API_KEY` | Your Roboflow API key for inference | ✅ Yes |

Get your free API key at [roboflow.com](https://roboflow.com).

---

## 🧪 How It Works

1. **Input** — User uploads a video file or provides a YouTube URL
2. **Frame Extraction** — OpenCV extracts N frames (based on scan mode) at strategic timestamps
3. **AI Inference** — Each frame is sent to a Roboflow custom workflow (`custom-workflow-2`) running on `serverless.roboflow.com`
4. **Verdict Logic**:
   - If ≥2 frames have confidence < 60% → All frames flagged as **AI**
   - Otherwise → Verdict based on the highest-confidence prediction
5. **Report** — Results compiled into a detailed verdict with optional PDF export

---

## 📦 Dependencies

```
flask
opencv-python-headless
requests
yt-dlp
reportlab
python-dotenv
gunicorn
inference-sdk
waitress
```

---

## 📋 .env.example

```env
# Roboflow API Configuration
ROBOFLOW_API_KEY=your_api_key_here
```

---

## 🏆 Built For

This project was built for a **Cyber Hackathon** — targeting the growing threat of AI-generated deepfake videos in misinformation campaigns, fraud, and identity attacks.

---

## ⚖️ Disclaimer

TrueSight AI is a forensic aid tool and should not be the sole basis for legal decisions. Results are probabilistic. Always combine AI analysis with human judgment and additional verification steps.

---

## 📄 License

This project is open source. See [LICENSE](LICENSE) for details.

---

<p align="center">Made with ❤️ for a safer digital world</p>
