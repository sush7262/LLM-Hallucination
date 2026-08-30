# Claim-Level Hallucination Detection in RAG Systems

An entailment-based verification module designed for Retrieval-Augmented Generation (RAG) pipelines. It decomposes RAG-generated answers into atomic factual claims, aligns each claim with relevant retrieved context, and performs 3-way NLI classification (`supported`, `unsupported`, `contradicted`) using a fine-tuned DeBERTa model.

---

## 🚀 Quick Setup & Installation Guide

Anyone cloning or pulling this project can get it running in 3 simple steps!

### Prerequisites
- **Python 3.10+** (Python 3.11, 3.12, or 3.13 recommended)
- **Git**

---

### Step 1: Clone the Repository

```bash
git clone https://github.com/sush7262/LLM-Hallucination.git
cd LLM-Hallucination
```

---

### Step 2: Create & Activate Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### Step 3: Install Required Dependencies

First, install PyTorch (with CUDA support if GPU is available, or CPU):

```bash
# PyTorch with CUDA 12.6 (or default pip install torch for CPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Then install all remaining project dependencies:

```bash
pip install -r requirements.txt
```

---

## 🏃 How to Run the Application

### 1. Start the FastAPI Backend Server

```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```
> *Note: On first startup, the backend will automatically download the lightweight SentenceTransformer and DeBERTa models (~500MB total) from HuggingFace.*

---

### 2. Open the Web Application UI

Open `frontend/index.html` directly in any web browser (Chrome, Edge, Firefox, Safari):
- Simply double-click `frontend/index.html` or drag it into your browser.
- Click **"Example A"** or **"Example B"** preset buttons to test instant verification!

---

## 🔑 Optional: Gemini LLM Decomposition Mode

By default, the system uses **Rule-Based Decomposition** (fast, offline, no API key required). 

If you wish to use **LLM (Gemini) Decomposition**, create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```

---

## 🛠️ Tech Stack & Models Used

- **Backend Framework**: FastAPI + Pydantic v2
- **Claim Alignment Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Entailment Classifier Model**: `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`
- **Frontend Dashboard**: Glassmorphism Single Page Application (HTML5 / Vanilla CSS / JavaScript + Canvas Chart)
