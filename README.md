# 📄 DocLens - Intelligent Document Analysis

DocLens is a RAG-powered document analysis tool. Ask questions about any PDF and get intelligent, context-aware answers powered by Groq's LLaMA model.

---

## 🏗️ Architecture

```
[Streamlit Frontend]  ──►  [FastAPI Backend on Vercel]  ──►  [Groq LLM API]
  (Streamlit Cloud)              (Serverless)
```

---

## 🚀 Deployment Guide

### Step 1 — Deploy Backend to Vercel

1. Push this project to a GitHub repository
2. Go to [vercel.com](https://vercel.com) → **Add New Project** → import your repo
3. In the Vercel dashboard, go to **Settings → Environment Variables** and add:
   - `GROQ_API_KEY` → your key from [groq.com](https://console.groq.com)
4. Click **Deploy**
5. Note your backend URL: `https://your-project.vercel.app`

### Step 2 — Deploy Frontend to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub account and select this repo
3. Set **Main file path** to: `streamlit_app.py`
4. Under **Advanced settings → Secrets**, add:
   ```
   BACKEND_DOCLENS_API_URL = "https://your-project.vercel.app/DocLens"
   ```
5. Click **Deploy**

---

## 🛠️ Local Development

```bash
# Install backend deps
pip install -r requirements.txt

# Create .env from example
cp .env.example .env
# Fill in GROQ_API_KEY in .env

# Terminal 1 - Backend
python main.py

# Terminal 2 - Frontend
streamlit run streamlit_app.py
```

---

## 📊 API Reference

### `POST /DocLens`
```json
{
  "documents": "https://example.com/file.pdf",
  "questions": ["What is the coverage limit?", "What are exclusions?"]
}
```
**Response:**
```json
{
  "answers": ["The coverage limit is ...", "Exclusions include ..."]
}
```

### `GET /health`
Returns `{"status": "healthy"}`.

---

## ⚠️ Vercel Limitations to Know

- **60s max timeout** on Pro plan (10s on Hobby) — large PDFs may time out
- **No persistent storage** — each request is stateless
- **Bundle size limit** — heavy ML libs (FAISS, sentence-transformers) are excluded by design

---

## 🔍 How It Works

1. PDF downloaded from the provided URL
2. Text split into overlapping chunks (2000 chars, 200 overlap)
3. TF-IDF retrieval finds the top 5 most relevant chunks per question
4. Groq's `llama-3.1-8b-instant` generates concise answers from the retrieved context
