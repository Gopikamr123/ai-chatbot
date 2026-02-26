# 🤖 AI RAG Chatbot
#live website link:
https://gopikamr.pythonanywhere.com/

A full-stack AI-powered chatbot that lets you **chat with your own documents**. Add any text content as a knowledge base, ask questions, and get grounded answers — powered by Groq (free), OpenAI, Gemini, or a local Ollama model.

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-orange?style=flat-square&logo=flask)
![SQLite](https://img.shields.io/badge/Database-SQLite-green?style=flat-square&logo=sqlite)
![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)

---

## ✨ Features

| Feature | Details |
|---|---|
| 🔐 Authentication | Register, login, logout with hashed passwords |
| 📄 Document CRUD | Add, view, edit, delete your knowledge base documents |
| 🤖 AI Chat | Ask questions — answers come directly from your documents |
| 🔍 Smart Retrieval | TF-IDF chunk-level scoring finds the right passage |
| 🌐 Web Search Fallback | If answer isn't in your docs, searches DuckDuckGo automatically |
| 💬 Chat History | Last 30 conversations saved per user |
| 🎨 Beautiful UI | Soft gradient glassmorphism design |

---

## 📁 Project Structure

```
RAG/
├── app.py                  ← All backend logic (Flask + RAG + auth + CRUD)
├── requirements.txt        ← Python dependencies
├── .env                    ← Your API keys (never commit this!)
├── .gitignore              ← Keeps secrets and junk out of Git
├── index.html              ← Landing page
├── venv/                   ← Virtual environment (not committed to Git)
└── templates/
    ├── base.html           ← Shared sidebar layout
    ├── login.html          ← Login page
    ├── register.html       ← Register page
    ├── dashboard.html      ← Home after login
    ├── documents.html      ← List all documents
    ├── document_form.html  ← Add / Edit document
    └── chat.html           ← AI chat interface
```

---

## 🚀 Setup & Run

### 1. Clone or download the project

```bash
git clone https://github.com/gopikamr123/ai-rag-chatbot.git
cd ai-rag-chatbot
```

### 2. Create and activate virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create your `.env` file

Create a file named `.env` in the root folder (same level as `app.py`):

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here
SECRET_KEY=any-random-secret-string-here
```

> 🆓 Get a **free** Groq API key at [console.groq.com](https://console.groq.com) — no credit card needed!

### 5. Run the app

```bash
python app.py
```

### 6. Open in your browser

```
http://localhost:5000
```

Register an account → Add documents → Start chatting!

---

## 🧠 How the RAG Pipeline Works

**RAG = Retrieval Augmented Generation**

```
Your Question
      │
      ▼
┌──────────────────────────┐
│  1. RETRIEVE             │
│  Split docs into chunks  │
│  Score with TF-IDF       │
│  Pick top 5 passages     │
└─────────────┬────────────┘
              │
              ▼
┌──────────────────────────┐
│  2. AUGMENT              │
│  Build prompt with the   │
│  retrieved chunks        │
└─────────────┬────────────┘
              │
              ▼
┌──────────────────────────┐
│  3. GENERATE             │
│  Send to LLM (Groq etc.) │
│  Get a grounded answer   │
└─────────────┬────────────┘
              │
              ▼
     Answer + 📄 Source badges
```

### Smart fallback logic

```
Has documents?
    ├── YES → Search chunks with TF-IDF
    │         ├── Answer found → return with 📄 "From your documents" badge
    │         └── Not found   → search web → return with 🌐 "Web search" badge
    └── NO  → Search web directly
```

### Special handling for summary questions

Questions like *"What topics are covered?"*, *"Summarise my documents"*, *"Key points"* automatically skip TF-IDF and return content from **all** your documents at once.

---

## ⚙️ LLM Provider Options

Change `LLM_PROVIDER` in your `.env` file to switch providers:

| Provider | Cost | `.env` setting |
|----------|------|----------------|
| **Groq** | ✅ Free | `LLM_PROVIDER=groq` |
| **Gemini** | Free tier | `LLM_PROVIDER=gemini` |
| **OpenAI** | Paid | `LLM_PROVIDER=openai` |
| **Ollama** | Free (runs locally) | `LLM_PROVIDER=ollama` |

### Full `.env` examples

**Groq (recommended — free & fast):**
```env
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
SECRET_KEY=my-secret-key
```

**OpenAI:**
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
OPENAI_MODEL=gpt-4o-mini
SECRET_KEY=my-secret-key
```

**Gemini:**
```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=AIzaxxxxxxxxxxxxxxxxxxxx
GEMINI_MODEL=gemini-1.5-flash
SECRET_KEY=my-secret-key
```

**Ollama (fully local, no internet needed):**
```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434
SECRET_KEY=my-secret-key
```
> Make sure Ollama is running with `ollama serve` before starting the app.

---

## 🗄️ Database

SQLite database (`chatbot.db`) is created automatically on first run. Three tables:

| Table | What it stores |
|-------|----------------|
| `user` | Accounts — username, email, hashed password |
| `document` | Knowledge base — title, content, owner |
| `chat_message` | History — question, answer, source type |

---

## 🔒 Security

- Passwords are **never stored in plain text** — hashed with Werkzeug (bcrypt-style)
- Each user can **only see and edit their own documents**
- `.env` is in `.gitignore` — your API keys are never committed to Git
- `SECRET_KEY` signs Flask sessions — use a long random string in production

---

## 🛠️ Tech Stack

- **Backend** — Python 3.14, Flask 3.0, SQLAlchemy, Flask-Login
- **Database** — SQLite (file-based, zero configuration)
- **AI / LLM** — Groq (Llama 3.1), OpenAI GPT, Google Gemini, or Ollama
- **Retrieval** — Custom TF-IDF chunking (no external vector database needed)
- **Web Search** — DuckDuckGo API (no API key required)
- **Frontend** — HTML / CSS / Vanilla JS, Cormorant Garamond + DM Sans fonts

---

## 🐛 Common Issues

| Error | Fix |
|-------|-----|
| `Groq Error: api_key must be set` | Add `GROQ_API_KEY=...` to your `.env` file |
| `ModuleNotFoundError: flask` | Run `pip install -r requirements.txt` with venv activated |
| `chatbot.db not found` | Normal on first run — it's created automatically |
| Chat says "not in documents" | Rephrase your question, or check your document actually has the info |
| Ollama error | Make sure you ran `ollama serve` in a separate terminal |

---

## 📝 License

MIT — free to use, modify, and distribute.
