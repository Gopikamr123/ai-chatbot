# ============================================================
#  RAG CHATBOT - app.py  (v3 — Reliable Document Chat)
#
#  Key improvements:
#  1. Documents are split into CHUNKS (paragraphs) — not whole-doc search
#  2. TF-IDF scoring — finds relevant passages even when exact words differ
#  3. System prompt forces LLM to answer FROM document content
#  4. If user has docs → always search docs first
#     If not found in docs → fallback to web search with clear label
#  5. If user has NO docs at all → use web search
# ============================================================

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import os, re, math
from collections import Counter

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ── APP SETUP ──────────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY']               = os.environ.get('SECRET_KEY', 'alena-secret-2024')
app.config['SQLALCHEMY_DATABASE_URI']  = 'sqlite:///chatbot.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db            = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view    = 'login'
login_manager.login_message = 'Please log in to access this page.'


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ── MODELS ─────────────────────────────────────────────────

class User(UserMixin, db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    username   = db.Column(db.String(80),  unique=True, nullable=False)
    email      = db.Column(db.String(120), unique=True, nullable=False)
    password   = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    documents  = db.relationship('Document',    backref='owner', lazy=True, cascade='all, delete-orphan')
    chats      = db.relationship('ChatMessage', backref='user',  lazy=True, cascade='all, delete-orphan')

    def set_password(self, pw):   self.password = generate_password_hash(pw)
    def check_password(self, pw): return check_password_hash(self.password, pw)
    def __repr__(self):           return f'<User {self.username}>'


class Document(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    title      = db.Column(db.String(200), nullable=False)
    content    = db.Column(db.Text,        nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user_id    = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    def __repr__(self): return f'<Document {self.title}>'


class ChatMessage(db.Model):
    id          = db.Column(db.Integer, primary_key=True)
    question    = db.Column(db.Text,       nullable=False)
    answer      = db.Column(db.Text,       nullable=False)
    sources     = db.Column(db.String(500))
    source_type = db.Column(db.String(20), default='document')
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)
    user_id     = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    def __repr__(self): return f'<ChatMessage {self.id}>'


# ── TEXT UTILITIES ─────────────────────────────────────────

STOP = {
    'a','an','the','is','it','in','on','at','to','for','of','and','or','but',
    'this','that','was','are','be','been','with','from','have','has','had',
    'do','does','did','will','would','can','could','should','what','how',
    'why','when','where','who','which','i','me','my','we','you','your',
    'he','she','they','them','its','our','us','as','by','so','if','than',
    'not','no','about','also','just','more','very','get','got','make','made'
}

# Questions that mean "give me everything" — skip TF-IDF, return all docs
SUMMARY_PHRASES = [
    'what topics', 'what is covered', "what's covered", 'summarize', 'summarise',
    'summary', 'overview', 'give me an overview', 'tell me about', 'what do',
    'what does', 'what is in', "what's in", 'key points', 'main points',
    'key ideas', 'main ideas', 'everything', 'all topics', 'list topics',
    'what information', 'what are the topics', 'what can you tell',
    'what do you know', 'explain everything', 'explain all',
]

def is_summary_question(question):
    """Return True if the question is a broad/summary request."""
    q = question.lower().strip()
    return any(phrase in q for phrase in SUMMARY_PHRASES)


def tokenize(text):
    words = re.findall(r'[a-zA-Z0-9]+', text.lower())
    return [w for w in words if w not in STOP and len(w) > 1]


def chunk_document(doc, chunk_size=150, overlap=30):
    """Split a document into overlapping word-chunks for fine-grained retrieval."""
    words  = doc.content.split()
    chunks = []
    step   = max(1, chunk_size - overlap)
    i      = 0
    while i < len(words):
        chunk_words = words[i : i + chunk_size]
        chunks.append({
            'doc_id':    doc.id,
            'doc_title': doc.title,
            'text':      ' '.join(chunk_words),
        })
        i += step
    return chunks


def tfidf_score(query_tokens, chunk_text, idf_map):
    """Score a chunk against the query using TF-IDF."""
    chunk_tokens = tokenize(chunk_text)
    if not chunk_tokens:
        return 0.0
    tf_counter = Counter(chunk_tokens)
    total      = len(chunk_tokens)
    score      = 0.0
    for word in query_tokens:
        if word in tf_counter:
            tf    = tf_counter[word] / total
            idf   = idf_map.get(word, 0.0)
            score += tf * idf
    return score


def get_first_chunk_per_doc(all_chunks, all_docs):
    """Return the first (and last, if long) chunk from every document."""
    seen, result = set(), []
    # First pass: first chunk of each doc
    for chunk in all_chunks:
        if chunk['doc_id'] not in seen:
            result.append(chunk)
            seen.add(chunk['doc_id'])
    # Second pass: also add the last chunk of each doc (captures end content)
    last_seen = set()
    for chunk in reversed(all_chunks):
        if chunk['doc_id'] not in last_seen and chunk not in result:
            result.append(chunk)
            last_seen.add(chunk['doc_id'])
    return result


def retrieve_chunks(question, user_id, top_k=6):
    """
    Retrieve the most relevant document chunks for a question.
    - Summary/broad questions → return opening chunks from ALL docs
    - Specific questions      → TF-IDF scoring, but ALWAYS return at least
                                one chunk per doc as fallback
    Returns: (top_chunks, all_docs)
    """
    all_docs = Document.query.filter_by(user_id=user_id).all()
    if not all_docs:
        return [], []

    all_chunks = []
    for doc in all_docs:
        all_chunks.extend(chunk_document(doc))

    if not all_chunks:
        return [], all_docs

    # ── CASE 1: Summary / overview question → return all docs ──
    if is_summary_question(question):
        return get_first_chunk_per_doc(all_chunks, all_docs), all_docs

    # ── CASE 2: TF-IDF scoring ──
    N        = len(all_chunks)
    doc_freq = Counter()
    for chunk in all_chunks:
        for w in set(tokenize(chunk['text'])):
            doc_freq[w] += 1
    idf_map = {w: math.log((N + 1) / (df + 1)) + 1 for w, df in doc_freq.items()}

    q_tokens = tokenize(question)
    if not q_tokens:
        # No meaningful tokens → return first chunk of each doc
        return get_first_chunk_per_doc(all_chunks, all_docs), all_docs

    scored = []
    for chunk in all_chunks:
        score = tfidf_score(q_tokens, chunk['text'], idf_map)
        # Title match bonus
        for w in q_tokens:
            if w in tokenize(chunk['doc_title']):
                score += 0.5
        scored.append((score, chunk))   # ← include ALL chunks, even score=0

    scored.sort(key=lambda x: x[0], reverse=True)

    # Take top chunks, max 2 per doc
    seen_docs, top_chunks = Counter(), []
    for score, chunk in scored:
        if seen_docs[chunk['doc_id']] < 2:
            top_chunks.append(chunk)
            seen_docs[chunk['doc_id']] += 1
        if len(top_chunks) >= top_k:
            break

    # Safety: if best score is 0 (no keyword overlap at all),
    # fall back to first chunk of every doc so LLM always has context
    best_score = scored[0][0] if scored else 0
    if best_score == 0:
        return get_first_chunk_per_doc(all_chunks, all_docs), all_docs

    return top_chunks, all_docs


# ── LLM CALL ──────────────────────────────────────────────

def call_llm(question, context, source_type='document'):
    if source_type == 'document':
        system = (
            "You are a helpful assistant. The user has provided documents as their knowledge base.\n"
            "Your job is to answer the user's question using ONLY the document excerpts provided below.\n"
            "Rules:\n"
            "- Answer directly and clearly based on what the documents say.\n"
            "- If the answer spans multiple excerpts, combine them into a clear answer.\n"
            "- Quote or paraphrase the document content to support your answer.\n"
            "- If the information is NOT in the documents at all, respond with exactly: "
            "\"I couldn't find that in your documents.\"\n"
            "- Do NOT make up information. Do NOT use outside knowledge.\n"
            "- Write in clear, readable paragraphs."
        )
    else:
        system = (
            "You are a helpful assistant. Answer the user's question using the web search results below. "
            "Be clear, concise and accurate. Mention the source when relevant."
        )

    user_msg = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    provider = os.environ.get('LLM_PROVIDER', 'groq').lower()

    if provider == 'groq':
        try:
            from groq import Groq
            client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
            resp   = client.chat.completions.create(
                model    = os.environ.get('GROQ_MODEL', 'llama-3.1-8b-instant'),
                messages = [{'role':'system','content':system},{'role':'user','content':user_msg}],
                max_tokens=900, temperature=0.2,
            )
            return resp.choices[0].message.content
        except ImportError: return "Error: Run 'pip install groq'"
        except Exception as e: return f"Groq Error: {e}"

    elif provider == 'openai':
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
            resp   = client.chat.completions.create(
                model    = os.environ.get('OPENAI_MODEL', 'gpt-4o-mini'),
                messages = [{'role':'system','content':system},{'role':'user','content':user_msg}],
                max_tokens=900, temperature=0.2,
            )
            return resp.choices[0].message.content
        except ImportError: return "Error: Run 'pip install openai'"
        except Exception as e: return f"OpenAI Error: {e}"

    elif provider == 'gemini':
        try:
            import google.generativeai as genai
            genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))
            model = genai.GenerativeModel(
                model_name        = os.environ.get('GEMINI_MODEL', 'gemini-1.5-flash'),
                system_instruction= system,
            )
            return model.generate_content(user_msg).text
        except ImportError: return "Error: Run 'pip install google-generativeai'"
        except Exception as e: return f"Gemini Error: {e}"

    elif provider == 'ollama':
        try:
            import requests as req
            resp = req.post(
                f"{os.environ.get('OLLAMA_BASE_URL','http://localhost:11434')}/api/chat",
                json={'model':os.environ.get('OLLAMA_MODEL','llama3.2'),
                      'messages':[{'role':'system','content':system},{'role':'user','content':user_msg}],
                      'stream':False},
                timeout=120,
            )
            return resp.json()['message']['content']
        except Exception as e: return f"Ollama Error: {e}"

    return "Error: Unknown LLM_PROVIDER. Set to: groq, openai, gemini, or ollama"


# ── WEB SEARCH FALLBACK ────────────────────────────────────

def web_search(query, max_results=4):
    results = []
    try:
        import requests
        from urllib.parse import quote_plus
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; RAGBot/3.0)'}
        data    = requests.get(
            f"https://api.duckduckgo.com/?q={quote_plus(query)}&format=json&no_html=1&skip_disambig=1",
            headers=headers, timeout=8
        ).json()
        if data.get('AbstractText'):
            results.append({'title': data.get('Heading','Answer'),
                            'snippet': data['AbstractText'][:600],
                            'url': data.get('AbstractURL','')})
        for t in data.get('RelatedTopics', [])[:3]:
            if isinstance(t, dict) and t.get('Text'):
                results.append({'title': t['Text'][:80], 'snippet': t['Text'][:400],
                                'url': t.get('FirstURL','')})
        if len(results) < 2:
            r2       = requests.get(f"https://html.duckduckgo.com/html/?q={quote_plus(query)}",
                                    headers=headers, timeout=8)
            snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', r2.text, re.DOTALL)
            titles   = re.findall(r'class="result__a"[^>]*>(.*?)</a>',       r2.text, re.DOTALL)
            urls     = re.findall(r'class="result__url"[^>]*>\s*(.*?)\s*</a>',r2.text, re.DOTALL)
            for i in range(min(max_results, len(snippets))):
                s = re.sub(r'<[^>]+>','', snippets[i]).strip()
                t = re.sub(r'<[^>]+>','', titles[i] if i < len(titles) else '').strip()
                if s:
                    results.append({'title': t or f'Result {i+1}', 'snippet': s[:500],
                                    'url': urls[i].strip() if i < len(urls) else ''})
    except Exception as e:
        results.append({'title':'Search Error','snippet':str(e),'url':''})
    return results[:max_results]


# ── SMART RAG PIPELINE ─────────────────────────────────────

def smart_rag(question, user_id):
    """
    Decision flow:
    1. User has documents → search chunks with TF-IDF
       a. LLM finds answer in docs   → return doc answer
       b. LLM says "not in docs"     → fall back to web, label it clearly
    2. User has no documents         → web search only
    """
    top_chunks, all_docs = retrieve_chunks(question, user_id, top_k=5)
    has_docs = len(all_docs) > 0

    # ── Case 1: user has documents ──
    if has_docs:
        if top_chunks:
            # Group chunks by document for clean context
            doc_sections = {}
            for chunk in top_chunks:
                did = chunk['doc_id']
                if did not in doc_sections:
                    doc_sections[did] = {'title': chunk['doc_title'], 'texts': []}
                doc_sections[did]['texts'].append(chunk['text'])

            context = '\n\n'.join(
                f"=== {info['title']} ===\n" + ' [...] '.join(info['texts'])
                for info in doc_sections.values()
            )
            answer = call_llm(question, context, source_type='document')

            if "couldn't find that in your documents" not in answer.lower():
                # Great — answered from documents
                seen, srcs = set(), []
                for c in top_chunks:
                    if c['doc_id'] not in seen:
                        srcs.append({'id': c['doc_id'], 'title': c['doc_title'], 'type': 'document'})
                        seen.add(c['doc_id'])
                return answer, srcs, 'document'
        else:
            answer = "I couldn't find that in your documents."

        # Fall through to web
        web_results = web_search(question)
        if web_results and not (len(web_results)==1 and 'Search Error' in web_results[0]['title']):
            web_ctx    = '\n\n'.join(f"[{r['title']}]\n{r['snippet']}" for r in web_results if r.get('snippet'))
            web_answer = call_llm(question, web_ctx, source_type='web')
            final      = f"*Not found in your documents — here's what the web says:*\n\n{web_answer}"
            srcs       = [{'title': r['title'], 'url': r['url'], 'type': 'web'}
                          for r in web_results if r.get('snippet')]
            return final, srcs, 'web'

        return (
            "I couldn't find that in your documents, and web search didn't return results either. "
            "Try adding more documents or rephrasing your question.",
            [], 'none'
        )

    # ── Case 2: no documents at all → web search ──
    web_results = web_search(question)
    if not web_results:
        return (
            "You have no documents yet. Go to **Documents → Add Document** to add content "
            "so I can answer questions about it!",
            [], 'none'
        )
    web_ctx = '\n\n'.join(f"[{r['title']}]\n{r['snippet']}" for r in web_results if r.get('snippet'))
    answer  = call_llm(question, web_ctx, source_type='web')
    srcs    = [{'title': r['title'], 'url': r['url'], 'type': 'web'}
               for r in web_results if r.get('snippet')]
    return f"*(No documents — answered from web search)*\n\n{answer}", srcs, 'web'


# ── AUTH ROUTES ────────────────────────────────────────────

@app.route('/')
def home():
    return redirect(url_for('dashboard') if current_user.is_authenticated else url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form['username'].strip()
        email    = request.form['email'].strip().lower()
        password = request.form['password']
        if User.query.filter_by(username=username).first():
            flash('Username already taken!', 'error'); return redirect(url_for('register'))
        if User.query.filter_by(email=email).first():
            flash('Email already registered!', 'error'); return redirect(url_for('register'))
        u = User(username=username, email=email)
        u.set_password(password)
        db.session.add(u); db.session.commit()
        flash('Account created! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        email    = request.form['email'].strip().lower()
        password = request.form['password']
        user     = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            login_user(user)
            flash(f'Welcome back, {user.username}!', 'success')
            return redirect(url_for('dashboard'))
        flash('Wrong email or password.', 'error')
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


# ── DASHBOARD ─────────────────────────────────────────────

@app.route('/dashboard')
@login_required
def dashboard():
    doc_count    = Document.query.filter_by(user_id=current_user.id).count()
    chat_count   = ChatMessage.query.filter_by(user_id=current_user.id).count()
    recent_chats = ChatMessage.query.filter_by(user_id=current_user.id)\
                              .order_by(ChatMessage.created_at.desc()).limit(5).all()
    return render_template('dashboard.html',
                           doc_count=doc_count, chat_count=chat_count,
                           recent_chats=recent_chats)


# ── CRUD: DOCUMENTS ────────────────────────────────────────

@app.route('/documents')
@login_required
def documents():
    docs = Document.query.filter_by(user_id=current_user.id)\
                   .order_by(Document.updated_at.desc()).all()
    return render_template('documents.html', docs=docs)


@app.route('/documents/add', methods=['GET', 'POST'])
@login_required
def add_document():
    if request.method == 'POST':
        title   = request.form['title'].strip()
        content = request.form['content'].strip()
        if not title or not content:
            flash('Title and content are required!', 'error')
            return redirect(url_for('add_document'))
        db.session.add(Document(title=title, content=content, user_id=current_user.id))
        db.session.commit()
        flash(f'Document "{title}" added to your knowledge base!', 'success')
        return redirect(url_for('documents'))
    return render_template('document_form.html', doc=None, action='Add')


@app.route('/documents/<int:doc_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_document(doc_id):
    doc = Document.query.filter_by(id=doc_id, user_id=current_user.id).first_or_404()
    if request.method == 'POST':
        doc.title      = request.form['title'].strip()
        doc.content    = request.form['content'].strip()
        doc.updated_at = datetime.utcnow()
        db.session.commit()
        flash(f'Document "{doc.title}" updated!', 'success')
        return redirect(url_for('documents'))
    return render_template('document_form.html', doc=doc, action='Edit')


@app.route('/documents/<int:doc_id>/delete', methods=['POST'])
@login_required
def delete_document(doc_id):
    doc = Document.query.filter_by(id=doc_id, user_id=current_user.id).first_or_404()
    title = doc.title
    db.session.delete(doc); db.session.commit()
    flash(f'Document "{title}" deleted.', 'info')
    return redirect(url_for('documents'))


# ── CHAT ──────────────────────────────────────────────────

@app.route('/chat')
@login_required
def chat():
    history   = ChatMessage.query.filter_by(user_id=current_user.id)\
                           .order_by(ChatMessage.created_at.desc()).limit(30).all()
    history.reverse()
    doc_count = Document.query.filter_by(user_id=current_user.id).count()
    return render_template('chat.html', history=history, doc_count=doc_count)


@app.route('/chat/ask', methods=['POST'])
@login_required
def ask():
    question = request.form.get('question', '').strip()
    if not question:
        return jsonify({'error': 'Please enter a question.'}), 400

    answer, sources, source_type = smart_rag(question, current_user.id)

    sources_str = ', '.join([str(s.get('id', s.get('url', ''))) for s in sources])
    db.session.add(ChatMessage(
        question=question, answer=answer,
        sources=sources_str, source_type=source_type,
        user_id=current_user.id
    ))
    db.session.commit()

    return jsonify({'answer': answer, 'sources': sources, 'source_type': source_type})


@app.route('/chat/clear', methods=['POST'])
@login_required
def clear_chat():
    ChatMessage.query.filter_by(user_id=current_user.id).delete()
    db.session.commit()
    flash('Chat history cleared!', 'info')
    return redirect(url_for('chat'))


# ── INIT ──────────────────────────────────────────────────
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)
