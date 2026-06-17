import os
from concurrent.futures import ThreadPoolExecutor
from groq import Groq
import streamlit as st
from dotenv import load_dotenv

# .env file se keys load karein
load_dotenv()

# Streamlit Page Configuration
st.set_page_config(
    page_title="Parallel RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================
# 🎨 PROFESSIONAL CUSTOM CSS STYLING
# =============================================
st.markdown("""
<style>
    /* ===== Google Font Import ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ===== Root Variables ===== */
    :root {
        --bg-primary: #0f0f1a;
        --bg-secondary: #1a1a2e;
        --bg-card: rgba(30, 30, 60, 0.6);
        --accent-1: #7c3aed;
        --accent-2: #06b6d4;
        --accent-3: #8b5cf6;
        --text-primary: #e2e8f0;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --border-color: rgba(124, 58, 237, 0.2);
        --glow-purple: rgba(124, 58, 237, 0.15);
        --glow-cyan: rgba(6, 182, 212, 0.15);
        --gradient-1: linear-gradient(135deg, #7c3aed, #06b6d4);
        --gradient-2: linear-gradient(135deg, #1a1a2e, #16213e);
        --shadow-lg: 0 20px 60px rgba(0, 0, 0, 0.4);
        --shadow-glow: 0 0 30px rgba(124, 58, 237, 0.2);
    }

    /* ===== Global Styles ===== */
    .stApp {
        font-family: 'Inter', sans-serif !important;
        background: var(--bg-primary) !important;
        color: var(--text-primary) !important;
    }

    /* Preserve Material Symbols for Streamlit icons */
    .stChatMessage .stMarkdown, .stChatMessage p, .stChatMessage li {
        font-family: 'Inter', sans-serif !important;
    }

    /* Fix avatar styling - hide text, show emoji */
    [data-testid="chatAvatarIcon-user"],
    [data-testid="chatAvatarIcon-assistant"] {
        font-family: 'Material Symbols Rounded', sans-serif !important;
    }

    /* ===== Hide Default Streamlit Elements ===== */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ===== Animated Background Gradient ===== */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background: 
            radial-gradient(ellipse at 20% 50%, var(--glow-purple) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 20%, var(--glow-cyan) 0%, transparent 50%),
            radial-gradient(ellipse at 50% 80%, rgba(139, 92, 246, 0.08) 0%, transparent 50%);
        pointer-events: none;
        z-index: 0;
        animation: bgPulse 8s ease-in-out infinite alternate;
    }

    @keyframes bgPulse {
        0% { opacity: 0.6; }
        100% { opacity: 1; }
    }

    /* ===== Sidebar Styling ===== */
    section[data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border-color) !important;
        backdrop-filter: blur(20px);
    }

    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
    }

    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown span {
        color: var(--text-secondary) !important;
    }

    /* ===== Hero Header ===== */
    .hero-container {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem 1rem;
        margin-bottom: 1rem;
        position: relative;
        z-index: 1;
    }

    .hero-badge {
        display: inline-block;
        background: rgba(124, 58, 237, 0.15);
        border: 1px solid rgba(124, 58, 237, 0.3);
        border-radius: 50px;
        padding: 6px 18px;
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--accent-3);
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 1rem;
        animation: fadeInDown 0.6s ease-out;
    }

    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: var(--gradient-1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        line-height: 1.2;
        animation: fadeInUp 0.8s ease-out;
    }

    .hero-subtitle {
        font-size: 1rem;
        color: var(--text-secondary);
        font-weight: 400;
        max-width: 500px;
        margin: 0 auto;
        line-height: 1.6;
        animation: fadeInUp 1s ease-out;
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* ===== Sidebar Document Cards ===== */
    .sidebar-header {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 1rem 0 0.5rem 0;
    }

    .sidebar-header-icon {
        font-size: 1.5rem;
        background: var(--gradient-1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .sidebar-header-text {
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--text-primary);
    }

    .sidebar-metric {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin: 0.8rem 0;
        backdrop-filter: blur(10px);
    }

    .sidebar-metric-label {
        font-size: 0.7rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }

    .sidebar-metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        background: var(--gradient-1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .doc-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        display: flex;
        align-items: center;
        gap: 10px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(10px);
    }

    .doc-card:hover {
        border-color: var(--accent-1);
        transform: translateX(4px);
        box-shadow: var(--shadow-glow);
    }

    .doc-icon {
        width: 36px;
        height: 36px;
        background: rgba(124, 58, 237, 0.15);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
        flex-shrink: 0;
    }

    .doc-name {
        font-size: 0.85rem;
        font-weight: 500;
        color: var(--text-primary);
    }

    .doc-status {
        font-size: 0.65rem;
        color: #22c55e;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* ===== Chat Container ===== */
    .stChatMessage {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 16px !important;
        padding: 1rem 1.2rem !important;
        margin-bottom: 1rem !important;
        backdrop-filter: blur(10px) !important;
        animation: messageSlideIn 0.4s ease-out !important;
    }

    @keyframes messageSlideIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .stChatMessage p, .stChatMessage li {
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
        line-height: 1.7 !important;
    }

    /* Don't override font on spans (material icons use spans) */
    .stChatMessage .message-content span {
        color: var(--text-primary) !important;
    }

    /* ===== Chat Input Box ===== */
    .stChatInput {
        border-top: 1px solid var(--border-color) !important;
    }

    .stChatInput textarea {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 14px !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
        padding: 0.8rem 1.2rem !important;
        transition: all 0.3s ease !important;
    }

    .stChatInput textarea:focus {
        border-color: var(--accent-1) !important;
        box-shadow: 0 0 20px rgba(124, 58, 237, 0.15) !important;
    }

    .stChatInput button {
        background: var(--gradient-1) !important;
        border: none !important;
        border-radius: 12px !important;
        transition: all 0.3s ease !important;
    }

    .stChatInput button:hover {
        transform: scale(1.05) !important;
        box-shadow: var(--shadow-glow) !important;
    }

    /* ===== Reference Badges ===== */
    .ref-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 16px;
        border-radius: 10px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
        animation: fadeInUp 0.5s ease-out;
    }

    .ref-badge-success {
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.3);
        color: #4ade80;
    }

    .ref-badge-info {
        background: rgba(6, 182, 212, 0.1);
        border: 1px solid rgba(6, 182, 212, 0.3);
        color: #22d3ee;
    }

    /* ===== Spinner Override ===== */
    .stSpinner > div {
        border-color: var(--accent-1) !important;
    }

    /* ===== Alert / Success / Info / Error boxes ===== */
    .stAlert {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
    }

    /* ===== Scrollbar ===== */
    ::-webkit-scrollbar {
        width: 6px;
    }
    ::-webkit-scrollbar-track {
        background: var(--bg-primary);
    }
    ::-webkit-scrollbar-thumb {
        background: var(--accent-1);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: var(--accent-3);
    }

    /* ===== Divider ===== */
    .sidebar-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border-color), transparent);
        margin: 1rem 0;
    }

    /* ===== Status Dot Pulse ===== */
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        background: #22c55e;
        border-radius: 50%;
        margin-right: 6px;
        animation: pulse 2s ease-in-out infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.4); }
        50% { opacity: 0.8; box-shadow: 0 0 0 6px rgba(34, 197, 94, 0); }
    }

    /* ===== Footer / Powered By ===== */
    .powered-by {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
        font-size: 0.7rem;
        color: var(--text-muted);
        letter-spacing: 0.5px;
    }

    .powered-by a {
        color: var(--accent-3);
        text-decoration: none;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# =============================================
# 🔑 GROQ CLIENT SETUP
# =============================================
# Groq Client Setup (Aapne direct key lagayi thi, toh yahan direct paste kar dein)
api_key = "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" # <-- Yahan apni sahi wali key lagayein

if not api_key or "YOUR_ACTUAL" in api_key:
    st.error("🔑 Please set your valid GROQ_API_KEY in the code.")
    st.stop()

client = Groq(api_key=api_key)
DATA_FOLDER = "data"


# =============================================
# 📄 1. PARALLEL DOCUMENT LOADING
# =============================================
def load_single_file(file_name):
    file_path = os.path.join(DATA_FOLDER, file_name)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return {"file": file_name, "content": f.read()}
    except Exception as e:
        return None

def load_all_documents_parallel():
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
        return []
    files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".txt")]
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(load_single_file, files))
    return [r for r in results if r is not None]


# =============================================
# 🔍 2. PARALLEL RETRIEVAL (KNOWLEDGE MATCHING)
# =============================================
def score_document(doc_obj, query_words):
    content = doc_obj["content"].lower()
    score = sum(1 for word in query_words if word.lower() in content)
    return (score, doc_obj["content"], doc_obj["file"])

def retrieve_context_parallel(query, documents):
    if not documents:
        return "No documents available.", "None"
    
    # Common words (is, what, the) ko nikal dein taake ghalat match na ho
    stopwords = {"what", "is", "the", "a", "an", "and", "or", "of", "about", "for", "to", "in", "are", "tell", "me"}
    query_words = [word.lower() for word in query.split() if word.lower() not in stopwords]
    
    # Agar query mein koi khas keyword bacha hi nahi, toh match 'None'
    if not query_words:
        return "No specific database context found.", "None"
        
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda d: score_document(d, query_words), documents))
    
    results.sort(key=lambda x: x[0], reverse=True)
    best_score, best_content, file_name = results[0]
    
    # STRICT CONDITION: Agar score 0 se bara hai (yani asli word match hua hai) toh hi file bhejo
    if best_score > 0:
        return best_content, file_name
        
    # Agar kuch match nahi hua toh file_name 'None' jayega
    return "No specific database context found.", "None"


# =============================================
# 🎨 HERO HEADER
# =============================================
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">⚡ Parallel Processing Engine</div>
    <div class="hero-title">RAG Chatbot</div>
    <div class="hero-subtitle">
        Intelligent document retrieval powered by parallel threading & Groq AI — 
        Ask anything about your knowledge base.
    </div>
</div>
""", unsafe_allow_html=True)


# =============================================
# 📂 SIDEBAR — KNOWLEDGE BASE PANEL
# =============================================
with st.sidebar:
    # Sidebar Logo / Brand
    st.markdown("""
    <div style="text-align:center; padding: 1.5rem 0 0.5rem 0;">
        <div style="font-size: 2.5rem; margin-bottom: 0.3rem;">🧠</div>
        <div style="font-size: 1.1rem; font-weight: 700; color: #e2e8f0;">Knowledge Hub</div>
        <div style="font-size: 0.7rem; color: #64748b; margin-top: 2px;">Manage your documents</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # Load documents
    docs_db = load_all_documents_parallel()

    # Metric Card
    st.markdown(f"""
    <div class="sidebar-metric">
        <div class="sidebar-metric-label">Documents Loaded</div>
        <div class="sidebar-metric-value">{len(docs_db)}</div>
        <div style="font-size: 0.72rem; color: #22c55e; font-weight: 500; margin-top: 2px;">
            <span class="status-dot"></span> All systems operational
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # Document List
    if docs_db:
        st.markdown("""
        <div style="font-size: 0.75rem; font-weight: 600; color: #94a3b8; 
             text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem;">
            📚 Available Documents
        </div>
        """, unsafe_allow_html=True)

        for d in docs_db:
            file_size = os.path.getsize(os.path.join(DATA_FOLDER, d['file']))
            size_kb = round(file_size / 1024, 1)
            st.markdown(f"""
            <div class="doc-card">
                <div class="doc-icon">📄</div>
                <div>
                    <div class="doc-name">{d['file']}</div>
                    <div class="doc-status">● Indexed • {size_kb} KB</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center; padding: 2rem 1rem; color: #64748b;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">📭</div>
            <div style="font-size: 0.85rem;">No documents found</div>
            <div style="font-size: 0.72rem; margin-top: 4px;">Add .txt files to the <code>data/</code> folder</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # Engine Info
    st.markdown("""
    <div style="padding: 0.5rem 0;">
        <div style="font-size: 0.7rem; font-weight: 600; color: #94a3b8; 
             text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.6rem;">
            ⚙️ Engine Details
        </div>
        <div style="font-size: 0.78rem; color: #64748b; line-height: 2;">
            <span style="color:#94a3b8;">Model:</span> Llama 3.1 8B<br>
            <span style="color:#94a3b8;">Provider:</span> Groq Cloud<br>
            <span style="color:#94a3b8;">Retrieval:</span> Parallel Threading<br>
            <span style="color:#94a3b8;">Temperature:</span> 0.4
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Powered By Footer
    st.markdown("""
    <div class="powered-by" style="margin-top: 2rem;">
        Built with ❤️ using<br>
        <a>Streamlit</a> • <a>Groq</a> • <a>Python</a>
    </div>
    """, unsafe_allow_html=True)


# =============================================
# 💬 CHAT INTERFACE
# =============================================

# Chat History Initialize karna
if "messages" not in st.session_state:
    st.session_state.messages = []

# Purani chat display karna
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# User Input (Chat Box)
if user_query := st.chat_input("💬 Ask something about your documents..."):
    
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🔍 Searching documents & generating answer..."):
            
            # Step A: Context dhoondo parallelly
            context, matched_file = retrieve_context_parallel(user_query, docs_db)
            
            # REFERENCE BADGE: Custom styled badges
            if matched_file != "None":
                st.markdown(f"""
                <div class="ref-badge ref-badge-success">
                    🎯 <strong>Knowledge Base Match</strong> — Retrieved from <code>{matched_file}</code>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="ref-badge ref-badge-info">
                    🌐 <strong>AI Knowledge</strong> — Answer from Groq AI base knowledge
                </div>
                """, unsafe_allow_html=True)
            
            # Step B: System prompt ready karo
            system_prompt = (
                "You are an advanced Parallel & Distributed Systems AI assistant.\n"
                f"Use this Database Context to answer accurately if relevant:\n{context}"
            )
            
            # Step C: Groq API Call
            try:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_query}
                    ],
                    model="llama-3.1-8b-instant",
                    temperature=0.4
                )
                response = chat_completion.choices[0].message.content
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                st.error(f"⚠️ Groq API Error: {e}")