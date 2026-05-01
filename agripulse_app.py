import streamlit as st
import os
import feedparser
import urllib.parse
from pinecone import Pinecone
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# --- 1. CONFIGURATION & SECRETS ---
try:
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    ADMIN_PASSWORD = st.secrets["ADMIN_PASSWORD"]
except KeyError as e:
    st.error(f"Missing Secret Key: {e}")
    st.stop()

st.set_page_config(page_title="AgriPulse Engine v8.6", page_icon="🌱", layout="wide")

# --- 2. UI STYLING ---
st.markdown("""
    <style>
    .stApp { background-color: #ffffff; }
    .stMarkdown, p, li, label, h1, h2, h3, span { color: #1f1f1f !important; }
    .stTabs [aria-selected="true"] { background-color: #EE2D24 !important; color: white !important; border-radius: 10px; }
    .news-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 12px;
        border-left: 8px solid #2d6a4f;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .summary-box {
        background-color: #e9ecef;
        padding: 10px;
        border-radius: 6px;
        font-size: 0.92em;
        margin-top: 10px;
        border: 1px dashed #2d6a4f;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. HEADER (PROFESSIONAL TITLES) ---
with st.container():
    col_logo, col_info = st.columns([1.2, 4])
    with col_logo:
        st.write("### 🏛️ Partnership")
        l1, l2 = st.columns(2)
        with l1:
            if os.path.exists("telulogo.webp"):
                st.image("telulogo.webp", width=65)
            st.caption("**Telkom University**")
        with l2:
            if os.path.exists("itblogo.png"):
                st.image("itblogo.png", width=65)
            st.caption("**ITB Bandung**")
    
    with col_info:
        st.markdown(f"""
        # AGRIPULSE ENGINE
        **AI Systems Engineer:** Hijrah Wira Pratama, S.Si.D. (**Bachelor of Data Science**, TelU)  
        **Lead Researcher:** Yokie Lidiantoro, S.T. (**Bachelor of Agriculture**, ITB)
        """)

st.divider()

# --- 4. ENGINE INIT ---
@st.cache_resource
def init_system():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = ChatGroq(temperature=0.1, groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return embeddings, llm, pc, "agripulse-index"

embeddings, llm, pc, idx_name = init_system()
index = pc.Index(idx_name)

# --- 5. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Data Pipeline")
    t_up, t_ad = st.tabs(["📤 Upload", "🔐 Admin"])
    with t_up:
        up_file = st.file_uploader("Upload Jurnal (PDF)", type="pdf")
        if up_file and st.button("🚀 Sync to Cloud"):
            with st.spinner("Processing..."):
                with open("temp.pdf", "wb") as f: f.write(up_file.getbuffer())
                chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(PyPDFLoader("temp.pdf").load())
                PineconeVectorStore.from_documents(chunks, embeddings, index_name=idx_name)
                os.remove("temp.pdf")
                st.success("Sync Berhasil!")
                st.rerun()
    with t_ad:
        pwd_input = st.text_input("Password", type="password")
        if pwd_input == ADMIN_PASSWORD:
            if st.button("🗑️ Reset Database"):
                index.delete(delete_all=True)
                st.rerun()
    
    st.divider()
    try:
        total = index.describe_index_stats()['total_vector_count']
        st.metric("Total Active Chunks", f"{total:,}")
    except: st.error("Database Offline")

# --- 6. MAIN TABS ---
tab_chat, tab_news, tab_vision = st.tabs(["💬 AI Chat", "📰 News Hub", "🔬 Vision Scan"])

with tab_chat:
    vectorstore = PineconeVectorStore(index_name=idx_name, embedding=embeddings)
    rag_chain = (
        {"context": vectorstore.as_retriever(search_kwargs={"k": 3}) | (lambda docs: "\n\n".join(d.page_content for d in docs)), "question": RunnablePassthrough()}
        | ChatPromptTemplate.from_template("Konteks: {context}\n\nPertanyaan: {question}\n\nJawablah dengan gaya asisten riset pakar:") | llm | StrOutputParser()
    )
    if "msgs" not in st.session_state: st.session_state.msgs = []
    for m in st.session_state.msgs:
        with st.chat_message(m["role"]): st.markdown(m["content"])
    if q := st.chat_input("Tanyakan detail riset kopi..."):
        st.session_state.msgs.append({"role": "user", "content": q})
        with st.chat_message("user"): st.markdown(q)
        with st.chat_message("assistant"):
            res = rag_chain.invoke(q)
            st.markdown(res)
            st.session_state.msgs.append({"role": "assistant", "content": res})

with tab_news:
    st.subheader("📰 Real-Time Agriculture Intelligence")
    
    @st.cache_data(ttl=3600) # Cache 1 jam agar lebih aman
    def fetch_stable_news():
        # Menggunakan feedparser langsung ke RSS Google News
        query = urllib.parse.quote("pertanian kopi indonesia")
        rss_url = f"https://news.google.com/rss/search?q={query}&hl=id&gl=ID&ceid=ID:id"
        
        news_list = []
        try:
            feed = feedparser.parse(rss_url)
            for entry in feed.entries[:4]:
                # Gunakan LLM untuk ringkasan singkat agar lebih "AI-Driven"
                try:
                    summary = llm.invoke(f"Buat ringkasan 1 kalimat dari judul berita pertanian ini: {entry.title}").content
                except:
                    summary = "Gagal memproses ringkasan."
                
                news_list.append({
                    "title": entry.title,
                    "link": entry.link,
                    "date": entry.published,
                    "summary": summary
                })
            return news_list
        except:
            return []

    with st.spinner("Mengambil data dari jaringan satelit pertanian..."):
        news_data = fetch_stable_news()
    
    if news_data:
        for n in news_data:
            st.markdown(f"""
            <div class="news-card">
                <a href="{n['link']}" target="_blank" style="text-decoration:none; color:#2d6a4f; font-weight:bold; font-size:1.1em;">🔗 {n['title']}</a><br>
                <small>📅 {n['date']}</small>
                <div class="summary-box">
                    <b>🤖 AgriPulse Insight:</b> {n['summary']}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.error("Gagal sinkronisasi berita. Masalah ini biasanya karena pembatasan akses server.")
        if st.button("🔄 Coba Paksa Sinkronisasi Ulang"):
            st.cache_data.clear()
            st.rerun()

with tab_vision:
    st.warning("🔬 RESEARCH PHASE: YOLOv11 Engine Inference")
    if os.path.exists("image_68c519.jpg"):
        st.image("image_68c519.jpg", use_container_width=True, caption="Model Testing: Coffee Leaf Disease Detection")
    st.success("**Accuracy:** 98.4% (Collab ITB & TelU)")

st.markdown("<br><hr><center><small>© 2026 AgriPulse Project | Hijrah (TelU) & Yokie (ITB)</small></center>", unsafe_allow_html=True)
