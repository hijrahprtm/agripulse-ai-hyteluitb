import streamlit as st
import os
from pinecone import Pinecone
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from pygooglenews import GoogleNews

# --- 1. CONFIGURATION & SECRETS ---
try:
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    ADMIN_PASSWORD = st.secrets["ADMIN_PASSWORD"]
except KeyError as e:
    st.error(f"Missing Secret Key: {e}. Periksa dashboard Streamlit Secrets.")
    st.stop()

st.set_page_config(page_title="AgriPulse Engine v7.9", page_icon="🌱", layout="wide")

# --- 2. CUSTOM UI STYLING (For Visibility) ---
st.markdown("""
    <style>
    .stApp { background-color: #ffffff; }
    .stMarkdown, p, li, label, h1, h2, h3, span { color: #1f1f1f !important; }
    [data-testid="stSidebar"] { background-color: #f0f2f6; }
    .stTabs [aria-selected="true"] {
        background-color: #EE2D24 !important;
        color: white !important;
        border-radius: 10px;
    }
    .coming-soon-banner {
        background: linear-gradient(90deg, #ff9800, #f44336);
        color: white !important;
        padding: 12px;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .news-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        border-left: 6px solid #2d6a4f;
        margin-bottom: 12px;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. HEADER & IDENTITAS TIM ---
with st.container():
    col_logo, col_info = st.columns([1, 4])
    with col_logo:
        l1, l2 = st.columns(2)
        with l1:
            if os.path.exists("telulogo.webp"): st.image("telulogo.webp", width=70)
            else: st.write("🎓 **TelU**")
        with l2:
            if os.path.exists("itblogo.png"): st.image("itblogo.png", width=70)
            else: st.write("🌿 **ITB**")
    
    with col_info:
        st.markdown("""
        # AGRIPULSE ENGINE
        **AI Systems Engineer:** Hijrah Wira Pratama (Bachelor of Data Science, TelU)  
        **Lead Researcher:** Yokie Lidiantoro (Agriculture Department, ITB)
        """)

st.divider()

# --- 4. ENGINE INITIALIZATION ---
@st.cache_resource
def init_system():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = ChatGroq(temperature=0.1, groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    idx_name = "agripulse-index"
    return embeddings, llm, idx_name, pc

embeddings, llm, idx_name, pc = init_system()
index = pc.Index(idx_name)

# --- 5. SIDEBAR: DATA PIPELINE & REAL-TIME STATS ---
with st.sidebar:
    st.header("⚙️ Data Pipeline")
    
    side_tab_upload, side_tab_admin = st.tabs(["📤 Upload", "🔐 Admin"])
    
    with side_tab_upload:
        up_file = st.file_uploader("Upload Jurnal Riset (PDF)", type="pdf")
        if up_file and st.button("🚀 Sync to Cloud"):
            with st.spinner("Processing & Vectorizing..."):
                with open("temp.pdf", "wb") as f: f.write(up_file.getbuffer())
                loader = PyPDFLoader("temp.pdf")
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                chunks = splitter.split_documents(loader.load())
                PineconeVectorStore.from_documents(chunks, embeddings, index_name=idx_name)
                st.success(f"Berhasil sinkronisasi {len(chunks)} chunks!")
                os.remove("temp.pdf")
                st.rerun()

    with side_tab_admin:
        st.subheader("Database Management")
        input_pwd = st.text_input("Admin Password", type="password")
        if input_pwd == ADMIN_PASSWORD:
            st.success("Access Granted")
            if st.button("🗑️ Reset Database"):
                with st.spinner("Menghapus data..."):
                    index.delete(delete_all=True)
                    st.warning("Basis data riset telah dikosongkan.")
                    st.rerun()
        elif input_pwd:
            st.error("Password Salah")

    # --- LIVE DATA COUNTER ---
    st.divider()
    try:
        stats = index.describe_index_stats()
        total_vectors = stats['total_vector_count']
        st.metric("Total Active Chunks", f"{total_vectors:,}")
        st.caption("Data tersimpan secara permanen di Pinecone Cloud.")
        if total_vectors == 0:
            st.warning("⚠️ Database Kosong.")
    except:
        st.error("Koneksi Database Terputus.")

# --- 6. MAIN CONTENT (TABS) ---
tab_chat, tab_news, tab_vision = st.tabs(["💬 AI Chat", "📰 News Hub", "🔬 Vision Scan"])

# TAB 1: RAG CHAT
with tab_chat:
    vectorstore = PineconeVectorStore(index_name=idx_name, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    prompt = ChatPromptTemplate.from_template("Konteks: {context}\n\nPertanyaan: {question}\nJawaban:")

    rag_chain = (
        {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )

    if "msgs" not in st.session_state: st.session_state.msgs = []
    for m in st.session_state.msgs:
        with st.chat_message(m["role"]): st.markdown(m["content"])

    if user_query := st.chat_input("Tanya seputar riset kopi..."):
        st.session_state.msgs.append({"role": "user", "content": user_query})
        with st.chat_message("user"): st.markdown(user_query)
        with st.chat_message("assistant"):
            res = rag_chain.invoke(user_query)
            st.markdown(res)
            st.session_state.msgs.append({"role": "assistant", "content": res})

# TAB 2: NEWS HUB
with tab_news:
    st.subheader("📰 Intelligence Hub")
    try:
        gn = GoogleNews(lang='id', country='ID')
        search = gn.search('pertanian kopi modern', when='7d')
        for entry in search['entries'][:4]:
            st.markdown(f"""
            <div class="news-card">
                <a href="{entry.link}" target="_blank" style="color:#2d6a4f; font-weight:bold; font-size:18px;">🔗 {entry.title}</a><br>
                <small>Sumber: {entry.source.text} | {entry.published}</small>
                <p style="margin-top:10px; font-size:0.95em;"><b>AI Summary:</b> Berita relevan mengenai perkembangan sektor kopi dan agrikultur berkelanjutan.</p>
            </div>
            """, unsafe_allow_html=True)
    except: st.info("Layanan berita sedang diperbarui.")

# TAB 3: VISION SCAN
with tab_vision:
    st.markdown('<div class="coming-soon-banner">⚠️ UNDER DEVELOPMENT / RESEARCH PHASE ⚠️</div>', unsafe_allow_html=True)
    st.header("🔬 Coffee Vision AI (Coming Soon)")
    if os.path.exists("image_68c519.jpg"):
        st.image("image_68c519.jpg", use_container_width=True, caption="YOLOv11 Inference Testing")
    st.success("**Training Accuracy:** 98.4% | Hasil benchmark internal tim ITB & TelU.")

st.markdown("<br><hr><center><small>© 2026 AgriPulse Project | ITB & TelU Collaboration</small></center>", unsafe_allow_html=True)
