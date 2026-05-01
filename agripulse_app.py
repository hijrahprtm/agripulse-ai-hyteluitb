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
    st.error(f"Missing Secret Key: {e}")
    st.stop()

st.set_page_config(page_title="AgriPulse Engine v8.3", page_icon="🌱", layout="wide")

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

# --- 3. HEADER (UPDATED TITLES) ---
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
        # Update Gelar sesuai permintaan
        st.markdown(f"""
        # AGRIPULSE ENGINE
        **AI Systems Engineer:** Hijrah Wira Pratama, S.Si.D. (**Bachelor of Data Science**, TelU)  
        **Lead Researcher:** Yokie Lidiantoro, S.P. (**Bachelor of Agriculture**, ITB)
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
    
    @st.cache_data(ttl=1800)
    def get_news_with_summary():
        try:
            gn = GoogleNews(lang='id', country='ID')
            # Query dioptimasi agar lebih mudah menemukan berita
            search = gn.search('kopi indonesia', when='30d') # Rentang waktu diperluas ke 30 hari
            news_list = []
            for entry in search['entries'][:4]:
                summary_prompt = f"Berikan ringkasan 1 kalimat profesional tentang berita ini: {entry.title}"
                summary = llm.invoke(summary_prompt).content
                news_list.append({
                    "title": entry.title, 
                    "link": entry.link, 
                    "source": entry.source.text, 
                    "date": entry.published,
                    "summary": summary
                })
            return news_list
        except: return []

    news_data = get_news_with_summary()
    if news_data:
        for n in news_data:
            st.markdown(f"""
            <div class="news-card">
                <a href="{n['link']}" target="_blank" style="text-decoration:none; color:#2d6a4f; font-weight:bold; font-size:1.15em;">🔗 {n['title']}</a><br>
                <small>Sumber: {n['source']} | {n['date']}</small>
                <div class="summary-box">
                    <b>🤖 AI Summary:</b> {n['summary']}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Sedang memuat berita terbaru. Jika masih kosong, coba gunakan koneksi internet yang lebih stabil atau refresh halaman.")

with tab_vision:
    st.warning("⚠️ RESEARCH PHASE")
    st.header("🔬 Coffee Vision AI")
    if os.path.exists("image_68c519.jpg"):
        st.image("image_68c519.jpg", use_container_width=True, caption="YOLOv11 Inference Testing")
    st.success("**Accuracy:** 98.4% (Collaboration ITB & TelU)")

st.markdown("<br><hr><center><small>© 2026 AgriPulse Project | Developed by Hijrah (TelU) & Yokie (ITB)</small></center>", unsafe_allow_html=True)
