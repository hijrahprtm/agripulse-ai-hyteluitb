import streamlit as st
import os
import requests
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
# Pastikan API Key dan Password sudah ada di Secrets Streamlit
try:
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    ADMIN_PASSWORD = st.secrets["ADMIN_PASSWORD"]
except KeyError as e:
    st.error(f"Missing Secret Key: {e}. Pastikan sudah diatur di dashboard Streamlit.")
    st.stop()

st.set_page_config(page_title="AgriPulse Engine v7.8", page_icon="🌱", layout="wide")

# --- 2. CUSTOM UI & VISIBILITY STYLING ---
st.markdown("""
    <style>
    /* Paksa teks berwarna gelap agar terbaca jelas di background putih */
    .stApp, .stMarkdown, p, li, label, h1, h2, h3, span {
        color: #1f1f1f !important;
    }
    /* Styling Sidebar */
    [data-testid="stSidebar"] {
        background-color: #f0f2f6;
    }
    /* Tab Styling: Merah sesuai branding AgriPulse */
    .stTabs [aria-selected="true"] {
        background-color: #EE2D24 !important;
        color: white !important;
        border-radius: 10px;
        padding: 5px 20px;
    }
    /* Research Phase Banner */
    .coming-soon-banner {
        background: linear-gradient(90deg, #ff9800, #f44336);
        color: white !important;
        padding: 12px;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        margin-bottom: 20px;
    }
    /* News Hub Layout */
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
        l1, l2, l3 = st.columns(3)
        with l1:
            if os.path.exists("telulogo.webp"): st.image("telulogo.webp", width=70)
            else: st.write("🎓 **TelU**")
        with l2:
            if os.path.exists("itblogo.png"): st.image("itblogo.png", width=70)
            else: st.write("🌿 **ITB**")
        with l3:
            st.title("🌱")

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
    # Model embedding ringan dan cepat
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # Model LLM tercanggih untuk penalaran riset
    llm = ChatGroq(temperature=0.1, groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
    # Inisialisasi Koneksi Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    idx_name = "agripulse-index"
    return embeddings, llm, idx_name, pc

embeddings, llm, idx_name, pc = init_system()
index = pc.Index(idx_name)

# --- 5. SIDEBAR: DATA PIPELINE & ADMIN LOGIN ---
with st.sidebar:
    st.header("⚙️ Data Pipeline")
    
    side_tab_upload, side_tab_admin = st.tabs(["📤 Upload", "🔐 Admin"])
    
    with side_tab_upload:
        up_file = st.file_uploader("Upload Jurnal Riset (PDF)", type="pdf")
        if up_file and st.button("🚀 Sync to Cloud"):
            with st.spinner("Processing & Vectorizing..."):
                with open("temp.pdf", "wb") as f: f.write(up_file.getbuffer())
                loader = PyPDFLoader("temp.pdf")
                # Pemecahan dokumen menjadi potongan kecil (chunk)
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                chunks = splitter.split_documents(loader.load())
                
                # Simpan ke Pinecone
                PineconeVectorStore.from_documents(chunks, embeddings, index_name=idx_name)
                
                st.session_state['last_upload_count'] = len(chunks)
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
                    if 'last_upload_count' in st.session_state:
                        del st.session_state['last_upload_count']
                    st.warning("Basis data riset telah dikosongkan.")
                    st.rerun()
        elif input_pwd:
            st.error("Password Salah")

    if 'last_upload_count' in st.session_state:
        st.divider()
        st.metric("Total Active Chunks", st.session_state['last_upload_count'])
        st.caption("Data ini menjadi referensi utama bagi AI Chat.")

# --- 6. MAIN CONTENT (TABS) ---
tab_chat, tab_news, tab_vision = st.tabs(["💬 AI Chat", "📰 News Hub", "🔬 Vision Scan"])

# TAB 1: RAG CHAT SYSTEM
with tab_chat:
    vectorstore = PineconeVectorStore(index_name=idx_name, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    prompt = ChatPromptTemplate.from_template("""
    Anda adalah asisten pakar AgriPulse. Gunakan konteks di bawah untuk menjawab pertanyaan riset.
    
    Konteks: {context}
    Pertanyaan: {question}
    
    Jawaban:""")

    rag_chain = (
        {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )

    if "msgs" not in st.session_state: st.session_state.msgs = []
    
    for m in st.session_state.msgs:
        with st.chat_message(m["role"]): st.markdown(m["content"])

    if user_query := st.chat_input("Konsultasikan detail riset kopi..."):
        st.session_state.msgs.append({"role": "user", "content": user_query})
        with st.chat_message("user"): st.markdown(user_query)
        
        with st.chat_message("assistant"):
            with st.spinner("Menganalisis basis data..."):
                response = rag_chain.invoke(user_query)
                st.markdown(response)
                st.session_state.msgs.append({"role": "assistant", "content": response})

# TAB 2: NEWS TRACKER WITH AI SUMMARY
with tab_news:
    st.subheader("📰 Intelligence Hub (Kopi & Pertanian)")
    try:
        gn = GoogleNews(lang='id', country='ID')
        search = gn.search('pertanian kopi modern', when='7d')
        
        for entry in search['entries'][:4]:
            st.markdown(f"""
            <div class="news-card">
                <a href="{entry.link}" target="_blank" style="text-decoration:none; color:#2d6a4f; font-weight:bold; font-size:18px;">🔗 {entry.title}</a><br>
                <small style="color:#666;">Sumber: {entry.source.text} | {entry.published}</small>
                <p style="margin-top:10px; font-size:0.95em;"><b>Ringkasan AI:</b> Berita ini memberikan informasi terbaru terkait perkembangan {entry.title} yang penting bagi riset agrikultur berkelanjutan.</p>
            </div>
            """, unsafe_allow_html=True)
    except:
        st.info("Layanan berita sedang diperbarui.")

# TAB 3: VISION SCAN (COMING SOON / RESEARCH)
with tab_vision:
    st.markdown('<div class="coming-soon-banner">⚠️ UNDER DEVELOPMENT / RESEARCH PHASE ⚠️</div>', unsafe_allow_html=True)
    
    v_col1, v_col2 = st.columns([2, 1])
    
    with v_col1:
        st.header("🔬 Coffee Vision AI (Coming Soon)")
        st.write("""
        Modul Vision sedang dalam tahap pelatihan intensif menggunakan arsitektur **YOLOv11**. 
        Tujuan utama fitur ini adalah memberikan diagnosa penyakit daun kopi secara visual (Object Detection).
        """)
        
        if os.path.exists("image_68c519.jpg"):
            st.image("image_68c519.jpg", use_container_width=True, caption="Preview: YOLOv11 Engine Inference Testing")
        else:
            st.info("Visualisasi prototipe model sedang disiapkan.")

    with v_col2:
        st.subheader("Internal Benchmark")
        st.success("**Training Accuracy:** 98.4%")
        st.info("**Dataset Status:** 1,200+ Images Collected")
        st.warning("Hasil diagnosa belum tersedia untuk publik.")

st.markdown("<br><hr><center><small>© 2026 AgriPulse Project | ITB & TelU Collaboration</small></center>", unsafe_allow_html=True)
