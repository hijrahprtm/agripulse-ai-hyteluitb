import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from pygooglenews import GoogleNews

# --- 1. CONFIGURATION & SECRETS ---
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# --- 2. APP UI SETUP & CREATIVE THEME ---
st.set_page_config(page_title="AgriPulse v5.0", page_icon="🌱", layout="wide")

# CSS Kreatif: Glassmorphism & Gradient Agriculture
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #1b4332 0%, #081c15 100%);
        background-image: url("https://www.transparenttextures.com/patterns/leaf.png"), 
                          linear-gradient(135deg, rgba(27, 67, 50, 0.9) 0%, rgba(8, 28, 21, 0.9) 100%);
    }
    
    /* Card Glassmorphism */
    [data-testid="stMetric"], .stTabs, .stSidebarContent {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px !important;
        padding: 20px;
        color: white !important;
    }

    /* Tab Styling (Merah TelU) */
    .stTabs [data-baseweb="tab-list"] { background: transparent; }
    .stTabs [aria-selected="true"] {
        background-color: #EE2D24 !important;
        color: white !important;
        border-radius: 12px;
    }
    
    /* Button Styling (Hijau ITB) */
    div.stButton > button {
        background: linear-gradient(90deg, #2d6a4f, #1b4332) !important;
        color: white !important;
        border-radius: 15px !important;
        border: none !important;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 15px rgba(45, 106, 79, 0.5);
    }

    /* News Box */
    .news-card {
        background: rgba(0, 0, 0, 0.2);
        padding: 15px;
        border-left: 4px solid #EE2D24;
        margin-bottom: 10px;
        border-radius: 0 10px 10px 0;
    }
    
    h1, h2, h3, p, span { color: #ffffff !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. CUSTOM HEADER & BRANDING ---
col_logo1, col_logo2, col_text = st.columns([0.6, 0.6, 4.8])

with col_logo1:
    st.image("telulogo.webp", width=100) if os.path.exists("telulogo.webp") else st.write("🎓 TelU")
with col_logo2:
    st.image("itblogo.png", width=100) if os.path.exists("itblogo.png") else st.write("🌿 ITB")

with col_text:
    st.title("🌱 AGRIPULSE ENGINE")
    st.markdown("#### **RAG-Integrated Precision Agriculture & Localized Synthesis**")
    st.markdown(
        """
        **AI Engineer:** Hijrah Wira Pratama, S.S.id. (Data Science, TelU)  
        **Agricultural Researcher:** Yokie Lidiantoro, S.T. (Agriculture, ITB)
        """
    )
st.divider()

# --- 4. MODELS INITIALIZATION ---
@st.cache_resource
def init_system():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = ChatGroq(temperature=0.1, groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index("agripulse-index")
    stats = index.describe_index_stats()
    return embeddings, llm, stats['total_vector_count']

embeddings, llm, total_chunks = init_system()

# --- 5. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Data Pipeline")
    uploaded_file = st.file_uploader("Upload Riset Kopi (PDF)", type="pdf")
    if uploaded_file and st.button("🚀 Sinkronkan Pengetahuan"):
        with st.spinner("AI sedang membedah dokumen..."):
            with open("temp.pdf", "wb") as f: f.write(uploaded_file.getbuffer())
            loader = PyPDFLoader("temp.pdf")
            chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(loader.load())
            PineconeVectorStore.from_documents(chunks, embeddings, index_name="agripulse-index")
            st.success("Cloud Memory diperbarui!")
            os.remove("temp.pdf")
            st.rerun()

# --- 6. MAIN TABS ---
tab1, tab2, tab3 = st.tabs(["💬 RAG Chat", "📰 Intelligence Hub", "🔬 Vision Scan"])

with tab1:
    # Arsitektur Chat Mas Hijrah
    vector_store = PineconeVectorStore(index_name="agripulse-index", embedding=embeddings)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever(search_kwargs={"k": 5}))

    if "messages" not in st.session_state: st.session_state.messages = []
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("Konsultasikan data riset pertanian..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            res = qa_chain.invoke(prompt)
            st.markdown(res["result"])
            st.session_state.messages.append({"role": "assistant", "content": res["result"]})

with tab2:
    st.header("📊 Research & Market Intelligence")
    c1, c2 = st.columns(2)
    c1.metric("Neural Chunks", total_chunks)
    c2.metric("System Health", "Optimal")
    
    st.divider()
    st.subheader("📰 Live Agricultural Summaries")
    
    # Fitur Baru: News Summarization
    with st.spinner("AI sedang merangkum berita terbaru untuk Anda..."):
        try:
            gn = GoogleNews(lang='id', country='ID')
            search = gn.search('budidaya penyakit kopi indonesia', when='7d')
            for entry in search['entries'][:3]:
                # AI membuat ringkasan singkat (Simpulan)
                summary_prompt = f"Berikan ringkasan 1 kalimat dari judul berita ini: {entry.title}"
                summary = llm.invoke(summary_prompt).content
                
                st.markdown(f"""
                <div class="news-card">
                    <a href="{entry.link}" style="color: #EE2D24; font-weight: bold;">{entry.title}</a><br>
                    <p style="font-size: 0.9em; margin-top: 5px;"><b>Simpulan AI:</b> {summary}</p>
                    <small>Sumber: {entry.source.text} | {entry.published}</small>
                </div>
                """, unsafe_allow_html=True)
        except:
            st.error("Koneksi berita terputus.")

with tab3:
    st.header("🔬 Mobile Computer Vision")
    v1, v2 = st.columns([3, 2])
    with v1:
        st.markdown("### Diagnosis Lapangan Otomatis")
        st.info("Teknologi: YOLOv11 & FastSAM")
        st.write("- **Deteksi:** Karat daun, Antraknosa, Bubuk Buah.")
        st.write("- **Output:** Rekomendasi pestisida organik berdasarkan database Mas Yoki.")
    with v2:
        # Gambar Kreatif: Smartphone menscan daun kopi yang sakit
        st.image("https://images.unsplash.com/photo-1592982537447-7440770cbfc9?q=80&w=400", 
                 caption="Prototyping Vision-RAG Hybrid")
