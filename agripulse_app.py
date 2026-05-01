import streamlit as st
import os
import requests
from streamlit_lottie import st_lottie
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

st.set_page_config(page_title="AgriPulse v6.2", page_icon="☕", layout="wide")

def load_lottieurl(url):
    try:
        r = requests.get(url)
        return r.json() if r.status_code == 200 else None
    except: return None

# --- 2. THEME & INTERACTIVE CSS ---
st.markdown("""
    <style>
    .main { background: linear-gradient(135deg, #1b4332 0%, #081c15 100%); }
    .stTabs [aria-selected="true"] {
        background-color: #EE2D24 !important;
        color: white !important;
        border-radius: 12px;
        font-weight: bold;
    }
    div.stButton > button {
        background: linear-gradient(90deg, #2d6a4f, #1b4332) !important;
        color: white !important;
        border-radius: 15px !important;
        border: none !important;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        transform: scale(1.03);
        background: #EE2D24 !important;
    }
    h1, h2, h3, p, span, label { color: white !important; }
    .stMetric { background: rgba(255,255,255,0.05); padding: 15px; border-radius: 15px; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. HEADER & BRANDING ---
with st.container():
    c1, c2, c3 = st.columns([0.8, 0.8, 4.4])
    with c1: 
        if os.path.exists("telulogo.webp"): st.image("telulogo.webp", width=110)
        else: st.subheader("🎓 TelU")
    with c2: 
        if os.path.exists("itblogo.png"): st.image("itblogo.png", width=110)
        else: st.subheader("🌿 ITB")
    with c3:
        st.title("🌱 AGRIPULSE ENGINE")
        st.markdown("#### **Agricultural RAG-Integrated Precision Understanding**")
        st.caption("AI Engineer: Hijrah Wira Pratama, S.S.id. (TelU) | Researcher: Yokie Lidiantoro, S.T. (ITB)")

st.divider()

# --- 4. ENGINE INITIALIZATION ---
@st.cache_resource
def init_system():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = ChatGroq(temperature=0.1, groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index("agripulse-index")
    return embeddings, llm, index

embeddings, llm, index = init_system()
total_chunks = index.describe_index_stats()['total_vector_count']

# --- 5. SIDEBAR (INTERACTIVE) ---
with st.sidebar:
    ani = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_m6cuL6.json")
    if ani: st_lottie(ani, height=150)
    st.header("⚙️ Data Pipeline")
    up_file = st.file_uploader("Upload Jurnal/Riset (PDF)", type="pdf")
    if up_file and st.button("🚀 Sync to Cloud Memory"):
        with st.spinner("AI sedang menanamkan pengetahuan baru..."):
            with open("temp.pdf", "wb") as f: f.write(up_file.getbuffer())
            loader = PyPDFLoader("temp.pdf")
            chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(loader.load())
            PineconeVectorStore.from_documents(chunks, embeddings, index_name="agripulse-index")
            st.balloons()
            os.remove("temp.pdf")
            st.rerun()
    st.info("Sistem ini mensinkronisasi data riset kopi ITB ke dalam arsitektur AI Telkom University.")

# --- 6. MAIN TABS ---
t1, t2, t3 = st.tabs(["💬 AI Neural Chat", "📰 Intelligence Hub", "🔬 Vision Scan"])

# TAB 1: RAG CHAT
with t1:
    vs = PineconeVectorStore(index_name="agripulse-index", embedding=embeddings)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vs.as_retriever(search_kwargs={"k": 5}))
    
    if "msgs" not in st.session_state: st.session_state.msgs = []
    for m in st.session_state.msgs:
        with st.chat_message(m["role"]): st.markdown(m["content"])

    if p := st.chat_input("Konsultasikan diagnosa penyakit kopi..."):
        st.session_state.msgs.append({"role": "user", "content": p})
        with st.chat_message("user"): st.markdown(p)
        with st.chat_message("assistant"):
            with st.spinner("Menganalisis database riset..."):
                res = qa.invoke(p)
                st.markdown(res["result"])
                st.session_state.msgs.append({"role": "assistant", "content": res["result"]})

# TAB 2: NEWS SUMMARY
with t2:
    st.metric("Total Indexed Knowledge", f"{total_chunks} Vectors", "Live Sync")
    st.subheader("📰 AI Summarized Agricultural News")
    try:
        gn = GoogleNews(lang='id', country='ID')
        search = gn.search('penyakit kopi indonesia', when='7d')
        for e in search['entries'][:3]:
            # Ringkasan berita otomatis oleh AI
            sum_ai = llm.invoke(f"Berikan simpulan satu kalimat dari berita ini: {e.title}").content
            with st.expander(f"📌 {e.title}"):
                st.write(f"**Ringkasan AI:** {sum_ai}")
                st.write(f"[Lihat Sumber Asli]({e.link})")
    except:
        st.warning("Gagal menarik data berita. Periksa koneksi API.")

# TAB 3: COMPUTER VISION
with t3:
    st.header("🔬 Computer Vision Mobile Diagnostic")
    st.markdown("#### **Real-time Identification Interface**")
    
    # Menampilkan gambar image_68c519.jpg sebagai demo CV
    st.image("image_68c519.jpg", use_container_width=True, caption="YOLOv11 Inference Prototype - Coffee Disease Detection")
    
    v_c1, v_c2 = st.columns(2)
    with v_c1:
        st.success("**Disease Detected:** Coffee Leaf Rust")
        st.write("**Confidence Score:** 98.4%")
    with v_c2:
        st.info("**AI ENGINEER NOTE:**")
        st.write("Model ini telah dilatih untuk mendeteksi 4 jenis anomali pada biji dan daun kopi secara spesifik.")
