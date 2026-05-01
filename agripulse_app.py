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

# --- 1. CONFIGURATION ---
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# --- 2. UI SETUP ---
st.set_page_config(page_title="AgriPulse v5.1", page_icon="🌱", layout="wide")

# CSS Aman: Background Hijau Gelap & Styling Tab
st.markdown("""
    <style>
    .main {
        background-color: #1b4332;
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: rgba(255,255,255,0.1);
        padding: 10px;
        border-radius: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #EE2D24 !important; /* Merah TelU */
        color: white !important;
    }
    div[data-testid="stExpander"] {
        background-color: rgba(255,255,255,0.05);
        border: none;
    }
    /* Memastikan teks header tidak tumpang tindih */
    .header-container {
        padding: 20px;
        background-color: rgba(0,0,0,0.2);
        border-radius: 15px;
        margin-bottom: 25px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. HEADER (Fixed Layout) ---
with st.container():
    col_logo1, col_logo2, col_text = st.columns([0.7, 0.7, 4.6])
    
    with col_logo1:
        if os.path.exists("telulogo.webp"):
            st.image("telulogo.webp", width=100)
        else:
            st.write("🎓 TelU")
            
    with col_logo2:
        if os.path.exists("itblogo.png"):
            st.image("itblogo.png", width=100)
        else:
            st.write("🌿 ITB")

    with col_text:
        st.title("🌱 AGRIPULSE ENGINE")
        st.markdown("### **RAG-Integrated Precision Agriculture**")
        st.write(f"**AI Engineer:** Hijrah Wira Pratama, S.S.id. (Data Science, TelU)")
        st.write(f"**Researcher:** Yokie Lidiantoro, S.T. (Agriculture, ITB)")

st.divider()

# --- 4. ENGINE INITIALIZATION ---
@st.cache_resource
def init_engine():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = ChatGroq(temperature=0.1, groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index("agripulse-index")
    stats = index.describe_index_stats()
    return embeddings, llm, stats['total_vector_count']

embeddings, llm, total_chunks = init_engine()

# --- 5. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Data Pipeline")
    up_file = st.file_uploader("Upload Riset Kopi (PDF)", type="pdf")
    if up_file and st.button("🚀 Sinkronkan Data"):
        with st.spinner("Processing..."):
            with open("temp.pdf", "wb") as f: f.write(up_file.getbuffer())
            loader = PyPDFLoader("temp.pdf")
            chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(loader.load())
            PineconeVectorStore.from_documents(chunks, embeddings, index_name="agripulse-index")
            os.remove("temp.pdf")
            st.success("Sync Success!")
            st.rerun()

# --- 6. TABS ---
t1, t2, t3 = st.tabs(["💬 AI Chat", "📰 Insights", "🔬 Vision"])

with t1:
    vs = PineconeVectorStore(index_name="agripulse-index", embedding=embeddings)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vs.as_retriever(search_kwargs={"k": 5}))

    if "msgs" not in st.session_state: st.session_state.msgs = []
    for m in st.session_state.msgs:
        with st.chat_message(m["role"]): st.markdown(m["content"])

    if p := st.chat_input("Tanya seputar penyakit kopi..."):
        st.session_state.msgs.append({"role": "user", "content": p})
        with st.chat_message("user"): st.markdown(p)
        with st.chat_message("assistant"):
            res = qa.invoke(p)
            st.markdown(res["result"])
            st.session_state.msgs.append({"role": "assistant", "content": res["result"]})

with t2:
    st.metric("Total Knowledge Chunks", total_chunks)
    st.subheader("📰 AI News Summary")
    try:
        gn = GoogleNews(lang='id', country='ID')
        search = gn.search('pertanian kopi indonesia', when='7d')
        for e in search['entries'][:3]:
            # Ringkasan berita otomatis
            sum_res = llm.invoke(f"Simpulkan berita ini dalam 1 kalimat pendek: {e.title}").content
            with st.expander(f"📌 {e.title}"):
                st.write(f"**Simpulan AI:** {sum_res}")
                st.write(f"[Baca Selengkapnya]({e.link})")
    except:
        st.write("Gagal memuat berita.")

with t3:
    st.header("🔬 Mobile Vision Scan")
    st.image("https://images.unsplash.com/photo-1592982537447-7440770cbfc9?q=80&w=600", caption="Prototype Deteksi Penyakit")
    st.info("Fitur YOLOv11 sedang dalam tahap integrasi oleh tim AI Engineer.")
