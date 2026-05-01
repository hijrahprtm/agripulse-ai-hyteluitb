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

# --- 2. APP UI SETUP & AGRI-THEME ENGINE ---
st.set_page_config(page_title="AgriPulse v4.1", page_icon="🌱", layout="wide")

# Custom CSS: Agriculture Background & Brand Colors
st.markdown("""
    <style>
    /* Background dengan nuansa perkebunan kopi */
    .stApp {
        background: linear-gradient(rgba(255, 255, 255, 0.9), rgba(255, 255, 255, 0.9)), 
                    url("https://images.unsplash.com/photo-1559056199-641a0ac8b55e?q=80&w=2070");
        background-size: cover;
        background-attachment: fixed;
    }
    
    /* Card Container agar teks terbaca jelas di atas background */
    .stTabs, .stSidebar, [data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.8) !important;
        padding: 15px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }

    /* Tab Styling (Merah TelU) */
    .stTabs [aria-selected="true"] {
        background-color: #EE2D24 !important;
        color: white !important;
        border-radius: 8px;
    }

    /* Button Styling (Hijau ITB) */
    div.stButton > button:first-child {
        background-color: #006633 !important;
        color: white !important;
        border-radius: 10px;
        border: none;
        height: 3em;
        width: 100%;
        font-weight: bold;
    }
    
    div.stButton > button:hover {
        background-color: #EE2D24 !important;
        border: 1px solid white;
    }

    /* Title & Text Colors */
    h1, h2, h3 {
        color: #1B4332 !important; /* Hijau Tua Daun */
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. CUSTOM HEADER & BRANDING ---
col_logo1, col_logo2, col_text = st.columns([0.6, 0.6, 4.8])

with col_logo1:
    if os.path.exists("telulogo.webp"):
        st.image("telulogo.webp", width=100)
    else:
        st.image("https://upload.wikimedia.org/wikipedia/id/0/03/Logo_Telkom_University_potrait.png", width=100)

with col_logo2:
    if os.path.exists("itblogo.png"):
        st.image("itblogo.png", width=100)

with col_text:
    st.title("🌱 AGRIPULSE")
    st.markdown("### **Agricultural RAG-Integrated Precision Understanding & Localized Synthesis Engine**")
    st.markdown(
        """
        **AI Systems Engineer:** Hijrah Wira Pratama, S.S.id. (Bachelor of Data Science, TelU)  
        **Lead Agricultural Researcher:** Yokie Lidiantoro, S.T. (Bachelor of Agriculture, ITB)
        """
    )
st.divider()

# --- 4. INITIALIZE MODELS ---
@st.cache_resource
def init_models():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "agripulse-index"
    
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name, dimension=384, metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
    
    index = pc.Index(index_name)
    stats = index.describe_index_stats()
    return embeddings, index_name, stats['total_vector_count']

embeddings, index_name, total_chunks = init_models()

# --- 5. SIDEBAR ---
with st.sidebar:
    st.header("🧠 Knowledge Management")
    uploaded_file = st.file_uploader("Upload Jurnal Pertanian (PDF)", type="pdf")
    
    if uploaded_file and st.button("Indeks Data Baru"):
        with st.spinner("AI sedang memproses riset..."):
            with open("temp_upload.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            loader = PyPDFLoader("temp_upload.pdf")
            data = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(data)
            PineconeVectorStore.from_documents(chunks, embeddings, index_name=index_name)
            st.success(f"Berhasil! {len(chunks)} chunks ditambahkan.")
            os.remove("temp_upload.pdf")
            st.rerun()
    st.divider()
    st.info("Sistem ini mengintegrasikan data riset ITB dengan arsitektur AI dari Telkom University.")

# --- 6. MAIN TABS ---
tab1, tab2, tab3 = st.tabs(["💬 AI Assistant", "📊 Research Insights", "🔬 CV Diagnostic"])

# TAB 1: RAG CHAT
with tab1:
    vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    llm = ChatGroq(temperature=0.1, groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever(search_kwargs={"k": 7}))

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Tanyakan tentang penyakit atau budidaya kopi..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Mencari referensi riset..."):
                response = qa_chain.invoke(prompt)
                st.markdown(response["result"])
                st.session_state.messages.append({"role": "assistant", "content": response["result"]})

# TAB 2: LIVE NEWS & METRICS
with tab2:
    st.header("📈 Research Dashboard")
    m1, m2 = st.columns(2)
    m1.metric("Sync Status", "Cloud Active", "Stable")
    m2.metric("Knowledge Base", f"{total_chunks} Chunks", "Updated")
    
    st.divider()
    st.subheader("📰 Berita Terkini Pertanian Kopi")
    
    try:
        gn = GoogleNews(lang='id', country='ID')
        search = gn.search('penyakit tanaman kopi indonesia', when='30d')
        if search['entries']:
            for entry in search['entries'][:5]:
                st.markdown(f"✅ **[{entry.title}]({entry.link})**")
                st.caption(f"Sumber: {entry.source.text} | {entry.published}")
        else:
            st.write("Belum ada berita terbaru bulan ini.")
    except Exception as e:
        st.info("Fitur berita akan aktif setelah library 'pygooglenews' terpasang.")

# TAB 3: COMPUTER VISION
with tab3:
    st.header("🔬 Computer Vision Mobile Diagnostic")
    st.markdown("#### *Deteksi Penyakit Daun & Biji Kopi via Smartphone*")
    
    cv_col1, cv_col2 = st.columns([3, 2])
    with cv_col1:
        st.info("Status: Prototype (YOLOv11 Integration)")
        st.markdown("""
        **Cara Kerja untuk Peneliti:**
        - Gunakan aplikasi mobile untuk memotret gejala penyakit di lahan.
        - AI akan melakukan segmentasi pada area yang terkena karat daun (*leaf rust*) atau jamur.
        - Hasil diagnosa akan disinkronkan dengan database riset di Tab 1.
        """)
    
    with cv_col2:
        # Gambar ilustrasi smartphone mendeteksi tanaman (Aman & Relevan)
        st.image("https://images.unsplash.com/photo-1581091226825-a6a2a5aee158?q=80&w=400", 
                 caption="Mobile AI Diagnostic Interface Concept")
