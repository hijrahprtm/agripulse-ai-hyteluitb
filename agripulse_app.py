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

# --- 2. APP UI SETUP & THEME ENGINE ---
st.set_page_config(page_title="AgriPulse v4.0", page_icon="🌱", layout="wide")

# Custom CSS: Dark Theme, Tab Colors (TelU/ITB), and Button Styling
st.markdown("""
    <style>
    /* Background Utama */
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #161B22;
        padding: 10px;
        border-radius: 15px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #8B949E;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #EE2D24 !important; /* Merah TelU */
        color: white !important;
        border-radius: 10px;
    }
    /* Button Styling (Hijau ITB) */
    div.stButton > button:first-child {
        background-color: #006633;
        color: white;
        border-radius: 10px;
        border: none;
        font-weight: bold;
    }
    div.stButton > button:hover {
        border: 2px solid #EE2D24;
        color: #EE2D24;
    }
    /* Sidebar Background */
    [data-testid="stSidebar"] {
        background-color: #161B22;
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

# --- 4. INITIALIZE MODELS & DATA ---
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
    uploaded_file = st.file_uploader("Input Data Riset (PDF)", type="pdf")
    
    if uploaded_file and st.button("Tanamkan ke Cloud Memory"):
        with st.spinner("AI sedang mengindeks data..."):
            with open("temp_upload.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            loader = PyPDFLoader("temp_upload.pdf")
            data = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(data)
            PineconeVectorStore.from_documents(chunks, embeddings, index_name=index_name)
            st.success(f"Berhasil mengindeks {len(chunks)} chunks!")
            os.remove("temp_upload.pdf")
            st.rerun()
    st.divider()
    st.caption("Developed by Hijrah (TelU) & Yokie (ITB)")

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

    if prompt := st.chat_input("Tanyakan detail teknis riset..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            response = qa_chain.invoke(prompt)
            st.markdown(response["result"])
            st.session_state.messages.append({"role": "assistant", "content": response["result"]})

# TAB 2: LIVE NEWS & METRICS
with tab2:
    st.header("📈 Research & Market Insights")
    m1, m2 = st.columns(2)
    m1.metric("Cloud Sync Status", "Active", "Online")
    m2.metric("Total Indexed Knowledge", f"{total_chunks} Chunks", "Live from Pinecone")
    
    st.divider()
    st.subheader("📰 Latest Coffee Industry News")
    
    # Fungsi Live News menggunakan pygooglenews
    try:
        gn = GoogleNews(lang='id', country='ID')
        search = gn.search('pertanian kopi indonesia', when='7d')
        for entry in search['entries'][:5]:
            st.markdown(f"**[{entry.title}]({entry.link})**")
            st.caption(f"Published: {entry.published}")
    except:
        st.warning("Gagal memuat berita. Pastikan 'pygooglenews' terinstal.")

# TAB 3: COMPUTER VISION
with tab3:
    st.header("🔬 Computer Vision Mobile Diagnostic")
    st.warning("🚀 **PROTOTYPE PHASE**")
    
    cv_col1, cv_col2 = st.columns([3, 2])
    with cv_col1:
        st.markdown("""
        ### Mobile Scanning Workflow:
        1. **Capture:** Peneliti mengambil foto biji kopi menggunakan smartphone.
        2. **Process:** Model YOLOv11 melakukan *inference* di server.
        3. **Analyze:** AI mendiagnosa jenis penyakit secara otomatis.
        """)
        st.info("Fitur ini sedang dikembangkan untuk membantu petani di lapangan.")
    
    with cv_col2:
        # Gambar Placeholder yang AMAN (Menggunakan URL resmi Unsplash yang spesifik kopi)
        st.image("https://images.unsplash.com/photo-1511537190424-bbbab87ac5eb?q=80&w=400", 
                 caption="Konsep Deteksi Penyakit via Smartphone")
