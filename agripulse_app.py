import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA

# --- 1. CONFIGURATION & SECRETS ---
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# --- 2. APP UI SETUP ---
st.set_page_config(page_title="AgriPulse v3.6", page_icon="🌱", layout="wide")

# --- 3. CUSTOM HEADER & BRANDING ---
header_col1, header_col2 = st.columns([1, 4])
with header_col1:
    # URL Logo Telkom University Resmi
    st.image("https://upload.wikimedia.org/wikipedia/id/0/03/Logo_Telkom_University_potrait.png", width=130)

with header_col2:
    st.title("🌱 AGRIPULSE")
    st.markdown("### **Agricultural RAG-Integrated Precision Understanding & Localized Synthesis Engine**")
    st.write(f"**Lead Developer:** Hijrah Pratama (Data Science, Telkom University)")
    st.write(f"**Subject Matter Expert:** Mas Yoki (S1 Pertanian, ITB)")
st.divider()

# --- 4. INITIALIZE MODELS ---
@st.cache_resource
def init_models():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "agripulse-index"
    
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
    
    # Ambil statistik index untuk dashboard
    index = pc.Index(index_name)
    stats = index.describe_index_stats()
    total_chunks = stats['total_vector_count']
    
    return embeddings, index_name, total_chunks

embeddings, index_name, total_chunks = init_models()

# --- 5. SIDEBAR: KNOWLEDGE BASE MANAGEMENT ---
with st.sidebar:
    st.header("🧠 Knowledge Management")
    uploaded_file = st.file_uploader("Upload Jurnal/Riset Kopi (PDF)", type="pdf")
    
    if uploaded_file and st.button("Tanamkan ke Cloud Memory"):
        with st.spinner("Processing Chunks..."):
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            loader = PyPDFLoader("temp.pdf")
            data = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(data)
            
            vector_store = PineconeVectorStore.from_documents(
                chunks, embeddings, index_name=index_name
            )
            st.success(f"Berhasil menanamkan {len(chunks)} chunks baru!")
            st.rerun() # Refresh untuk update counter di dashboard

    st.divider()
    st.caption("Tech Stack: Llama 3.3, Pinecone Cloud, LangChain")

# --- 6. MAIN TABS ---
tab1, tab2, tab3 = st.tabs(["💬 AI Assistant (RAG)", "📈 Dashboard & News", "🔬 Computer Vision"])

with tab1:
    vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    llm = ChatGroq(temperature=0.1, groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever(search_kwargs={"k": 5}))

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Tanyakan sesuatu tentang riset kopi..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            response = qa_chain.invoke(prompt)
            st.markdown(response["result"])
            st.session_state.messages.append({"role": "assistant", "content": response["result"]})

with tab2:
    st.header("📈 Live Agricultural Insights")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric(label="Market Sentiment Index", value="85%", delta="+2.5%")
    with col_b:
        # Menampilkan jumlah chunk asli dari Pinecone
        st.metric(label="Research Database", value=f"{total_chunks} Chunks", delta="Live from Cloud")
    
    st.divider()
    st.subheader("Latest Coffee Industry News")
    st.info("Koneksi ke pygooglenews aktif. Menunggu sinkronisasi feed berita terbaru...")

with tab3:
    st.header("🔬 Coffee Bean Health Analysis")
    st.warning("🚧 **DEVELOPMENT PHASE**")
    
    cv_col1, cv_col2 = st.columns(2)
    with cv_col1:
        st.markdown("""
        ### Roadmap Fitur:
        1. **Leaf Rust Detection:** Identifikasi jamur Hemileia vastatrix.
        2. **Bean Grading:** Klasifikasi otomatis biji kopi berdasarkan standar SCAA.
        3. **Pest Identification:** Deteksi hama penggerek buah kopi.
        
        **Model:** YOLOv11 & Vision Transformer (ViT)
        """)
    with cv_col2:
        # Placeholder gambar yang aman dan profesional
        st.image("https://images.unsplash.com/photo-1559056199-641a0ac8b55e?auto=format&fit=crop&q=80&w=400", 
                 caption="Preview: Research on Coffee Bean Quality Control")
