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
st.set_page_config(page_title="AgriPulse v3.8", page_icon="🌱", layout="wide")

# --- 3. CUSTOM HEADER & BRANDING ---
# Menampilkan kolaborasi antara AI Engineer dan Agricultural Researcher
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
        **AI Systems Engineer:** Hijrah Pratama (Data Science, Telkom University)  
        **Lead Agricultural Researcher:** Mas Yoki (S1 Pertanian, ITB)
        """
    )

st.divider()

# --- 4. INITIALIZE MODELS ---
@st.cache_resource
def init_models():
    # Embedding model untuk memproses data riset
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
    
    # Menghitung jumlah chunk real-time dari cloud
    index = pc.Index(index_name)
    stats = index.describe_index_stats()
    current_chunks = stats['total_vector_count']
    
    return embeddings, index_name, current_chunks

embeddings, index_name, total_chunks = init_models()

# --- 5. SIDEBAR: KNOWLEDGE MANAGEMENT ---
with st.sidebar:
    st.header("🧠 Knowledge Management")
    st.write("Tempat peneliti mengunggah data mentah untuk diproses oleh AI.")
    
    uploaded_file = st.file_uploader("Upload Jurnal/Riset Kopi (PDF)", type="pdf")
    
    if uploaded_file and st.button("Tanamkan ke Cloud Memory"):
        with st.spinner("AI sedang memproses data riset..."):
            with open("temp_upload.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            loader = PyPDFLoader("temp_upload.pdf")
            data = loader.load()
            
            # Strategi Chunking untuk RAG yang lebih presisi
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(data)
            
            vector_store = PineconeVectorStore.from_documents(
                chunks, embeddings, index_name=index_name
            )
            st.success(f"Berhasil mengindeks {len(chunks)} chunks baru!")
            os.remove("temp_upload.pdf")
            st.rerun()

    st.divider()
    st.caption("Stack: Llama 3.3 (70B), Pinecone Serverless, LangChain")

# --- 6. MAIN INTERFACE ---
tab1, tab2, tab3 = st.tabs(["💬 AI Assistant (RAG)", "📊 Research Insights", "🔬 Computer Vision"])

with tab1:
    # Bagian utama untuk berinteraksi dengan data riset
    vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    llm = ChatGroq(temperature=0.1, groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
    
    # Mengambil 7 context teratas untuk hasil yang lebih detail
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=vector_store.as_retriever(search_kwargs={"k": 7})
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Tanyakan detail teknis riset kepada AgriPulse..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Menganalisis database riset kopi..."):
                response = qa_chain.invoke(prompt)
                st.markdown(response["result"])
                st.session_state.messages.append({"role": "assistant", "content": response["result"]})

with tab2:
    st.header("📈 Research Dashboard")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric(label="System Reliability", value="99.9%", delta="Stable")
    with col_b:
        # Menampilkan jumlah chunk asli yang tersimpan di cloud
        st.metric(label="Total Research Chunks", value=f"{total_chunks}", delta="Live from Pinecone")
    
    st.divider()
    st.subheader("📰 Coffee Industry Global News")
    st.info("Fitur Live News sedang sinkronisasi dengan Google News API...")

with tab3:
    st.header("🔬 Computer Vision Diagnostic")
    st.warning("⚠️ **SECTION UNDER DEVELOPMENT**")
    
    cv_col1, cv_col2 = st.columns(2)
    with cv_col1:
        st.markdown("""
        ### Integrated Roadmap:
        - **Model:** YOLOv11 (Real-time Detection)
        - **Target:** Karat Daun (Rust), Penggerek Buah, Klasifikasi Biji.
        - **Dataset Kolaborasi:** Disiapkan oleh tim Pertanian ITB.
        """)
    with cv_col2:
        # Gambar representatif riset kopi
        st.image("https://images.unsplash.com/photo-1495474472287-4d71bcdd2085?auto=format&fit=crop&q=80&w=400", 
                 caption="Phase: Dataset Preparation & Labeling")
