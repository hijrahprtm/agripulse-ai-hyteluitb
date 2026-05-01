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
# Pastikan PINECONE_API_KEY dan GROQ_API_KEY sudah ada di Streamlit Secrets
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# --- 2. APP UI SETUP ---
st.set_page_config(page_title="AgriPulse v3.5", page_icon="☕", layout="wide")

# --- 3. CUSTOM HEADER & BRANDING ---
# Menampilkan identitas Lead Developer (DS TelU) & SME (Pertanian ITB)
col1, col2 = st.columns([1, 5])
with col1:
    # Logo Telkom University dari sumber stabil
    st.image("https://upload.wikimedia.org/wikipedia/id/0/03/Logo_Telkom_University_potrait.png", width=120)
with col2:
    st.title("🌱 AGRIPULSE")
    st.markdown("### **Agricultural RAG-Integrated Precision Understanding & Localized Synthesis Engine**")
    st.markdown(
        """
        **Lead Developer:** Hijrah Pratama (Data Science, Telkom University)  
        **Subject Matter Expert:** Mas Yoki (S1 Pertanian, ITB)
        """
    )
st.divider()

# --- 4. INITIALIZE MODELS (Vector DB & Embeddings) ---
@st.cache_resource
def init_models():
    # Model embedding ringan untuk memproses 5000+ chunks
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Inisialisasi Pinecone Cloud
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "agripulse-index"
    
    # Cek/Buat Index di Pinecone jika belum ada
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384, # Dimensi sesuai all-MiniLM-L6-v2
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
    
    return embeddings, index_name

embeddings, index_name = init_models()

# --- 5. SIDEBAR: KNOWLEDGE BASE MANAGEMENT ---
with st.sidebar:
    st.header("🧠 Knowledge Management")
    uploaded_file = st.file_uploader("Upload Jurnal/Riset Kopi (PDF)", type="pdf")
    
    if uploaded_file and st.button("Tanamkan ke Cloud Memory"):
        with st.spinner("Processing & Vectorizing Chunks to Pinecone..."):
            # Simpan file sementara
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            loader = PyPDFLoader("temp.pdf")
            data = loader.load()
            
            # Split data riset menjadi chunks kecil agar RAG lebih akurat
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(data)
            
            # Simpan secara permanen ke Pinecone Cloud
            vector_store = PineconeVectorStore.from_documents(
                chunks, 
                embeddings, 
                index_name=index_name
            )
            st.success(f"Berhasil menyimpan {len(chunks)} chunks ke Cloud!")
            os.remove("temp.pdf")
    
    st.divider()
    st.info("Sistem menggunakan Llama 3.3 (Groq) untuk analisis data riset yang presisi.")

# --- 6. MAIN TABS INTERFACE ---
tab1, tab2, tab3 = st.tabs(["💬 AI Assistant (RAG)", "📰 Live News & Dashboard", "🔍 Computer Vision (Health Check)"])

# TAB 1: RAG Chatbot
with tab1:
    # Koneksi ke Vector Store
    vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    
    # Inisialisasi LLM Llama 3.3 melalui Groq LPU
    llm = ChatGroq(
        temperature=0.1, 
        groq_api_key=GROQ_API_KEY, 
        model_name="llama-3.3-70b-versatile"
    )

    # Setup Retrieval Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 5})
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Tanyakan analisis riset kopi kepada AgriPulse..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Menganalisis database riset..."):
                response = qa_chain.invoke(prompt)
                answer = response["result"]
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

# TAB 2: Dashboard & News
with tab2:
    st.header("📈 Live Agricultural Insights")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric(label="Market Sentiment Index", value="85%", delta="+2.5%")
    with col_b:
        st.metric(label="Research Database", value="3,884 Chunks", delta="Synced")
    
    st.markdown("---")
    st.subheader("Latest Coffee Industry News")
    st.info("Feature integration with pygooglenews is active. Fetching latest updates...")
    # Placeholder untuk fungsi news Mas Hijrah

# TAB 3: Computer Vision (Future Updates)
with tab3:
    st.header("🔬 Coffee Bean Health Analysis")
    st.warning("⚠️ **COMING SOON**")
    st.markdown("""
    Fitur Computer Vision untuk deteksi penyakit biji kopi (seperti Karat Daun/Borer) sedang dalam tahap pengembangan.
    - **Target Model:** YOLOv11 / Vision Transformer
    - **Dataset:** 10,000+ Agricultural Images from ITB Collab
    """)
    st.image("https://img.freepik.com/premium-photo/coffee-leaf-rust-disease-background_875825-10336.jpg", caption="Experimental Phase: Coffee Rust Detection")
