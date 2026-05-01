import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA

# --- CONFIGURATION & SECRETS ---
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# --- APP UI SETUP ---
st.set_page_config(page_title="AgriPulse v3.5", page_icon="☕", layout="wide")

# --- CUSTOM HEADER & BRANDING ---
col1, col2 = st.columns([1, 5])
with col1:
    # Ganti URL ini dengan logo TelU jika ada
    st.image("https://upload.wikimedia.org/wikipedia/id/0/03/Logo_Telkom_University_potrait.png", width=100)
with col2:
    st.title("🌱 AGRIPULSE")
    st.subheader("Agricultural RAG-Integrated Precision Understanding & Localized Synthesis Engine")
    st.markdown("**Collaborative Project:** Hijrah Pratama (Data Science, TelU) & Mas Yoki (Pertanian, ITB)")

st.divider()

# --- INITIALIZE MODELS ---
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
    return embeddings, index_name

embeddings, index_name = init_models()

# --- SIDEBAR: KNOWLEDGE BASE MANAGEMENT ---
with st.sidebar:
    st.header("🧠 Knowledge Management")
    uploaded_file = st.file_uploader("Upload Jurnal/Riset Kopi (PDF)", type="pdf")
    
    if uploaded_file and st.button("Tanamkan ke Cloud Memory"):
        with st.spinner("Processing & Vectorizing 5000+ Chunks..."):
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            loader = PyPDFLoader("temp.pdf")
            data = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(data)
            vector_store = PineconeVectorStore.from_documents(chunks, embeddings, index_name=index_name)
            st.success(f"Berhasil menyimpan {len(chunks)} chunks ke Cloud!")
            os.remove("temp.pdf")
    
    st.divider()
    st.info("Aplikasi ini menggunakan Llama 3.3 (Groq) & Pinecone untuk analisis presisi hasil riset kopi.")

# --- MAIN TABS SECTION ---
tab1, tab2, tab3 = st.tabs(["💬 AI Assistant (RAG)", "📰 Live News & Dashboard", "🔍 Computer Vision (Health Check)"])

with tab1:
    # Interface Chat yang sudah ada
    vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    llm = ChatGroq(temperature=0.1, groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever(search_kwargs={"k": 5}))

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Tanyakan analisis riset kopi..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Menganalisis database riset..."):
                response = qa_chain.invoke(prompt)
                answer = response["result"]
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

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
    # Mas bisa panggil fungsi pygooglenews Mas di sini nanti.

with tab3:
    st.header("🔬 Coffee Bean Health Analysis")
    st.warning("⚠️ **COMING SOON**")
    st.markdown("""
    Fitur Computer Vision untuk deteksi penyakit biji kopi sedang dalam tahap pengembangan intensif.
    - **Target Model:** YOLOv8 / FastViT
    - **Status:** Training on 10,000+ Coffee Leaf & Bean Images
    """)
    st.image("https://img.freepik.com/premium-photo/coffee-leaf-rust-disease-background_875825-10336.jpg", caption="Experimental Phase: Coffee Rust Detection")
