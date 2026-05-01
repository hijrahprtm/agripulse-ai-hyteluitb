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
# Memastikan API Key aman di Streamlit Secrets
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# --- APP UI SETUP ---
st.set_page_config(page_title="AgriPulse v3.4", page_icon="☕", layout="wide")
st.title("🌱 AGRIPULSE")
st.caption("Agricultural RAG-Integrated Precision Understanding & Localized Synthesis Engine")

# --- INITIALIZE MODELS ---
@st.cache_resource
def init_models():
    # Menggunakan model embedding yang ringan namun akurat
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Inisialisasi Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "agripulse-index"
    
    # Cek jika index belum ada, buat baru (Serverless - Zero Cost)
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384, # Dimensi untuk all-MiniLM-L6-v2
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
            # Simpan file sementara untuk diproses
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            loader = PyPDFLoader("temp.pdf")
            data = loader.load()
            
            # Split data menjadi chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(data)
            
            # Upload ke Pinecone (Permanen)
            vector_store = PineconeVectorStore.from_documents(
                chunks, 
                embeddings, 
                index_name=index_name
            )
            st.success(f"Berhasil menyimpan {len(chunks)} chunks ke Cloud!")
            os.remove("temp.pdf")

# --- MAIN CHAT INTERFACE ---
# Inisialisasi Vector Store untuk Retrieval
vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)

# Inisialisasi LLM (Llama 3.3 70B via Groq)
llm = ChatGroq(
    temperature=0.1, 
    groq_api_key=GROQ_API_KEY, 
    model_name="llama-3.3-70b-versatile"
)

# Setup RAG Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 5})
)

# Chat UI
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
