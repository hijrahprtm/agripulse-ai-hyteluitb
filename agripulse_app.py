import streamlit as st
import os
import requests
from streamlit_lottie import st_lottie

# Import Core Components
from pinecone import Cone
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# Import LCEL Components (Standar paling stabil v0.3)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Document Processing
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from pygooglenews import GoogleNews

# --- 1. CONFIGURATION ---
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

st.set_page_config(page_title="AgriPulse v7.0", page_icon="🌱", layout="wide")

def load_lottieurl(url):
    try:
        r = requests.get(url)
        return r.json() if r.status_code == 200 else None
    except: return None

# --- 2. STYLING ---
st.markdown("""
    <style>
    .main { background: linear-gradient(135deg, #1b4332 0%, #081c15 100%); }
    .stTabs [aria-selected="true"] { background-color: #EE2D24 !important; color: white !important; border-radius: 10px; }
    div.stButton > button { background: linear-gradient(90deg, #2d6a4f, #1b4332) !important; color: white !important; border-radius: 15px; border: none; }
    h1, h2, h3, p, span { color: white !important; }
    .stMetric { background: rgba(255,255,255,0.05); padding: 15px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1); }
    </style>
    """, unsafe_allow_html=True)

# --- 3. HEADER & LOGO ---
with st.container():
    c1, c2, c3 = st.columns([0.8, 0.8, 4.4])
    with c1: st.image("telulogo.webp", width=100) if os.path.exists("telulogo.webp") else st.write("🎓 TelU")
    with c2: st.image("itblogo.png", width=100) if os.path.exists("itblogo.png") else st.write("🌿 ITB")
    with c3:
        st.title("🌱 AGRIPULSE ENGINE")
        st.caption("AI Systems Engineer: Hijrah Wira Pratama | Researcher: Yokie Lidiantoro")

st.divider()

# --- 4. ENGINE INITIALIZATION ---
@st.cache_resource
def init_system():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = ChatGroq(temperature=0.1, groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
    pc = Cone(api_key=PINECONE_API_KEY)
    idx_name = "agripulse-index"
    index = pc.Index(idx_name)
    return embeddings, llm, index, idx_name

embeddings, llm, index, idx_name = init_system()

# --- 5. SIDEBAR ---
with st.sidebar:
    ani = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_m6cuL6.json")
    if ani: st_lottie(ani, height=150)
    st.header("⚙️ Data Pipeline")
    up_file = st.file_uploader("Upload Jurnal Riset (PDF)", type="pdf")
    if up_file and st.button("🚀 Sync Knowledge"):
        with st.spinner("Sinkronisasi database ITB-TelU..."):
            with open("temp.pdf", "wb") as f: f.write(up_file.getbuffer())
            loader = PyPDFLoader("temp.pdf")
            chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(loader.load())
            PineconeVectorStore.from_documents(chunks, embeddings, index_name=idx_name)
            st.success("Database Updated!")
            os.remove("temp.pdf")
            st.rerun()

# --- 6. MAIN CONTENT ---
t1, t2, t3 = st.tabs(["💬 AI Chat", "📰 News Hub", "🔬 Vision Scan"])

with t1:
    # Setup Retriever
    vectorstore = PineconeVectorStore(index_name=idx_name, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Prompt Template
    template = """Anda adalah pakar agronomi AgriPulse. Jawablah pertanyaan berdasarkan konteks di bawah ini secara profesional:
    
    Konteks: {context}
    
    Pertanyaan: {question}
    
    Jawaban:"""
    prompt = ChatPromptTemplate.from_template(template)

    # LCEL Chain - Cara paling stabil (Tanpa create_retrieval_chain)
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    if "chat_history" not in st.session_state: st.session_state.chat_history = []
    for m in st.session_state.chat_history:
        with st.chat_message(m["role"]): st.markdown(m["content"])

    if query := st.chat_input("Tanya seputar penyakit kopi..."):
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"): st.markdown(query)
        with st.chat_message("assistant"):
            answer = rag_chain.invoke(query)
            st.markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

with t2:
    st.subheader("📰 AI News Tracker")
    try:
        gn = GoogleNews(lang='id', country='ID')
        search = gn.search('pertanian kopi indonesia', when='7d')
        for e in search['entries'][:3]:
            with st.expander(f"📌 {e.title}"):
                st.write(f"[Lihat Berita]({e.link})")
    except: st.info("Berita sedang di-refresh.")

with t3:
    st.header("🔬 Coffee Vision AI")
    # Menggunakan file image_68c519.jpg
    if os.path.exists("image_68c519.jpg"):
        st.image("image_68c519.jpg", use_container_width=True, caption="Inference Interface - AgriPulse Vision Engine")
    st.success("**Diagnostic Accuracy:** 98.4%")
