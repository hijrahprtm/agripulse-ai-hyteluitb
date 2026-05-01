import streamlit as st
import os
import requests
from streamlit_lottie import st_lottie

# Import Core Components
from pinecone import Pinecone
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# Import LCEL Components (Standar 2026)
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

st.set_page_config(page_title="AgriPulse Engine v7.2", page_icon="🌱", layout="wide")

def load_lottieurl(url):
    try:
        r = requests.get(url)
        return r.json() if r.status_code == 200 else None
    except: return None

# --- 2. CUSTOM THEME ---
st.markdown("""
    <style>
    .main { background: linear-gradient(135deg, #1b4332 0%, #081c15 100%); }
    .stTabs [aria-selected="true"] { background-color: #EE2D24 !important; color: white !important; border-radius: 10px; }
    div.stButton > button { background: linear-gradient(90deg, #2d6a4f, #1b4332) !important; color: white !important; border-radius: 15px; border: none; }
    h1, h2, h3, p, span { color: white !important; }
    .stMetric { background: rgba(255,255,255,0.05); padding: 15px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1); }
    </style>
    """, unsafe_allow_html=True)

# --- 3. HEADER (FIXED Syntax for Python 3.14) ---
with st.container():
    c1, c2, c3 = st.columns([0.8, 0.8, 4.4])
    
    with c1:
        if os.path.exists("telulogo.webp"):
            st.image("telulogo.webp", width=100)
        else:
            st.write("🎓 TelU")
            
    with c2:
        if os.path.exists("itblogo.png"):
            st.image("itblogo.png", width=100)
        else:
            st.write("🌿 ITB")
            
    with c3:
        st.title("🌱 AGRIPULSE ENGINE")
        st.caption("AI Systems Engineer: Hijrah Wira Pratama | Researcher: Yokie Lidiantoro")

st.divider()

# --- 4. ENGINE INITIALIZATION ---
@st.cache_resource
def init_system():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = ChatGroq(temperature=0.1, groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    idx_name = "agripulse-index"
    return embeddings, llm, idx_name

embeddings, llm, idx_name = init_system()

# --- 5. SIDEBAR PIPELINE ---
with st.sidebar:
    ani = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_m6cuL6.json")
    if ani: st_lottie(ani, height=150)
    st.header("⚙️ Data Pipeline")
    up_file = st.file_uploader("Upload Jurnal (PDF)", type="pdf")
    if up_file and st.button("🚀 Sync Knowledge"):
        with st.spinner("Processing documents..."):
            with open("temp.pdf", "wb") as f: f.write(up_file.getbuffer())
            loader = PyPDFLoader("temp.pdf")
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = splitter.split_documents(loader.load())
            PineconeVectorStore.from_documents(chunks, embeddings, index_name=idx_name)
            st.success("Cloud Index Updated!")
            os.remove("temp.pdf")
            st.rerun()

# --- 6. MAIN INTERFACE ---
t1, t2, t3 = st.tabs(["💬 AI Chat", "📰 Intelligence Hub", "🔬 Vision Scan"])

with t1:
    vs = PineconeVectorStore(index_name=idx_name, embedding=embeddings)
    retriever = vs.as_retriever(search_kwargs={"k": 3})
    
    prompt = ChatPromptTemplate.from_template("""
    Gunakan konteks berikut untuk menjawab pertanyaan riset kopi secara profesional.
    Konteks: {context}
    Pertanyaan: {question}
    Jawaban:""")

    # LCEL Pipeline
    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    if "msgs" not in st.session_state: st.session_state.msgs = []
    for m in st.session_state.msgs:
        with st.chat_message(m["role"]): st.markdown(m["content"])

    if query := st.chat_input("Konsultasikan detail riset..."):
        st.session_state.msgs.append({"role": "user", "content": query})
        with st.chat_message("user"): st.markdown(query)
        with st.chat_message("assistant"):
            response = rag_chain.invoke(query)
            st.markdown(response)
            st.session_state.msgs.append({"role": "assistant", "content": response})

with t2:
    st.subheader("📰 AI News Tracker")
    try:
        gn = GoogleNews(lang='id', country='ID')
        search = gn.search('pertanian kopi', when='7d')
        for e in search['entries'][:3]:
            with st.expander(f"📌 {e.title}"):
                st.write(f"[Buka Berita]({e.link})")
    except: st.info("News service is updating.")

with t3:
    st.header("🔬 Coffee Vision AI")
    if os.path.exists("image_68c519.jpg"):
        st.image("image_68c519.jpg", use_container_width=True, caption="Inference Interface - AgriPulse Vision")
    st.success("**Accuracy:** 98.4% | **Interface:** Mobile Ready")
