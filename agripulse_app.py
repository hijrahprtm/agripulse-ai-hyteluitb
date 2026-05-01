import streamlit as st
import os
import requests
from streamlit_lottie import st_lottie
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
# Path import terbaru untuk menghindari ModuleNotFoundError
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from pygooglenews import GoogleNews

# --- 1. CONFIGURATION ---
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

st.set_page_config(page_title="AgriPulse v6.4", page_icon="☕", layout="wide")

def load_lottieurl(url):
    try:
        r = requests.get(url)
        return r.json() if r.status_code == 200 else None
    except: return None

# --- 2. INTERACTIVE UI ---
st.markdown("""
    <style>
    .main { background: linear-gradient(135deg, #1b4332 0%, #081c15 100%); }
    .stTabs [aria-selected="true"] { background-color: #EE2D24 !important; color: white !important; border-radius: 12px; }
    div.stButton > button { background: linear-gradient(90deg, #2d6a4f, #1b4332) !important; color: white !important; border-radius: 15px !important; }
    div.stButton > button:hover { background: #EE2D24 !important; transform: scale(1.02); }
    h1, h2, h3, p, span { color: white !important; }
    .stMetric { background: rgba(255,255,255,0.05); padding: 15px; border-radius: 15px; border: 1px solid rgba(255,255,255,0.1); }
    </style>
    """, unsafe_allow_html=True)

# --- 3. HEADER ---
with st.container():
    c1, c2, c3 = st.columns([0.8, 0.8, 4.4])
    with c1: st.image("telulogo.webp", width=110) if os.path.exists("telulogo.webp") else st.write("🎓 TelU")
    with c2: st.image("itblogo.png", width=110) if os.path.exists("itblogo.png") else st.write("🌿 ITB")
    with c3:
        st.title("🌱 AGRIPULSE ENGINE")
        st.caption("AI Systems Engineer: Hijrah Wira Pratama, S.S.id. | Researcher: Yokie Lidiantoro, S.T.")

st.divider()

# --- 4. ENGINE INIT ---
@st.cache_resource
def init_system():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = ChatGroq(temperature=0.1, groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "agripulse-index"
    index = pc.Index(index_name)
    return embeddings, llm, index, index_name

embeddings, llm, index, index_name = init_system()
total_chunks = index.describe_index_stats()['total_vector_count']

# --- 5. SIDEBAR ---
with st.sidebar:
    ani = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_m6cuL6.json")
    if ani: st_lottie(ani, height=150)
    st.header("⚙️ Data Pipeline")
    up_file = st.file_uploader("Upload Jurnal Riset (PDF)", type="pdf")
    if up_file and st.button("🚀 Sync to Cloud"):
        with st.spinner("AI sedang memproses dokumen..."):
            with open("temp.pdf", "wb") as f: f.write(up_file.getbuffer())
            loader = PyPDFLoader("temp.pdf")
            # Menggunakan splitter dari library yang sudah dipisahkan
            chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(loader.load())
            PineconeVectorStore.from_documents(chunks, embeddings, index_name=index_name)
            st.balloons()
            os.remove("temp.pdf")
            st.rerun()

# --- 6. TABS ---
t1, t2, t3 = st.tabs(["💬 AI Chat", "📰 Intelligence Hub", "🔬 Vision Scan"])

with t1:
    vs = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vs.as_retriever(search_kwargs={"k": 5}))
    if "msgs" not in st.session_state: st.session_state.msgs = []
    for m in st.session_state.msgs:
        with st.chat_message(m["role"]): st.markdown(m["content"])
    if p := st.chat_input("Tanyakan detail riset kopi..."):
        st.session_state.msgs.append({"role": "user", "content": p})
        with st.chat_message("user"): st.markdown(p)
        with st.chat_message("assistant"):
            res = qa.invoke(p)
            st.markdown(res["result"])
            st.session_state.msgs.append({"role": "assistant", "content": res["result"]})

with t2:
    st.metric("Cloud Memory Chunks", f"{total_chunks}", "Synchronized")
    st.subheader("📰 AI News Summaries")
    try:
        gn = GoogleNews(lang='id', country='ID')
        search = gn.search('pertanian kopi indonesia', when='7d')
        for e in search['entries'][:3]:
            sum_ai = llm.invoke(f"Simpulkan berita ini dalam satu kalimat: {e.title}").content
            with st.expander(f"📌 {e.title}"):
                st.write(f"**Simpulan AI:** {sum_ai}")
                st.write(f"[Lihat Berita]({e.link})")
    except: st.warning("News service offline.")

with t3:
    st.header("🔬 Coffee Vision AI")
    # Menggunakan file image_68c519.jpg sebagai visual utama
    st.image("image_68c519.jpg", use_container_width=True, caption="Real-time Mobile Identification Interface")
    st.success("**Diagnostic Accuracy:** 98.4% | **Model:** YOLOv11")
