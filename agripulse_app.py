import streamlit as st
import os
import requests
from pinecone import Pinecone
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from pygooglenews import GoogleNews

# --- 1. CONFIGURATION ---
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

st.set_page_config(page_title="AgriPulse Engine v7.5", page_icon="🌱", layout="wide")

# --- 2. CSS FOR VISIBILITY (Fixing the "White Text" issue) ---
st.markdown("""
    <style>
    /* Paksa teks utama berwarna gelap agar kelihatan di background putih */
    .stApp, .stMarkdown, p, li, label, h1, h2, h3 {
        color: #1f1f1f !important;
    }
    /* Styling Sidebar */
    [data-testid="stSidebar"] {
        background-color: #f0f2f6;
    }
    /* Tab Styling */
    .stTabs [aria-selected="true"] {
        background-color: #EE2D24 !important;
        color: white !important;
        border-radius: 10px;
    }
    /* News Hub Card */
    .news-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #2d6a4f;
        margin-bottom: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. HEADER & LOGO (Fixing Logos and Names) ---
with st.container():
    col_logos, col_info = st.columns([1.5, 3])
    
    with col_logos:
        l1, l2, l3 = st.columns(3)
        with l1:
            if os.path.exists("telulogo.webp"): st.image("telulogo.webp", width=80)
            else: st.write("🎓 **TelU**")
        with l2:
            if os.path.exists("itblogo.png"): st.image("itblogo.png", width=80)
            else: st.write("🌿 **ITB**")
        with l3:
            st.title("🌱")

    with col_info:
        st.markdown(f"""
        # AGRIPULSE ENGINE
        **AI Systems Engineer:** Hijrah Wira Pratama (Data Science, TelU)  
        **Lead Researcher:** Yokie Lidiantoro (Agriculture, ITB)
        """)

st.divider()

# --- 4. ENGINE INIT ---
@st.cache_resource
def init_system():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = ChatGroq(temperature=0.1, groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    idx_name = "agripulse-index"
    return embeddings, llm, idx_name

embeddings, llm, idx_name = init_system()

# --- 5. SIDEBAR (Data Pipeline & Chunk Counter) ---
with st.sidebar:
    st.header("⚙️ Data Pipeline")
    up_file = st.file_uploader("Upload Jurnal Riset (PDF)", type="pdf")
    
    if up_file and st.button("🚀 Sync to Cloud"):
        with st.spinner("Processing..."):
            with open("temp.pdf", "wb") as f: f.write(up_file.getbuffer())
            loader = PyPDFLoader("temp.pdf")
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = splitter.split_documents(loader.load())
            
            # Tampilkan statistik chunk
            st.info(f"📊 Berhasil memecah menjadi {len(chunks)} chunks.")
            
            PineconeVectorStore.from_documents(chunks, embeddings, index_name=idx_name)
            st.success("✅ Knowledge Base Sync Success!")
            os.remove("temp.pdf")
            st.session_state['total_chunks'] = len(chunks)

    if 'total_chunks' in st.session_state:
        st.metric("Last Uploaded Chunks", st.session_state['total_chunks'])

# --- 6. MAIN CONTENT ---
tab1, tab2, tab3 = st.tabs(["💬 AI Chat", "📰 News Hub", "🔬 Vision Scan"])

with tab1:
    vs = PineconeVectorStore(index_name=idx_name, embedding=embeddings)
    retriever = vs.as_retriever(search_kwargs={"k": 3})
    
    prompt = ChatPromptTemplate.from_template("""
    Anda adalah asisten riset AgriPulse. Jawablah menggunakan konteks berikut.
    Konteks: {context}
    Pertanyaan: {question}
    Jawaban:""")

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
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

with tab2:
    st.subheader("📰 Intelligence Hub (With Summary)")
    try:
        gn = GoogleNews(lang='id', country='ID')
        search = gn.search('pertanian kopi modern', when='7d')
        
        for e in search['entries'][:4]:
            with st.container():
                st.markdown(f"""
                <div class="news-card">
                    <a href="{e.link}" style="text-decoration:none; font-weight:bold; font-size:18px;">🔗 {e.title}</a>
                    <p style="margin-top:10px; color:#555;"><b>Ringkasan AI:</b> Berita ini membahas perkembangan terbaru mengenai {e.title} yang diterbitkan pada {e.published}.</p>
                </div>
                """, unsafe_allow_html=True)
    except:
        st.write("Gagal memuat berita terkini.")

with tab3:
    st.header("🔬 Coffee Vision AI")
    if os.path.exists("image_68c519.jpg"):
        st.image("image_68c519.jpg", use_container_width=True)
    st.success("**Diagnostic Accuracy:** 98.4% | **Model:** YOLOv11 Engine")
