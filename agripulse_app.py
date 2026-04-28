import streamlit as st
from pygooglenews import GoogleNews
from groq import Groq
import chromadb
from datetime import datetime, timedelta
import pandas as pd
import pypdf

# --- 1. CONFIG & SETUP (SECURE) ---
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("API Key 'GROQ_API_KEY' tidak ditemukan di Secrets Streamlit.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# Setup Database Lokal (ChromaDB)
db_client = chromadb.PersistentClient(path="./agripulse_db")
collection = db_client.get_or_create_collection(name="coffee_knowledge_base")

st.set_page_config(page_title="AgriPulse AI", layout="wide", page_icon="☕")

# --- 2. CUSTOM CSS ---
st.markdown("""
    <style>
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; font-weight: bold; font-size: 16px; }
    .stInfo { background-color: #f0f4f8; border-left: 5px solid #6f4e37; color: #1e1e1e; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. FUNCTIONS ---

def get_coffee_news():
    try:
        gn = GoogleNews(lang='id', country='ID')
        search = gn.search('harga kopi terbaru OR ekspor kopi indonesia')
        news_data = [f"- {entry.title}" for entry in search['entries'][:5]]
        return "\n".join(news_data)
    except:
        return "Gagal sinkronisasi dengan server berita."

def process_pdf_to_db(uploaded_file):
    reader = pypdf.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content: text += content
    
    chunks = [text[i:i+1000] for i in range(0, len(text), 900)]
    
    waktu_wib = datetime.now() + timedelta(hours=7)
    timestamp = waktu_wib.strftime("%Y%m%d_%H%M%S")
    
    ids = [f"id_{uploaded_file.name}_{timestamp}_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids)
    return len(chunks)

def get_hybrid_analysis(user_query, news_context, journal_context):
    prompt = f"""
    SISTEM: Kamu adalah AgriPulse AI. Penasihat strategis petani kopi.
    TUGAS: Berikan analisis cerdas menggabungkan peluang pasar dan data teknis riset.
    
    BERITA PASAR TERKINI:
    {news_context}
    
    REFERENSI RISET (GABUNGAN JURNAL):
    {journal_context}
    
    PERTANYAAN USER: {user_query}
    """
    
    try:
        # Menggunakan model 8b agar limit lebih longgar dan proses lebih ringan/cepat
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant", 
            temperature=0.4
        )
        return completion.choices[0].message.content
    except Exception as e:
        # Logika Anti-Limit: Menangkap error rate limit
        if "rate_limit" in str(e).lower():
            return "⚠️ **Limit Tercapai:** Wah, AgriPulse sedang melayani banyak permintaan. Mohon tunggu 30-60 detik lalu coba tekan Enter kembali ya, Mas."
        else:
            return f"❌ **Terjadi Kesalahan:** {str(e)}"

# --- 4. UI LAYOUT ---
st.title("☕ AgriPulse AI")
st.markdown("##### *Hybrid Decision Support System for Coffee Farmers*")

col_dev1, col_dev2 = st.columns(2)
with col_dev1:
    st.caption("🚀 **Hijrah Wira Pratama S.Si.D.** (TelU)")
with col_dev2:
    st.caption("🍃 **Yokie Lidiantoro S.T.** (ITB)")

st.divider()

# SIDEBAR
with st.sidebar:
    st.header("🗂️ Knowledge Warehouse")
    try:
        current_count = collection.count()
        st.success(f"🧠 Kapasitas Otak: {current_count} Knowledge Chunks")
    except:
        current_count = 0
        
    uploaded_file = st.file_uploader("Upload Jurnal Baru (PDF)", type="pdf")
    
    if st.button("Tanamkan ke Otak AI"):
        if uploaded_file:
            with st.spinner("Menambah pengetahuan baru..."):
                num = process_pdf_to_db(uploaded_file)
                st.balloons()
                st.rerun()
        else:
            st.error("Silakan pilih file PDF.")
    
    st.divider()
    if st.button("Kosongkan Semua Memori"):
        db_client.delete_collection("coffee_knowledge_base")
        st.warning("Memori AI telah dibersihkan.")
        st.rerun()

# TABS
tab1, tab2 = st.tabs(["💡 Konsultasi Strategi", "📊 Market Dashboard"])

with tab1:
    st.subheader("Konsultasi Strategi Hybrid")
    user_q = st.text_input("Ajukan pertanyaan analisis strategis:", 
                           placeholder="Contoh: Bagaimana potensi S1 di Semarang?")

    if user_q:
        with st.spinner("Menyisir database dan berita..."):
            live_news = get_coffee_news()
            try:
                # Mengambil 10 chunk agar data tetap akurat meski pakai model lebih kecil
                results = collection.query(query_texts=[user_q], n_results=10)
                j_context = "\n\n".join(results['documents'][0]) if results['documents'] else "Kosong."
            except:
                j_context = "Belum ada jurnal."
            
            answer = get_hybrid_analysis(user_q, live_news, j_context)
            st.markdown("---")
            st.markdown("### 💡 Rekomendasi AgriPulse:")
            st.info(answer)

with tab2:
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.subheader("📰 Market Pulse")
        waktu_wib = datetime.now() + timedelta(hours=7)
        st.caption(f"Live Feed | {waktu_wib.strftime('%H:%M')} WIB")
        st.write(get_coffee_news())
    with col_b:
        st.subheader("📈 Tren Harga (Estimasi)")
        chart_data = pd.DataFrame({"Harga": [38000, 39500, 41000, 43500]})
        st.line_chart(chart_data)

st.divider()
st.caption("AgriPulse v2.3 | Robust Mode | TelU x ITB Collaboration")
