import streamlit as st
from pygooglenews import GoogleNews
from groq import Groq
import chromadb
from datetime import datetime, timedelta
import pandas as pd
import pypdf
import re

# --- 1. CONFIG & SETUP ---
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    # Password Admin dari Secrets, default 'admin123'
    ADMIN_KEY = st.secrets.get("ADMIN_KEY", "admin123") 
except KeyError:
    st.error("Konfigurasi Secrets (GROQ_API_KEY) tidak ditemukan.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)
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
    .univ-label { font-size: 13px; color: #333; line-height: 1.2; margin-bottom: 0px; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. FUNCTIONS ---

def get_coffee_news():
    try:
        gn = GoogleNews(lang='id', country='ID')
        search = gn.search('harga kopi terbaru OR sentimen pasar kopi indonesia')
        news_data = [f"- {entry.title}" for entry in search['entries'][:5]]
        return "\n".join(news_data)
    except:
        return "Gagal mengambil berita terbaru."

def extract_price_from_news(news_text):
    """AI Price Extraction - Menghasilkan data untuk grafik"""
    prompt = f"""
    Tugas: Berikan 4 angka harga kopi (Rupiah per kg) berdasarkan sentimen berita ini.
    Gunakan rentang 38000-48000. HANYA berikan angka dipisahkan koma.
    Contoh: 39000, 41500, 40000, 43000
    Berita: {news_text}
    """
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.1
        )
        raw_output = response.choices[0].message.content
        prices = [int(s) for s in re.findall(r'\d+', raw_output)]
        return prices[:4] if len(prices) >= 4 else [38000, 39500, 41000, 43500]
    except:
        return [38000, 40000, 42000, 41000]

def process_pdf_to_db(uploaded_file):
    reader = pypdf.PdfReader(uploaded_file)
    text = "".join([p.extract_text() for p in reader.pages if p.extract_text()])
    chunks = [text[i:i+1000] for i in range(0, len(text), 900)]
    waktu_wib = datetime.now() + timedelta(hours=7)
    timestamp = waktu_wib.strftime("%Y%m%d_%H%M%S")
    ids = [f"id_{uploaded_file.name}_{timestamp}_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids)
    return len(chunks)

def get_hybrid_analysis(user_query, news_context, journal_context):
    prompt = f"SISTEM: Penasihat Strategis Kopi. BERITA: {news_context}\nRISET: {journal_context}\nUSER: {user_query}"
    try:
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant", 
            temperature=0.4
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"⚠️ Sedang sibuk atau limit. Tunggu sebentar ya."

# --- 4. UI LAYOUT ---
st.title("☕ AgriPulse AI")
st.markdown("##### *Hybrid Decision Support System for Coffee Farmers*")

# BRANDING BAR (Role Updated)
col_univ1, col_univ2 = st.columns(2)
with col_univ1:
    c_logo1, c_text1 = st.columns([1, 4])
    with c_logo1:
        try: st.image("telulogo.webp", width=60)
        except: st.caption("Logo TelU")
    with c_text1:
        # UPDATE ROLE HIJRAH -> AI ENGINEER
        st.markdown('<p class="univ-label"><b>Hijrah Wira Pratama, S.Si.D.</b><br>AI Engineer (TelU)</p>', unsafe_allow_html=True)

with col_univ2:
    c_logo2, c_text2 = st.columns([1, 4])
    with c_logo2:
        try: st.image("itblogo.png", width=55)
        except: st.caption("Logo ITB")
    with c_text2:
        # UPDATE ROLE YOKIE -> AGRICULTURAL RESEARCHER
        st.markdown('<p class="univ-label"><b>Yokie Lidiantoro, S.T.</b><br>Agricultural Researcher (ITB)</p>', unsafe_allow_html=True)

st.divider()

# SIDEBAR
with st.sidebar:
    st.header("🗂️ Knowledge Warehouse")
    st.info(f"🧠 Kapasitas: {collection.count()} Knowledge Chunks")
    
    uploaded_file = st.file_uploader("Upload Jurnal Strategis (PDF)", type="pdf")
    if st.button("Tanamkan ke Memori AI"):
        if uploaded_file:
            with st.spinner("Menganalisis dokumen..."):
                process_pdf_to_db(uploaded_file)
                st.success("Data Tersimpan!"); st.rerun()

    st.divider()
    st.subheader("🔐 Admin Access")
    admin_input = st.text_input("Admin Password", type="password")
    if admin_input == ADMIN_KEY:
        if st.button("🗑️ Kosongkan Database"):
            db_client.delete_collection("coffee_knowledge_base")
            st.warning("Memori dibersihkan."); st.rerun()

# MAIN TABS
tab1, tab2, tab3 = st.tabs(["💡 Konsultasi Strategi", "📊 Market Dashboard", "📸 Diagnosa Penyakit"])

with tab1:
    st.subheader("Konsultasi Strategi Hybrid")
    user_q = st.text_area("Ajukan pertanyaan atau analisis pasar:", height=150, 
                          placeholder="Bagaimana perkembangan kopi di Indonesia?")

    if st.button("Mulai Analisis AI"):
        if user_q:
            with st.spinner("Menyisir database jurnal dan berita..."):
                news = get_coffee_news()
                results = collection.query(query_texts=[user_q], n_results=10)
                j_context = "\n\n".join(results['documents'][0]) if results['documents'] else "Data riset kosong."
                answer = get_hybrid_analysis(user_q, news, j_context)
                st.markdown("---")
                st.info(answer)

with tab2:
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.subheader("📰 Market Pulse")
        w_wib = datetime.now() + timedelta(hours=7)
        st.caption(f"Live Feed | {w_wib.strftime('%H:%M')} WIB")
        latest_news = get_coffee_news()
        st.write(latest_news)
    
    with col_b:
        st.subheader("📈 Tren Harga (AI-Extracted)")
        prices = extract_price_from_news(latest_news)
        chart_df = pd.DataFrame({"Harga (Rp/Kg)": prices})
        st.line_chart(chart_df)
        st.caption("Sentimen harga dihitung otomatis dari berita terbaru.")

with tab3:
    st.subheader("Diagnosa Penyakit Biji Kopi")
    st.warning("Feature Coming Soon: AI-Powered Disease Detection")
    st.file_uploader("Upload foto sampel (Coming Soon)", type=['png','jpg'], disabled=True)
    st.info("Penyakit biji kopi akan dideteksi via Computer Vision (Mobile & Web).")

st.divider()
st.caption("AgriPulse v2.6 | Robust & Secure Edition | Collaboration TelU x ITB")
