import streamlit as st
from pygooglenews import GoogleNews
from groq import Groq
import chromadb
from datetime import datetime, timedelta
import pandas as pd
import pypdf
import re
import os

# --- 0. ENV SETUP ---
# Menjamin folder database dibuat di server Streamlit, bukan di GitHub
DB_PATH = "./agripulse_db"
if not os.path.exists(DB_PATH):
    os.makedirs(DB_PATH)

# --- 1. CONFIG & CLIENT ---
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    ADMIN_KEY = st.secrets.get("ADMIN_KEY", "hijrahxyokixteluxitbangkatan2021") 
except KeyError:
    st.error("Konfigurasi Secrets (GROQ_API_KEY) tidak ditemukan di dashboard Streamlit.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# Inisialisasi Database Vektor
try:
    db_client = chromadb.PersistentClient(path=DB_PATH)
    collection = db_client.get_or_create_collection(name="coffee_knowledge_base")
except Exception as e:
    st.error(f"Gagal memuat Database: {e}")
    st.stop()

st.set_page_config(page_title="AgriPulse AI v3.0", layout="wide", page_icon="☕")

# --- 2. CORE FUNCTIONS ---

def get_coffee_news():
    try:
        gn = GoogleNews(lang='id', country='ID')
        search = gn.search('harga kopi terbaru indonesia OR sentimen pasar kopi')
        news_data = [f"- {entry.title}" for entry in search['entries'][:5]]
        return "\n".join(news_data) if news_data else "Berita tidak tersedia."
    except:
        return "Gagal mengambil data berita."

def extract_price_from_news(news_text):
    # Menggunakan model ringan agar hemat token
    prompt = f"Berikan 4 angka harga kopi (Rp/kg) dari teks ini: {news_text}. Range 38000-48000. HANYA ANGKA pisahkan koma."
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.1
        )
        prices = [int(s) for s in re.findall(r'\d+', response.choices[0].message.content)]
        return prices[:4] if len(prices) >= 4 else [39000, 40500, 42000, 41500]
    except:
        return [38500, 40000, 41500, 41000]

def process_pdf_to_db(uploaded_file):
    reader = pypdf.PdfReader(uploaded_file)
    text = "".join([p.extract_text() for p in reader.pages if p.extract_text()])
    # Ukuran chunk 800 karakter adalah titik tengah terbaik untuk akurasi & efisiensi token
    chunks = [text[i:i+800] for i in range(0, len(text), 700)]
    waktu_wib = datetime.now() + timedelta(hours=7)
    timestamp = waktu_wib.strftime("%Y%m%d_%H%M%S")
    ids = [f"id_{timestamp}_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids)
    return len(chunks)

def get_hybrid_analysis(user_query, news_context, journal_context):
    prompt = f"""ANDA ADALAH: Penasihat Strategis Kopi Profesional.
    Tugas: Berikan solusi cerdas dengan memadukan Riset Jurnal dan Berita Pasar.
    
    RISET JURNAL: {journal_context}
    BERITA PASAR: {news_context}
    PERTANYAAN PETANI: {user_query}
    
    INSTRUKSI:
    1. Jika jurnal menyebutkan metode teknis (seperti AHP atau efisiensi), gunakan itu sebagai dasar jawaban.
    2. Hubungkan dengan tren harga di berita.
    3. Jawab dengan nada profesional namun mudah dimengerti."""
    
    try:
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile", 
            temperature=0.4,
            max_tokens=1000
        )
        return completion.choices[0].message.content
    except Exception as e:
        if "rate_limit" in str(e).lower():
            return "⚠️ **Limit Trafik API Terdeteksi.**\n\nAnalisis riset mendalam membutuhkan daya proses besar. Harap **TUNGGU 60 DETIK** tanpa menekan tombol apapun agar sistem dapat me-reset kuota Anda."
        return f"Terjadi gangguan teknis: {e}"

# --- 3. UI STYLE ---
st.markdown("""
    <style>
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    .stInfo { background-color: #f0f4f8; border-left: 5px solid #6f4e37; color: #1e1e1e; }
    .univ-label { font-size: 13px; color: #333; line-height: 1.2; margin-bottom: 0px; }
    </style>
    """, unsafe_allow_html=True)

# --- 4. HEADER & LOGO ---
st.title("☕ AgriPulse AI v3.0")
st.markdown("##### *Hybrid Decision Support System for Coffee Farmers*")

col_univ1, col_univ2 = st.columns(2)
with col_univ1:
    c_logo1, c_text1 = st.columns([1, 4])
    with c_logo1:
        try: st.image("telulogo.webp", width=60)
        except: st.caption("TelU")
    with c_text1:
        st.markdown('<p class="univ-label"><b>Hijrah Wira Pratama, S.Si.D.</b><br>AI Engineer (TelU)</p>', unsafe_allow_html=True)

with col_univ2:
    c_logo2, c_text2 = st.columns([1, 4])
    with c_logo2:
        try: st.image("itblogo.png", width=55)
        except: st.caption("ITB")
    with c_text2:
        st.markdown('<p class="univ-label"><b>Yokie Lidiantoro, S.T.</b><br>Agricultural Researcher (ITB)</p>', unsafe_allow_html=True)

st.divider()

# --- 5. SIDEBAR ---
with st.sidebar:
    st.header("🗂️ Knowledge Warehouse")
    try:
        count = collection.count()
    except:
        count = 0
    st.info(f"🧠 Kapasitas Memori: {count} Chunks")
    
    uploaded_file = st.file_uploader("Upload Jurnal PDF (Riset)", type="pdf")
    if st.button("Tanamkan ke Memori AI"):
        if uploaded_file:
            with st.spinner("Mengekstrak data riset..."):
                num = process_pdf_to_db(uploaded_file)
                st.success(f"Berhasil menambahkan {num} poin data!"); st.rerun()

    st.divider()
    st.subheader("🔐 Admin Panel")
    admin_input = st.text_input("Password", type="password")
    if admin_input == ADMIN_KEY:
        if st.button("🗑️ Reset Database"):
            db_client.delete_collection("coffee_knowledge_base")
            st.rerun()

# --- 6. MAIN CONTENT (TABS) ---
tab1, tab2, tab3 = st.tabs(["💡 Konsultasi Strategi", "📊 Market Dashboard", "📸 Diagnosa Penyakit"])

with tab1:
    st.subheader("Konsultasi Strategi Hybrid")
    user_q = st.text_area("Ajukan pertanyaan strategis Anda:", height=150, placeholder="Contoh: Bagaimana strategi AHP memitigasi risiko gagal panen saat harga kopi turun?")

    if st.button("Mulai Analisis AI"):
        if user_q:
            with st.spinner("Menghubungkan riset jurnal & sentimen pasar..."):
                news_ctx = get_coffee_news()
                # Ambil 3 chunk paling relevan (Anti-Limit Token)
                try:
                    search_results = collection.query(query_texts=[user_q], n_results=3)
                    j_ctx = "\n\n".join(search_results['documents'][0]) if search_results['documents'] else "Database kosong."
                except:
                    j_ctx = "Database tidak terjangkau."
                
                answer = get_hybrid_analysis(user_q, news_ctx, j_ctx)
                st.markdown("---")
                st.info(answer)
        else:
            st.warning("Silakan ketik pertanyaan Anda.")

with tab2:
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.subheader("📰 Sentimen Pasar")
        news_text = get_coffee_news()
        st.write(news_text)
    with col_b:
        st.subheader("📈 Proyeksi Harga")
        prices = extract_price_from_news(news_text)
        st.line_chart(pd.DataFrame({"Harga (Rp/Kg)": prices}))

with tab3:
    st.subheader("Diagnosa Penyakit")
    st.warning("Fitur Computer Vision dalam tahap pengembangan (V4.0).")

st.divider()
st.caption("AgriPulse v3.0 | Stable Edition | TelU x ITB")
