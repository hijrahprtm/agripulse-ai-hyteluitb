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
DB_PATH = "./agripulse_db"
if not os.path.exists(DB_PATH):
    os.makedirs(DB_PATH)

# --- 1. CONFIG & CLIENT ---
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    ADMIN_KEY = st.secrets.get("ADMIN_KEY", "hijrahxyokixteluxitbangkatan2021") 
except KeyError:
    st.error("Konfigurasi Secrets tidak ditemukan.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

try:
    db_client = chromadb.PersistentClient(path=DB_PATH)
    collection = db_client.get_or_create_collection(name="coffee_knowledge_base")
except Exception as e:
    st.error(f"Database Error: {e}")
    st.stop()

st.set_page_config(page_title="AgriPulse AI", layout="wide", page_icon="☕")

# --- 2. LOGIC FUNCTIONS ---

def get_coffee_news():
    try:
        gn = GoogleNews(lang='id', country='ID')
        search = gn.search('harga kopi terbaru indonesia')
        return "\n".join([f"- {e.title}" for e in search['entries'][:5]])
    except:
        return "Gagal memuat berita."

def extract_price_from_news(news_text):
    # Menggunakan model kecil agar tidak makan jatah token model utama
    prompt = f"Ekstrak 4 angka harga kopi (Rp/kg) dari: {news_text}. HANYA ANGKA pisahkan koma."
    try:
        res = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.1
        )
        prices = [int(s) for s in re.findall(r'\d+', res.choices[0].message.content)]
        return prices[:4] if len(prices) >= 4 else [39000, 41000, 42500, 41500]
    except:
        return [40000, 41000, 42000, 41000]

def process_pdf_to_db(uploaded_file):
    reader = pypdf.PdfReader(uploaded_file)
    text = "".join([p.extract_text() for p in reader.pages if p.extract_text()])
    # Chunking lebih cerdas (per 800 karakter agar token irit)
    chunks = [text[i:i+800] for i in range(0, len(text), 700)]
    waktu_wib = datetime.now() + timedelta(hours=7)
    timestamp = waktu_wib.strftime("%Y%m%d_%H%M")
    ids = [f"id_{timestamp}_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids)
    return len(chunks)

def get_hybrid_analysis(user_query, news_context, journal_context):
    prompt = f"""ANDA: Pakar Strategi Kopi (Hybrid Expert).
    Tugas: Jawab pertanyaan user dengan memadukan data berita pasar dan riset internal.
    
    KONTEKS PASAR (BERITA):
    {news_context}
    
    KONTEKS RISET (JURNAL):
    {journal_context}
    
    PERTANYAAN: {user_query}
    
    INSTRUKSI: Jika data jurnal relevan, sebutkan poin teknisnya (seperti AHP atau efisiensi). Jika data berita relevan, hubungkan dengan harga pasar saat ini."""
    
    try:
        # MENGGUNAKAN MODEL TERBAIK (Llama 3.3 70B) untuk kualitas jawaban maksimal
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile", 
            temperature=0.3,
            max_tokens=1000
        )
        return completion.choices[0].message.content
    except Exception as e:
        if "rate_limit" in str(e).lower():
            return "⚠️ **Sistem Sedang Padat (Rate Limit).**\n\nAkun gratis Groq membatasi jumlah kata yang diproses per menit. Karena kita menganalisis jurnal yang cukup panjang, harap **TUNGGU TEPAT 60 DETIK** sebelum menekan tombol lagi agar kuota token Anda terisi kembali."
        return f"Terjadi kesalahan teknis: {e}"

# --- 3. UI INTERFACE ---
st.title("☕ AgriPulse AI v3.0")
st.markdown("##### *Decision Support System: Market Pulse & Research Insight*")

# Logo & Header
c1, c2 = st.columns(2)
with c1:
    st.markdown("🧑‍🔬 **Researcher:** Hijrah Wira Pratama (TelU)")
with c2:
    st.markdown("🌍 **Partner:** Yokie Lidiantoro (ITB)")

st.divider()

# Sidebar
with st.sidebar:
    st.header("📦 Knowledge Base")
    try:
        count = collection.count()
    except:
        count = 0
    st.metric("Total Data Chunks", count)
    
    up_file = st.file_uploader("Tambah Jurnal (PDF)", type="pdf")
    if st.button("Inject to Memory"):
        if up_file:
            with st.spinner("Processing..."):
                n = process_pdf_to_db(up_file)
                st.success(f"Added {n} chunks!"); st.rerun()

    st.divider()
    adm = st.text_input("Admin Key", type="password")
    if adm == ADMIN_KEY:
        if st.button("Clear Memory"):
            db_client.delete_collection("coffee_knowledge_base")
            st.rerun()

# Tabs
t1, t2 = st.tabs(["💡 Strategic Analysis", "📈 Market Watch"])

with t1:
    q = st.text_area("Apa yang ingin Anda konsultasikan malam ini?", placeholder="Contoh: Bandingkan efisiensi biaya pupuk kimia vs organik...")
    if st.button("Run Hybrid Analysis"):
        if q:
            with st.spinner("Syncing data..."):
                news_data = get_coffee_news()
                # AMBIL HANYA 3 CHUNKS TERBAIK (Agar Irit Token/Anti-Limit)
                try:
                    res = collection.query(query_texts=[q], n_results=3)
                    j_context = "\n".join(res['documents'][0]) if res['documents'] else "No journal data."
                except:
                    j_context = "Database query failed."
                
                output = get_hybrid_analysis(q, news_data, j_context)
                st.markdown("---")
                st.markdown(output)
        else:
            st.warning("Input query dulu.")

with t2:
    col_news, col_chart = st.columns(2)
    with col_news:
        st.subheader("Latest Market News")
        n_text = get_coffee_news()
        st.info(n_text)
    with col_chart:
        st.subheader("Price Trends")
        p = extract_price_from_news(n_text)
        st.line_chart(p)

st.caption("AgriPulse v3.0 | Stable Edition | Powered by Llama 3.3 70B")
