import streamlit as st
from pygooglenews import GoogleNews
from groq import Groq
import chromadb
from datetime import datetime
import pytz
import pandas as pd
import pypdf
import re
import os

# --- 0. ENV & TIME SETUP ---
DB_PATH = "./agripulse_db"
if not os.path.exists(DB_PATH): os.makedirs(DB_PATH)

wib = pytz.timezone('Asia/Jakarta')
now_wib = datetime.now(wib)
jam = now_wib.hour

if jam < 11: salam = "Selamat Pagi"
elif jam < 15: salam = "Selamat Siang"
elif jam < 19: salam = "Selamat Sore"
else: salam = "Selamat Malam"

# --- 1. CONFIG & CLIENT ---
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    ADMIN_KEY = st.secrets.get("ADMIN_KEY", "hijrahxyokixteluxitbangkatan2021") 
except KeyError:
    st.error("Konfigurasi Secrets tidak ditemukan.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)
db_client = chromadb.PersistentClient(path=DB_PATH)
collection = db_client.get_or_create_collection(name="coffee_knowledge_base")

st.set_page_config(page_title="AgriPulse AI v3.3", layout="wide", page_icon="☕")

# --- 2. CORE LOGIC ---

def get_coffee_news():
    try:
        gn = GoogleNews(lang='id', country='ID')
        search = gn.search('harga kopi terbaru indonesia')
        return "\n".join([f"- {e.title}" for e in search['entries'][:5]])
    except: return "Gagal memuat berita."

def extract_price_from_news(news_text):
    prompt = f"Ekstrak 4 angka harga kopi (Rp/kg) dari: {news_text}. HANYA ANGKA pisahkan koma."
    try:
        res = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.1
        )
        prices = [int(s) for s in re.findall(r'\d+', res.choices[0].message.content)]
        return prices[:4] if len(prices) >= 4 else [39000, 41000, 42500, 41500]
    except: return [40000, 41000, 42000, 41000]

def get_hybrid_analysis(user_query, news_context, journal_context):
    prompt = f"""
    SISTEM: Pakar Kopi Spesialis (TelU x ITB). Waktu: {now_wib.strftime('%H:%M')} WIB.
    INSTRUKSI: Jawab TEKNIS & FOKUS. Jika tanya penyakit, JANGAN bicara harga/lahan.
    GEJALA PBKo: Fokus ke lubang buah dan biji hancur.
    PENGENDALIAN: Fokus ke sanitasi, perangkap, dan agens hayati.
    
    JURNAL: {journal_context}
    BERITA: {news_context}
    USER: {user_query}
    """
    try:
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-specdec", 
            temperature=0.1,
            max_tokens=1000
        )
        return completion.choices[0].message.content
    except Exception as e: return f"Error: {e}"

# --- 3. UI STYLE ---
st.markdown("""<style>
    .stInfo { border-left: 5px solid #6f4e37; }
    .univ-label { font-size: 13px; font-weight: bold; }
</style>""", unsafe_allow_html=True)

# --- 4. HEADER (LOGO KAMPUS KEMBALI) ---
st.write(f"🕒 {now_wib.strftime('%H:%M')} WIB")
st.title(f"☕ {salam}, AgriPulse v3.3")

col_univ1, col_univ2 = st.columns(2)
with col_univ1:
    c1, c2 = st.columns([1, 4])
    with c1: 
        if os.path.exists("telulogo.webp"): st.image("telulogo.webp", width=60)
        else: st.write("🏫")
    with c2: st.markdown('<p class="univ-label">Hijrah Wira Pratama, S.Si.D.<br>AI Engineer (TelU)</p>', unsafe_allow_html=True)

with col_univ2:
    c1, c2 = st.columns([1, 4])
    with c1: 
        if os.path.exists("itblogo.png"): st.image("itblogo.png", width=55)
        else: st.write("🏫")
    with c2: st.markdown('<p class="univ-label">Yokie Lidiantoro, S.T.<br>Agricultural Researcher (ITB)</p>', unsafe_allow_html=True)

st.divider()

# --- 5. SIDEBAR (LOGIN ADMIN KEMBALI) ---
with st.sidebar:
    st.header("🗂️ Knowledge Base")
    st.info(f"🧠 Memori: {collection.count()} Chunks")
    
    up_file = st.file_uploader("Update Jurnal (PDF)", type="pdf")
    if st.button("Update Memori"):
        if up_file:
            reader = pypdf.PdfReader(up_file)
            text = "".join([p.extract_text() for p in reader.pages if p.extract_text()])
            chunks = [text[i:i+800] for i in range(0, len(text), 700)]
            ids = [f"id_{now_wib.timestamp()}_{i}" for i in range(len(chunks))]
            collection.add(documents=chunks, ids=ids)
            st.success("Database Updated!"); st.rerun()

    st.divider()
    st.subheader("🔐 Admin Access")
    pwd = st.text_input("Password", type="password")
    if pwd == ADMIN_KEY:
        if st.button("🗑️ Hapus Database Jurnal"):
            db_client.delete_collection("coffee_knowledge_base")
            st.warning("Database dibersihkan!"); st.rerun()

# --- 6. MAIN CONTENT (TABS & DASHBOARD KEMBALI) ---
t1, t2, t3 = st.tabs(["💡 Analisis Spesialis", "📊 Dashboard Market", "📸 Computer Vision"])

with t1:
    q = st.text_area("Konsultasi Teknis:", placeholder="Contoh: Gejala serangan PBKo...")
    if st.button("Jalankan Analisis"):
        if q:
            with st.spinner("Menganalisis..."):
                res = collection.query(query_texts=[q], n_results=3)
                j_ctx = "\n\n".join(res['documents'][0]) if res['documents'] else ""
                news_ctx = get_coffee_news() if any(x in q.lower() for x in ['harga', 'pasar']) else ""
                answer = get_hybrid_analysis(q, news_ctx, j_ctx)
                st.markdown("---")
                st.info(answer)

with t2:
    st.subheader("Market Trend Analysis")
    col_news, col_chart = st.columns(2)
    n_data = get_coffee_news()
    with col_news:
        st.info(n_data)
    with col_chart:
        p_data = extract_price_from_news(n_data)
        st.line_chart(pd.DataFrame({"Harga (Rp/Kg)": p_data}))

with t3:
    st.subheader("📸 Diagnosa Penyakit (Visual)")
    st.warning("Feature Coming Soon: Integrasi Model YOLO untuk deteksi karat daun dan PBKo.")

st.divider()
st.caption("AgriPulse v3.3 | TelU x ITB | Model: Llama-3.3-70b-specdec")
