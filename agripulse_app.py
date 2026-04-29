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
    st.error("Secrets GROQ_API_KEY tidak ditemukan.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)
db_client = chromadb.PersistentClient(path=DB_PATH)
collection = db_client.get_or_create_collection(name="coffee_knowledge_base")

st.set_page_config(page_title="AgriPulse AI v3.2", layout="wide", page_icon="☕")

# --- 2. CORE LOGIC ---

def get_hybrid_analysis(user_query, news_context, journal_context):
    # MENGGUNAKAN MODEL TERBARU: llama-3.3-70b-specdec
    prompt = f"""
    SISTEM: Anda adalah Pakar Proteksi Tanaman Kopi. Waktu: {now_wib.strftime('%H:%M')} WIB.
    
    PERINTAH KHUSUS:
    1. Jawablah secara TEKNIS, MENDALAM, dan TO THE POINT.
    2. Jika pertanyaan tentang HAMA/PENYAKIT: Berikan deskripsi GEJALA visual pada tanaman/buah secara detail dan berikan LANGKAH PENGENDALIAN ORGANIK yang aplikatif.
    3. DILARANG membahas tren harga, pasar, atau kesesuaian lahan jika tidak ditanya.
    4. Hapus kalimat pembuka dan penutup yang bersifat template bisnis.
    
    REFERENSI JURNAL: {journal_context}
    REFERENSI BERITA: {news_context}
    PERTANYAAN: {user_query}
    """
    
    try:
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-specdec", 
            temperature=0.1,
            max_tokens=1500
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Sistem sedang transisi model. Mohon tunggu sejenak. (Detail: {e})"

# --- 3. UI STYLE ---
st.markdown("""<style>.stInfo { border-left: 5px solid #6f4e37; font-size: 16px; }</style>""", unsafe_allow_html=True)

# --- 4. HEADER ---
st.write(f"🕒 {now_wib.strftime('%H:%M')} WIB")
st.title(f"☕ {salam}, AgriPulse v3.2")
st.markdown(f"*{now_wib.strftime('%d %B %Y')} | Specialized Advisor*")

col_univ1, col_univ2 = st.columns(2)
with col_univ1:
    st.markdown(f'**Hijrah Wira Pratama, S.Si.D.** (TelU)')
with col_univ2:
    st.markdown(f'**Yokie Lidiantoro, S.T.** (ITB)')

st.divider()

# --- 5. MAIN CONTENT ---
tab1, tab2 = st.tabs(["💡 Konsultasi Teknis", "📊 Dashboard Harga"])

with tab1:
    q = st.text_area("Masukkan pertanyaan teknis:", placeholder="Contoh: Gejala PBKo dan pengendaliannya...")
    if st.button("Jalankan Analisis Spesialis"):
        if q:
            with st.spinner("Menganalisis..."):
                res = collection.query(query_texts=[q], n_results=3)
                j_ctx = "\n\n".join(res['documents'][0]) if res['documents'] else ""
                
                news_ctx = ""
                if any(word in q.lower() for word in ['harga', 'pasar', 'bisnis']):
                    try:
                        gn = GoogleNews(lang='id', country='ID')
                        news_ctx = "\n".join([f"- {e.title}" for e in gn.search('harga kopi')['entries'][:3]])
                    except: pass

                answer = get_hybrid_analysis(q, news_ctx, j_ctx)
                st.markdown("---")
                st.info(answer)

with tab2:
    st.subheader("Update Harga & Pasar")
    gn = GoogleNews(lang='id', country='ID')
    for e in gn.search('harga kopi terbaru')['entries'][:8]:
        st.write(f"📌 {e.title}")

st.divider()
st.caption("AgriPulse v3.2 | Model: Llama-3.3-70b-specdec | Fix Decommissioned Error")
