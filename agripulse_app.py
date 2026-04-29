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

# Set Waktu Indonesia Barat (WIB) secara real-time
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
    st.error("Secrets tidak ditemukan. Pastikan GROQ_API_KEY sudah disetel.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

try:
    db_client = chromadb.PersistentClient(path=DB_PATH)
    collection = db_client.get_or_create_collection(name="coffee_knowledge_base")
except Exception as e:
    st.error(f"Database Error: {e}")
    st.stop()

st.set_page_config(page_title="AgriPulse AI v3.1", layout="wide", page_icon="☕")

# --- 2. CORE LOGIC ---

def get_hybrid_analysis(user_query, news_context, journal_context):
    # Menggunakan Llama 3.1 70B: Kuota rate limit lebih besar untuk penggunaan intensif
    prompt = f"""
    SISTEM: Anda adalah Pakar Kopi Spesialis. Waktu: {now_wib.strftime('%H:%M')} WIB.
    
    PERINTAH TEGAS:
    1. Jawab HANYA apa yang ditanyakan. Fokus penuh pada inti pertanyaan.
    2. Jika bertanya PENYAKIT: Bedah GEJALA secara detail dan berikan SOLUSI teknis penanggulangannya.
    3. JANGAN memberikan analisis ekonomi/bisnis jika tidak relevan dengan pertanyaan.
    4. JANGAN membahas kesesuaian lahan jika user bertanya tentang hama/penyakit.
    5. Jawab dengan gaya profesional, padat, dan langsung ke poin utama.
    
    DATA REFERENSI (Hanya gunakan jika benar-benar relevan):
    {journal_context}
    
    KONTEKS TAMBAHAN (Hanya gunakan jika user bertanya pasar):
    {news_context}
    
    PERTANYAAN USER: {user_query}
    
    JAWABAN:
    """
    
    try:
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-70b-versatile",
            temperature=0.2,
            max_tokens=1000
        )
        return completion.choices[0].message.content
    except Exception as e:
        if "rate_limit" in str(e).lower():
            return "⚠️ Sistem sedang ramai. Harap tunggu 30-60 detik untuk reset kuota API."
        return f"Gangguan teknis: {e}"

# --- 3. UI STYLE ---
st.markdown("""
    <style>
    .block-container { padding-top: 1.5rem; }
    .stInfo { background-color: #fcfcfc; border: 1px solid #eee; border-left: 5px solid #6f4e37; color: #333; }
    .univ-label { font-size: 13px; color: #444; line-height: 1.3; }
    </style>
    """, unsafe_allow_html=True)

# --- 4. HEADER ---
st.write(f"🕒 {now_wib.strftime('%H:%M')} WIB")
st.title(f"☕ {salam}, AgriPulse v3.1")
st.markdown(f"*{now_wib.strftime('%d %B %Y')} | Research & Decision Support*")

col_univ1, col_univ2 = st.columns(2)
with col_univ1:
    c_logo1, c_text1 = st.columns([1, 4])
    with c_logo1:
        if os.path.exists("telulogo.webp"): st.image("telulogo.webp", width=60)
        else: st.caption("TelU")
    with c_text1:
        st.markdown('<p class="univ-label"><b>Hijrah Wira Pratama, S.Si.D.</b><br>AI Engineer (TelU)</p>', unsafe_allow_html=True)

with col_univ2:
    c_logo2, c_text2 = st.columns([1, 4])
    with c_logo2:
        if os.path.exists("itblogo.png"): st.image("itblogo.png", width=55)
        else: st.caption("ITB")
    with c_text2:
        st.markdown('<p class="univ-label"><b>Yokie Lidiantoro, S.T.</b><br>Agricultural Researcher (ITB)</p>', unsafe_allow_html=True)

st.divider()

# --- 5. SIDEBAR ---
with st.sidebar:
    st.header("🗂️ Knowledge Base")
    st.info(f"🧠 Memori Aktif: {collection.count()} Chunks")
    
    up_file = st.file_uploader("Upload Jurnal Strategis (PDF)", type="pdf")
    if st.button("Update Memori"):
        if up_file:
            with st.spinner("Memproses..."):
                reader = pypdf.PdfReader(up_file)
                text = "".join([p.extract_text() for p in reader.pages if p.extract_text()])
                chunks = [text[i:i+800] for i in range(0, len(text), 700)]
                ids = [f"id_{now_wib.timestamp()}_{i}" for i in range(len(chunks))]
                collection.add(documents=chunks, ids=ids)
                st.success("Database berhasil diperbarui!"); st.rerun()

    st.divider()
    admin_key = st.text_input("Admin Key", type="password")
    if admin_key == ADMIN_KEY:
        if st.button("🗑️ Reset Database"):
            db_client.delete_collection("coffee_knowledge_base")
            st.rerun()

# --- 6. MAIN CONTENT ---
tab1, tab2 = st.tabs(["💡 Analisis Spesialis", "📊 Dashboard Harga"])

with tab1:
    st.subheader("Konsultasi Pakar")
    q = st.text_area("Apa yang ingin Anda tanyakan hari ini?", placeholder="Contoh: Jelaskan gejala penyakit karat daun...")
    
    if st.button("Jalankan Analisis"):
        if q:
            with st.spinner("Menyisir database riset..."):
                # Pencarian database tetap jalan untuk context
                try:
                    res = collection.query(query_texts=[q], n_results=3)
                    j_ctx = "\n\n".join(res['documents'][0]) if res['documents'] else "Data riset tidak ditemukan."
                except:
                    j_ctx = ""

                # Ambil berita HANYA jika ada kata kunci ekonomi
                news_ctx = ""
                keywords = ['harga', 'pasar', 'bisnis', 'ekonomi', 'tren', 'profit', 'untung', 'jual']
                if any(word in q.lower() for word in keywords):
                    try:
                        gn = GoogleNews(lang='id', country='ID')
                        news_ctx = "\n".join([f"- {e.title}" for e in gn.search('harga kopi')['entries'][:3]])
                    except:
                        news_ctx = "Gagal memuat berita pasar."

                answer = get_hybrid_analysis(q, news_ctx, j_ctx)
                st.markdown("---")
                st.info(answer)

with tab2:
    st.subheader("Sentimen Harga Pasar")
    try:
        gn = GoogleNews(lang='id', country='ID')
        search = gn.search('harga kopi terbaru indonesia')
        for e in search['entries'][:10]:
            st.write(f"📌 {e.title}")
            st.caption(f"Sumber: {e.source.get('title', 'Berita Online')}")
    except:
        st.write("Berita tidak dapat dimuat.")

st.divider()
st.caption("AgriPulse v3.1 | Powered by Llama 3.1 70B | Fixed & Specialized")
