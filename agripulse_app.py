import streamlit as st
from pygooglenews import GoogleNews
from groq import Groq
import chromadb
from datetime import datetime
import pytz
import pd
import pypdf
import re
import os

# --- 0. ENV & TIME SETUP ---
DB_PATH = "./agripulse_db"
if not os.path.exists(DB_PATH): os.makedirs(DB_PATH)

# Set Waktu Indonesia Barat (WIB)
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
    st.error("Secrets tidak ditemukan.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)
db_client = chromadb.PersistentClient(path=DB_PATH)
collection = db_client.get_or_create_collection(name="coffee_knowledge_base")

st.set_page_config(page_title="AgriPulse AI v3.1", layout="wide", page_icon="☕")

# --- 2. CORE LOGIC ---

def get_hybrid_analysis(user_query, news_context, journal_context):
    # Logika Model: Menggunakan Llama 3.1 70B (Quota lebih besar/jarang limit dibanding 3.3)
    prompt = f"""
    SISTEM: Anda adalah Pakar Kopi Spesialis. Hari ini: {now_wib.strftime('%A, %d %B %Y %H:%M')} WIB.
    
    TUGAS UTAMA:
    1. Jawablah dengan FOKUS HANYA pada inti pertanyaan user. 
    2. JANGAN memasukkan konteks lahan jika user bertanya tentang penyakit. 
    3. JANGAN memasukkan konteks bisnis/harga jika user tidak bertanya tentang ekonomi.
    4. Jika bertanya PENYAKIT: Jelaskan GEJALA secara detail dan CARA PENANGGULANGANNYA secara teknis.
    
    DATA RISET INTERNAL (Gunakan hanya jika relevan dengan pertanyaan):
    {journal_context}
    
    DATA PASAR (Gunakan hanya jika user bertanya tentang harga/pasar):
    {news_context}
    
    PERTANYAAN USER: {user_query}
    
    JAWABAN:
    (Berikan jawaban langsung, detail, tanpa pembuka yang bertele-tele atau penutup tren harga yang tidak relevan).
    """
    
    try:
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-70b-versatile", # Model dengan limit prompting lebih longgar
            temperature=0.2, # Lebih kaku dan fokus pada fakta
            max_tokens=1200
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Mohon maaf, sistem sedang sinkronisasi. Silakan coba kembali dalam 30 detik. ({e})"

# --- 3. UI STYLE ---
st.markdown("""
    <style>
    .block-container { padding-top: 1rem; }
    .stInfo { background-color: #ffffff; border-left: 5px solid #6f4e37; border-right: 1px solid #ddd; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd; }
    .univ-label { font-size: 13px; color: #333; line-height: 1.2; }
    </style>
    """, unsafe_allow_html=True)

# --- 4. HEADER ---
st.write(f"🕒 {now_wib.strftime('%H:%M')} WIB")
st.title(f"☕ {salam}, AgriPulse v3.1")
st.markdown(f"*{now_wib.strftime('%d %B %Y')} - Decision Support System*")

col_univ1, col_univ2 = st.columns(2)
with col_univ1:
    c_logo1, c_text1 = st.columns([1, 4])
    with c_logo1: st.image("telulogo.webp", width=60) if os.path.exists("telulogo.webp") else st.caption("TelU")
    with c_text1: st.markdown('<p class="univ-label"><b>Hijrah Wira Pratama, S.Si.D.</b><br>AI Engineer</p>', unsafe_allow_html=True)

with col_univ2:
    c_logo2, c_text2 = st.columns([1, 4])
    with c_logo2: st.image("itblogo.png", width=55) if os.path.exists("itblogo.png") else st.caption("ITB")
    with c_text2: st.markdown('<p class="univ-label"><b>Yokie Lidiantoro, S.T.</b><br>Agricultural Researcher</p>', unsafe_allow_html=True)

st.divider()

# --- 5. MAIN UI ---
with st.sidebar:
    st.header("🗂️ Knowledge Base")
    st.info(f"🧠 Memori: {collection.count()} Chunks")
    up_file = st.file_uploader("Upload Jurnal (PDF)", type="pdf")
    if st.button("Update Database"):
        if up_file:
            reader = pypdf.PdfReader(up_file)
            text = "".join([p.extract_text() for p in reader.pages if p.extract_text()])
            chunks = [text[i:i+800] for i in range(0, len(text), 700)]
            ids = [f"id_{now_wib.timestamp()}_{i}" for i in range(len(chunks))]
            collection.add(documents=chunks, ids=ids)
            st.success("Database Diperbarui!"); st.rerun()

tab1, tab2 = st.tabs(["💡 Analisis Spesialis", "📊 Dashboard Harga"])

with tab1:
    user_q = st.text_area("Masukkan pertanyaan (Contoh: Apa gejala karat daun dan cara mengatasinya?)", height=120)
    if st.button("Dapatkan Jawaban"):
        if user_q:
            with st.spinner("Menganalisis data spesifik..."):
                # Pencarian database tetap jalan tapi AI diperingatkan untuk memfilter
                results = collection.query(query_texts=[user_q], n_results=3)
                j_ctx = "\n\n".join(results['documents'][0]) if results['documents'] else ""
                
                # Hanya ambil berita jika user tanya pasar/harga
                news_ctx = ""
                if any(x in user_q.lower() for x in ['harga', 'pasar', 'ekonomi', 'bisnis', 'tren']):
                    gn = GoogleNews(lang='id', country='ID')
                    news_ctx = "\n".join([f"- {e.title}" for e in gn.search('harga kopi')['entries'][:3]])
                
                answer = get_hybrid_analysis(user_q, news_ctx, j_ctx)
                st.markdown("---")
                st.info(answer)

with tab2:
    st.subheader("Tren Harga Real-Time")
    st.caption("Data diperbarui otomatis dari berita pasar terbaru.")
    # Kode Dashboard simpel agar tidak membebani token
    gn = GoogleNews(lang='id', country='ID')
    entries = gn.search('harga kopi')['entries'][:5]
    for e in entries: st.write(f"• {e.title}")

st.divider()
st.caption("AgriPulse v3.1 | Powered by Llama 3.1 70B | Spesialis Edition")
