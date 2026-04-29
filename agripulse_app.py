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

# --- 0. KONFIGURASI LOKASI & WAKTU ---
DB_PATH = "./agripulse_db"
if not os.path.exists(DB_PATH): 
    os.makedirs(DB_PATH)

# Sinkronisasi Waktu Indonesia Barat (WIB)
wib = pytz.timezone('Asia/Jakarta')
now_wib = datetime.now(wib)
jam = now_wib.hour

if jam < 11: salam = "Selamat Pagi"
elif jam < 15: salam = "Selamat Siang"
elif jam < 19: salam = "Selamat Sore"
else: salam = "Selamat Malam"

# --- 1. KONEKSI API & DATABASE ---
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    ADMIN_KEY = st.secrets.get("ADMIN_KEY", "hijrahxyokixteluxitbangkatan2021") 
except KeyError:
    st.error("Konfigurasi Secrets (GROQ_API_KEY) tidak ditemukan di dashboard Streamlit.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# Inisialisasi ChromaDB
try:
    db_client = chromadb.PersistentClient(path=DB_PATH)
    collection = db_client.get_or_create_collection(name="coffee_knowledge_base")
except Exception as e:
    st.error(f"Gagal memuat Database Vektor: {e}")
    st.stop()

st.set_page_config(page_title="AgriPulse AI v3.4", layout="wide", page_icon="☕")

# --- 2. FUNGSI INTI ---

def get_coffee_news():
    """Mengambil berita kopi terbaru untuk dashboard dan konteks pasar"""
    try:
        gn = GoogleNews(lang='id', country='ID')
        search = gn.search('harga kopi terbaru indonesia')
        news_list = [f"- {e.title}" for e in search['entries'][:5]]
        return "\n".join(news_list) if news_list else "Berita tidak tersedia."
    except:
        return "Koneksi ke portal berita terputus."

def extract_price_from_news(news_text):
    """Menggunakan model cepat untuk ekstraksi angka harga bagi grafik"""
    prompt = f"Ekstrak 4 angka harga kopi (Rp/kg) dari teks: {news_text}. HANYA tulis angka pisahkan koma."
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

def get_hybrid_analysis(user_query, news_context, journal_context):
    """Logika utama AI Spesialis Kopi menggunakan Llama 3.3 70B Versatile"""
    prompt = f"""
    SISTEM: Anda adalah Pakar Kopi Spesialis (Kolaborasi TelU & ITB). Waktu: {now_wib.strftime('%H:%M')} WIB.
    
    PERINTAH UTAMA:
    1. Fokus HANYA pada tanaman KOPI.
    2. Jawab secara TEKNIS, MENDALAM, dan TO-THE-POINT.
    3. Jika user bertanya tentang PENYAKIT/HAMA: Jelaskan GEJALA visual secara detail dan CARA PENANGGULANGANNYA secara organik.
    4. DILARANG melantur ke masalah harga, tren pasar, atau kesesuaian lahan jika pertanyaan user murni teknis penyakit.
    5. Jika user bertanya tentang STRATEGI/HARGA: Padukan data riset jurnal dengan berita pasar.
    
    REFERENSI JURNAL: {journal_context}
    REFERENSI PASAR: {news_context}
    PERTANYAAN USER: {user_query}
    
    JAWABAN:
    """
    try:
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.1, # Rendah agar jawaban kaku pada fakta/teknis
            max_tokens=1200
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Mohon maaf, layanan AI sedang sibuk. Silakan coba 30 detik lagi. (Error: {e})"

# --- 3. TAMPILAN UI (CUSTOM CSS) ---
st.markdown("""
    <style>
    .stInfo { background-color: #ffffff; border-left: 5px solid #6f4e37; border-top: 1px solid #eee; border-right: 1px solid #eee; border-bottom: 1px solid #eee; color: #1e1e1e; }
    .univ-label { font-size: 13px; font-weight: bold; color: #333; line-height: 1.2; margin-bottom: 0px; }
    </style>
    """, unsafe_allow_html=True)

# --- 4. HEADER & LOGO ---
st.write(f"🕒 {now_wib.strftime('%H:%M')} WIB | {now_wib.strftime('%d %B %Y')}")
st.title(f"☕ {salam}, AgriPulse v3.4")
st.markdown("##### *Decision Support System for Sustainable Coffee Farming*")

col_univ1, col_univ2 = st.columns(2)
with col_univ1:
    c_logo1, c_text1 = st.columns([1, 4])
    with c_logo1:
        if os.path.exists("telulogo.webp"): st.image("telulogo.webp", width=60)
        else: st.caption("TelU")
    with c_text1:
        st.markdown('<p class="univ-label">Hijrah Wira Pratama, S.Si.D.<br>AI Engineer (TelU)</p>', unsafe_allow_html=True)

with col_univ2:
    c_logo2, c_text2 = st.columns([1, 4])
    with c_logo2:
        if os.path.exists("itblogo.png"): st.image("itblogo.png", width=55)
        else: st.caption("ITB")
    with c_text2:
        st.markdown('<p class="univ-label">Yokie Lidiantoro, S.T.<br>Agricultural Researcher (ITB)</p>', unsafe_allow_html=True)

st.divider()

# --- 5. SIDEBAR (KNOWLEDGE BASE & ADMIN) ---
with st.sidebar:
    st.header("🗂️ Knowledge Base")
    try:
        count = collection.count()
    except:
        count = 0
    st.info(f"🧠 Kapasitas Memori: {count} Chunks")
    
    uploaded_file = st.file_uploader("Upload Jurnal Riset (PDF)", type="pdf")
    if st.button("Tanamkan ke Memori AI"):
        if uploaded_file:
            with st.spinner("Mengekstrak data..."):
                reader = pypdf.PdfReader(uploaded_file)
                text = "".join([p.extract_text() for p in reader.pages if p.extract_text()])
                # Chunking untuk efisiensi context
                chunks = [text[i:i+800] for i in range(0, len(text), 700)]
                ids = [f"id_{now_wib.timestamp()}_{i}" for i in range(len(chunks))]
                collection.add(documents=chunks, ids=ids)
                st.success(f"Berhasil menambahkan {len(chunks)} data!"); st.rerun()

    st.divider()
    st.subheader("🔐 Admin Panel")
    admin_input = st.text_input("Password", type="password")
    if admin_input == ADMIN_KEY:
        if st.button("🗑️ Hapus Seluruh Database"):
            db_client.delete_collection("coffee_knowledge_base")
            st.warning("Database telah dibersihkan."); st.rerun()

# --- 6. KONTEN UTAMA (TABS) ---
tab1, tab2, tab3 = st.tabs(["💡 Analisis Spesialis", "📊 Dashboard Market", "📸 Computer Vision"])

with tab1:
    st.subheader("Konsultasi Pakar Kopi")
    user_q = st.text_area("Ajukan pertanyaan teknis atau strategis:", height=150, placeholder="Contoh: Apa gejala PBKo dan bagaimana pengendaliannya secara organik?")

    if st.button("Jalankan Analisis AI"):
        if user_q:
            with st.spinner("Menghubungkan data riset & pasar..."):
                # Pencarian Data Jurnal
                try:
                    search_results = collection.query(query_texts=[user_q], n_results=3)
                    j_ctx = "\n\n".join(search_results['documents'][0]) if search_results['documents'] else ""
                except:
                    j_ctx = ""
                
                # Pencarian Data Berita (Hanya jika user tanya soal harga/ekonomi)
                news_ctx = ""
                if any(word in user_q.lower() for word in ['harga', 'pasar', 'bisnis', 'ekonomi', 'untung', 'jual']):
                    news_ctx = get_coffee_news()
                
                answer = get_hybrid_analysis(user_q, news_ctx, j_ctx)
                st.markdown("---")
                st.info(answer)
        else:
            st.warning("Silakan ketik pertanyaan Anda.")

with tab2:
    st.subheader("Tren Harga & Sentimen Pasar")
    col_a, col_b = st.columns([1, 1])
    news_data = get_coffee_news()
    with col_a:
        st.markdown("##### 📰 Berita Pasar Terkini")
        st.write(news_data)
    with col_b:
        st.markdown("##### 📈 Proyeksi Tren Harga")
        prices = extract_price_from_news(news_data)
        st.line_chart(pd.DataFrame({"Harga (Rp/Kg)": prices}))

with tab3:
    st.subheader("📸 Diagnosa Penyakit (Visual)")
    st.warning("Fitur Coming Soon: Integrasi YOLOv11 untuk deteksi Karat Daun dan PBKo langsung melalui kamera.")
    st.write("Sistem sedang dikembangkan untuk memvalidasi diagnosa visual dengan data riset tekstual yang ada di database.")

st.divider()
st.caption("AgriPulse v3.4 | Powered by Llama 3.3 70B | TelU x ITB Collaboration")
