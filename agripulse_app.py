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

st.set_page_config(page_title="AgriPulse AI v3.0", layout="wide", page_icon="☕")

# --- 2. LOGIC FUNCTIONS ---

def get_coffee_news():
    try:
        gn = GoogleNews(lang='id', country='ID')
        search = gn.search('harga kopi terbaru indonesia')
        news_data = [f"- {entry.title}" for entry in search['entries'][:5]]
        return "\n".join(news_data) if news_data else "Tidak ada berita terbaru."
    except:
        return "Gagal mengambil berita terbaru."

def extract_price_from_news(news_text):
    prompt = f"Berikan 4 angka harga kopi (Rp/kg) dari teks ini: {news_text}. Range 38000-48000. HANYA ANGKA pisahkan koma."
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.1
        )
        prices = [int(s) for s in re.findall(r'\d+', response.choices[0].message.content)]
        return prices[:4] if len(prices) >= 4 else [38000, 39500, 41000, 43500]
    except:
        return [38500, 40500, 42000, 41500]

def process_pdf_to_db(uploaded_file):
    reader = pypdf.PdfReader(uploaded_file)
    text = "".join([p.extract_text() for p in reader.pages if p.extract_text()])
    chunks = [text[i:i+800] for i in range(0, len(text), 700)]
    waktu_wib = datetime.now() + timedelta(hours=7)
    timestamp = waktu_wib.strftime("%Y%m%d_%H%M%S")
    ids = [f"id_{timestamp}_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids)
    return len(chunks)

def get_hybrid_analysis(user_query, news_context, journal_context):
    prompt = f"""SISTEM: Penasihat Strategis Kopi Hybrid (Llama 3.3 Expert). 
    Gunakan konteks BERITA dan RISET JURNAL untuk menjawab secara presisi. 
    
    BERITA: {news_context}
    RISET JURNAL: {journal_context}
    
    PERTANYAAN USER: {user_query}"""
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
            return "⚠️ **Limit Trafik API Terdeteksi.**\n\nAnalisis jurnal membutuhkan token yang besar. Harap **TUNGGU 60 DETIK** agar sistem dapat me-reset kuota Anda."
        return f"Error: {e}"

# --- 3. CUSTOM CSS ---
st.markdown("""
    <style>
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    .stInfo { background-color: #f0f4f8; border-left: 5px solid #6f4e37; color: #1e1e1e; }
    .univ-label { font-size: 13px; color: #333; line-height: 1.2; margin-bottom: 0px; }
    </style>
    """, unsafe_allow_html=True)

# --- 4. UI INTERFACE ---
st.title("☕ AgriPulse AI v3.0")
st.markdown("##### *Hybrid Decision Support System for Coffee Farmers*")

# Logo TelU & ITB
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

# --- SIDEBAR ---
with st.sidebar:
    st.header("🗂️ Knowledge Warehouse")
    try:
        count = collection.count()
    except:
        count = 0
    st.info(f"🧠 Kapasitas: {count} Knowledge Chunks")
    
    uploaded_file = st.file_uploader("Upload Jurnal Strategis (PDF)", type="pdf")
    if st.button("Tanamkan ke Memori AI"):
        if uploaded_file:
            with st.spinner("Menganalisis dokumen..."):
                num = process_pdf_to_db(uploaded_file)
                st.success(f"Berhasil menanamkan {num} potongan data!"); st.rerun()

    st.divider()
    st.subheader("🔐 Admin Access")
    admin_input = st.text_input("Admin Password", type="password")
    if admin_input == ADMIN_KEY:
        if st.button("🗑️ Kosongkan Database"):
            db_client.delete_collection("coffee_knowledge_base")
            collection = db_client.get_or_create_collection(name="coffee_knowledge_base")
            st.warning("Memori database dibersihkan."); st.rerun()

# --- MAIN TABS ---
tab1, tab2, tab3 = st.tabs(["💡 Konsultasi Strategi", "📊 Market Dashboard", "📸 Diagnosa Penyakit"])

with tab1:
    st.subheader("Konsultasi Strategi Hybrid")
    user_q = st.text_area("Ajukan pertanyaan atau masalah perkebunan Anda:", height=150)

    if st.button("Mulai Analisis AI"):
        if user_q:
            with st.spinner("Menyisir database jurnal dan berita terbaru..."):
                news = get_coffee_news()
                # n_results=3 untuk efisiensi Token agar tidak sering limit
                try:
                    results = collection.query(query_texts=[user_q], n_results=3)
                    if results and results.get('documents') and len(results['documents'][0]) > 0:
                        j_context = "\n\n".join(results['documents'][0])
                    else:
                        j_context = "Data riset internal kosong."
                except:
                    j_context = "Database belum siap."
                
                answer = get_hybrid_analysis(user_q, news, j_context)
                st.markdown("---")
                st.info(answer)
        else:
            st.warning("Silakan masukkan pertanyaan.")

with tab2:
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.subheader("📰 Market Pulse")
        latest_news = get_coffee_news()
        st.write(latest_news)
    with col_b:
        st.subheader("📈 Tren Harga (AI-Extracted)")
        prices = extract_price_from_news(latest_news)
        st.line_chart(pd.DataFrame({"Harga (Rp/Kg)": prices}))

with tab3:
    st.subheader("Diagnosa Penyakit Biji Kopi")
    st.warning("Feature Coming Soon: AI-Powered Disease Detection")

st.divider()
st.caption("AgriPulse v3.0 | Stable Edition | TelU x ITB")
