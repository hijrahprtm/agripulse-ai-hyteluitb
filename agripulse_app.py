# --- 3. UI INTERFACE ---
st.title("☕ AgriPulse AI v3.0")
st.markdown("##### *Hybrid Decision Support System for Coffee Farmers*")

# Mengembalikan Logo TelU & ITB yang Berjejer
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
            st.warning("Memori database dibersihkan."); st.rerun()

# --- MAIN TABS ---
tab1, tab2, tab3 = st.tabs(["💡 Konsultasi Strategi", "📊 Market Dashboard", "📸 Diagnosa Penyakit"])

with tab1:
    st.subheader("Konsultasi Strategi Hybrid")
    user_q = st.text_area("Ajukan pertanyaan atau masalah perkebunan Anda:", height=150, placeholder="Contoh: Bagaimana strategi pupuk organik saat harga jual sedang turun?")

    if st.button("Mulai Analisis AI"):
        if user_q:
            with st.spinner("Menyisir database jurnal dan berita terbaru..."):
                news = get_coffee_news()
                
                # Mengambil hanya 3 hasil agar tidak terkena Rate Limit (TPM)
                try:
                    results = collection.query(query_texts=[user_q], n_results=3)
                    if results and results.get('documents') and len(results['documents'][0]) > 0:
                        j_context = "\n\n".join(results['documents'][0])
                    else:
                        j_context = "Data riset internal kosong. AI akan menjawab berdasarkan pengetahuan umum dan berita."
                except Exception:
                    j_context = "Database belum siap. Menggunakan mode pengetahuan umum."
                
                answer = get_hybrid_analysis(user_q, news, j_context)
                st.markdown("---")
                st.info(answer)
        else:
            st.warning("Silakan masukkan pertanyaan terlebih dahulu.")

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
