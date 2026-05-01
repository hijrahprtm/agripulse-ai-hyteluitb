# --- CUSTOM HEADER & BRANDING ---
col1, col2 = st.columns([1, 5])
with col1:
    # Opsi A: Menggunakan file lokal (Upload logo_telu.png ke GitHub Anda)
    if os.path.exists("logo_telu.png"):
        st.image("logo_telu.png", width=120)
    else:
        # Opsi B: Backup URL jika file lokal belum ada
        st.image("https://upload.wikimedia.org/wikipedia/id/0/03/Logo_Telkom_University_potrait.png", width=120)

with col2:
    st.title("🌱 AGRIPULSE")
    st.markdown("### **Agricultural RAG-Integrated Precision Understanding & Localized Synthesis Engine**")
    st.markdown(
        """
        **Lead Developer:** Hijrah Pratama (Data Science, Telkom University)  
        **Subject Matter Expert:** Mas Yoki (S1 Pertanian, ITB)
        """
    )
st.divider()
