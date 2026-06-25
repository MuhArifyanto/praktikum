import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import os

# Konfigurasi Halaman Streamlit
st.set_page_config(page_title="Klasifikasi Tingkat Obesitas", page_icon="🩺", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main-title {
        font-size: 3rem !important;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0px;
    }
    .sub-title {
        text-align: center;
        color: #6B7280;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🩺 Dashboard Prediksi Obesitas</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Klasifikasi tingkat obesitas menggunakan algoritma Machine Learning (Decision Tree)</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3006/3006876.png", width=100)
    st.header("⚙️ Pengaturan Dataset")
    
    default_file = 'ObesityDataSet_raw_and_data_sinthetic.csv'
    df = None
    
    if os.path.exists(default_file):
        st.success(f"✅ Dataset bawaan aktif")
        df = pd.read_csv(default_file)
    else:
        st.info("Upload dataset Anda di bawah ini:")
        uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
    st.markdown("---")
    st.caption("Dibuat menggunakan Streamlit & Scikit-Learn")

if df is not None:
    # Membagi antarmuka menjadi beberapa tab agar lebih profesional & rapi
    tab1, tab2, tab3 = st.tabs(["📊 Eksplorasi Data", "🤖 Pelatihan Model", "📈 Hasil & Visualisasi"])
    
    target_column = 'NObeyesdad'
    
    with tab1:
        st.subheader("Sekilas Tentang Data")
        st.markdown("Berikut adalah sampel 5 baris pertama dari dataset yang digunakan:")
        st.dataframe(df.head(), use_container_width=True)
        
        # Info Dataset dalam kolom
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Total Baris Data:** {df.shape[0]} baris")
        with col2:
            st.info(f"**Total Atribut/Kolom:** {df.shape[1]} kolom")
            
    if target_column in df.columns:
        # --- PREPROCESSING ---
        df_encoded = df.copy()
        le = LabelEncoder()
        
        for col in df_encoded.select_dtypes(include=['object']).columns:
            df_encoded[col] = le.fit_transform(df_encoded[col])
            
        X = df_encoded.drop(columns=[target_column])
        y = df_encoded[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        with tab2:
            st.subheader("Konfigurasi & Pelatihan Model")
            st.markdown("Model **Decision Tree Classifier** siap dilatih menggunakan proporsi **80% data latih** dan **20% data uji**.")
            
            if st.button("🚀 Latih Model Sekarang", type="primary", use_container_width=True):
                with st.spinner('Memproses data dan melatih model...'):
                    model = DecisionTreeClassifier(random_state=42)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    # Simpan hasil ke session_state agar tidak hilang saat pindah tab
                    st.session_state['trained'] = True
                    st.session_state['acc'] = accuracy_score(y_test, y_pred)
                    st.session_state['y_test'] = y_test
                    st.session_state['y_pred'] = y_pred
                    
                st.success("✅ Pelatihan selesai! Silakan buka tab **Hasil & Visualisasi** untuk melihat performanya.")
                
        with tab3:
            if st.session_state.get('trained', False):
                st.subheader("Evaluasi Performa Model")
                
                # Menampilkan metrik akurasi dengan gaya dashboard
                acc_percentage = st.session_state['acc'] * 100
                st.metric(label="🎯 Akurasi Pengujian (Test Accuracy)", value=f"{acc_percentage:.2f}%", delta="Sangat Baik")
                
                st.markdown("---")
                st.subheader("Grafik Analisis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**1. Distribusi Kelas Target**")
                    fig1, ax1 = plt.subplots(figsize=(6, 4))
                    sns.countplot(y=df[target_column], ax=ax1, palette="mako", order=df[target_column].value_counts().index)
                    ax1.set_xlabel("Jumlah Individu")
                    ax1.set_ylabel("Tingkat Obesitas")
                    sns.despine() # Menghilangkan border luar grafik agar terlihat lebih bersih
                    st.pyplot(fig1)
                    
                with col2:
                    st.markdown("**2. Confusion Matrix**")
                    fig2, ax2 = plt.subplots(figsize=(6, 4))
                    cm = confusion_matrix(st.session_state['y_test'], st.session_state['y_pred'])
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2, cbar=False)
                    ax2.set_xlabel("Prediksi Model")
                    ax2.set_ylabel("Data Aktual")
                    st.pyplot(fig2)
            else:
                st.info("⚠️ Silakan latih model terlebih dahulu di tab **Pelatihan Model** untuk melihat hasil visualisasi.")
                
    else:
        st.error(f"Dataset harus memiliki kolom target bernama '{target_column}'")
else:
    st.warning("⚠️ Menunggu dataset untuk diproses...")