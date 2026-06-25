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
st.set_page_config(page_title="Klasifikasi Tingkat Obesitas", page_icon="🩺", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* Menyembunyikan menu bawaan Streamlit agar lebih bersih */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Mempercantik Judul Utama */
    .main-title {
        font-size: 3rem !important;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #1E3A8A, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    
    /* Mempercantik tombol Latih Model */
    .stButton > button {
        background: linear-gradient(90deg, #3B82F6 0%, #8B5CF6 100%);
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
        color: white;
    }
    
    /* Mempercantik angka Akurasi */
    [data-testid="stMetricValue"] {
        color: #10B981 !important;
        font-weight: 800;
        font-size: 2.5rem !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">Aplikasi Prediksi Tingkat Obesitas 🩺</p>', unsafe_allow_html=True)
st.markdown("""
Aplikasi ini memprediksi tingkat obesitas menggunakan algoritma **Decision Tree**. 
Model akan mempelajari pola kebiasaan makan dan kondisi fisik dari dataset untuk melakukan klasifikasi.
""")

st.sidebar.header("1. Pengaturan Dataset")

# Logika cerdas: Cari file lokal dulu, jika tidak ada baru minta upload
default_file = 'ObesityDataSet_raw_and_data_sinthetic.csv'
df = None

if os.path.exists(default_file):
    st.sidebar.success(f"✅ Dataset lokal ditemukan!")
    df = pd.read_csv(default_file)
else:
    st.sidebar.info("Dataset lokal tidak ditemukan. Silakan upload file CSV.")
    uploaded_file = st.sidebar.file_uploader("Upload dataset", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

# Jika dataframe berhasil dimuat (baik dari lokal maupun upload)
if df is not None:
    st.markdown("---")
    st.subheader("📋 Pratinjau Data Asli")
    
    # Memasukkan dataframe ke dalam expander agar tidak memakan banyak tempat
    with st.expander("Klik untuk melihat detail dataset (5 Baris Pertama)"):
        st.dataframe(df.head(), use_container_width=True)
        st.caption(f"Total Data: **{df.shape[0]} baris** dan **{df.shape[1]} kolom**")

    target_column = 'NObeyesdad'
    
    if target_column in df.columns:
        # --- PREPROCESSING DATA ---
        df_encoded = df.copy()
        le = LabelEncoder()
        
        # Mengubah data teks menjadi angka (Label Encoding)
        for col in df_encoded.select_dtypes(include=['object']).columns:
            df_encoded[col] = le.fit_transform(df_encoded[col])
            
        # Memisahkan Fitur (X) dan Target (y)
        X = df_encoded.drop(columns=[target_column])
        y = df_encoded[target_column]
        
        # Split data latih (80%) dan data uji (20%)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        st.markdown("---")
        st.subheader("⚙️ Pelatihan & Evaluasi Model")
        st.write("Silakan tekan tombol di bawah ini untuk memulai pelatihan model Machine Learning.")
        
        # Tombol untuk mengeksekusi model
        if st.button("🚀 Latih Model (Decision Tree)"):
            with st.spinner('Sedang melatih model...'):
                # Inisiasi dan Pelatihan
                model = DecisionTreeClassifier(random_state=42)
                model.fit(X_train, y_train)
                
                # Prediksi
                y_pred = model.predict(X_test)
                
                # Evaluasi Akurasi
                acc = accuracy_score(y_test, y_pred)
                
            st.success("✅ Model berhasil dilatih!")
            
            # Menggunakan st.metric untuk akurasi agar terlihat jauh lebih modern
            st.metric(label="🎯 Akurasi Model", value=f"{acc * 100:.2f}%", delta="Sangat Baik")
            
            # --- VISUALISASI HASIL ---
            st.markdown("---")
            st.subheader("📈 Visualisasi Data & Hasil Prediksi")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**1. Distribusi Kelas Obesitas (Data Asli)**")
                fig1, ax1 = plt.subplots(figsize=(6, 4))
                # Menggunakan palet warna mako yang lebih elegan
                sns.countplot(y=df[target_column], ax=ax1, palette="mako", order=df[target_column].value_counts().index)
                ax1.set_xlabel("Jumlah Individu")
                ax1.set_ylabel("Tingkat Obesitas")
                sns.despine() # Menghapus garis tepi agar lebih bersih
                st.pyplot(fig1)
                
            with col2:
                st.write("**2. Confusion Matrix**")
                st.caption("Melihat detail tebakan benar (diagonal) vs tebakan salah")
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                cm = confusion_matrix(y_test, y_pred)
                # Menghilangkan colorbar agar kotak lebih luas dan rapi
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2, cbar=False)
                ax2.set_xlabel("Prediksi Model")
                ax2.set_ylabel("Data Aktual")
                st.pyplot(fig2)
                
    else:
        st.error(f"Error: Dataset harus memiliki kolom target bernama '{target_column}'")
else:
    st.warning("⚠️ Menunggu dataset untuk diproses...")