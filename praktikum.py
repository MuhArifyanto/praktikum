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

st.title("Aplikasi Prediksi Tingkat Obesitas 🩺")
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
    st.subheader("Pratinjau Data Asli")
    st.dataframe(df.head())

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
        
        st.divider()
        st.header("Pelatihan & Evaluasi Model")
        
        # Tombol untuk mengeksekusi model
        if st.button("Latih Model (Decision Tree)", type="primary"):
            with st.spinner('Sedang melatih model...'):
                # Inisiasi dan Pelatihan
                model = DecisionTreeClassifier(random_state=42)
                model.fit(X_train, y_train)
                
                # Prediksi
                y_pred = model.predict(X_test)
                
                # Evaluasi Akurasi
                acc = accuracy_score(y_test, y_pred)
                
            st.success("Model berhasil dilatih!")
            st.info(f"🎯 **Akurasi Model: {acc * 100:.2f}%**")
            
            # --- VISUALISASI HASIL ---
            st.subheader("Visualisasi Data & Hasil Prediksi")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**1. Distribusi Kelas Obesitas (Data Asli)**")
                fig1, ax1 = plt.subplots(figsize=(6, 4))
                sns.countplot(y=df[target_column], ax=ax1, palette="viridis", order=df[target_column].value_counts().index)
                ax1.set_xlabel("Jumlah Individu")
                ax1.set_ylabel("Tingkat Obesitas")
                st.pyplot(fig1)
                
            with col2:
                st.write("**2. Confusion Matrix**")
                st.caption("Melihat detail tebakan benar (diagonal) vs tebakan salah (luar diagonal)")
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                cm = confusion_matrix(y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2)
                ax2.set_xlabel("Prediksi Model")
                ax2.set_ylabel("Data Aktual")
                st.pyplot(fig2)
                
    else:
        st.error(f"Error: Dataset harus memiliki kolom target bernama '{target_column}'")
else:
    st.warning("⚠️ Menunggu dataset untuk diproses...")