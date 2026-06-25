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
st.set_page_config(page_title="Prediksi Obesitas AI", page_icon="🔮", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS ---
# Membuat tampilan sangat modern dengan warna-warna cerah, tombol kustom, dan card
st.markdown("""
<style>
    /* Global Font & Background adjustments */
    .stApp {
        background-color: #F8FAFC;
    }
    .main-title {
        font-size: 3.5rem !important;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #2563EB, #9333EA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0px;
        padding-top: 20px;
    }
    .sub-title {
        text-align: center;
        color: #475569;
        font-size: 1.2rem;
        font-weight: 500;
        margin-bottom: 3rem;
    }
    /* Styling Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        justify-content: center;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #ffffff;
        border-radius: 10px 10px 0px 0px;
        padding: 10px 20px;
        box-shadow: 0 -2px 5px rgba(0,0,0,0.05);
        color: #1E293B;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563EB !important;
        color: white !important;
    }
    /* Styling Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #3B82F6 0%, #8B5CF6 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.6);
        color: white;
        border: none;
    }
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 3rem !important;
        color: #10B981 !important;
        font-weight: 800;
    }
    /* Prediction Result Card */
    .prediction-card {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(16, 185, 129, 0.4);
        margin-top: 20px;
    }
    .prediction-text {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">AI Obesity Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Sistem Prediksi Cerdas Berbasis Machine Learning (Decision Tree)</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Konfigurasi Sistem")
    
    default_file = 'ObesityDataSet_raw_and_data_sinthetic.csv'
    df = None
    
    if os.path.exists(default_file):
        st.success(f"✅ Sistem terhubung dengan {default_file}")
        df = pd.read_csv(default_file)
    else:
        st.info("Upload dataset Anda di bawah ini:")
        uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
    st.markdown("---")
    st.caption("Engine: Scikit-Learn | UI: Streamlit")

if df is not None:
    # Memperbaiki Data Processing agar Encoder tersimpan dengan benar untuk Prediksi
    target_column = 'NObeyesdad'
    
    if target_column in df.columns:
        # Menyiapkan session state untuk model
        if 'model' not in st.session_state:
            st.session_state['model'] = None
            st.session_state['encoders'] = {}
            st.session_state['features'] = []
            
        # Tab Layout Modern
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Eksplorasi Data", "⚙️ Training Engine", "📉 Evaluasi Model", "🔮 Live Prediction"])
        
        with tab1:
            st.markdown("### 📋 Preview Dataset")
            st.dataframe(df.head(10), use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Baris", f"{df.shape[0]:,} Rows")
            col2.metric("Total Fitur", f"{df.shape[1]-1} Features")
            col3.metric("Kelas Target", f"{df[target_column].nunique()} Classes")
            
        with tab2:
            st.markdown("### ⚙️ Machine Learning Training Engine")
            st.info("Tekan tombol di bawah untuk melatih algoritma Decision Tree Classifier.")
            
            if st.button("🚀 Inisialisasi & Latih AI Model", use_container_width=True):
                with st.spinner("AI sedang mempelajari pola data..."):
                    # Preprocessing Data dengan benar
                    df_encoded = df.copy()
                    encoders = {}
                    
                    for col in df_encoded.select_dtypes(include=['object']).columns:
                        le = LabelEncoder()
                        df_encoded[col] = le.fit_transform(df_encoded[col])
                        encoders[col] = le
                        
                    X = df_encoded.drop(columns=[target_column])
                    y = df_encoded[target_column]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    model = DecisionTreeClassifier(random_state=42)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    # Simpan ke memori
                    st.session_state['model'] = model
                    st.session_state['encoders'] = encoders
                    st.session_state['features'] = X.columns.tolist()
                    st.session_state['acc'] = accuracy_score(y_test, y_pred)
                    st.session_state['y_test'] = y_test
                    st.session_state['y_pred'] = y_pred
                    
                st.success("✅ AI berhasil dilatih! Pindah ke tab Evaluasi atau Live Prediction.")
                
        with tab3:
            if st.session_state['model'] is not None:
                st.markdown("### 📉 Hasil Evaluasi Model AI")
                acc_percentage = st.session_state['acc'] * 100
                
                col_met, col_chart = st.columns([1, 2])
                with col_met:
                    st.metric(label="Akurasi Sistem", value=f"{acc_percentage:.2f}%", delta="High Accuracy")
                    st.markdown("Model memiliki tingkat keyakinan yang sangat tinggi dalam mengklasifikasikan data baru.")
                    
                with col_chart:
                    st.markdown("**Matriks Kebingungan (Confusion Matrix)**")
                    fig2, ax2 = plt.subplots(figsize=(5, 3))
                    cm = confusion_matrix(st.session_state['y_test'], st.session_state['y_pred'])
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", ax=ax2, cbar=False)
                    ax2.set_xlabel("Prediksi AI")
                    ax2.set_ylabel("Data Sebenarnya")
                    st.pyplot(fig2)
            else:
                st.warning("⚠️ Silakan jalankan Training Engine terlebih dahulu.")
                
        with tab4:
            st.markdown("### 🔮 Uji Coba Prediksi Mandiri")
            if st.session_state['model'] is not None:
                st.markdown("Masukkan data pasien/individu untuk memprediksi tingkat obesitasnya.")
                
                with st.form("predict_form"):
                    col_a, col_b, col_c = st.columns(3)
                    
                    input_data = {}
                    
                    with col_a:
                        st.markdown("**Profil Fisik**")
                        input_data['Gender'] = st.selectbox("Jenis Kelamin", ["Male", "Female"])
                        input_data['Age'] = st.number_input("Umur (Tahun)", 10, 100, 25)
                        input_data['Height'] = st.number_input("Tinggi Badan (Meter)", 1.0, 2.5, 1.70, step=0.01)
                        input_data['Weight'] = st.number_input("Berat Badan (Kg)", 30.0, 200.0, 70.0, step=0.5)
                        
                    with col_b:
                        st.markdown("**Gaya Hidup & Makanan**")
                        input_data['family_history_with_overweight'] = st.selectbox("Riwayat Keluarga Obesitas", ["yes", "no"])
                        input_data['FAVC'] = st.selectbox("Sering Makan Kalori Tinggi?", ["yes", "no"])
                        input_data['FCVC'] = st.slider("Konsumsi Sayur (1-3)", 1.0, 3.0, 2.0)
                        input_data['NCP'] = st.slider("Jumlah Makan Utama (1-4)", 1.0, 4.0, 3.0)
                        input_data['CAEC'] = st.selectbox("Makan di antara jam makan", ["no", "Sometimes", "Frequently", "Always"])
                        
                    with col_c:
                        st.markdown("**Aktivitas Lainnya**")
                        input_data['SMOKE'] = st.selectbox("Merokok?", ["yes", "no"])
                        input_data['CH2O'] = st.slider("Konsumsi Air (1-3 Liter)", 1.0, 3.0, 2.0)
                        input_data['SCC'] = st.selectbox("Monitoring Kalori?", ["yes", "no"])
                        input_data['FAF'] = st.slider("Aktivitas Fisik (0-3)", 0.0, 3.0, 1.0)
                        input_data['TUE'] = st.slider("Waktu di depan layar (0-2)", 0.0, 2.0, 1.0)
                        input_data['CALC'] = st.selectbox("Konsumsi Alkohol", ["no", "Sometimes", "Frequently", "Always"])
                        input_data['MTRANS'] = st.selectbox("Transportasi Utama", ["Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"])
                        
                    submitted = st.form_submit_button("🔮 Lakukan Prediksi")
                    
                if submitted:
                    # Proses data input
                    df_input = pd.DataFrame([input_data])
                    encoders = st.session_state['encoders']
                    
                    # Encode categorical inputs
                    for col in df_input.columns:
                        if col in encoders and col != target_column:
                            # Jika ada label baru yang tidak dikenal (misal), set ke label terdekat/default
                            # Di sini kita anggap input pasti ada di label encoder karena pilihan di selectbox sama
                            try:
                                df_input[col] = encoders[col].transform(df_input[col])
                            except:
                                pass
                                
                    # Pastikan urutan fitur sama persis
                    df_input = df_input[st.session_state['features']]
                    
                    # Prediksi
                    pred_encoded = st.session_state['model'].predict(df_input)[0]
                    # Decode prediksi
                    pred_label = encoders[target_column].inverse_transform([pred_encoded])[0]
                    
                    # Tampilkan hasil
                    pred_label_clean = str(pred_label).replace("_", " ")
                    st.markdown(f"""
                        <div class="prediction-card">
                            <p style="margin:0; font-size:1.2rem;">Hasil Prediksi AI:</p>
                            <p class="prediction-text">{pred_label_clean}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
            else:
                st.warning("⚠️ Harap latih AI Model di tab 'Training Engine' terlebih dahulu sebelum menggunakan Live Prediction.")
                
    else:
        st.error(f"Dataset harus memiliki kolom target bernama '{target_column}'")
else:
    st.warning("⚠️ Menunggu dataset untuk diproses...")