# =============================================================================
# Aplikasi Prediksi Tingkat Obesitas
# Production-Ready Version | Streamlit Community Cloud Compatible
# Menggunakan Plotly sebagai charting library (lebih stabil dari matplotlib
# di server headless/cloud, tidak ada risiko Segmentation Fault)
# =============================================================================

import io
import os
import warnings
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# =============================================================================
# KONSTANTA
# =============================================================================
DEFAULT_DATASET = 'ObesityDataSet_raw_and_data_sinthetic.csv'
TARGET_COLUMN   = 'NObeyesdad'
TEST_SIZE       = 0.2
RANDOM_STATE    = 42

# =============================================================================
# KONFIGURASI HALAMAN
# =============================================================================
st.set_page_config(
    page_title = "Klasifikasi Tingkat Obesitas",
    page_icon  = "🩺",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# =============================================================================
# CUSTOM CSS
# =============================================================================
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer     {visibility: hidden;}
    header     {visibility: hidden;}

    /* Light Mode */
    .stApp {
        background: linear-gradient(135deg, #ffffff 0%, #c1dfc4 100%);
        background-attachment: fixed;
    }
    [data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.5) !important;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-right: 1px solid rgba(255,255,255,0.3);
    }

    /* Dark Mode */
    @media (prefers-color-scheme: dark) {
        .stApp {
            background: linear-gradient(135deg, #0d1f14 0%, #0f3424 55%, #115937 100%) !important;
            background-attachment: fixed;
        }
        [data-testid="stSidebar"] {
            background-color: rgba(0, 0, 0, 0.35) !important;
        }
        .main-title {
            background: -webkit-linear-gradient(45deg, #a8e063, #e0ffe0) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
        }
    }

    /* Judul */
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #047857, #10B981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        line-height: 1.2;
    }
    .sub-desc {
        color: #374151;
        font-size: 1rem;
        margin-bottom: 1rem;
    }

    /* Tombol */
    .stButton > button {
        background: linear-gradient(90deg, #10B981 0%, #059669 100%);
        color: white !important;
        border: none;
        padding: 0.6rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(16, 185, 129, 0.45);
        color: white !important;
    }

    /* Metric */
    [data-testid="stMetricValue"] {
        color: #059669 !important;
        font-weight: 800;
        font-size: 2.4rem !important;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# FUNGSI CACHED
# =============================================================================

@st.cache_data(show_spinner="Memuat dataset...", ttl=3600)
def load_data(path: str) -> pd.DataFrame:
    """Baca CSV dari path lokal. Di-cache agar tidak dibaca ulang setiap re-run."""
    return pd.read_csv(path)


@st.cache_data(show_spinner="Memuat dataset yang di-upload...", ttl=3600)
def load_uploaded_data(file_bytes: bytes) -> pd.DataFrame:
    """Baca CSV dari bytes (file upload). Di-cache berdasarkan konten file."""
    return pd.read_csv(io.BytesIO(file_bytes))


@st.cache_data(show_spinner="Memproses data...", ttl=3600)
def preprocess_data(df: pd.DataFrame, target_col: str):
    """
    Label encoding + train/test split.
    Menggunakan pd.api.types untuk deteksi kolom string yang aman
    (menghindari Pandas4Warning dari select_dtypes(['object'])).
    """
    df_enc = df.copy()
    encoders = {}

    cat_cols = [
        c for c in df_enc.columns
        if pd.api.types.is_string_dtype(df_enc[c])
        or pd.api.types.is_object_dtype(df_enc[c])
    ]

    for col in cat_cols:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        encoders[col] = le

    X = df_enc.drop(columns=[target_col])
    y = df_enc[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify     = y,
    )
    return X_train, X_test, y_train, y_test, encoders


def run_training(X_train, X_test, y_train, y_test):
    """Latih Decision Tree dan kembalikan model + hasil evaluasi."""
    model = DecisionTreeClassifier(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    cm     = confusion_matrix(y_test, y_pred)
    return model, y_pred, acc, cm


# =============================================================================
# FUNGSI VISUALISASI — PLOTLY (Bebas Segfault, ringan, interaktif)
# =============================================================================

def plot_distribution(df: pd.DataFrame, target_col: str):
    """Bar chart distribusi kelas menggunakan Plotly Express."""
    counts = df[target_col].value_counts().reset_index()
    counts.columns = ['Kelas', 'Jumlah']
    fig = px.bar(
        counts,
        x            = 'Jumlah',
        y            = 'Kelas',
        orientation  = 'h',
        color        = 'Kelas',
        color_discrete_sequence = px.colors.sequential.Teal,
        title        = 'Distribusi Kelas Obesitas',
        labels       = {'Jumlah': 'Jumlah Individu', 'Kelas': 'Tingkat Obesitas'},
    )
    fig.update_layout(
        showlegend      = False,
        plot_bgcolor    = 'rgba(0,0,0,0)',
        paper_bgcolor   = 'rgba(0,0,0,0)',
        font_color      = '#1f2937',
        title_font_size = 14,
        margin          = dict(l=10, r=10, t=40, b=10),
    )
    return fig


def plot_confusion_matrix(cm, labels):
    """Heatmap Confusion Matrix menggunakan Plotly Figure Factory."""
    # Buat anotasi teks untuk setiap sel
    z_text = [[str(val) for val in row] for row in cm.tolist()]
    fig = ff.create_annotated_heatmap(
        z           = cm.tolist(),
        x           = list(labels),
        y           = list(labels),
        annotation_text = z_text,
        colorscale  = 'Blues',
        showscale   = False,
    )
    fig.update_layout(
        title           = 'Confusion Matrix',
        title_font_size = 14,
        xaxis_title     = 'Prediksi Model',
        yaxis_title     = 'Data Aktual',
        plot_bgcolor    = 'rgba(0,0,0,0)',
        paper_bgcolor   = 'rgba(0,0,0,0)',
        font_color      = '#1f2937',
        margin          = dict(l=10, r=10, t=60, b=10),
        yaxis_autorange = 'reversed',
    )
    return fig


# =============================================================================
# HEADER
# =============================================================================
st.markdown(
    '<p class="main-title">Aplikasi Prediksi Tingkat Obesitas 🩺</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="sub-desc">Sistem klasifikasi berbasis <b>Decision Tree</b> '
    'yang mempelajari pola kebiasaan makan dan kondisi fisik dari dataset.</p>',
    unsafe_allow_html=True,
)

# =============================================================================
# SIDEBAR — DATASET
# =============================================================================
with st.sidebar:
    st.header("⚙️ Pengaturan Dataset")
    st.caption("Sistem mencari dataset lokal otomatis. Jika tidak ada, Anda bisa upload secara manual.")

df: Optional[pd.DataFrame] = None

if os.path.exists(DEFAULT_DATASET):
    st.sidebar.success("✅ Dataset lokal ditemukan!")
    try:
        df = load_data(DEFAULT_DATASET)
    except Exception as e:
        st.sidebar.error(f"Gagal membaca dataset: {e}")
else:
    st.sidebar.info("Dataset lokal tidak ditemukan. Silakan upload file CSV.")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Dataset (CSV)", type=["csv"],
        help="Format: CSV dengan kolom NObeyesdad sebagai target",
    )
    if uploaded_file is not None:
        try:
            df = load_uploaded_data(uploaded_file.read())
        except Exception as e:
            st.sidebar.error(f"Gagal membaca file upload: {e}")

# =============================================================================
# MAIN CONTENT
# =============================================================================
if df is None:
    st.info(
        "⏳ Menunggu dataset... Pastikan file "
        "**ObesityDataSet_raw_and_data_sinthetic.csv** "
        "ada di direktori yang sama, atau upload secara manual melalui sidebar."
    )
    st.stop()

if TARGET_COLUMN not in df.columns:
    st.error(
        f"❌ Kolom target **'{TARGET_COLUMN}'** tidak ditemukan dalam dataset. "
        f"Kolom yang tersedia: `{', '.join(df.columns.tolist())}`"
    )
    st.stop()

# --- PRATINJAU DATA ---
st.markdown("---")
st.subheader("📋 Pratinjau Dataset")

with st.expander("Klik untuk melihat sampel data (10 baris pertama)"):
    st.dataframe(df.head(10), width=900)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Baris",    f"{df.shape[0]:,}")
c2.metric("Total Kolom",    str(df.shape[1]))
c3.metric("Kelas Target",   str(df[TARGET_COLUMN].nunique()))
c4.metric("Missing Values", str(int(df.isnull().sum().sum())))

# --- PREPROCESSING ---
try:
    X_train, X_test, y_train, y_test, encoders = preprocess_data(df, TARGET_COLUMN)
except Exception as e:
    st.error(f"❌ Gagal melakukan preprocessing: {e}")
    st.exception(e)
    st.stop()

# --- PELATIHAN MODEL ---
st.markdown("---")
st.subheader("⚙️ Pelatihan & Evaluasi Model")
st.write("Klik tombol di bawah untuk melatih model Machine Learning dengan data yang sudah dimuat.")

if st.button("🚀 Latih Model (Decision Tree)"):
    try:
        with st.spinner("Sedang melatih model... Mohon tunggu."):
            model, y_pred, acc, cm = run_training(X_train, X_test, y_train, y_test)

        st.success("✅ Model berhasil dilatih!")

        m1, m2, m3 = st.columns(3)
        m1.metric(label="🎯 Akurasi Test", value=f"{acc * 100:.2f}%", delta="Sangat Baik")
        m2.metric(label="📊 Data Latih",   value=f"{len(X_train):,}")
        m3.metric(label="🔬 Data Uji",     value=f"{len(X_test):,}")

        # --- VISUALISASI (Plotly — Stabil di Cloud) ---
        st.markdown("---")
        st.subheader("📈 Visualisasi Data & Hasil Evaluasi")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**1. Distribusi Kelas Obesitas**")
            fig_dist = plot_distribution(df, TARGET_COLUMN)
            st.plotly_chart(fig_dist, use_container_width=True)

        with col2:
            st.write("**2. Confusion Matrix**")
            labels = (
                encoders[TARGET_COLUMN].classes_
                if TARGET_COLUMN in encoders
                else sorted(y_test.unique())
            )
            fig_cm = plot_confusion_matrix(cm, labels)
            st.plotly_chart(fig_cm, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat pelatihan: {e}")
        st.exception(e)