# =============================================================================
# Aplikasi Prediksi Tingkat Obesitas
# Production-Ready Version | Streamlit Community Cloud Compatible
# =============================================================================

# --- IMPORTS ---
# matplotlib.use('Agg') HARUS dipanggil sebelum import pyplot untuk mencegah
# Segmentation Fault pada server headless (no display environment)
import matplotlib
matplotlib.use('Agg')

import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Suppress semua FutureWarning dan DeprecationWarning dari library pihak ketiga
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# =============================================================================
# KONSTANTA - Diletakkan di atas agar mudah dikonfigurasi
# =============================================================================
DEFAULT_DATASET = 'ObesityDataSet_raw_and_data_sinthetic.csv'
TARGET_COLUMN   = 'NObeyesdad'
TEST_SIZE       = 0.2
RANDOM_STATE    = 42

# =============================================================================
# KONFIGURASI HALAMAN - WAJIB baris pertama setelah import
# =============================================================================
st.set_page_config(
    page_title = "Klasifikasi Tingkat Obesitas",
    page_icon  = "🩺",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# =============================================================================
# CSS - Dipindahkan ke satu blok terpusat untuk maintainability
# =============================================================================
CUSTOM_CSS = """
<style>
    /* Sembunyikan branding Streamlit bawaan */
    #MainMenu {visibility: hidden;}
    footer     {visibility: hidden;}
    header     {visibility: hidden;}

    /* --- Light Mode Background --- */
    .stApp {
        background: linear-gradient(135deg, #ffffff 0%, #c1dfc4 100%);
        background-attachment: fixed;
    }
    /* Sidebar glassmorphism - Light */
    [data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.45) !important;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-right: 1px solid rgba(255, 255, 255, 0.3);
    }

    /* --- Dark Mode Background --- */
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

    /* --- Judul Utama (Light Mode) --- */
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

    /* --- Tombol Hijau Modern --- */
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

    /* --- Metric Value Hijau --- */
    [data-testid="stMetricValue"] {
        color: #059669 !important;
        font-weight: 800;
        font-size: 2.4rem !important;
    }

    /* --- Card wrapper tipis untuk section --- */
    .section-card {
        background: rgba(255, 255, 255, 0.35);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.4);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =============================================================================
# FUNGSI - Dipisahkan dari logika UI untuk separation of concerns
# =============================================================================

@st.cache_data(show_spinner="Memuat dataset...", ttl=3600)
def load_data(path: str) -> pd.DataFrame:
    """
    Memuat CSV dengan caching. @st.cache_data memastikan file hanya dibaca
    sekali dari disk, lalu disimpan di memori untuk request berikutnya.
    ttl=3600 berarti cache akan di-refresh setiap 1 jam sekali secara otomatis.
    """
    return pd.read_csv(path)


@st.cache_data(show_spinner="Memuat dataset yang di-upload...", ttl=3600)
def load_uploaded_data(uploaded_bytes: bytes) -> pd.DataFrame:
    """
    Terpisah dari load_data agar hash key berbeda untuk file upload.
    Menerima bytes agar bisa di-cache dengan benar oleh Streamlit.
    """
    import io
    return pd.read_csv(io.BytesIO(uploaded_bytes))


@st.cache_data(show_spinner=False)
def preprocess_data(df: pd.DataFrame, target_col: str):
    """
    Melakukan label encoding dan split data.
    Di-cache terpisah dari pelatihan model agar preprocessing tidak
    diulang setiap kali pengguna klik tombol.

    Menggunakan pd.api.types untuk deteksi kolom teks yang aman dan tidak
    menghasilkan Pandas4Warning (lebih akurat dari select_dtypes(['object'])).
    """
    df_encoded = df.copy()
    encoders = {}

    # Deteksi kolom kategorikal secara eksplisit menggunakan API publik Pandas
    # Ini menghindari Pandas4Warning terkait 'object' dtype yang deprecated
    cat_cols = [
        col for col in df_encoded.columns
        if pd.api.types.is_string_dtype(df_encoded[col])
        or pd.api.types.is_object_dtype(df_encoded[col])
    ]

    for col in cat_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        encoders[col] = le

    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test, encoders


@st.cache_resource(show_spinner=False)
def train_model(X_train_hash: int, y_train_hash: int):
    """
    DUMMY SIGNATURE: st.cache_resource tidak bisa meng-cache ndarray langsung.
    Solusi: gunakan cache di level pemanggil dengan hash custom.

    Catatan: @st.cache_resource cocok untuk objek berat (model ML) yang
    ingin di-share antar session tanpa deserialisasi ulang (berbeda dari
    cache_data yang di-pickle setiap session).
    """
    pass


def run_training(X_train, X_test, y_train, y_test) -> tuple:
    """
    Melatih model Decision Tree dan mengembalikan hasil evaluasi.
    Dipisahkan ke fungsi sendiri untuk testability.
    """
    model = DecisionTreeClassifier(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc   = accuracy_score(y_test, y_pred)
    cm    = confusion_matrix(y_test, y_pred)
    return model, y_pred, acc, cm


def create_distribution_figure(df: pd.DataFrame, target_col: str) -> plt.Figure:
    """
    Membuat figure distribusi kelas. Figur HARUS ditutup setelah di-render
    oleh st.pyplot() untuk mencegah memory leak (matplotlib tidak punya
    garbage collector otomatis untuk figure objects).
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    order   = df[target_col].value_counts().index
    sns.countplot(
        y      = df[target_col],
        hue    = df[target_col],
        ax     = ax,
        palette= "mako",
        order  = order,
        legend = False,
    )
    ax.set_xlabel("Jumlah Individu", fontsize=10)
    ax.set_ylabel("Tingkat Obesitas", fontsize=10)
    ax.set_title("Distribusi Kelas Target", fontsize=12, fontweight='bold')
    sns.despine(fig=fig, ax=ax)
    fig.tight_layout()
    return fig


def create_confusion_matrix_figure(cm, labels) -> plt.Figure:
    """
    Membuat figure Confusion Matrix menggunakan ConfusionMatrixDisplay
    (API sklearn terbaru) yang lebih akurat daripada heatmap manual.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap='Blues', colorbar=False, xticks_rotation='vertical')
    ax.set_title("Confusion Matrix", fontsize=12, fontweight='bold')
    ax.set_xlabel("Prediksi Model", fontsize=10)
    ax.set_ylabel("Data Aktual", fontsize=10)
    fig.tight_layout()
    return fig


# =============================================================================
# HEADER APLIKASI
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
# SIDEBAR - PENGATURAN DATASET
# =============================================================================
with st.sidebar:
    st.header("⚙️ Pengaturan Dataset")
    st.caption("Sistem mencari dataset lokal otomatis. Jika tidak ada, Anda bisa upload secara manual.")

df: pd.DataFrame | None = None

if os.path.exists(DEFAULT_DATASET):
    st.sidebar.success("✅ Dataset lokal ditemukan!")
    try:
        df = load_data(DEFAULT_DATASET)
    except Exception as e:
        st.sidebar.error(f"Gagal membaca dataset: {e}")
else:
    st.sidebar.info("Dataset lokal tidak ditemukan. Silakan upload file CSV.")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Dataset (CSV)", type=["csv"], help="Format: CSV dengan kolom NObeyesdad sebagai target"
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
    st.info("⏳ Menunggu dataset... Pastikan file **ObesityDataSet_raw_and_data_sinthetic.csv** "
            "ada di direktori yang sama, atau upload secara manual melalui sidebar.")
    st.stop()  # Hentikan eksekusi di sini, tidak perlu blok else raksasa

# --- VALIDASI KOLOM TARGET ---
if TARGET_COLUMN not in df.columns:
    st.error(f"❌ Kolom target **'{TARGET_COLUMN}'** tidak ditemukan dalam dataset. "
             f"Kolom yang tersedia: `{', '.join(df.columns.tolist())}`")
    st.stop()

# --- PRATINJAU DATA ---
st.markdown("---")
st.subheader("📋 Pratinjau Dataset")

with st.expander("Klik untuk melihat sampel data (10 baris pertama)"):
    st.dataframe(df.head(10), use_container_width=True)

# Statistik ringkas dataset ditampilkan sebagai metric cards
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Baris",   f"{df.shape[0]:,}")
c2.metric("Total Kolom",   str(df.shape[1]))
c3.metric("Kelas Target",  str(df[TARGET_COLUMN].nunique()))
c4.metric("Missing Values", str(df.isnull().sum().sum()))

# --- PREPROCESSING (CACHED) ---
try:
    X_train, X_test, y_train, y_test, encoders = preprocess_data(df, TARGET_COLUMN)
except Exception as e:
    st.error(f"❌ Gagal melakukan preprocessing: {e}")
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

        # Tampilkan metrik utama
        m1, m2, m3 = st.columns(3)
        m1.metric(label="🎯 Akurasi Test",  value=f"{acc * 100:.2f}%",  delta="Sangat Baik")
        m2.metric(label="📊 Data Latih",    value=f"{len(X_train):,}")
        m3.metric(label="🔬 Data Uji",      value=f"{len(X_test):,}")

        # --- VISUALISASI ---
        st.markdown("---")
        st.subheader("📈 Visualisasi Data & Hasil Evaluasi")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**1. Distribusi Kelas Obesitas**")
            fig1 = create_distribution_figure(df, TARGET_COLUMN)
            st.pyplot(fig1, use_container_width=True)
            plt.close(fig1)  # KRITIS: Tutup figure untuk bebaskan memori (cegah memory leak)

        with col2:
            st.write("**2. Confusion Matrix**")
            # Dapatkan label asli dari encoder target untuk confusion matrix yang terbaca
            if TARGET_COLUMN in encoders:
                labels = encoders[TARGET_COLUMN].classes_
            else:
                labels = sorted(y_test.unique())

            fig2 = create_confusion_matrix_figure(cm, labels)
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)  # KRITIS: Tutup figure untuk bebaskan memori

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat pelatihan: {e}")
        st.exception(e)  # Tampilkan traceback lengkap untuk debugging di development