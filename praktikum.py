# =============================================================================
# Aplikasi Prediksi Tingkat Obesitas
# Streamlit Community Cloud — Stable Build
# Charting: Plotly Express only (no matplotlib, no seaborn)
# =============================================================================

import io
import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# =============================================================================
# KONSTANTA
# =============================================================================
DEFAULT_DATASET = "ObesityDataSet_raw_and_data_sinthetic.csv"
TARGET_COLUMN   = "NObeyesdad"
TEST_SIZE       = 0.2
RANDOM_STATE    = 42

# =============================================================================
# KONFIGURASI HALAMAN
# =============================================================================
st.set_page_config(
    page_title            = "Klasifikasi Tingkat Obesitas",
    page_icon             = "🩺",
    layout                = "wide",
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

    .stApp {
        background: linear-gradient(135deg, #ffffff 0%, #c1dfc4 100%);
        background-attachment: fixed;
    }
    [data-testid="stSidebar"] {
        background-color: rgba(255,255,255,0.5) !important;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
    }
    @media (prefers-color-scheme: dark) {
        .stApp {
            background: linear-gradient(135deg, #0d1f14 0%, #0f3424 55%, #115937 100%) !important;
        }
        [data-testid="stSidebar"] {
            background-color: rgba(0,0,0,0.35) !important;
        }
        .main-title {
            background: -webkit-linear-gradient(45deg, #a8e063, #e0ffe0) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
        }
    }
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #047857, #10B981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        line-height: 1.2;
    }
    .sub-desc { color: #374151; font-size: 1rem; margin-bottom: 1rem; }
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
        box-shadow: 0 6px 16px rgba(16,185,129,0.45);
        color: white !important;
    }
    [data-testid="stMetricValue"] {
        color: #059669 !important;
        font-weight: 800;
        font-size: 2.4rem !important;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# FUNGSI — LOAD DATA (cache_data dengan tipe primitif sebagai key)
# =============================================================================

@st.cache_data(show_spinner="Memuat dataset...")
def load_csv_from_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner="Memuat dataset...")
def load_csv_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


# =============================================================================
# FUNGSI — PREPROCESSING (FIX: terima JSON string, bukan parquet/DataFrame)
# JSON dipilih karena: aman, tidak butuh pyarrow, kompatibel di semua env
# =============================================================================

@st.cache_data(show_spinner="Memproses data...")
def encode_and_split(df_json: str, target_col: str):
    """
    Menerima JSON string agar @st.cache_data bisa hash dengan aman.
    PERBAIKAN: tidak lagi menggunakan parquet (butuh pyarrow).
    """
    df = pd.read_json(io.StringIO(df_json))

    encoders: dict = {}
    df_enc = df.copy()

    # Gunakan is_string_dtype + is_object_dtype untuk hindari Pandas4Warning
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


# =============================================================================
# FUNGSI — TRAINING
# PERBAIKAN: Tidak menggunakan @st.cache_resource (tidak bisa hash numpy array).
# Hasil evaluasi disimpan di st.session_state untuk persistensi antar re-run.
# =============================================================================

def train_decision_tree(X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    cm     = confusion_matrix(y_test, y_pred)
    return float(acc), cm.tolist()


# =============================================================================
# FUNGSI VISUALISASI — Plotly Express (tidak butuh scipy/C extension)
# =============================================================================

def chart_distribution(df: pd.DataFrame, target_col: str):
    vc = df[target_col].value_counts().reset_index()
    vc.columns = ["Kelas", "Jumlah"]

    fig = px.bar(
        vc, x="Jumlah", y="Kelas", orientation="h",
        color="Kelas",
        color_discrete_sequence=px.colors.sequential.Teal,
        title="Distribusi Kelas Obesitas",
        labels={"Jumlah": "Jumlah Individu", "Kelas": "Tingkat Obesitas"},
    )
    fig.update_layout(
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        title_font_size=14,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def chart_confusion_matrix(cm_list: list, labels):
    """px.imshow tidak butuh scipy/figure_factory — bebas Segfault."""
    cm_arr     = np.array(cm_list)
    str_labels = [str(lbl) for lbl in labels]

    fig = px.imshow(
        cm_arr,
        x=str_labels, y=str_labels,
        color_continuous_scale="Blues",
        text_auto=True,
        title="Confusion Matrix",
        labels=dict(x="Prediksi Model", y="Data Aktual", color="Jumlah"),
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        title_font_size=14,
        margin=dict(l=0, r=0, t=60, b=0),
        coloraxis_showscale=False,
    )
    fig.update_xaxes(side="bottom")
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
    "yang mempelajari pola kebiasaan makan dan kondisi fisik dari dataset.</p>",
    unsafe_allow_html=True,
)

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.header("⚙️ Pengaturan Dataset")
    st.caption("Sistem mencari dataset lokal otomatis.")

df: Optional[pd.DataFrame] = None

if os.path.exists(DEFAULT_DATASET):
    st.sidebar.success("✅ Dataset lokal ditemukan!")
    try:
        df = load_csv_from_path(DEFAULT_DATASET)
    except Exception as e:
        st.sidebar.error(f"Gagal membaca dataset: {e}")
else:
    st.sidebar.info("Dataset tidak ditemukan. Silakan upload file CSV.")
    uploaded = st.sidebar.file_uploader(
        "Upload Dataset (CSV)", type=["csv"],
        help="Harus memiliki kolom 'NObeyesdad' sebagai target",
    )
    if uploaded is not None:
        try:
            df = load_csv_from_bytes(uploaded.read())
        except Exception as e:
            st.sidebar.error(f"Gagal membaca file: {e}")

# =============================================================================
# MAIN
# =============================================================================
if df is None:
    st.info(
        "⏳ Menunggu dataset... Pastikan file "
        "**ObesityDataSet_raw_and_data_sinthetic.csv** ada di repositori."
    )
    st.stop()

if TARGET_COLUMN not in df.columns:
    st.error(
        f"❌ Kolom **'{TARGET_COLUMN}'** tidak ditemukan. "
        f"Kolom tersedia: `{', '.join(df.columns.tolist())}`"
    )
    st.stop()

# --- PRATINJAU DATA ---
st.markdown("---")
st.subheader("📋 Pratinjau Dataset")

with st.expander("Klik untuk melihat 10 baris pertama"):
    st.dataframe(df.head(10))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Baris",    f"{df.shape[0]:,}")
c2.metric("Total Kolom",    str(df.shape[1]))
c3.metric("Kelas Target",   str(df[TARGET_COLUMN].nunique()))
c4.metric("Missing Values", str(int(df.isnull().sum().sum())))

# --- PREPROCESSING ---
try:
    df_json = df.to_json()
    X_train, X_test, y_train, y_test, encoders = encode_and_split(df_json, TARGET_COLUMN)
except Exception as e:
    st.error(f"❌ Gagal preprocessing: {e}")
    st.exception(e)
    st.stop()

# --- PELATIHAN ---
st.markdown("---")
st.subheader("⚙️ Pelatihan & Evaluasi Model")
st.write("Klik tombol di bawah untuk melatih model.")

if st.button("🚀 Latih Model (Decision Tree)"):
    try:
        with st.spinner("Melatih model..."):
            acc, cm_list = train_decision_tree(X_train, X_test, y_train, y_test)

        # Simpan hasil ke session_state agar tidak hilang saat re-run
        st.session_state["acc"]     = acc
        st.session_state["cm_list"] = cm_list
        st.session_state["trained"] = True

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat training: {e}")
        st.exception(e)

# Tampilkan hasil jika sudah pernah dilatih (persistent across reruns)
if st.session_state.get("trained"):
    acc     = st.session_state["acc"]
    cm_list = st.session_state["cm_list"]

    st.success("✅ Model berhasil dilatih!")

    m1, m2, m3 = st.columns(3)
    m1.metric("🎯 Akurasi Test", f"{acc * 100:.2f}%", delta="Sangat Baik")
    m2.metric("📊 Data Latih",   f"{len(X_train):,}")
    m3.metric("🔬 Data Uji",     f"{len(X_test):,}")

    st.markdown("---")
    st.subheader("📈 Visualisasi Data & Hasil Evaluasi")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**1. Distribusi Kelas Obesitas**")
        fig_dist = chart_distribution(df, TARGET_COLUMN)
        st.plotly_chart(fig_dist, key="chart_dist")

    with col2:
        st.write("**2. Confusion Matrix**")
        labels = (
            encoders[TARGET_COLUMN].classes_
            if TARGET_COLUMN in encoders
            else sorted(set(y_test.tolist()))
        )
        fig_cm = chart_confusion_matrix(cm_list, labels)
        st.plotly_chart(fig_cm, key="chart_cm")