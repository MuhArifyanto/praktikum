# Aplikasi Klasifikasi Tingkat Obesitas 🩺

Aplikasi web interaktif berbasis Streamlit ini memprediksi tingkat obesitas seseorang berdasarkan data kebiasaan makan dan kondisi fisiknya.

## Fitur Utama
- **Prediksi Otomatis**: Menggunakan algoritma *Machine Learning* **Decision Tree** untuk melakukan klasifikasi tingkat obesitas.
- **Data Skala Besar**: Menggunakan dataset `ObesityDataSet_expanded.csv` yang berisi lebih dari 6.000 baris data untuk memenuhi standar akurasi yang tinggi (Akurasi *Test* > 90%).
- **Visualisasi Interaktif**: Menampilkan distribusi kelas pada data dan evaluasi prediksi (*Confusion Matrix*).
- **Fleksibel**: Pengguna bisa memakai dataset bawaan ataupun mengunggah (upload) file CSV milik sendiri langsung dari antarmuka web.

## Cara Menjalankan Secara Lokal (Localhost)
1. Clone repository ini ke komputer Anda.
2. Instal semua dependensi yang diperlukan dengan menjalankan perintah:
   ```bash
   pip install -r requirements.txt
   ```
3. Jalankan aplikasi menggunakan Streamlit:
   ```bash
   streamlit run praktikum.py
   ```
4. Buka tautan lokal yang muncul di terminal (biasanya `http://localhost:8501`) melalui browser Anda.

## Deploy ke Streamlit Cloud
Repository ini juga sudah disiapkan agar siap di-*deploy* secara langsung (gratis) menggunakan [Streamlit Community Cloud](https://share.streamlit.io/). Hanya perlu menghubungkan repository GitHub ini, lalu arahkan ke file utama `praktikum.py`.
