# 📈 Hybrid SARIMA–FTS Forecast Dashboard — Waterpark Sumenep Asta Tinggi

Dashboard analitik berbasis **Streamlit** untuk melakukan preprocessing, forecasting, evaluasi, diagnosis error, dan proyeksi masa depan jumlah kunjungan **Waterpark Sumenep (Asta Tinggi)** menggunakan **Hybrid SARIMA–FTS**.

Project ini merupakan implementasi forecasting dalam kegiatan **MBKM Skema Penelitian/Riset**. Dashboard dirancang sebagai **Forecasting Dashboard**: pengguna dapat menggunakan dataset bawaan atau mengunggah dataset sendiri, memilih pembagian data, menjalankan pipeline Hybrid SARIMA–FTS secara kronologis, memeriksa hasil testing, melakukan diagnosis error, dan menyiapkan keluaran penelitian.

> **Catatan metodologis:** lapisan UI/UX berada di atas pipeline preprocessing, SARIMA, FTS, calibration, tuning, ensemble, evaluasi, dan future forecasting. Penyempurnaan tampilan tidak mengganti formula Hybrid SARIMA–FTS.

## 📌 Fitur Utama

### Dataset & Preprocessing
- Upload dataset **CSV, XLSX, atau XLS**.
- Auto-detection kolom periode/tanggal dan jumlah kunjungan.
- Mendukung format dua kolom (**tanggal/periode + jumlah kunjungan**) maupun format tiga kolom seperti Asta Tinggi (**No + Periode + Kunjungan**).
- Parsing tanggal ISO dan nama bulan dalam bahasa Indonesia.
- Nilai **0 dan Missing Value** diproses dengan **Interpolasi Linear** pada posisi yang dapat diinterpolasi.
- Tahun yang tidak tersedia pada sumber tidak dibuat secara sintetis.
- Audit preprocessing tersedia untuk meninjau perubahan data.
- Pembagian data tersedia secara kronologis: **70:30** dan **80:20**.

### Hybrid SARIMA–FTS
- **SARIMA:** `(2,1,0)(1,1,0,12)`.
- **FTS:** Chen Weighted.
- **Window:** `6 → 1` dan `12 → 1`.
- **Validation:** 12 observasi terakhir dari training.
- `n_intervals` FTS ditelusuri pada rentang **5–12**.
- `alpha` ditelusuri pada rentang **0,80–0,99** dengan interval **0,01**.
- Prediksi FTS dikalibrasi menggunakan **Linear Regression**.
- Konfigurasi dipilih pada validation dengan urutan kriteria **MAPE, MAE, RMSE**.
- Testing menggunakan **walk-forward/one-step-ahead** setelah konfigurasi ditetapkan.
- Forecast Hybrid menggunakan:
```text
Hybrid = α × SARIMA + (1 − α) × FTS_calibrated
```

### Evaluasi & Diagnosis
- Metrik **MAE, RMSE, MAPE, dan R²**.
- Perbandingan **SARIMA baseline, FTS terkalibrasi, dan Hybrid**.
- Ringkasan kontribusi/perubahan terhadap baseline.
- Error analysis dan residual diagnostics.
- **Uji Ljung–Box** untuk diagnosis autokorelasi residual.
- Tabel hasil testing dan detail prediksi.

### Forecast Masa Depan & Ekspor
- Forecast masa depan dengan pilihan **Window 6 → 1** atau **12 → 1**.
- Horizon forecast **1–24 periode**.
- Tabel forecast dan visualisasi historis + forecast.
- Download hasil historis, testing, tuning, dan forecast.
- **Complete Research Package** dalam bentuk ZIP untuk kebutuhan dokumentasi/arsip penelitian.

### Antarmuka
- Navigasi 9 tahap.
- **Plus Jakarta Sans** sebagai tipografi utama.
- **Bootstrap 5.3.3** untuk dukungan grid/layout responsif.
- **Bootstrap Icons 1.11.3** untuk ikon antarmuka.
- Desain research dashboard dengan deep navy, muted gold, white surface, spacing lega, dan card yang konsisten.
- Sidebar ringkas, page header bertahap, KPI cards, tabs, chart cards, dan responsive layout.

## 🔬 Alur Dashboard

```text
Upload / Dataset Bawaan
        ↓
Pemeriksaan Struktur Data
        ↓
Preprocessing + Interpolasi Linear
        ↓
Pembagian Data 70:30 / 80:20
        ↓
Validation 12 Observasi
        ↓
SARIMA One-Step-Ahead
        +
FTS Chen Weighted
        ↓
Linear Calibration
        ↓
Tuning n_intervals + alpha
        ↓
Final Walk-Forward Testing
        ↓
Model Comparison
        ↓
Evaluasi MAE, RMSE, MAPE, R²
        ↓
Error / Residual Diagnostics
        ↓
Forecast Masa Depan
        ↓
Research Export
```

## 🧩 Konfigurasi Hybrid

Konfigurasi model aktif pada dashboard:

```text
SARIMA Order       : (2,1,0)
Seasonal Order     : (1,1,0,12)
Seasonal Period    : 12
Validation Size    : 12
FTS Method         : Chen Weighted
Window             : 6 → 1 / 12 → 1
n_intervals        : 5–12
alpha              : 0,80–0,99
```

Formula ensemble:

```text
Hybrid = α × SARIMA + (1 − α) × FTS_calibrated
```

Pemilihan `n_intervals` dan `alpha` dilakukan pada **validation**, sedangkan testing digunakan sebagai evaluasi akhir.

## 📊 Menu Dashboard

Dashboard memiliki 9 menu:

```text
01  Beranda
02  Input Dataset
03  Data Historis
04  Hasil Forecasting
05  Windowing
06  Evaluasi Model
07  Forecast Masa Depan
08  Konfigurasi Model
09  Panduan Aplikasi
```

### 01 · Beranda
Ringkasan workspace penelitian, dataset aktif, workflow analisis, dan preview forecasting.

### 02 · Input Dataset
Upload dataset, pilih split 70:30 atau 80:20, jalankan pipeline, dan lihat audit preprocessing.

### 03 · Data Historis
Eksplorasi rentang waktu, statistik dasar, grafik historis, tabel data, dan download.

### 04 · Hasil Forecasting
Melihat prediksi one-step-ahead dan membandingkan **Aktual, SARIMA, FTS terkalibrasi, dan Hybrid**.

### 05 · Windowing
Meninjau Window 6 → 1 dan Window 12 → 1 secara terpisah melalui tab agar informasi tetap fokus.

### 06 · Evaluasi Model
Menampilkan metrics testing, baseline comparison, model comparison, tuning visualization, error analysis, residual diagnostics, Ljung–Box, dan research export.

### 07 · Forecast Masa Depan
Membangun proyeksi setelah observasi historis terakhir dengan konfigurasi Window yang dipilih dan horizon 1–24 periode.

### 08 · Konfigurasi Model
Menampilkan parameter SARIMA, konfigurasi FTS, alpha, n_intervals, validation, dan ringkasan tuning.

### 09 · Panduan Aplikasi
Panduan penggunaan dashboard, persiapan dataset, informasi program, metodologi, serta interpretasi metrik dan output.

## 🎨 Teknologi & Antarmuka

### Teknologi
- Python
- Pandas
- NumPy
- Statsmodels
- Scikit-learn
- Plotly
- Streamlit
- OpenPyXL / xlrd untuk input Excel

### UI/UX
- Plus Jakarta Sans
- Bootstrap 5.3.3 melalui CDN
- Bootstrap Icons 1.11.3 melalui CDN
- Responsive grid
- KPI cards
- Scenario tabs
- Research-oriented information hierarchy
- Luxury / clean / minimal visual system

Bootstrap dan Bootstrap Icons digunakan sebagai pendukung lapisan antarmuka; keduanya tidak mengubah proses perhitungan model.

## 📁 Struktur Repository

Struktur runtime minimum:

```text
Hybrid-SARIMA-FTS-Waterpark-Sumenep-Asta-Tinggi/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── Validasi Akhir.py
└── Streamlit Data/
    └── data_historis.csv
```

File CSV hasil eksperimen seperti `hasil_testing.csv`, `window_6.csv`, `window_12.csv`, `metrics.csv`, dan `konfigurasi_model.csv` dapat disimpan sebagai **arsip/output tambahan**, tetapi `app.py` terbaru tidak mengunci ketujuh CSV tersebut sebagai dependency runtime.

Dataset `Streamlit Data/data_historis.csv` digunakan sebagai **dataset bawaan** sehingga aplikasi dapat dibuka tanpa upload terlebih dahulu.

## ▶️ Menjalankan Secara Lokal

Buka terminal pada folder repository:

```powershell
pip install -r requirements.txt
```

Validasi project:

```powershell
python "Validasi CSV, Notebook & Dashboard.py"
```

Setelah validasi berhasil:

```powershell
streamlit run app.py
```

Dashboard biasanya dapat diakses melalui:

```text
http://localhost:8501
```

## ☁️ Deployment Streamlit Community Cloud

Repository minimal:

```text
app.py
requirements.txt
README.md
Streamlit Data/data_historis.csv
```

Pada Streamlit Community Cloud:

```text
Repository       : repository GitHub Anda
Branch            : main
Main file path    : app.py
```

## 🧪 Catatan Metodologis

- Split dilakukan **secara kronologis**, bukan random shuffle.
- Validation berada sebelum testing dan digunakan untuk tuning.
- `n_intervals` dan `alpha` dipilih pada validation sebelum konfigurasi digunakan untuk testing.
- Forecast testing menggunakan pendekatan **one-step-ahead / walk-forward**.
- Nilai `0` dan missing yang berada pada posisi yang dapat diinterpolasi diperlakukan sebagai observasi yang perlu ditangani melalui interpolasi linear.
- Untuk dataset Asta Tinggi, tahun **2020 tidak tersedia** pada sumber dan tidak dibuat secara sintetis.
- Future forecast merupakan **proyeksi model**, bukan nilai aktual yang sudah terjadi.
- Audit preprocessing sebaiknya diperiksa sebelum interpretasi hasil.
- Metrik digunakan bersama grafik dan diagnostic agar interpretasi tidak hanya bergantung pada satu indikator.

## ✅ Validasi Project

Script `Validasi Akhir.py` disediakan untuk memeriksa konsistensi:

```text
Struktur project
        ↓
app.py
        ↓
Konfigurasi Hybrid SARIMA–FTS
        ↓
Navigasi dashboard
        ↓
Dataset bawaan
        ↓
README ↔ app.py
        ↓
requirements.txt
        ↓
Output CSV opsional
        ↓
Notebook opsional
```

Validator tidak mengunci satu angka MAE/RMSE/MAPE/R² karena hasil dapat berubah sesuai dataset dan pembagian data yang dipilih pengguna.

## 👤 Project

**Hybrid SARIMA–FTS Forecast Dashboard — Waterpark Sumenep Asta Tinggi**

Bagian dari kegiatan:

**MBKM Skema Penelitian/Riset**

Topik penelitian:

**Pengembangan Model Hibrida CEEMDAN–GRU dengan Optimasi Bayesian untuk Peramalan Kunjungan Wisatawan dalam Mendukung Pengelolaan Pariwisata Berkelanjutan**

Dashboard ini digunakan sebagai media implementasi dan dokumentasi eksperimen **Hybrid SARIMA–FTS** pada data kunjungan Waterpark Sumenep (Asta Tinggi).

## 📄 Lisensi

Project digunakan untuk kebutuhan akademik, dokumentasi penelitian, dan demonstrasi implementasi.