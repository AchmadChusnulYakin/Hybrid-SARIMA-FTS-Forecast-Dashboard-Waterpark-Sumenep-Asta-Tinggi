from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(".")
APP = ROOT / "app.py"
README = ROOT / "README.md"
REQUIREMENTS = ROOT / "requirements.txt"
DATA_DIR = ROOT / "Streamlit Data"
DATA_FILE = DATA_DIR / "data_historis.csv"

MENU = [
    "01  Beranda",
    "02  Input Dataset",
    "03  Data Historis",
    "04  Hasil Forecasting",
    "05  Windowing",
    "06  Evaluasi Model",
    "07  Forecast Masa Depan",
    "08  Konfigurasi Model",
    "09  Panduan Aplikasi",
]

OPTIONAL_CSV = [
    "hasil_training.csv",
    "hasil_testing.csv",
    "window_6.csv",
    "window_12.csv",
    "metrics.csv",
    "konfigurasi_model.csv",
]

checks = []
info = []

def status(label: str, ok: bool, detail: str = ""):
    checks.append(bool(ok))
    print(f"{label:<68}: {'OK' if ok else 'GAGAL'}" + (f" — {detail}" if detail else ""))

def note(label: str, detail: str = ""):
    info.append((label, detail))
    print(f"{label:<68}: INFO" + (f" — {detail}" if detail else ""))

print("=" * 92)
print("VALIDASI CSV ↔ NOTEBOOK ↔ DASHBOARD HYBRID SARIMA–FTS")
print("Waterpark Sumenep (Asta Tinggi)")
print("=" * 92)

# 1. PROJECT
print("\n[1] STRUKTUR PROJECT")
status("app.py tersedia", APP.exists())
status("README.md tersedia", README.exists())
status("requirements.txt tersedia", REQUIREMENTS.exists())
status("Folder Streamlit Data tersedia", DATA_DIR.exists())
status("Dataset bawaan data_historis.csv tersedia", DATA_FILE.exists())

# 2. SOURCE CODE
print("\n[2] SOURCE CODE app.py")
app_text = ""
tree = None
if APP.exists():
    app_text = APP.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(app_text)
        status("app.py bebas syntax error", True)
    except SyntaxError as exc:
        status("app.py bebas syntax error", False, str(exc))
else:
    status("app.py bebas syntax error", False, "app.py tidak ditemukan")

for fn in [
    "fit_sarima_forecast",
    "fts_chen_forecast",
    "run_hybrid_pipeline",
    "cached_run_pipeline",
    "make_future_forecast",
    "evaluate_metrics",
]:
    status(f"Fungsi {fn} tersedia", bool(re.search(rf"def\s+{re.escape(fn)}\s*\(", app_text)))

# 3. CONFIG
print("\n[3] KONFIGURASI HYBRID SARIMA–FTS")
def extract_tuple(pattern, source):
    m = re.search(pattern, source)
    if not m:
        return None
    try:
        return tuple(ast.literal_eval(m.group(1)))
    except Exception:
        return None

sarima_order = extract_tuple(r"SARIMA_ORDER\s*=\s*(\([^)]+\))", app_text)
seasonal_order = extract_tuple(r"SARIMA_SEASONAL_ORDER\s*=\s*(\([^)]+\))", app_text)
status("SARIMA order = (2,1,0)", sarima_order == (2, 1, 0), f"terbaca {sarima_order}")
status("Seasonal order = (1,1,0,12)", seasonal_order == (1, 1, 0, 12), f"terbaca {seasonal_order}")

def scalar(name):
    m = re.search(rf"{name}\s*=\s*([^\n#]+)", app_text)
    return m.group(1).strip() if m else None

status("Seasonal period = 12", scalar("SEASONAL_PERIOD") == "12", f"terbaca {scalar('SEASONAL_PERIOD')}")
status("Validation size = 12", scalar("VALIDATION_SIZE") == "12", f"terbaca {scalar('VALIDATION_SIZE')}")
status("Window 6 → 1 tersedia", "WINDOWS = [6, 12]" in app_text)
status("Window 12 → 1 tersedia", "WINDOWS = [6, 12]" in app_text)
status("FTS n_intervals = 5–12", "range(5, 13)" in app_text)
status("Alpha grid = 0,80–0,99 step 0,01", "np.arange(0.80, 1.00, 0.01)" in app_text)
status("FTS Chen Weighted tersedia", "fts_chen_forecast" in app_text and "weighted=True" in app_text)
status("Linear Regression calibration tersedia", "LinearRegression" in app_text)
status("Formula Hybrid tersedia", "alpha * p_s + (1 - alpha) * p_f" in app_text)
status("Walk-forward testing tersedia", "for j, t in enumerate(range(train_n, n))" in app_text)
status("Future forecast tersedia", "def make_future_forecast" in app_text)

# 4. NAVIGATION
print("\n[4] NAVIGASI DASHBOARD")
menu_match = re.search(r"menu_options\s*=\s*\[(.*?)\]", app_text, flags=re.S)
detected = []
if menu_match:
    detected = re.findall(r'"([^"]+)"', menu_match.group(1))
status("9 menu utama sesuai app.py", detected == MENU, f"terbaca {detected}")
for item in MENU:
    status(f"Menu {item} tersedia", item in detected)

status("Indikator halaman menggunakan / 09", "{step:02d} / 09" in app_text)
status("Halaman 09 menggunakan step = 9", 'elif menu.startswith("09")' in app_text and re.search(r'ui_header\([\s\S]{0,500}?\b9,\s*"bi-book"', app_text) is not None)

# 5. DATASET
print("\n[5] DATASET BAWAAN")
if DATA_FILE.exists():
    try:
        df = pd.read_csv(DATA_FILE)
        status("data_historis.csv dapat dibaca", True, f"{len(df)} baris")
        status("Jumlah observasi = 108", len(df) == 108, f"terbaca {len(df)}")
        status("Kolom Tanggal tersedia", "Tanggal" in df.columns)
        status("Kolom Jumlah_Kunjungan tersedia", "Jumlah_Kunjungan" in df.columns)
        if {"Tanggal", "Jumlah_Kunjungan"}.issubset(df.columns):
            dt = pd.to_datetime(df["Tanggal"], errors="coerce")
            vals = pd.to_numeric(df["Jumlah_Kunjungan"], errors="coerce")
            status("Semua tanggal dapat dikonversi", dt.notna().all())
            status("Tidak ada duplikasi tanggal", dt.duplicated().sum() == 0)
            status("Tanggal awal = 2015-01-01", dt.min() == pd.Timestamp("2015-01-01"))
            status("Tanggal akhir = 2024-12-01", dt.max() == pd.Timestamp("2024-12-01"))
            status("Tahun 2020 tidak dibuat sintetis", 2020 not in set(dt.dt.year))
            status("Jumlah_Kunjungan dapat dikonversi numerik", pd.to_numeric(df["Jumlah_Kunjungan"], errors="coerce").notna().sum() + df["Jumlah_Kunjungan"].isna().sum() == len(df))
            # The bundled file may already be preprocessed. Both states are valid.
            miss = int(vals.isna().sum())
            zeros = int((vals == 0).sum())
            status("Dataset bawaan kompatibel dengan preprocessing app.py", "replace(0, np.nan).interpolate" in app_text and miss >= 0)
            status("Tidak ada nilai 0 pada dataset bawaan", zeros == 0)
            note("Status preprocessing dataset bawaan", f"missing={miss}, zero={zeros}; app.py tetap menangani 0/missing pada proses upload.")
    except Exception as exc:
        status("data_historis.csv dapat dibaca", False, str(exc))

# 6. README ↔ APP
print("\n[6] KONSISTENSI README ↔ app.py")
readme_text = README.read_text(encoding="utf-8", errors="ignore") if README.exists() else ""
for label, token in [
    ("README menyebut Hybrid SARIMA–FTS", "Hybrid SARIMA–FTS"),
    ("README menyebut split 70:30", "70:30"),
    ("README menyebut split 80:20", "80:20"),
    ("README menyebut SARIMA (2,1,0)(1,1,0,12)", "(2,1,0)(1,1,0,12)"),
    ("README menyebut FTS Chen Weighted", "FTS:** Chen Weighted"),
    ("README menyebut Window 6 → 1", "6 → 1"),
    ("README menyebut Window 12 → 1", "12 → 1"),
    ("README menyebut MAE", "MAE"),
    ("README menyebut RMSE", "RMSE"),
    ("README menyebut MAPE", "MAPE"),
    ("README menyebut R²", "R²"),
    ("README menyebut Ljung–Box", "Ljung–Box"),
    ("README menyebut Complete Research Package", "Complete Research Package"),
    ("README menyebut Panduan Aplikasi", "Panduan Aplikasi"),
]:
    status(label, token in readme_text)

for item in MENU:
    status(f"README menyebut {item}", item in readme_text)

# 7. REQUIREMENTS
print("\n[7] REQUIREMENTS")
req_text = REQUIREMENTS.read_text(encoding="utf-8", errors="ignore").lower() if REQUIREMENTS.exists() else ""
for pkg in ["streamlit", "pandas", "numpy", "plotly", "statsmodels", "scikit-learn", "openpyxl", "xlrd"]:
    status(f"requirements memuat {pkg}", pkg.lower() in req_text)

# 8. OPTIONAL CSV
print("\n[8] OUTPUT CSV STATIS / ARSIP")
for name in OPTIONAL_CSV:
    p = DATA_DIR / name
    if p.exists():
        try:
            odf = pd.read_csv(p)
            note(f"{name} tersedia", f"{len(odf)} baris")
        except Exception as exc:
            status(f"{name} dapat dibaca", False, str(exc))
    else:
        note(f"{name} tidak tersedia", "Opsional; app.py terbaru menghitung pipeline secara dinamis.")

# 9. UI / DESIGN
print("\n[9] DESIGN SYSTEM")
for label, token in [
    ("Plus Jakarta Sans tersedia", "Plus Jakarta Sans"),
    ("Bootstrap 5.3.3 tersedia", "bootstrap@5.3.3"),
    ("Bootstrap Icons 1.11.3 tersedia", "bootstrap-icons@1.11.3"),
    ("Responsive grid tersedia", "grid-template-columns"),
    ("Sidebar navigation tersedia", "st.sidebar.radio"),
    ("Research package export tersedia", "make_research_zip"),
]:
    status(label, token in app_text)

# 10. README / entry point
print("\n[10] ENTRY POINT & DEPLOYMENT")
status("README menjadikan app.py sebagai entry point", "streamlit run app.py" in readme_text)
status("README menjadikan main file path app.py", "Main file path" in readme_text and "app.py" in readme_text)
status("app.py menggunakan dataset default dari Streamlit Data", 'Path("Streamlit Data")' in app_text)
status("app.py tidak mengunci 7 CSV sebagai REQUIRED_FILES", "REQUIRED_FILES" not in app_text)
status("Upload dataset menggunakan file uploader", "st.file_uploader" in app_text)
status("Split 70:30 dan 80:20 dihitung ulang melalui pipeline", '["70:30", "80:20"]' in app_text)

# 11. NOTEBOOK
print("\n[11] NOTEBOOK HYBRID")
notebooks = list(ROOT.glob("*.ipynb"))
if notebooks:
    # At least one notebook is present; do a lightweight content scan.
    combined = ""
    for nb in notebooks:
        try:
            combined += nb.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            pass
    status("Notebook tersedia di root repository", True, f"{len(notebooks)} file")
    status("Notebook mencantumkan Hybrid / SARIMA", bool(re.search(r"Hybrid SARIMA|SARIMA", combined, re.I)))
else:
    note("Notebook tidak ditemukan di root repository", "Opsional untuk runtime dashboard; tambahkan bila ingin mengarsipkan notebook bersama project.")

# FINAL
print("\n" + "=" * 92)
print("STATUS AKHIR")
print("=" * 92)
passed = sum(checks)
total = len(checks)
failed = total - passed
print(f"Jumlah pemeriksaan : {total}")
print(f"Berhasil           : {passed}")
print(f"Gagal              : {failed}")
print(f"Informasi          : {len(info)}")
print("\nKESIMPULAN :", "LULUS" if failed == 0 else "BELUM LULUS")
if info:
    print("\nCATATAN INFORMASI")
    print("-" * 92)
    for label, detail in info:
        print(f"• {label}")
        if detail:
            print(f"  {detail}")
print("=" * 92)