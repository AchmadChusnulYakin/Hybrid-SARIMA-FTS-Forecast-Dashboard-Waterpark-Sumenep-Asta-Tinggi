import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Hybrid SARIMA–FTS Forecast Dashboard - Waterpark Sumenep Asta Tinggi",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path("Streamlit Data")
DEFAULT_DATA_FILE = DATA_DIR / "data_historis.csv"

# ============================================================
# GLOBAL CONFIGURATION — aligned with final V7 methodology
# ============================================================
SARIMA_ORDER = (2, 1, 0)
SARIMA_SEASONAL_ORDER = (1, 1, 0, 12)
SEASONAL_PERIOD = 12
VALIDATION_SIZE = 12
WINDOWS = [6, 12]
FTS_INTERVAL_GRID = list(range(5, 13))
ALPHA_GRID = np.round(np.arange(0.80, 1.00, 0.01), 2)
MIN_SARIMA_OBS = 24
MAXITER = 200

# ============================================================
# SINGLE DESIGN SYSTEM
# ============================================================
# ============================================================
def fmt_num(value, digits=0):
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_month(dt):
    if pd.isna(dt):
        return "—"
    months = [
        "Januari", "Februari", "Maret", "April", "Mei", "Juni",
        "Juli", "Agustus", "September", "Oktober", "November", "Desember",
    ]
    return f"{months[dt.month - 1]} {dt.year}"


def add_kpi(label, value, help_text="", icon="bi-graph-up"):
    st.markdown(
        f"""
        <div class='kpi-card'>
            <div class='kpi-icon'><i class='bi {icon}'></i></div>
            <div class='kpi-label'>{label}</div>
            <div class='kpi-value'>{value}</div>
            <div class='kpi-help'>{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(eyebrow, title, description, step):
    st.markdown(
        f"""
        <div class='page-header'>
            <div>
                <div class='page-eyebrow'>{eyebrow}</div>
                <h1 class='page-title'>{title}</h1>
                <div class='page-desc'>{description}</div>
            </div>
            <div class='page-step'>{step:02d} / 09</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title, subtitle=None, icon=None):
    icon_html = f"<i class='bi {icon}' style='color:#1f5f97;margin-right:7px;'></i>" if icon else ""
    sub = f"<div class='section-subtitle'>{subtitle}</div>" if subtitle else ""
    st.markdown(
        f"<div class='section-head'><div><div class='section-title'>{icon_html}{title}</div>{sub}</div></div>",
        unsafe_allow_html=True,
    )


def workflow_card(number, icon, title, text):
    return f"""
    <div class='workflow-card'>
        <div class='workflow-num'>Tahap {number:02d}</div>
        <div class='workflow-icon'><i class='bi {icon}'></i></div>
        <div class='workflow-title'>{title}</div>
        <div class='feature-text'>{text}</div>
    </div>
    """


def chart_layout(fig, title=None, height=430):
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=14, r=14, t=54 if title else 18, b=16),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        hovermode="x unified",
        colorway=["#183d65", "#2c75a3", "#8276f5", "#7d8ea3", "#3f8f73"],
        font=dict(family="Plus Jakarta Sans, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif", color="#31435b", size=11),
        title_font=dict(family="Plus Jakarta Sans, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif", color="#13243c", size=16),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=10)),
        xaxis=dict(showgrid=False, linecolor="#e4eaf1", tickfont=dict(size=10)),
        yaxis=dict(gridcolor="#edf1f5", zeroline=False, linecolor="#e4eaf1", tickfont=dict(size=10)),
    )
    return fig


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def evaluate_metrics(y_true, y_pred):
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[mask]
    yp = yp[mask]
    if len(yt) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE (%)": np.nan, "R²": np.nan}
    nonzero = yt != 0
    mape_value = (
        np.mean(np.abs((yt[nonzero] - yp[nonzero]) / yt[nonzero])) * 100
        if np.any(nonzero)
        else np.nan
    )
    return {
        "MAE": mean_absolute_error(yt, yp),
        "RMSE": np.sqrt(mean_squared_error(yt, yp)),
        "MAPE (%)": mape_value,
        "R²": r2_score(yt, yp),
    }


def looks_like_header(row):
    tokens = [str(x).strip().lower() for x in row.tolist()]
    joined = " ".join(tokens)
    keywords = ["tanggal", "periode", "kunjungan", "pengunjung", "visitor", "visits", "jumlah"]
    return any(k in joined for k in keywords)


def read_uploaded_bytes(file_bytes, filename):
    suffix = Path(filename).suffix.lower()
    bio = io.BytesIO(file_bytes)
    if suffix in [".xlsx", ".xls"]:
        raw = pd.read_excel(bio, header=None)
    elif suffix == ".csv":
        raw = pd.read_csv(bio, header=None)
    else:
        raise ValueError("Format file harus CSV atau Excel (.xlsx/.xls).")

    raw = raw.dropna(how="all").reset_index(drop=True)
    if raw.empty:
        raise ValueError("Dataset kosong.")

    if looks_like_header(raw.iloc[0]):
        raw2 = raw.iloc[1:].copy().reset_index(drop=True)
        raw2.columns = raw.iloc[0].astype(str).str.strip()
        return raw2
    return raw


def detect_columns(raw):
    cols = list(raw.columns)
    lower = {c: str(c).strip().lower() for c in cols}

    date_keywords = ["tanggal", "date", "periode", "bulan", "month", "time", "waktu"]
    value_keywords = ["kunjungan", "pengunjung", "visitor", "visits", "jumlah", "value", "count"]

    date_col = next((c for c in cols if any(k in lower[c] for k in date_keywords)), None)
    value_col = next((c for c in cols if any(k in lower[c] for k in value_keywords)), None)

    if date_col is None and len(cols) >= 2:
        date_col = cols[1] if len(cols) >= 3 else cols[0]
    if value_col is None and len(cols) >= 3:
        value_col = cols[2]
    if value_col is None:
        numeric_candidates = []
        for c in cols:
            s = safe_numeric(raw[c])
            if s.notna().sum() >= max(2, int(len(raw) * 0.6)):
                numeric_candidates.append(c)
        if numeric_candidates:
            value_col = numeric_candidates[-1]

    if date_col is None or value_col is None:
        raise ValueError(
            "Kolom tanggal/periode dan kolom jumlah kunjungan tidak dapat dikenali. "
            "Gunakan minimal dua kolom: periode/tanggal dan jumlah kunjungan."
        )
    return date_col, value_col


def parse_indonesian_periods(values):
    months = {
        "januari": 1, "februari": 2, "maret": 3, "april": 4,
        "mei": 5, "juni": 6, "juli": 7, "agustus": 8,
        "september": 9, "oktober": 10, "november": 11, "desember": 12,
    }
    parsed = []
    current_year = None

    for value in values:
        if pd.isna(value):
            parsed.append(pd.NaT)
            continue
        if isinstance(value, (pd.Timestamp, np.datetime64)):
            parsed.append(pd.Timestamp(value))
            continue

        text_value = " ".join(str(value).strip().split())
        p = text_value.lower()

        # Parse ISO dates explicitly first so YYYY-MM-DD is not interpreted
        # as YYYY-DD-MM when dayfirst=True is enabled for Indonesian inputs.
        iso_match = pd.Series([text_value]).str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$").iloc[0]
        if iso_match:
            direct = pd.to_datetime(text_value, errors="coerce", dayfirst=False)
        else:
            direct = pd.to_datetime(text_value, errors="coerce", dayfirst=True)
        if pd.notna(direct):
            # Avoid treating Indonesian month-only text incorrectly if a parser guesses.
            if not (len(text_value.split()) == 1 and p in months):
                parsed.append(pd.Timestamp(direct).replace(day=1))
                current_year = pd.Timestamp(direct).year
                continue

        parts = p.split()
        if len(parts) == 2 and parts[1].isdigit() and parts[0] in months:
            current_year = int(parts[1])
            parsed.append(pd.Timestamp(current_year, months[parts[0]], 1))
            continue
        if len(parts) == 1 and parts[0] in months:
            if current_year is None:
                parsed.append(pd.NaT)
            else:
                parsed.append(pd.Timestamp(current_year, months[parts[0]], 1))
            continue

        parsed.append(pd.NaT)

    return pd.to_datetime(parsed, errors="coerce")


def prepare_dataset(raw):
    date_col, value_col = detect_columns(raw)
    work = raw[[date_col, value_col]].copy()
    work.columns = ["periode_input", "kunjungan"]

    work["tanggal"] = parse_indonesian_periods(work["periode_input"])
    work["kunjungan"] = safe_numeric(work["kunjungan"])
    # Preserve rows whose visitor value is 0/blank because the research
    # preprocessing rule treats those observations as values to interpolate.
    work = work.dropna(subset=["tanggal"]).copy()
    work["tanggal"] = work["tanggal"].dt.to_period("M").dt.to_timestamp()
    work = work.sort_values("tanggal").drop_duplicates("tanggal", keep="last").reset_index(drop=True)

    if len(work) < 36:
        raise ValueError(
            f"Dataset hanya memiliki {len(work)} observasi valid. "
            "Gunakan minimal 36 observasi bulanan agar model musiman periode 12 dapat dievaluasi dengan layak."
        )

    work["kunjungan_asli"] = work["kunjungan"].astype(float)
    missing_or_zero = work["kunjungan_asli"].isna() | work["kunjungan_asli"].eq(0)
    work["status_preprocessing"] = np.where(missing_or_zero, "Interpolasi linear", "Aktual")
    work["kunjungan_preprocessed"] = (
        work["kunjungan_asli"].replace(0, np.nan).interpolate(method="linear", limit_area="inside")
    )

    if work["kunjungan_preprocessed"].isna().any():
        unresolved = work.loc[work["kunjungan_preprocessed"].isna(), "tanggal"].dt.strftime("%Y-%m").tolist()
        raise ValueError(
            "Terdapat nilai yang belum dapat diinterpolasi pada posisi awal/akhir: "
            + ", ".join(unresolved)
        )

    series = pd.Series(
        work["kunjungan_preprocessed"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(work["tanggal"]),
        name="Jumlah_Kunjungan",
    )

    full_calendar = pd.date_range(
        series.index.min(), series.index.max(), freq="MS"
    )
    calendar_df = pd.DataFrame({"tanggal": full_calendar})
    calendar_df["kunjungan_preprocessed"] = (
        calendar_df["tanggal"].map(series.to_dict()).astype(float)
    )

    # Audit perubahan preprocessing.
    audit = work[
        ["tanggal", "periode_input", "kunjungan_asli", "kunjungan_preprocessed", "status_preprocessing"]
    ].copy()
    audit["perubahan"] = ~np.isclose(
        audit["kunjungan_asli"], audit["kunjungan_preprocessed"], equal_nan=True
    )

    return {
        "data": work,
        "series": series,
        "calendar_df": calendar_df,
        "audit": audit,
        "date_col": date_col,
        "value_col": value_col,
    }


def calendar_history_until(calendar_df, test_date):
    return calendar_df.loc[
        calendar_df["tanggal"] < pd.Timestamp(test_date),
        "kunjungan_preprocessed",
    ].to_numpy(dtype=float)


def fit_sarima_forecast(history):
    hist = np.asarray(history, dtype=float)
    valid_n = np.isfinite(hist).sum()
    if valid_n == 0:
        return np.nan

    if valid_n < MIN_SARIMA_OBS:
        observed = hist[np.isfinite(hist)]
        if len(observed) == 1:
            return max(0.0, float(observed[-1]))
        model = SARIMAX(
            observed,
            order=(1, 0, 1),
            seasonal_order=(0, 0, 0, 0),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
    else:
        model = SARIMAX(
            hist,
            order=SARIMA_ORDER,
            seasonal_order=SARIMA_SEASONAL_ORDER,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

    fitted = model.fit(disp=False, maxiter=MAXITER)
    pred = float(np.asarray(fitted.forecast(1)).ravel()[0])
    return max(0.0, pred)


def fts_chen_forecast(history, n_intervals=8, weighted=True):
    x = np.asarray(history, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    if len(x) == 1 or np.isclose(x.min(), x.max()):
        return float(x[-1])

    xmin, xmax = float(x.min()), float(x.max())
    pad = 0.001 * (xmax - xmin + 1.0)
    bounds = np.linspace(xmin - pad, xmax + pad, n_intervals + 1)
    midpoints = (bounds[:-1] + bounds[1:]) / 2

    def state(v):
        idx = np.searchsorted(bounds, v, side="right") - 1
        return int(np.clip(idx, 0, n_intervals - 1))

    states = [state(v) for v in x]
    flr = {}
    for a, b in zip(states[:-1], states[1:]):
        flr.setdefault(a, []).append(b)

    last_state = states[-1]
    consequents = flr.get(last_state, [])
    if not consequents:
        return float(midpoints[last_state])

    values = np.array([midpoints[j] for j in consequents], dtype=float)
    if weighted:
        weights = np.arange(1, len(values) + 1, dtype=float)
        return float(np.average(values, weights=weights))
    return float(values.mean())


def run_hybrid_pipeline(file_bytes, filename, split_ratio):
    raw = read_uploaded_bytes(file_bytes, filename)
    prepared = prepare_dataset(raw)
    series = prepared["series"]
    calendar_df = prepared["calendar_df"]
    audit = prepared["audit"]

    y = series.to_numpy(dtype=float)
    dates = series.index
    n = len(y)
    train_n = int(np.floor(n * split_ratio))
    if train_n <= VALIDATION_SIZE + max(WINDOWS) + 3:
        raise ValueError("Jumlah training terlalu sedikit setelah pembagian data.")

    test_n = n - train_n
    if test_n < 6:
        raise ValueError("Data testing terlalu sedikit. Pilih dataset yang lebih panjang.")

    val_start = train_n - VALIDATION_SIZE
    val_end = train_n
    y_val = y[val_start:val_end]

    # SARIMA validation/testing one-step-ahead.
    sarima_val_pred = np.array([
        fit_sarima_forecast(calendar_history_until(calendar_df, dates[t]))
        for t in range(val_start, val_end)
    ])
    sarima_test_pred = np.array([
        fit_sarima_forecast(calendar_history_until(calendar_df, dates[t]))
        for t in range(train_n, n)
    ])

    sarima_val_metrics = evaluate_metrics(y_val, sarima_val_pred)
    sarima_test_metrics = evaluate_metrics(y[train_n:], sarima_test_pred)

    v_tuning_rows = []
    full_calibration_models = {}
    best_cfg = {}
    results = {}

    for window in WINDOWS:
        for n_intervals in FTS_INTERVAL_GRID:
            raw_cal = []
            actual_cal = []

            cal_start = max(18, window)
            for t in range(cal_start, val_start):
                raw_pred = fts_chen_forecast(
                    y[:t][-window:],
                    n_intervals=n_intervals,
                    weighted=True,
                )
                if np.isfinite(raw_pred) and np.isfinite(y[t]):
                    raw_cal.append(raw_pred)
                    actual_cal.append(y[t])

            if len(raw_cal) < 2:
                continue

            reg = LinearRegression()
            reg.fit(np.asarray(raw_cal).reshape(-1, 1), np.asarray(actual_cal))

            raw_val = np.array([
                fts_chen_forecast(
                    y[:t][-window:],
                    n_intervals=n_intervals,
                    weighted=True,
                )
                for t in range(val_start, val_end)
            ])
            fts_val = np.maximum(0.0, reg.predict(raw_val.reshape(-1, 1)))

            for alpha in ALPHA_GRID:
                hybrid_val = alpha * sarima_val_pred + (1 - alpha) * fts_val
                met = evaluate_metrics(y_val, hybrid_val)
                v_tuning_rows.append({
                    "Window": window,
                    "n_intervals": n_intervals,
                    "alpha": float(alpha),
                    "Calib_N": len(raw_cal),
                    **met,
                })

    v7_tuning = pd.DataFrame(v_tuning_rows)
    if v7_tuning.empty:
        raise ValueError("Tuning validation tidak menghasilkan konfigurasi yang valid.")

    for window in WINDOWS:
        part = (
            v7_tuning[v7_tuning["Window"] == window]
            .sort_values(["MAPE (%)", "MAE", "RMSE"])
        )
        row = part.iloc[0]
        best_cfg[window] = {
            "n_intervals": int(row["n_intervals"]),
            "alpha": float(row["alpha"]),
        }

        n_intervals = best_cfg[window]["n_intervals"]
        raw_full = []
        actual_full = []
        for t in range(max(18, window), train_n):
            raw_pred = fts_chen_forecast(
                y[:t][-window:],
                n_intervals=n_intervals,
                weighted=True,
            )
            if np.isfinite(raw_pred) and np.isfinite(y[t]):
                raw_full.append(raw_pred)
                actual_full.append(y[t])
        reg_full = LinearRegression()
        reg_full.fit(np.asarray(raw_full).reshape(-1, 1), np.asarray(actual_full))
        full_calibration_models[window] = reg_full

        rows = []
        alpha = best_cfg[window]["alpha"]
        for j, t in enumerate(range(train_n, n)):
            p_s = sarima_test_pred[j]
            p_f_raw = fts_chen_forecast(
                y[:t][-window:],
                n_intervals=n_intervals,
                weighted=True,
            )
            p_f = max(0.0, float(reg_full.predict(np.array([[p_f_raw]]))[0]))
            p_h = alpha * p_s + (1 - alpha) * p_f
            rows.append({
                "index": t,
                "Tanggal": dates[t],
                "Aktual": y[t],
                "SARIMA": p_s,
                "FTS_raw": p_f_raw,
                "FTS_calibrated": p_f,
                "Hybrid_SARIMA_FTS": p_h,
                "Window": window,
                "n_intervals": n_intervals,
                "alpha": alpha,
            })

        out = pd.DataFrame(rows)
        out["Error_Hybrid"] = out["Aktual"] - out["Hybrid_SARIMA_FTS"]
        out["Abs_Error_Hybrid"] = np.abs(out["Error_Hybrid"])
        results[window] = out

    metrics_rows = []
    for window, out in results.items():
        for model_col, label in [
            ("SARIMA", "SARIMA"),
            ("FTS_calibrated", "FTS_calibrated"),
            ("Hybrid_SARIMA_FTS", "Hybrid_SARIMA_FTS"),
        ]:
            met = evaluate_metrics(out["Aktual"], out[model_col])
            metrics_rows.append({
                "Skenario": f"{label} | Window {window} → 1",
                "Window": window,
                "Model": label,
                **met,
            })

    metrics_df = pd.DataFrame(metrics_rows)

    # In-sample SARIMA fit for training display.
    train_end_date = dates[train_n - 1]
    train_calendar = calendar_df.loc[
        calendar_df["tanggal"] <= train_end_date,
        "kunjungan_preprocessed",
    ].to_numpy(dtype=float)
    train_model = SARIMAX(
        train_calendar,
        order=SARIMA_ORDER,
        seasonal_order=SARIMA_SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    train_fit = train_model.fit(disp=False, maxiter=MAXITER)
    train_pred_calendar = np.asarray(train_fit.get_prediction().predicted_mean, dtype=float)
    calendar_train_dates = calendar_df.loc[
        calendar_df["tanggal"] <= train_end_date, "tanggal"
    ].to_numpy()
    train_lookup = pd.DataFrame({
        "Tanggal": pd.to_datetime(calendar_train_dates),
        "Prediksi": train_pred_calendar,
    })
    train_actual = series.iloc[:train_n].copy()
    train_actual.index.name = "Tanggal"
    train_export = (
        train_actual.rename("Aktual").reset_index()
        .merge(train_lookup, on="Tanggal", how="left")
    )

    selected_df = pd.DataFrame(best_cfg).T.reset_index().rename(columns={"index": "Window"})

    return {
        **prepared,
        "raw": raw,
        "split_ratio": split_ratio,
        "train_n": train_n,
        "test_n": test_n,
        "val_start": val_start,
        "validation": series.iloc[val_start:val_end].copy(),
        "train": series.iloc[:train_n].copy(),
        "test": series.iloc[train_n:].copy(),
        "sarima_val_pred": sarima_val_pred,
        "sarima_test_pred": sarima_test_pred,
        "sarima_val_metrics": sarima_val_metrics,
        "sarima_test_metrics": sarima_test_metrics,
        "v7_tuning": v7_tuning,
        "best_cfg": best_cfg,
        "best_cfg_df": selected_df,
        "results": results,
        "metrics": metrics_df,
        "train_export": train_export,
        "full_calibration_models": full_calibration_models,
    }


@st.cache_data(show_spinner=False, max_entries=8)
def cached_run_pipeline(file_bytes, filename, split_ratio):
    return run_hybrid_pipeline(file_bytes, filename, split_ratio)


def load_default_source():
    if not DEFAULT_DATA_FILE.exists():
        return None
    return DEFAULT_DATA_FILE.read_bytes(), DEFAULT_DATA_FILE.name


def make_future_forecast(bundle, periods, window):
    series = bundle["series"]
    calendar_df = bundle["calendar_df"]
    reg = bundle["full_calibration_models"][window]
    alpha = bundle["best_cfg"][window]["alpha"]
    n_intervals = bundle["best_cfg"][window]["n_intervals"]

    full_hist = calendar_df["kunjungan_preprocessed"].to_numpy(dtype=float)
    model = SARIMAX(
        full_hist,
        order=SARIMA_ORDER,
        seasonal_order=SARIMA_SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False, maxiter=MAXITER)
    sarima_future = np.asarray(fit.forecast(periods), dtype=float)

    history = list(series.to_numpy(dtype=float))
    future_dates = pd.date_range(
        start=series.index[-1] + pd.offsets.MonthBegin(1),
        periods=periods,
        freq="MS",
    )

    rows = []
    for i in range(periods):
        fts_raw = fts_chen_forecast(
            np.asarray(history[-window:], dtype=float),
            n_intervals=n_intervals,
            weighted=True,
        )
        fts_cal = max(0.0, float(reg.predict(np.array([[fts_raw]]))[0]))
        hybrid = max(0.0, alpha * sarima_future[i] + (1 - alpha) * fts_cal)
        rows.append({
            "Tanggal": future_dates[i],
            "SARIMA": sarima_future[i],
            "FTS_calibrated": fts_cal,
            "Prediksi_Hybrid": hybrid,
        })
        history.append(hybrid)

    return pd.DataFrame(rows)

# ============================================================

# ============================================================
# RESEARCH REPORTING DASHBOARD — UI LAYER ONLY
# ============================================================
# Catatan: seluruh fungsi preprocessing, SARIMA, FTS, tuning,
# ensemble, evaluasi, dan future forecasting di atas blok ini
# dipertahankan. Bagian di bawah hanya mengatur antarmuka.

import html
import zipfile

# ============================================================
# DESIGN SYSTEM
# ============================================================
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">
    <style>
        :root {
            --lux-navy: #0b1930;
            --lux-navy-2: #122844;
            --lux-ink: #172338;
            --lux-muted: #6d7c90;
            --lux-soft: #f5f7fa;
            --lux-line: #e5eaf0;
            --lux-white: #ffffff;
            --lux-gold: #b89962;
            --lux-gold-soft: #f6f0e6;
            --lux-blue: #356a9c;
            --lux-blue-soft: #edf4fa;
            --lux-green: #2f7b5a;
            --lux-green-soft: #ecf7f1;
            --lux-shadow: 0 10px 32px rgba(18, 37, 62, .065);
            --lux-shadow-lg: 0 18px 48px rgba(11, 25, 48, .10);
        }
        html, body, [class*="css"] {
            font-family: "Plus Jakarta Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
        }
        .stApp {
            background:
                radial-gradient(circle at 92% 0%, rgba(184,153,98,.05), transparent 22%),
                linear-gradient(180deg, #f7f8fa 0%, #f4f6f9 100%);
            color: var(--lux-ink);
        }
        .main .block-container {
            max-width: 1500px;
            padding: 1.6rem 2.15rem 3rem;
        }
        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #081526 0%, #0b1930 48%, #10233e 100%);
            border-right: 1px solid rgba(255,255,255,.07);
        }
        [data-testid="stSidebar"] > div:first-child { padding: 1.05rem .92rem 1.1rem; }
        [data-testid="stSidebar"] * { color: #eef3f9; }
        [data-testid="stSidebar"] .stRadio > label {
            font-size: 9px !important;
            font-weight: 800 !important;
            letter-spacing: .17em !important;
            color: #8ea1b9 !important;
            text-transform: uppercase;
            margin: 11px 0 7px;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] { gap: 5px; }
        [data-testid="stSidebar"] div[role="radio"] {
            border-radius: 12px;
            padding: 8px 11px;
            border: 1px solid transparent;
            transition: all .18s ease;
        }
        [data-testid="stSidebar"] div[role="radio"]:hover {
            background: rgba(255,255,255,.055);
            border-color: rgba(255,255,255,.07);
        }
        [data-testid="stSidebar"] div[role="radio"][aria-checked="true"] {
            background: linear-gradient(90deg, rgba(73,111,149,.34), rgba(184,153,98,.10));
            border-color: rgba(184,153,98,.24);
            box-shadow: inset 3px 0 0 var(--lux-gold);
        }
        [data-testid="stSidebar"] div[role="radio"] > div:first-child { display:none; }
        [data-testid="stSidebar"] div[role="radio"] p {
            font-size: 12px !important;
            font-weight: 600 !important;
            margin: 0 !important;
        }
        .side-brand {
            padding: 2px 3px 16px;
            border-bottom: 1px solid rgba(255,255,255,.08);
            margin-bottom: 14px;
        }
        .side-logo {
            width: 40px; height: 40px; border-radius: 13px;
            display: inline-flex; align-items: center; justify-content:center;
            background: linear-gradient(135deg, rgba(184,153,98,.24), rgba(255,255,255,.04));
            border: 1px solid rgba(184,153,98,.25);
            color: #ecd7ae; font-size: 17px; margin-right: 9px;
            vertical-align: middle;
        }
        .side-title {
            display:inline-block; vertical-align:middle;
            font-size:17px; font-weight:800; letter-spacing:-.02em;
        }
        .side-subtitle {
            margin: 10px 1px 0; color:#aab8ca !important;
            font-size:10px; line-height:1.65;
        }
        .side-label {
            color:#8296b0; font-size:8px; font-weight:800;
            letter-spacing:.16em; text-transform:uppercase; margin: 13px 0 7px;
        }
        .side-status {
            padding: 12px; border-radius: 14px;
            background: rgba(255,255,255,.045);
            border: 1px solid rgba(255,255,255,.08);
        }
        .side-status .caption { color:#8296b0; font-size:8px; text-transform:uppercase; letter-spacing:.12em; font-weight:800; }
        .side-status .value { color:#f2f6fb; font-size:11px; font-weight:700; margin-top:4px; word-break:break-word; }
        .side-chip {
            display:inline-flex; align-items:center; gap:6px;
            margin-top:8px; padding:5px 8px; border-radius:999px;
            background: rgba(77,157,117,.11); border:1px solid rgba(77,157,117,.19);
            color:#9fdfc0 !important; font-size:8.5px; font-weight:800;
        }
        .side-mini {
            margin-top: 12px; padding-top: 11px;
            border-top: 1px solid rgba(255,255,255,.07);
            color:#8296b0 !important; font-size:9px; line-height:1.6;
        }
        /* Global UI */
        .page-shell { margin-bottom: 24px; }
        .eyebrow {
            color: var(--lux-gold); font-size: 9px; font-weight:800;
            letter-spacing:.18em; text-transform:uppercase; margin-bottom:7px;
        }
        .page-title {
            margin:0; color:var(--lux-ink); font-size:30px; line-height:1.1;
            font-weight:800; letter-spacing:-.045em;
        }
        .page-description {
            margin-top:8px; max-width:860px; color:var(--lux-muted);
            font-size:12.5px; line-height:1.7;
        }
        .page-meta {
            display:inline-flex; align-items:center; gap:8px;
            padding:8px 11px; border-radius:999px; background:#fff;
            border:1px solid var(--lux-line); color:var(--lux-muted);
            font-size:9px; font-weight:800; letter-spacing:.08em;
            box-shadow: 0 5px 16px rgba(18,37,62,.04);
        }
        .page-meta i { color:var(--lux-gold); }
        /* =====================================================
           HERO — first-impression research dashboard
           Visual language: premium research lab, analytical, calm
           ===================================================== */
        .hero {
            position:relative;
            overflow:hidden;
            min-height:318px;
            border-radius:28px;
            padding:34px 34px 30px 38px;
            color:#fff;
            background:
                radial-gradient(circle at 86% 16%, rgba(94,151,202,.24), transparent 22%),
                radial-gradient(circle at 74% 92%, rgba(52,112,164,.30), transparent 27%),
                linear-gradient(135deg, #061321 0%, #0a1d33 48%, #17466b 100%);
            box-shadow:0 24px 60px rgba(10,25,45,.17);
            margin-bottom:24px;
            isolation:isolate;
        }
        /* Decorative circular arcs removed for a cleaner first impression. */
        .hero::before,
        .hero::after {
            display:none !important;
            content:none;
        }
        .hero-grid {
            position:relative;
            z-index:2;
            display:grid;
            grid-template-columns:minmax(0,1.45fr) minmax(300px,.75fr);
            gap:28px;
            align-items:center;
        }
        .hero-main { min-width:0; }
        .hero-kicker {
            display:inline-flex;
            align-items:center;
            gap:7px;
            font-size:8.5px;
            font-weight:800;
            letter-spacing:.18em;
            text-transform:uppercase;
            color:#d5b979;
            opacity:.95;
        }
        .hero-kicker-dot {
            width:6px; height:6px; border-radius:50%;
            background:#d5b979;
            box-shadow:0 0 0 5px rgba(213,185,121,.12);
        }
        .hero-title {
            margin-top:12px;
            font-size:42px;
            line-height:1.04;
            letter-spacing:-.052em;
            font-weight:800;
            max-width:760px;
        }
        .hero-accent {
            color:#d5b979;
        }
        .hero-copy {
            margin-top:14px;
            max-width:760px;
            font-size:12.5px;
            line-height:1.72;
            color:rgba(255,255,255,.74);
        }
        .hero-actions {
            margin-top:20px;
            display:flex;
            gap:8px;
            flex-wrap:wrap;
        }
        .hero-pill {
            display:inline-flex;
            align-items:center;
            gap:7px;
            padding:8px 11px;
            border-radius:999px;
            background:rgba(255,255,255,.07);
            border:1px solid rgba(255,255,255,.10);
            font-size:8.8px;
            font-weight:700;
            color:#f5f7fb;
            backdrop-filter:blur(6px);
        }
        .hero-pill strong { color:#fff; font-weight:800; }
        .hero-side {
            position:relative;
            z-index:3;
            width:100%;
            max-width:350px;
            justify-self:end;
        }
        .hero-panel {
            position:relative;
            overflow:hidden;
            border-radius:20px;
            padding:15px;
            background:linear-gradient(180deg, rgba(86,145,193,.16), rgba(255,255,255,.055));
            border:1px solid rgba(255,255,255,.13);
            box-shadow:0 20px 42px rgba(0,0,0,.18);
            backdrop-filter:blur(12px);
        }
        .hero-panel::after {
            content:"";
            position:absolute;
            width:110px; height:110px;
            right:-44px; bottom:-50px;
            border:1px solid rgba(255,255,255,.08);
            border-radius:50%;
        }
        .hero-panel-head {
            display:flex;
            align-items:center;
            justify-content:space-between;
            gap:10px;
        }
        .hero-panel-label {
            font-size:7.5px;
            letter-spacing:.16em;
            text-transform:uppercase;
            font-weight:800;
            color:rgba(255,255,255,.50);
        }
        .hero-live {
            display:inline-flex;
            align-items:center;
            gap:5px;
            padding:5px 7px;
            border-radius:999px;
            font-size:7.5px;
            font-weight:800;
            color:#cfe6f7;
            border:1px solid rgba(116,176,218,.24);
            background:rgba(75,138,190,.12);
        }
        .hero-live-dot {
            width:5px; height:5px; border-radius:50%;
            background:#86bde5;
            box-shadow:0 0 0 4px rgba(110,173,218,.10);
        }
        .hero-panel-title {
            margin-top:9px;
            font-size:17px;
            line-height:1.15;
            font-weight:800;
            letter-spacing:-.025em;
            color:#fff;
        }
        .hero-panel-subtitle {
            margin-top:4px;
            font-size:8.5px;
            color:rgba(255,255,255,.48);
        }
        .hero-mini-chart {
            margin-top:11px;
            border-radius:14px;
            padding:8px 9px 4px;
            background:rgba(4,14,26,.25);
            border:1px solid rgba(255,255,255,.07);
        }
        .hero-mini-chart svg {
            display:block;
            width:100%;
            height:82px;
        }
        .hero-stat-row {
            display:grid;
            grid-template-columns:repeat(3,1fr);
            gap:7px;
            margin-top:8px;
        }
        .hero-stat {
            padding:8px 8px 9px;
            border-radius:11px;
            background:rgba(255,255,255,.055);
            border:1px solid rgba(255,255,255,.065);
        }
        .hero-stat-label {
            font-size:6.8px;
            text-transform:uppercase;
            letter-spacing:.10em;
            color:rgba(255,255,255,.42);
            font-weight:800;
        }
        .hero-stat-value {
            margin-top:4px;
            font-size:12px;
            color:#fff;
            font-weight:800;
        }
        .hero-stat-value.gold { color:#d5b979; }
        .hero-flow {
            display:flex;
            align-items:center;
            gap:6px;
            margin-top:9px;
            padding-top:9px;
            border-top:1px solid rgba(255,255,255,.07);
        }
        .hero-flow-step {
            flex:1;
            text-align:center;
            font-size:6.8px;
            color:rgba(255,255,255,.45);
            font-weight:700;
            line-height:1.2;
        }
        .hero-flow-node {
            width:20px; height:20px;
            margin:0 auto 4px;
            display:flex;
            align-items:center;
            justify-content:center;
            border-radius:7px;
            color:#fff;
            background:rgba(255,255,255,.09);
            border:1px solid rgba(255,255,255,.10);
            font-size:7px;
        }
        .hero-flow-arrow {
            color:rgba(255,255,255,.28);
            font-size:9px;
        }
        /* =====================================================
           V12 HERO PALETTE — BLUE ONLY
           No gold/yellow accents or warm glows inside the hero.
           ===================================================== */
        .hero, .hero * {
            --hero-blue-1: #7fb8df;
            --hero-blue-2: #a9d2ef;
            --hero-blue-3: #86bde5;
        }
        .hero .hero-accent, .hero .hero-stat-value.gold { color:#d5b979 !important; }
        .hero .hero-kicker { color:#d5b979 !important; }
        .hero .hero-kicker-dot { background:#d5b979 !important; box-shadow:0 0 0 5px rgba(213,185,121,.12) !important; }
        .hero .hero-live { color:#cfe6f7 !important; border-color:rgba(116,176,218,.24) !important; background:rgba(75,138,190,.12) !important; }
        .hero .hero-live-dot { background:var(--hero-blue-3) !important; box-shadow:0 0 0 4px rgba(110,173,218,.10) !important; }
        .hero .hero-stat-value.gold { color:#d5b979 !important; }
        .hero .hero-panel { background:linear-gradient(180deg, rgba(86,145,193,.16), rgba(255,255,255,.055)) !important; }

        .hero-empty-note {
            margin-top:9px;
            padding:9px 10px;
            border-radius:11px;
            background:rgba(255,255,255,.045);
            border:1px solid rgba(255,255,255,.065);
            color:rgba(255,255,255,.52);
            font-size:8px;
            line-height:1.55;
        }
        .section { margin-top:24px; }
        .section-head { display:flex; justify-content:space-between; gap:14px; align-items:flex-end; margin-bottom:12px; }
        .section-kicker { color:var(--lux-gold); font-size:8.5px; text-transform:uppercase; font-weight:800; letter-spacing:.15em; }
        .section-title { color:var(--lux-ink); font-size:17px; line-height:1.2; font-weight:800; letter-spacing:-.025em; margin-top:3px; }
        .section-desc { color:var(--lux-muted); font-size:11px; line-height:1.55; margin-top:3px; }
        .section-icon {
            width:34px; height:34px; border-radius:11px; display:inline-flex;
            align-items:center; justify-content:center; background:var(--lux-gold-soft, #f6f0e6);
            color:var(--lux-gold); font-size:14px;
        }
        .surface {
            background:#fff; border:1px solid var(--lux-line); border-radius:18px;
            padding:18px; box-shadow:var(--lux-shadow); margin-bottom:15px;
        }
        .surface-tight { padding:14px 16px; }
        .metric-card {
            background:#fff; border:1px solid var(--lux-line); border-radius:17px;
            padding:15px 16px 16px; box-shadow:var(--lux-shadow); min-height:114px;
            position:relative; overflow:hidden;
        }
        .metric-card::after {
            content:""; position:absolute; width:100px; height:100px; border-radius:50%;
            top:-56px; right:-44px; background:rgba(184,153,98,.08);
        }
        .metric-icon {
            width:31px; height:31px; border-radius:10px; display:flex; align-items:center; justify-content:center;
            background:var(--lux-blue-soft); color:var(--lux-blue); font-size:13px; margin-bottom:10px;
        }
        .metric-label { color:#8290a2; font-size:8.5px; text-transform:uppercase; letter-spacing:.12em; font-weight:800; }
        .metric-value { color:var(--lux-ink); font-size:24px; line-height:1.1; font-weight:800; letter-spacing:-.04em; margin-top:4px; }
        .metric-help { color:#8b97a7; font-size:9.5px; line-height:1.45; margin-top:5px; }
        .feature {
            height:100%; background:#fff; border:1px solid var(--lux-line); border-radius:17px;
            padding:17px; box-shadow:var(--lux-shadow);
        }
        .feature-icon { width:38px;height:38px;border-radius:12px;display:flex;align-items:center;justify-content:center;background:var(--lux-blue-soft);color:var(--lux-blue);font-size:15px; }
        .feature-title { margin-top:11px; color:var(--lux-ink); font-size:12px; font-weight:800; }
        .feature-text { margin-top:5px; color:var(--lux-muted); font-size:10px; line-height:1.65; }
        .workflow {
            height:100%; padding:16px; background:#fbfcfd; border:1px solid var(--lux-line);
            border-radius:16px; position:relative;
        }
        .workflow-num { color:var(--lux-gold); font-size:8px; font-weight:800; letter-spacing:.12em; }
        .workflow-icon { color:var(--lux-blue); font-size:16px; margin-top:9px; }
        .workflow-title { color:var(--lux-ink); font-size:10.5px; font-weight:800; margin-top:7px; }
        .workflow-text { color:var(--lux-muted); font-size:9.5px; line-height:1.5; margin-top:4px; }
        .tag {
            display:inline-flex; align-items:center; gap:6px; padding:6px 9px; border-radius:999px;
            border:1px solid var(--lux-line); background:#fff; color:var(--lux-muted); font-size:8.5px; font-weight:800;
        }
        .tag-gold { background:var(--lux-gold-soft); border-color:#eadfcf; color:#896d3d; }
        .tag-blue { background:var(--lux-blue-soft); border-color:#d8e6f2; color:#456f94; }
        .tag-green { background:var(--lux-green-soft); border-color:#d7ebdf; color:var(--lux-green); }
        .note {
            display:flex; gap:10px; align-items:flex-start; padding:13px 14px; border-radius:14px;
            background:#f8fafc; border:1px solid var(--lux-line); color:var(--lux-muted);
            font-size:10px; line-height:1.6;
        }
        .note i { color:var(--lux-gold); font-size:13px; margin-top:1px; }
        .split-line { height:1px; background:var(--lux-line); margin:15px 0; }
        .stepper { display:flex; flex-wrap:wrap; align-items:center; gap:8px; padding:11px 12px; border:1px solid var(--lux-line); background:#fbfcfd; border-radius:14px; }
        .step { display:flex; align-items:center; gap:7px; color:#8592a1; font-size:9.5px; font-weight:700; }
        .step.on { color:var(--lux-blue); }
        .step-dot { width:22px;height:22px;border-radius:8px;display:flex;align-items:center;justify-content:center;background:#eef2f6;font-size:8px;font-weight:800; }
        .step.on .step-dot { background:var(--lux-blue);color:#fff; }
        .step-arrow { color:#b8c1cb;font-size:10px; }
        .empty {
            text-align:center; padding:58px 20px; border-radius:20px;
            border:1px dashed #cfd7e1; background:#fbfcfd;
        }
        .empty-icon { width:50px;height:50px;border-radius:16px;margin:0 auto 12px;background:var(--lux-blue-soft);color:var(--lux-blue);display:flex;align-items:center;justify-content:center;font-size:20px; }
        .empty-title { color:var(--lux-ink);font-size:15px;font-weight:800; }
        .empty-text { max-width:510px;margin:6px auto 0;color:var(--lux-muted);font-size:10.5px;line-height:1.65; }
        /* Native Streamlit widgets */
        .stButton > button, .stDownloadButton > button {
            min-height:42px; border-radius:12px !important; font-family:inherit !important;
            font-size:10.5px !important; font-weight:800 !important; border:1px solid #dfe5ec !important;
            background:#fff !important; color:var(--lux-ink) !important; box-shadow:none !important;
        }
        .stButton > button:hover, .stDownloadButton > button:hover { transform:translateY(-1px); box-shadow:0 8px 18px rgba(18,37,62,.08) !important; }
        .stButton > button[kind="primary"] { background:linear-gradient(135deg,#17395e,#285b82) !important; color:#fff !important; border:none !important; }
        .stTextInput input, .stNumberInput input, .stDateInput input, [data-baseweb="select"] > div { border-radius:11px !important; min-height:40px !important; }
        [data-testid="stFileUploader"] section { border-radius:15px !important; border:1px dashed #cad4df !important; background:#fbfcfd !important; }
        [data-testid="stFileUploader"] section:hover { border-color:#8eabc6 !important; }
        div[data-testid="stMetric"] { background:#fff; border:1px solid var(--lux-line); border-radius:15px; box-shadow:var(--lux-shadow); padding:12px 14px; }
        div[data-testid="stDataFrame"] { border:1px solid var(--lux-line); border-radius:14px; overflow:hidden; }
        div[data-testid="stExpander"] { border:1px solid var(--lux-line); border-radius:14px; overflow:hidden; }
        .stRadio label, .stSelectbox label, .stDateInput label, .stSlider label, .stFileUploader label { font-size:10px !important; font-weight:700 !important; color:#5f6e81 !important; }
        .stCaption { color:#8693a2 !important; }
        .footer { margin-top:35px; padding:18px 0 5px; border-top:1px solid var(--lux-line); color:#8a95a3; font-size:9.5px; line-height:1.6; }
        @media (max-width: 900px) {
            .main .block-container { padding:1rem .9rem 2rem; }
            .hero { min-height:auto; padding:25px 22px; border-radius:22px; }
            .hero-grid { grid-template-columns:1fr; gap:18px; }
            .hero-title { font-size:31px; }
            .hero-side { max-width:none; }
            .hero-panel { padding:13px; }
            .page-title { font-size:26px; }
        }

        /* =====================================================
           V4 LAYOUT REFINEMENT
           Focus: quieter sidebar, single-window analysis, generous spacing
           ===================================================== */
        [data-testid="stSidebar"] { width: 255px !important; min-width: 255px !important; }
        [data-testid="stSidebar"] > div:first-child { padding: .8rem .72rem 1rem !important; }
        [data-testid="stSidebar"] .side-brand { padding: 5px 4px 13px !important; }
        [data-testid="stSidebar"] .side-logo { width: 34px !important; height: 34px !important; border-radius: 10px !important; margin-right: 8px !important; }
        [data-testid="stSidebar"] .side-title { font-size: 15px !important; letter-spacing: -.02em !important; }
        [data-testid="stSidebar"] .side-subtitle { display:none !important; }
        [data-testid="stSidebar"] .side-label { margin: 10px 4px 6px !important; font-size: 8px !important; letter-spacing: .14em !important; }
        [data-testid="stSidebar"] .side-status { margin: 0 1px 10px !important; padding: 8px 10px !important; border-radius: 11px !important; background: rgba(255,255,255,.035) !important; }
        [data-testid="stSidebar"] .side-status .caption { font-size: 8px !important; }
        [data-testid="stSidebar"] .side-status .value { font-size: 10px !important; line-height: 1.35 !important; margin-top: 2px !important; }
        [data-testid="stSidebar"] .side-status .value + .value { color:#9aacbf !important; font-size:9px !important; }
        [data-testid="stSidebar"] .side-chip { font-size: 8px !important; padding: 3px 6px !important; margin-top: 5px !important; }
        [data-testid="stSidebar"] .side-mini { display:none !important; }
        [data-testid="stSidebar"] div[role="radiogroup"] { gap: 2px !important; }
        [data-testid="stSidebar"] div[role="radio"] { padding: 6px 9px !important; border-radius: 9px !important; min-height: 32px !important; }
        [data-testid="stSidebar"] div[role="radio"] p { font-size: 11px !important; font-weight: 650 !important; letter-spacing: 0 !important; white-space: nowrap !important; }
        [data-testid="stSidebar"] div[role="radio"][aria-checked="true"] { box-shadow: inset 2px 0 0 #9ec7ef !important; }
        [data-testid="stSidebar"] .side-divider { margin: 9px 3px 7px !important; opacity:.55 !important; }
        .page-header { margin-bottom: 30px !important; }
        .section { margin-top: 30px !important; }
        .section-head { margin-bottom: 18px !important; }
        .section-title { line-height: 1.25 !important; }
        .section-desc { margin-top: 7px !important; line-height: 1.7 !important; }
        .chart-slot { height: 18px; }
        .chart-card { margin-top: 2px; padding: 6px 4px 2px; border-radius: 18px; background:#fff; border:1px solid var(--lux-line); box-shadow:var(--lux-shadow); }
        .window-tabs-spacer { height: 12px; }
        div[data-testid="stTabs"] { margin-top: 8px !important; }
        div[data-testid="stTabs"] button[role="tab"] { font-family:inherit !important; font-size:11px !important; font-weight:800 !important; padding: 10px 16px !important; }
        div[data-testid="stTabsContent"] { padding-top: 20px !important; }
        .scenario-banner { display:flex; align-items:center; justify-content:space-between; gap:16px; padding:12px 15px; border:1px solid var(--lux-line); background:#fbfcfe; border-radius:14px; margin-bottom:18px; }
        .scenario-banner .left { display:flex; align-items:center; gap:9px; }
        .scenario-banner .icon { width:30px;height:30px;border-radius:9px;display:flex;align-items:center;justify-content:center;background:var(--lux-blue-soft);color:var(--lux-blue); }
        .scenario-banner .title { font-size:11px;font-weight:800;color:var(--lux-ink); }
        .scenario-banner .text { font-size:9.5px;color:var(--lux-muted);margin-top:2px; }
        .scenario-banner .badge { padding:6px 9px;border-radius:999px;background:#f3eee5;color:#9b7640;font-size:9px;font-weight:800;white-space:nowrap; }
        .method-band { margin-top:12px; padding:13px 15px; border:1px solid var(--lux-line); border-radius:15px; background:#fbfcfd; }
        .method-band-title { color:var(--lux-ink); font-size:10.5px; font-weight:800; display:flex; align-items:center; gap:7px; }
        .method-band-title i { color:var(--lux-gold); }
        .method-band-items { display:flex; flex-wrap:wrap; align-items:center; gap:7px; margin-top:9px; }
        .method-chip { display:inline-flex; align-items:center; gap:5px; padding:6px 9px; border-radius:999px; font-size:8.5px; font-weight:700; border:1px solid; }
        .method-chip.blue { background:var(--lux-blue-soft); color:#456f94; border-color:#d8e6f2; }
        .method-chip.gold { background:var(--lux-gold-soft); color:#896d3d; border-color:#eadfcf; }
        .method-chip.green { background:var(--lux-green-soft); color:var(--lux-green); border-color:#d7ebdf; }
        .method-arrow { color:#aeb9c4; font-size:10px; }
        .method-note { color:var(--lux-muted); font-size:8.5px; margin-left:2px; }
        .research-note { margin:10px 0 14px; padding:11px 13px; border-radius:13px; border:1px solid #eadfcf; background:#fcf8f1; color:#7a694f; font-size:9.5px; line-height:1.6; }
        .diagnostic-card { background:#fff; border:1px solid var(--lux-line); border-radius:17px; padding:14px; box-shadow:var(--lux-shadow); }
        @media(max-width:900px){
            [data-testid="stSidebar"] { width: 235px !important; min-width:235px !important; }
            .section { margin-top:24px !important; }
            .section-head { margin-bottom:15px !important; }
            .scenario-banner { align-items:flex-start; flex-direction:column; }
            div[data-testid="stTabsContent"] { padding-top:14px !important; }
        }

        /* =====================================================
           09 — PANDUAN APLIKASI
           Buku panduan mini yang terintegrasi di dashboard
           ===================================================== */
        .guide-hero {
            padding:22px 24px;
            border-radius:18px;
            background:linear-gradient(135deg,#edf4fa 0%,#f8fbfd 100%);
            border:1px solid #d8e6f2;
            box-shadow:var(--lux-shadow);
            margin-bottom:18px;
        }
        .guide-hero-title {
            color:var(--lux-ink);
            font-size:18px;
            font-weight:800;
            letter-spacing:-.025em;
        }
        .guide-hero-text {
            margin-top:6px;
            max-width:900px;
            color:var(--lux-muted);
            font-size:10.5px;
            line-height:1.7;
        }
        .guide-badge {
            display:inline-flex;
            align-items:center;
            gap:6px;
            margin-top:11px;
            padding:6px 9px;
            border-radius:999px;
            background:#fff;
            border:1px solid #d8e6f2;
            color:#456f94;
            font-size:8.5px;
            font-weight:800;
        }
        .guide-card {
            height:100%;
            background:#fff;
            border:1px solid var(--lux-line);
            border-radius:17px;
            padding:17px;
            box-shadow:var(--lux-shadow);
        }
        .guide-number {
            color:var(--lux-blue);
            font-size:8px;
            font-weight:800;
            letter-spacing:.14em;
            text-transform:uppercase;
        }
        .guide-card-title {
            margin-top:7px;
            color:var(--lux-ink);
            font-size:12px;
            font-weight:800;
        }
        .guide-card-text {
            margin-top:5px;
            color:var(--lux-muted);
            font-size:9.5px;
            line-height:1.65;
        }
        .guide-callout {
            display:flex;
            gap:10px;
            align-items:flex-start;
            padding:13px 14px;
            border-radius:14px;
            background:#f8fafc;
            border:1px solid var(--lux-line);
            color:var(--lux-muted);
            font-size:9.5px;
            line-height:1.65;
        }
        .guide-callout i {
            color:var(--lux-blue);
            font-size:13px;
            margin-top:1px;
        }
        .guide-table {
            width:100%;
            border-collapse:collapse;
            font-size:9.5px;
            color:var(--lux-muted);
        }
        .guide-table th {
            text-align:left;
            padding:9px 10px;
            background:#f6f9fc;
            color:var(--lux-ink);
            border-bottom:1px solid var(--lux-line);
            font-weight:800;
        }
        .guide-table td {
            padding:9px 10px;
            border-bottom:1px solid #eef2f5;
            vertical-align:top;
            line-height:1.5;
        }
        .guide-table tr:last-child td { border-bottom:none; }
        .guide-code {
            display:inline-block;
            padding:4px 7px;
            border-radius:7px;
            background:#eef4f9;
            border:1px solid #dbe7f0;
            color:#456f94;
            font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;
            font-size:8.5px;
        }
        .guide-list {
            margin:0;
            padding-left:17px;
            color:var(--lux-muted);
            font-size:9.5px;
            line-height:1.75;
        }
        .guide-list li { margin-bottom:3px; }
        .guide-section-space { margin-top:18px; }

    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# UI HELPERS
# ============================================================
def ui_metric(label, value, help_text="", icon="bi-bar-chart"):
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-icon'><i class='bi {icon}'></i></div>
            <div class='metric-label'>{html.escape(str(label))}</div>
            <div class='metric-value'>{value}</div>
            <div class='metric-help'>{html.escape(str(help_text))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def ui_section(kicker, title, description="", icon="bi-grid"):
    st.markdown(
        f"""
        <div class='section'>
            <div class='section-head'>
                <div>
                    <div class='section-kicker'>{html.escape(kicker)}</div>
                    <div class='section-title'>{html.escape(title)}</div>
                    {f"<div class='section-desc'>{html.escape(description)}</div>" if description else ""}
                </div>
                <div class='section-icon'><i class='bi {icon}'></i></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def ui_header(eyebrow, title, description, step, icon):
    safe_title = html.escape(title)
    st.markdown(
        f"""
        <div class='page-shell'>
            <div class='row g-3 align-items-end'>
                <div class='col-12 col-lg-9'>
                    <div class='eyebrow'><i class='bi {icon}'></i> &nbsp;{html.escape(eyebrow)}</div>
                    <h1 class='page-title'>{safe_title}</h1>
                    <div class='page-description'>{html.escape(description)}</div>
                </div>
                <div class='col-12 col-lg-3 text-lg-end'>
                    <span class='page-meta'><i class='bi bi-layers'></i> {step:02d} / 09</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def ui_chart(fig, height=450):
    fig.update_layout(
        height=height,
        margin=dict(l=15, r=15, t=32, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font=dict(family="Plus Jakarta Sans, sans-serif", size=10, color="#627286"),
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="left", x=0, font=dict(size=9)
        ),
        xaxis=dict(showgrid=False, showline=True, linecolor="#e1e7ee", linewidth=1),
        yaxis=dict(showgrid=True, gridcolor="#edf0f4", zeroline=False, showline=False),
    )
    return fig


def render_chart(fig, use_container_width=True, gap=18):
    """Render a Plotly chart with deliberate vertical spacing from its section text."""
    st.markdown(f"<div class='chart-slot' style='height:{gap}px;'></div>", unsafe_allow_html=True)
    return st.plotly_chart(fig, use_container_width=use_container_width)


def empty_state(title, text, icon="bi-cloud-arrow-up"):
    st.markdown(
        f"""
        <div class='empty'>
            <div class='empty-icon'><i class='bi {icon}'></i></div>
            <div class='empty-title'>{html.escape(title)}</div>
            <div class='empty-text'>{html.escape(text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_row_for_window(bundle, window):
    out = bundle["results"][window]
    return evaluate_metrics(out["Aktual"], out["Hybrid_SARIMA_FTS"])


def make_forecast_chart(history, future, title="Historis dan Forecast Masa Depan"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history.index, y=history.values, mode="lines", name="Historis",
        line=dict(color="#536579", width=2.2)
    ))
    fig.add_trace(go.Scatter(
        x=future["Tanggal"], y=future["Prediksi_Hybrid"], mode="lines+markers", name="Forecast Hybrid",
        line=dict(color="#b89962", width=2.5), marker=dict(size=6)
    ))
    if len(history) and len(future):
        split_x = future["Tanggal"].iloc[0]
        fig.add_vline(x=split_x, line_dash="dot", line_color="#d4dae2", line_width=1.2)
        fig.add_annotation(
            x=split_x, y=1.02, xref="x", yref="paper", text="Mulai Forecast",
            showarrow=False, font=dict(size=9, color="#8994a2"), xanchor="left"
        )
    fig = ui_chart(fig, 480)
    fig.update_yaxes(title="Jumlah Kunjungan")
    fig.update_xaxes(title="Tanggal")
    fig.update_layout(title=title, title_font=dict(size=13, color="#1b2738", family="Plus Jakarta Sans, sans-serif"))
    return fig


def pct_change(base, new):
    if not np.isfinite(base) or base == 0 or not np.isfinite(new):
        return np.nan
    return ((base - new) / abs(base)) * 100.0


def render_research_result_summary(bundle):
    metrics = bundle["metrics"].copy()
    hybrid = metrics[metrics["Model"].eq("Hybrid_SARIMA_FTS")].copy().sort_values("Window")
    if hybrid.empty:
        return

    split_pct = int(bundle["split_ratio"] * 100)
    test_pct = 100 - split_pct
    st.markdown(
        """
        <div class='surface' style='padding:20px 20px 18px; background:linear-gradient(135deg,#ffffff 0%,#fbfcfe 100%);'>
            <div style='display:flex;justify-content:space-between;align-items:flex-start;gap:18px;flex-wrap:wrap;'>
                <div>
                    <div class='section-kicker'>RESEARCH RESULT SUMMARY</div>
                    <div class='section-title' style='font-size:18px;margin-top:4px;'>Ringkasan hasil eksperimen</div>
                    <div class='section-desc'>Konfigurasi terpilih dan performa testing ditampilkan di satu tempat agar hasil penelitian mudah dibaca dan dipresentasikan.</div>
                </div>
                <span class='tag tag-green'><i class='bi bi-check2-circle'></i> Testing terpisah dari tuning</span>
            </div>
        </div>
        """, unsafe_allow_html=True
    )

    cols = st.columns([1.15, 1, 1])
    with cols[0]:
        rows = []
        for _, r in hybrid.iterrows():
            w = int(r["Window"])
            cfg = bundle["best_cfg"][w]
            rows.append(
                f"<div style='display:flex;justify-content:space-between;gap:10px;padding:7px 0;border-bottom:1px solid #eef2f5;'>"
                f"<span style='color:#7b8998;font-size:9px;'>Window {w} → 1</span>"
                f"<b style='color:#1b2738;font-size:10px;'>n={cfg['n_intervals']} · α={cfg['alpha']:.2f}</b></div>"
            )
        st.markdown(
            "<div class='feature'><div class='feature-title'>Konfigurasi terpilih</div>"
            f"<div class='feature-text'>SARIMA{SARIMA_ORDER}{SARIMA_SEASONAL_ORDER} · FTS Chen Weighted · Split {split_pct}:{test_pct}</div>"
            + "".join(rows) + "</div>", unsafe_allow_html=True
        )
    with cols[1]:
        r = hybrid[hybrid["Window"].eq(6)].iloc[0] if (hybrid["Window"] == 6).any() else hybrid.iloc[0]
        st.markdown(
            f"<div class='feature'><div class='feature-title'>Testing snapshot · W{int(r['Window'])}</div>"
            f"<div class='feature-text'>N testing = {len(bundle['test'])} observasi</div>"
            f"<div style='margin-top:10px;display:grid;grid-template-columns:1fr 1fr;gap:8px;'>"
            f"<div><span class='metric-label'>MAPE</span><div class='metric-value' style='font-size:20px;'>{r['MAPE (%)']:.2f}%</div></div>"
            f"<div><span class='metric-label'>MAE</span><div class='metric-value' style='font-size:20px;'>{fmt_num(r['MAE'],1)}</div></div>"
            f"<div><span class='metric-label'>RMSE</span><div class='metric-value' style='font-size:20px;'>{fmt_num(r['RMSE'],1)}</div></div>"
            f"<div><span class='metric-label'>R²</span><div class='metric-value' style='font-size:20px;'>{r['R²']:.4f}</div></div>"
            f"</div></div>", unsafe_allow_html=True
        )
    with cols[2]:
        st.markdown(
            "<div class='feature'><div class='feature-title'>Cara membaca hasil</div>"
            "<div class='feature-text'>MAPE, MAE, dan RMSE mengukur besarnya error pada testing; R² menggambarkan variasi target yang dijelaskan pada evaluasi.</div>"
            "<div class='guide-callout' style='margin-top:10px;'><i class='bi bi-info-circle'></i><div>Konfigurasi dipilih pada validation menggunakan urutan MAPE → MAE → RMSE, bukan menggunakan nilai testing.</div></div>"
            "</div>", unsafe_allow_html=True
        )


def evaluation_comparison_frame(metrics):
    rows = []
    sarima = metrics[metrics["Model"].eq("SARIMA")].sort_values("Window")
    fts = metrics[metrics["Model"].eq("FTS_calibrated")].sort_values("Window")
    hybrid = metrics[metrics["Model"].eq("Hybrid_SARIMA_FTS")].sort_values("Window")

    # Baseline SARIMA is technically the same forecast across FTS windows; keep one row per window
    # only to preserve the pipeline's evaluation structure. Improvement is calculated against the
    # SARIMA metric of the corresponding test row.
    for w in sorted(set(metrics["Window"].astype(int))):
        s = sarima[sarima["Window"].eq(w)]
        f = fts[fts["Window"].eq(w)]
        h = hybrid[hybrid["Window"].eq(w)]
        if s.empty or f.empty or h.empty:
            continue
        s = s.iloc[0]; f = f.iloc[0]; h = h.iloc[0]
        rows.append({
            "Window": w,
            "SARIMA MAPE (%)": s["MAPE (%)"],
            "FTS MAPE (%)": f["MAPE (%)"],
            "Hybrid MAPE (%)": h["MAPE (%)"],
            "Δ MAPE vs SARIMA (%)": pct_change(s["MAPE (%)"], h["MAPE (%)"]),
            "SARIMA RMSE": s["RMSE"],
            "Hybrid RMSE": h["RMSE"],
            "Δ RMSE vs SARIMA (%)": pct_change(s["RMSE"], h["RMSE"]),
            "SARIMA MAE": s["MAE"],
            "Hybrid MAE": h["MAE"],
            "Δ MAE vs SARIMA (%)": pct_change(s["MAE"], h["MAE"]),
        })
    return pd.DataFrame(rows)


def render_tuning_charts(bundle):
    tuning = bundle["v7_tuning"].copy()
    if tuning.empty:
        return
    ui_section("TUNING VISUAL", "Jejak Pemilihan Konfigurasi", "Tampilkan hubungan alpha dan jumlah interval dengan MAPE validation sebelum konfigurasi digunakan pada testing.", "bi-sliders2")
    c1, c2 = st.columns(2)
    for col, window in zip((c1, c2), WINDOWS):
        with col:
            part = tuning[tuning["Window"].eq(window)].copy()
            if part.empty:
                st.info(f"Tidak ada tuning valid untuk Window {window}.")
                continue
            best = bundle["best_cfg"][window]
            alpha_part = part[part["n_intervals"].eq(best["n_intervals"])].sort_values("alpha")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=alpha_part["alpha"], y=alpha_part["MAPE (%)"], mode="lines+markers", name="MAPE validation"))
            selected = alpha_part[alpha_part["alpha"].sub(best["alpha"]).abs().lt(1e-9)]
            if not selected.empty:
                fig.add_trace(go.Scatter(x=selected["alpha"], y=selected["MAPE (%)"], mode="markers", name="Terpilih", marker=dict(size=10, symbol="diamond")))
            fig = ui_chart(fig, 350)
            fig.update_layout(title=f"Window {window} · MAPE vs Alpha", title_font=dict(size=12, color="#1b2738"))
            fig.update_xaxes(title="Alpha (bobot SARIMA)")
            fig.update_yaxes(title="MAPE Validation (%)")
            render_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    for col, window in zip((c3, c4), WINDOWS):
        with col:
            part = tuning[tuning["Window"].eq(window)].copy()
            agg = part.groupby("n_intervals", as_index=False)["MAPE (%)"].min().sort_values("n_intervals")
            if agg.empty:
                continue
            best = bundle["best_cfg"][window]
            fig = go.Figure(go.Scatter(x=agg["n_intervals"], y=agg["MAPE (%)"], mode="lines+markers", name="Min MAPE per interval"))
            sel = agg[agg["n_intervals"].eq(best["n_intervals"])]
            if not sel.empty:
                fig.add_trace(go.Scatter(x=sel["n_intervals"], y=sel["MAPE (%)"], mode="markers", name="Terpilih", marker=dict(size=10, symbol="diamond")))
            fig = ui_chart(fig, 350)
            fig.update_layout(title=f"Window {window} · MAPE vs Interval", title_font=dict(size=12, color="#1b2738"))
            fig.update_xaxes(title="Jumlah interval FTS")
            fig.update_yaxes(title="Minimum MAPE Validation (%)")
            render_chart(fig, use_container_width=True)


# ============================================================
# RESEARCH / EXPORT HELPERS
# ============================================================
def preprocessing_summary(bundle):
    audit = bundle["audit"].copy()
    raw_n = int(bundle["raw"].shape[0]) if isinstance(bundle.get("raw"), pd.DataFrame) else len(audit)
    audit_n = len(audit)
    changed_n = int(audit["perubahan"].sum()) if "perubahan" in audit.columns else 0
    interpolated_n = int((audit["status_preprocessing"] == "Interpolasi linear").sum()) if "status_preprocessing" in audit.columns else 0
    duplicate_n = max(0, raw_n - audit_n)
    actual_n = max(0, audit_n - interpolated_n)
    return {
        "Input rows": raw_n,
        "Aktual": actual_n,
        "Diinterpolasi": interpolated_n,
        "Perubahan nilai": changed_n,
        "Duplikat periode": duplicate_n,
        "Observasi akhir": len(bundle["series"]),
    }


def make_research_zip(bundle, future=None):
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        raw = bundle.get("raw")
        if isinstance(raw, pd.DataFrame):
            zf.writestr("01_raw_dataset.csv", raw.to_csv(index=False))
        zf.writestr("02_preprocessed_dataset.csv", bundle["data"].to_csv(index=False))
        zf.writestr("03_preprocessing_audit.csv", bundle["audit"].to_csv(index=False))
        zf.writestr("04_validation_tuning.csv", bundle["v7_tuning"].to_csv(index=False))
        zf.writestr("05_metrics_testing.csv", bundle["metrics"].to_csv(index=False))
        zf.writestr("06_train_predictions.csv", bundle["train_export"].to_csv(index=False))
        for window, out in bundle["results"].items():
            zf.writestr(f"07_testing_window_{window}.csv", out.to_csv(index=False))
        if future is not None:
            zf.writestr("08_future_forecast.csv", future.to_csv(index=False))

        cfg_lines = [
            "Hybrid SARIMA–FTS Forecast Dashboard",
            "Waterpark Sumenep Asta Tinggi",
            "",
            "MODEL CONFIGURATION",
            f"SARIMA order: {SARIMA_ORDER}",
            f"SARIMA seasonal order: {SARIMA_SEASONAL_ORDER}",
            f"Seasonal period: {SEASONAL_PERIOD}",
            f"Validation size: {VALIDATION_SIZE}",
            f"Split: {int(bundle['split_ratio']*100)}:{100-int(bundle['split_ratio']*100)}",
            "FTS: Chen Weighted",
            f"Windows: {WINDOWS}",
            f"FTS interval grid: {FTS_INTERVAL_GRID}",
            f"Alpha grid: {ALPHA_GRID.tolist()}",
            "Selection order: MAPE → MAE → RMSE",
            "",
            "SELECTED CONFIGURATION",
        ]
        for w in WINDOWS:
            cfg = bundle["best_cfg"][w]
            cfg_lines.append(f"Window {w}: n_intervals={cfg['n_intervals']}, alpha={cfg['alpha']:.2f}")
        zf.writestr("00_model_configuration.txt", "\n".join(cfg_lines))
    mem.seek(0)
    return mem.getvalue()


def render_data_split_band(bundle):
    split_pct = int(bundle["split_ratio"] * 100)
    test_pct = 100 - split_pct
    train_n = bundle["train_n"]
    val_n = VALIDATION_SIZE
    test_n = bundle["test_n"]
    train = bundle["series"].iloc[:train_n - val_n]
    validation = bundle["validation"]
    test = bundle["test"]

    def span(series):
        if len(series) == 0:
            return "—"
        return f"{fmt_month(series.index.min())} – {fmt_month(series.index.max())}"

    st.markdown(
        f"""
        <div class='method-band'>
            <div class='method-band-title'><i class='bi bi-signpost-2'></i> Kronologi Eksperimen</div>
            <div class='method-band-items'>
                <span class='method-chip blue'><b>Train</b> {train_n - val_n} obs</span>
                <span class='method-arrow'>→</span>
                <span class='method-chip gold'><b>Validation</b> {val_n} obs</span>
                <span class='method-arrow'>→</span>
                <span class='method-chip green'><b>Test</b> {test_n} obs</span>
                <span class='method-note'>Split {split_pct}:{test_pct} · validation berada di bagian akhir training</span>
            </div>
            <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:10px;'>
                <div style='padding:8px 9px;border:1px solid #e8edf2;border-radius:10px;background:#fff;'><div class='metric-label'>Train</div><div style='font-size:9.5px;font-weight:700;color:#30445b;margin-top:3px;'>{span(train)}</div></div>
                <div style='padding:8px 9px;border:1px solid #eadfcf;border-radius:10px;background:#fffaf3;'><div class='metric-label'>Validation</div><div style='font-size:9.5px;font-weight:700;color:#7a633c;margin-top:3px;'>{span(validation)}</div></div>
                <div style='padding:8px 9px;border:1px solid #d9ece1;border-radius:10px;background:#f9fdfb;'><div class='metric-label'>Test</div><div style='font-size:9.5px;font-weight:700;color:#2f6d52;margin-top:3px;'>{span(test)}</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def model_comparison_frame(metrics):
    cols = ["Model", "Window", "MAE", "RMSE", "MAPE (%)", "R²"]
    out = metrics[cols].copy()
    out["Skenario"] = out.apply(lambda r: f"{r['Model']} · W{int(r['Window'])}", axis=1)
    return out[["Skenario", "Model", "Window", "MAE", "RMSE", "MAPE (%)", "R²"]]

# ============================================================
# SESSION STATE & SOURCE
# ============================================================
if "future_result" not in st.session_state:
    st.session_state.future_result = None
if "future_signature" not in st.session_state:
    st.session_state.future_signature = None

if "bundle" not in st.session_state:
    default_source = load_default_source()
    if default_source is not None:
        try:
            st.session_state.bundle = cached_run_pipeline(default_source[0], default_source[1], 0.70)
            st.session_state.source_name = "Dataset bawaan repository"
        except Exception:
            st.session_state.bundle = None
    else:
        st.session_state.bundle = None

bundle = st.session_state.get("bundle")
source_name = st.session_state.get("source_name", "Dataset belum dipilih")

# ============================================================
# SIDEBAR — INFORMATION ARCHITECTURE
# ============================================================
st.sidebar.markdown(
    """
    <div class='side-brand'>
        <span class='side-logo'><i class='bi bi-activity'></i></span>
        <span class='side-title'>Hybrid Analytics</span>
        <div class='side-subtitle'>Waterpark Sumenep<br>Forecast Dashboard</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("<div class='side-label'>Workspace</div>", unsafe_allow_html=True)
if bundle is None:
    status_html = """
    <div class='side-status'>
        <div class='caption'>Dataset</div>
        <div class='value'>Belum diproses</div>
        <div class='side-chip'><i class='bi bi-circle'></i> Ready for input</div>
    </div>
    """
else:
    split_pct = f"{int(bundle['split_ratio']*100)}:{100-int(round(bundle['split_ratio']*100))}"
    status_html = f"""
    <div class='side-status'>
        <div class='caption'>Dataset aktif</div>
        <div class='value'>{html.escape(str(source_name))}</div>
        <div class='value' style='margin-top:6px;color:#aebdd0;font-weight:600;'>n = {len(bundle['series'])} · split {split_pct}</div>
        <div class='side-chip'><i class='bi bi-check2-circle'></i> Pipeline ready</div>
    </div>
    """
st.sidebar.markdown(status_html, unsafe_allow_html=True)

st.sidebar.markdown("<div class='side-label'>Navigation</div>", unsafe_allow_html=True)
menu_options = [
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
menu = st.sidebar.radio("Navigation", menu_options, label_visibility="collapsed")

st.sidebar.markdown(
    "<div class='side-divider' style='border-top:1px solid rgba(255,255,255,.08);'></div>",
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    "<div style='font-size:8px;color:#7189a5;line-height:1.5;padding:0 4px;'>Hybrid SARIMA–FTS · Forecast Dashboard</div>",
    unsafe_allow_html=True,
)

# ============================================================
# 01 — BERANDA
# ============================================================
if menu.startswith("01"):
    if bundle is None:
        st.markdown(
            """
            <div class='hero'>
                <div class='hero-grid'>
                    <div class='hero-main'>
                        <div class='hero-kicker'><span class='hero-kicker-dot'></span> MBKM Skema Riset · Time Series Forecasting</div>
                        <div class='hero-title'>Hybrid <span class='hero-accent'>SARIMA–FTS</span><br>Waterpark Sumenep</div>
                        <div class='hero-copy'>
                            Dashboard penelitian yang dirancang untuk memahami pola dan karakteristik kunjungan Waterpark Sumenep melalui pendekatan forecasting time series. Aplikasi ini mengintegrasikan 
                            Hybrid SARIMA–FTS, proses tuning dan validasi, evaluasi performa menggunakan berbagai metrik, serta proyeksi kunjungan masa depan dalam satu workspace analitik yang terstruktur.
                        </div>
                        <div class='hero-actions'>
                            <span class='hero-pill'><i class='bi bi-diagram-3'></i> Chronological workflow</span>
                            <span class='hero-pill'><i class='bi bi-shield-check'></i> Validation before testing</span>
                            <span class='hero-pill'><i class='bi bi-stars'></i> Research reporting</span>
                        </div>
                    </div>
                    <div class='hero-side'>
                        <div class='hero-panel'>
                            <div class='hero-panel-head'>
                                <div class='hero-panel-label'>Research Workspace</div>
                                <div class='hero-live'><span class='hero-live-dot'></span> Ready to analyze</div>
                            </div>
                            <div class='hero-panel-title'>From history to forecast</div>
                            <div class='hero-panel-subtitle'>SARIMA + FTS Chen Weighted → Hybrid Ensemble</div>
                            <div class='hero-mini-chart'>
                                <svg viewBox='0 0 320 82' preserveAspectRatio='none' aria-hidden='true'>
                                    <defs>
                                        <linearGradient id='heroFillEmpty' x1='0' x2='0' y1='0' y2='1'>
                                            <stop offset='0%' stop-color='#7fb8df' stop-opacity='.24'/>
                                            <stop offset='100%' stop-color='#7fb8df' stop-opacity='0'/>
                                        </linearGradient>
                                    </defs>
                                    <path d='M0 63 C24 60 34 67 57 53 S91 45 111 52 S141 34 160 41 S190 30 214 34 S239 22 258 28 S290 15 320 20 L320 82 L0 82 Z'
                                          fill='url(#heroFillEmpty)'/>
                                    <path d='M0 63 C24 60 34 67 57 53 S91 45 111 52 S141 34 160 41 S190 30 214 34 S239 22 258 28 S290 15 320 20'
                                          fill='none' stroke='#d5b979' stroke-width='2.4' stroke-linecap='round'/>
                                    <path d='M0 71 C28 66 46 69 72 61 S105 52 131 54 S166 45 188 48 S222 39 246 42 S280 34 320 36'
                                          fill='none' stroke='rgba(255,255,255,.55)' stroke-width='1.25' stroke-dasharray='4 4'/>
                                    <circle cx='320' cy='20' r='3.5' fill='#d5b979'/>
                                </svg>
                            </div>
                            <div class='hero-stat-row'>
                                <div class='hero-stat'><div class='hero-stat-label'>Seasonality</div><div class='hero-stat-value'>12 bln</div></div>
                                <div class='hero-stat'><div class='hero-stat-label'>Window</div><div class='hero-stat-value'>6 / 12</div></div>
                                <div class='hero-stat'><div class='hero-stat-label'>Model</div><div class='hero-stat-value gold'>Hybrid</div></div>
                            </div>
                            <div class='hero-flow'>
                                <div class='hero-flow-step'><div class='hero-flow-node'>01</div>Data</div>
                                <div class='hero-flow-arrow'>→</div>
                                <div class='hero-flow-step'><div class='hero-flow-node'>02</div>Tuning</div>
                                <div class='hero-flow-arrow'>→</div>
                                <div class='hero-flow-step'><div class='hero-flow-node'>03</div>Testing</div>
                                <div class='hero-flow-arrow'>→</div>
                                <div class='hero-flow-step'><div class='hero-flow-node'>04</div>Forecast</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        empty_state(
            "Workspace belum memiliki hasil",
            "Gunakan menu Input Dataset untuk memuat data dan menjalankan pipeline Hybrid SARIMA–FTS.",
            "bi-database-add",
        )
    else:
        split_pct = f"{int(bundle['split_ratio']*100)}:{100-int(round(bundle['split_ratio']*100))}"
        m6 = metric_row_for_window(bundle, 6)
        m12 = metric_row_for_window(bundle, 12)
        active_window = 6 if m6["MAPE (%)"] <= m12["MAPE (%)"] else 12
        active_metric = m6 if active_window == 6 else m12
        active_cfg = bundle["best_cfg"][active_window]
        alpha_pct = round(active_cfg["alpha"] * 100)
        fts_pct = 100 - alpha_pct
        test_n = bundle["test_n"]

        st.markdown(
            f"""
            <div class='hero'>
                <div class='hero-grid'>
                    <div class='hero-main'>
                        <div class='hero-kicker'><span class='hero-kicker-dot'></span> MBKM Skema Riset · Time Series Forecasting</div>
                        <div class='hero-title'>Hybrid <span class='hero-accent'>SARIMA–FTS</span><br>Waterpark Sumenep</div>
                        <div class='hero-copy'>
                            Dashboard penelitian yang dirancang untuk memahami pola dan karakteristik kunjungan Waterpark Sumenep melalui pendekatan forecasting time series. Aplikasi ini mengintegrasikan 
                            Hybrid SARIMA–FTS, proses tuning dan validasi, evaluasi performa menggunakan berbagai metrik, serta proyeksi kunjungan masa depan dalam satu workspace analitik yang terstruktur.
                        </div>
                        <div class='hero-actions'>
                            <span class='hero-pill'><i class='bi bi-database'></i> <strong>{html.escape(str(source_name))}</strong></span>
                            <span class='hero-pill'><i class='bi bi-layout-split'></i> Split Data {split_pct}</span>
                            <span class='hero-pill'><i class='bi bi-check2-circle'></i> Pipeline Processed</span>
                        </div>
                    </div>
                    <div class='hero-side'>
                        <div class='hero-panel'>
                            <div class='hero-panel-head'>
                                <div class='hero-panel-label'>Live experiment snapshot</div>
                                <div class='hero-live'><span class='hero-live-dot'></span> Processed</div>
                            </div>
                            <div class='hero-panel-title'>Hybrid Performance</div>
                            <div class='hero-panel-subtitle'>Ringkasan testing dari skenario aktif dashboard</div>
                            <div class='hero-mini-chart'>
                                <svg viewBox='0 0 320 82' preserveAspectRatio='none' aria-hidden='true'>
                                    <defs>
                                        <linearGradient id='heroFillActive' x1='0' x2='0' y1='0' y2='1'>
                                            <stop offset='0%' stop-color='#7fb8df' stop-opacity='.28'/>
                                            <stop offset='100%' stop-color='#7fb8df' stop-opacity='0'/>
                                        </linearGradient>
                                    </defs>
                                    <path d='M0 64 C22 56 38 66 57 54 S91 48 112 50 S142 39 162 43 S188 28 212 34 S238 22 258 27 S290 14 320 18 L320 82 L0 82 Z'
                                          fill='url(#heroFillActive)'/>
                                    <path d='M0 64 C22 56 38 66 57 54 S91 48 112 50 S142 39 162 43 S188 28 212 34 S238 22 258 27 S290 14 320 18'
                                          fill='none' stroke='#d5b979' stroke-width='2.5' stroke-linecap='round'/>
                                    <path d='M0 70 C28 66 48 68 71 61 S106 57 131 57 S161 50 188 51 S219 43 245 45 S282 37 320 38'
                                          fill='none' stroke='rgba(255,255,255,.56)' stroke-width='1.25' stroke-dasharray='4 4'/>
                                    <circle cx='320' cy='18' r='3.5' fill='#d5b979'/>
                                </svg>
                            </div>
                            <div class='hero-stat-row'>
                                <div class='hero-stat'><div class='hero-stat-label'>MAPE aktif</div><div class='hero-stat-value gold'>{active_metric["MAPE (%)"]:.2f}%</div></div>
                                <div class='hero-stat'><div class='hero-stat-label'>Testing</div><div class='hero-stat-value'>{test_n} obs</div></div>
                                <div class='hero-stat'><div class='hero-stat-label'>Window</div><div class='hero-stat-value'>W{active_window}</div></div>
                            </div>
                            <div class='hero-flow'>
                                <div class='hero-flow-step'><div class='hero-flow-node'>S</div>α {alpha_pct}%</div>
                                <div class='hero-flow-arrow'>+</div>
                                <div class='hero-flow-step'><div class='hero-flow-node'>F</div>FTS {fts_pct}%</div>
                                <div class='hero-flow-arrow'>→</div>
                                <div class='hero-flow-step'><div class='hero-flow-node'>H</div>Hybrid</div>
                                <div class='hero-flow-arrow'>→</div>
                                <div class='hero-flow-step'><div class='hero-flow-node'>W</div>W{active_window} → 1</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        ui_section("Overview", "Workspace Snapshot", "Ringkasan dataset dan konfigurasi aktif.", "bi-grid-1x2")
        c1, c2, c3, c4 = st.columns(4)
        with c1: ui_metric("Observasi", fmt_num(len(bundle["series"])), "Data bulanan valid", "bi-database")
        with c2: ui_metric("Training", fmt_num(bundle["train_n"]), f"{int(bundle['split_ratio']*100)}% dari data", "bi-stack")
        with c3: ui_metric("Testing", fmt_num(bundle["test_n"]), f"{100-int(bundle['split_ratio']*100)}% dari data", "bi-clipboard-data")
        with c4: ui_metric("Validation", fmt_num(VALIDATION_SIZE), "Bagian akhir training", "bi-shield-check")

        render_data_split_band(bundle)

        ui_section("Workflow", "Alur Penelitian", "Setiap tahap disusun mengikuti urutan penggunaan dashboard.", "bi-signpost-split")
        workflow = [
            ("01", "bi-upload", "Input Dataset", "Unggah data dan pilih split."),
            ("02", "bi-funnel", "Preprocessing", "Normalisasi periode dan interpolasi."),
            ("03", "bi-cpu", "Hybrid Model", "SARIMA + FTS dengan tuning."),
            ("04", "bi-graph-up", "Evaluation", "MAE, RMSE, MAPE, dan R²."),
            ("05", "bi-calendar3", "Future Forecast", "Proyeksi setelah historis."),
        ]
        flow_html = "<div class='row g-3'>"
        workflow_open = "<div class='workflow'>"
        for num, icon, title, text_desc in workflow:
            flow_html += (
                f"<div class='col-12 col-sm-6 col-lg'>{workflow_open}"
                f"<div class='workflow-num'>{num}</div>"
                f"<div class='workflow-icon'><i class='bi {icon}'></i></div>"
                f"<div class='workflow-title'>{title}</div>"
                f"<div class='workflow-text'>{text_desc}</div>"
                "</div></div>"
            )
        flow_html += "</div>"
        st.markdown(flow_html, unsafe_allow_html=True)

        render_research_result_summary(bundle)

        ui_section("Visual", "Aktual vs Prediksi Hybrid", "Testing Window 6 → 1 sebagai preview utama.", "bi-bar-chart-line")
        out = bundle["results"][6]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=out["Tanggal"], y=out["Aktual"], mode="lines+markers", name="Aktual", line=dict(color="#52667b", width=2.2), marker=dict(size=5)))
        fig.add_trace(go.Scatter(x=out["Tanggal"], y=out["Hybrid_SARIMA_FTS"], mode="lines+markers", name="Prediksi Hybrid", line=dict(color="#b89962", width=2.4), marker=dict(size=5)))
        fig = ui_chart(fig, 430)
        fig.update_layout(title="Testing — Window 6 → 1", title_font=dict(size=13, color="#1b2738"))
        fig.update_xaxes(title="Tanggal")
        fig.update_yaxes(title="Jumlah Kunjungan")
        render_chart(fig, use_container_width=True)

# ============================================================
# 02 — INPUT DATASET
# ============================================================
elif menu.startswith("02"):
    ui_header("DATA WORKFLOW", "Input Dataset", "Unggah dataset, tentukan split, dan jalankan pipeline Hybrid SARIMA–FTS secara kronologis.", 2, "bi-cloud-arrow-up")
    st.markdown(
        """
        <div class='stepper'>
            <div class='step on'><span class='step-dot'>01</span> Upload</div><span class='step-arrow'>→</span>
            <div class='step on'><span class='step-dot'>02</span> Split</div><span class='step-arrow'>→</span>
            <div class='step'><span class='step-dot'>03</span> Preprocess</div><span class='step-arrow'>→</span>
            <div class='step'><span class='step-dot'>04</span> Run Model</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.28, .92], gap="large")
    with left:
        ui_section("01 · Source", "Upload Dataset", "CSV atau Excel dengan kolom periode/tanggal dan jumlah kunjungan.", "bi-file-earmark-arrow-up")
        st.markdown("<div class='surface'>", unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Pilih dataset",
            type=["xlsx", "xls", "csv"],
            help="Format yang didukung: 2 kolom (periode + kunjungan) atau 3 kolom Waterpark Asta Tinggi (No + Periode + Kunjungan).",
        )
        if uploaded is None:
            st.markdown("<div class='note'><i class='bi bi-info-circle'></i><div>Tanpa upload, dashboard dapat menggunakan dataset bawaan repository sebagai sumber default.</div></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<span class='tag tag-green'><i class='bi bi-file-earmark-check'></i> {html.escape(uploaded.name)}</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        ui_section("02 · Split", "Pembagian Data", "Gunakan pembagian kronologis untuk menjaga urutan waktu.", "bi-layout-split")
        st.markdown("<div class='surface'>", unsafe_allow_html=True)
        split_choice = st.selectbox("Train : Test", ["70:30", "80:20"], index=0 if (bundle is None or bundle["split_ratio"] == 0.70) else 1)
        split_ratio = 0.70 if split_choice == "70:30" else 0.80
        st.markdown(
            f"<div class='row g-2 mt-1'><div class='col-6'><span class='tag tag-blue'><i class='bi bi-collection'></i> Train {int(split_ratio*100)}%</span></div><div class='col-6'><span class='tag tag-gold'><i class='bi bi-clipboard-data'></i> Test {100-int(split_ratio*100)}%</span></div></div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div class='split-line'></div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-explain' style='font-size:10px;color:#6f7d8f;'>Validation menggunakan 12 observasi terakhir dari bagian training.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    ui_section("03 · Run", "Eksekusi Analisis", "Perhitungan model dan tuning hanya dijalankan setelah tombol berikut ditekan.", "bi-play-circle")
    run_clicked = st.button("Jalankan Hybrid SARIMA–FTS", type="primary", use_container_width=True)
    if run_clicked:
        if uploaded is None:
            default_source = load_default_source()
            if default_source is None:
                st.error("Dataset bawaan tidak tersedia. Silakan unggah dataset.")
                st.stop()
            file_bytes, filename = default_source
            source_label = "Dataset Bawaan Repository"
        else:
            file_bytes, filename = uploaded.getvalue(), uploaded.name
            source_label = uploaded.name
        with st.spinner("Menjalankan preprocessing, tuning, dan evaluasi model..."):
            try:
                new_bundle = cached_run_pipeline(file_bytes, filename, split_ratio)
                st.session_state.bundle = new_bundle
                st.session_state.source_name = source_label
                st.session_state.future_result = None
                st.session_state.future_signature = None
                bundle = new_bundle
                source_name = source_label
            except Exception as exc:
                st.error(f"Proses gagal: {exc}")
                st.stop()
        st.success(f"Dataset berhasil diproses dengan split {split_choice}.")

    if bundle is None:
        empty_state("Belum ada dataset aktif", "Upload dataset atau gunakan sumber bawaan lalu jalankan analisis.", "bi-database-add")
    else:
        ui_section("Active Dataset", "Dataset Yang Sedang Digunakan", "Status dataset dan ringkasan preprocessing.", "bi-database-check")
        st.markdown(
            f"<div class='surface'><span class='tag tag-green'><i class='bi bi-check-circle'></i> Aktif</span> &nbsp; <b style='font-size:11px'>{html.escape(str(st.session_state.get('source_name','Dataset')))}</b><br><span style='display:block;color:#7f8c9c;font-size:10px;margin-top:7px;'>{len(bundle['series'])} observasi valid · {bundle['date_col']} → {bundle['value_col']}</span></div>",
            unsafe_allow_html=True,
        )
        c1,c2,c3,c4=st.columns(4)
        with c1: ui_metric("Observasi", fmt_num(len(bundle["series"])), "Setelah preprocessing", "bi-database")
        with c2: ui_metric("Training", fmt_num(bundle["train_n"]), f"{int(bundle['split_ratio']*100)}%", "bi-stack")
        with c3: ui_metric("Validation", fmt_num(VALIDATION_SIZE), "Observasi terakhir training", "bi-shield-check")
        with c4: ui_metric("Testing", fmt_num(bundle["test_n"]), f"{100-int(bundle['split_ratio']*100)}%", "bi-clipboard-data")

        prep = preprocessing_summary(bundle)
        ui_section("Data Quality", "Ringkasan Preprocessing", "Jejak perubahan data ditampilkan agar proses pembersihan dapat diaudit.", "bi-shield-check")
        p1,p2,p3,p4,p5 = st.columns(5)
        with p1: ui_metric("Input", fmt_num(prep["Input rows"]), "Baris dataset terbaca", "bi-file-earmark-text")
        with p2: ui_metric("Aktual", fmt_num(prep["Aktual"]), "Tidak diinterpolasi", "bi-check2-circle")
        with p3: ui_metric("Interpolasi", fmt_num(prep["Diinterpolasi"]), "Nilai 0 / missing", "bi-bezier2")
        with p4: ui_metric("Perubahan", fmt_num(prep["Perubahan nilai"]), "Baris yang berubah", "bi-pencil-square")
        with p5: ui_metric("Final", fmt_num(prep["Observasi akhir"]), "Observasi siap model", "bi-database-check")

        render_data_split_band(bundle)

        ui_section("Trend", "Data Setelah Preprocessing", "Periode dinormalisasi dan nilai 0/missing diproses melalui interpolasi linear.", "bi-graph-up-arrow")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=bundle["series"].index, y=bundle["series"].values, mode="lines+markers", name="Aktual", line=dict(color="#356a9c", width=2.1), marker=dict(size=4)))
        fig = ui_chart(fig, 430)
        fig.update_layout(title="Data Historis Setelah Preprocessing", title_font=dict(size=13, color="#1b2738"))
        fig.update_xaxes(title="Tanggal"); fig.update_yaxes(title="Jumlah Kunjungan")
        render_chart(fig, use_container_width=True)
        with st.expander("Audit preprocessing"):
            st.dataframe(bundle["audit"], use_container_width=True, hide_index=True)

# ============================================================
# 03 — DATA HISTORIS
# ============================================================
elif menu.startswith("03"):
    ui_header("DATA EXPLORATION", "Data Historis", "Eksplorasi tren, statistik dasar, dan observasi bulanan dari dataset aktif.", 3, "bi-clock-history")
    if bundle is None:
        empty_state("Dataset belum diproses", "Buka Input Dataset untuk memulai analisis.", "bi-database-add")
        st.stop()
    series = bundle["series"]
    min_date, max_date = series.index.min().date(), series.index.max().date()
    st.markdown("<div class='surface'>", unsafe_allow_html=True)
    date_range = st.date_input("Rentang tanggal", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    st.markdown("</div>", unsafe_allow_html=True)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        filtered = series[(series.index.date >= date_range[0]) & (series.index.date <= date_range[1])]
    else:
        filtered = series
    c1,c2,c3,c4=st.columns(4)
    with c1: ui_metric("Observasi", fmt_num(len(filtered)), "Rentang terpilih", "bi-calendar3")
    with c2: ui_metric("Minimum", fmt_num(filtered.min(),1), "Jumlah kunjungan", "bi-arrow-down-circle")
    with c3: ui_metric("Maksimum", fmt_num(filtered.max(),1), "Jumlah kunjungan", "bi-arrow-up-circle")
    with c4: ui_metric("Rata-rata", fmt_num(filtered.mean(),1), "Jumlah kunjungan", "bi-bar-chart-line")
    ui_section("Trend", "Pergerakan Kunjungan", "Visualisasi seri waktu pada rentang yang dipilih.", "bi-activity")
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=filtered.index, y=filtered.values, mode="lines+markers", name="Aktual", line=dict(color="#356a9c", width=2.2), marker=dict(size=5)))
    fig=ui_chart(fig,460); fig.update_layout(title="Data Historis", title_font=dict(size=13,color="#1b2738")); fig.update_xaxes(title="Tanggal"); fig.update_yaxes(title="Jumlah Kunjungan")
    render_chart(fig,use_container_width=True)
    ui_section("Table", "Data Observasi", "Tabel ringkas dengan format tanggal dan jumlah kunjungan.", "bi-table")
    hist_df=filtered.rename("Jumlah_Kunjungan").reset_index().rename(columns={"index":"Tanggal"})
    st.dataframe(hist_df,use_container_width=True,hide_index=True)
    st.download_button("Download Data Historis",hist_df.to_csv(index=False).encode("utf-8"),"data_historis.csv","text/csv")

# ============================================================
# 04 — HASIL FORECASTING
# ============================================================
elif menu.startswith("04"):
    ui_header("MODEL OUTPUT", "Hasil Forecasting", "Lihat prediksi one-step-ahead dan bandingkan komponen SARIMA, FTS, serta Hybrid.", 4, "bi-bar-chart-line")
    if bundle is None:
        empty_state("Dataset belum diproses", "Jalankan pipeline dari Input Dataset terlebih dahulu.", "bi-cpu")
        st.stop()
    st.markdown("<div class='surface surface-tight'>", unsafe_allow_html=True)
    window_choice=st.radio("Skenario",["Window 6 → 1","Window 12 → 1"],horizontal=True)
    compare_all=st.toggle("Tampilkan SARIMA dan FTS",value=True)
    st.markdown("</div>",unsafe_allow_html=True)
    window=6 if window_choice.startswith("Window 6") else 12
    out=bundle["results"][window]
    met=evaluate_metrics(out["Aktual"],out["Hybrid_SARIMA_FTS"])
    cfg=bundle["best_cfg"][window]
    st.markdown(
        f"<div class='method-band'><div class='method-band-title'><i class='bi bi-info-circle'></i> Skenario Aktif</div>"
        f"<div class='method-band-items'><span class='method-chip blue'><b>Testing</b> N={len(out)}</span><span class='method-arrow'>→</span>"
        f"<span class='method-chip gold'><b>MAPE</b> {met['MAPE (%)']:.2f}%</span><span class='method-arrow'>→</span>"
        f"<span class='method-chip green'><b>α</b> {cfg['alpha']:.2f}</span><span class='method-note'>Detail metrik lengkap tersedia di Evaluasi Model</span></div></div>",
        unsafe_allow_html=True,
    )
    ui_section("Forecast", window_choice, "Garis utama menunjukkan hasil Hybrid; detail model dapat diaktifkan.", "bi-graph-up")
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=out["Tanggal"],y=out["Aktual"],mode="lines+markers",name="Aktual",line=dict(color="#52667b",width=2.4),marker=dict(size=5)))
    if compare_all:
        fig.add_trace(go.Scatter(x=out["Tanggal"],y=out["SARIMA"],mode="lines",name="SARIMA",line=dict(color="#8ea0b2",width=1.7,dash="dot")))
        fig.add_trace(go.Scatter(x=out["Tanggal"],y=out["FTS_calibrated"],mode="lines",name="FTS Terkalibrasi",line=dict(color="#6f84a1",width=1.8,dash="dash")))
    fig.add_trace(go.Scatter(x=out["Tanggal"],y=out["Hybrid_SARIMA_FTS"],mode="lines+markers",name="Prediksi Hybrid",line=dict(color="#b89962",width=2.6),marker=dict(size=5)))
    fig=ui_chart(fig,500); fig.update_layout(title=f"Aktual vs Prediksi — {window_choice}",title_font=dict(size=13,color="#1b2738")); fig.update_xaxes(title="Tanggal"); fig.update_yaxes(title="Jumlah Kunjungan")
    render_chart(fig,use_container_width=True)
    ui_section("Error Analysis", "Selisih Aktual dan Prediksi", "Error Hybrid dihitung pada data testing.", "bi-bar-chart-steps")
    err_fig=go.Figure()
    err_fig.add_trace(go.Bar(x=out["Tanggal"], y=out["Error_Hybrid"], name="Error Hybrid", marker_color="#c5a76b"))
    err_fig=ui_chart(err_fig,340); err_fig.update_layout(title="Error per Periode",title_font=dict(size=12,color="#1b2738")); err_fig.update_xaxes(title="Tanggal"); err_fig.update_yaxes(title="Aktual − Prediksi")
    render_chart(err_fig,use_container_width=True)
    ui_section("Detail", "Tabel Prediksi", "Tabel lengkap untuk dokumentasi dan ekspor.", "bi-table")
    display_cols=["Tanggal","Aktual","SARIMA","FTS_calibrated","Hybrid_SARIMA_FTS","Error_Hybrid","alpha","n_intervals"]
    st.dataframe(out[display_cols],use_container_width=True,hide_index=True)
    st.download_button("Download Hasil Forecasting",out.to_csv(index=False).encode("utf-8"),f"hasil_hybrid_window_{window}.csv","text/csv")

# ============================================================
# 05 — WINDOWING
# ============================================================
elif menu.startswith("05"):
    ui_header(
        "SCENARIO ANALYSIS",
        "Windowing",
        "Tinjau hasil forecasting pada dua skenario window secara terpisah agar fokus tetap terjaga dan tidak terjadi penumpukan informasi.",
        5,
        "bi-grid-3x3-gap",
    )
    if bundle is None:
        empty_state("Dataset belum diproses", "Jalankan pipeline terlebih dahulu.", "bi-sliders2")
        st.stop()

    st.markdown(
        """
        <div class='scenario-banner'>
            <div class='left'>
                <div class='icon'><i class='bi bi-grid-3x3-gap'></i></div>
                <div>
                    <div class='title'>Skenario Forecasting</div>
                    <div class='text'>Pilih satu window untuk melihat metrik, grafik, dan tabel secara fokus.</div>
                </div>
            </div>
            <div class='badge'>One scenario at a time</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab6, tab12 = st.tabs(["Window 6 → 1", "Window 12 → 1"])

    def render_window_detail(window):
        out = bundle["results"][window]
        metric = evaluate_metrics(out["Aktual"], out["Hybrid_SARIMA_FTS"])
        cfg = bundle["best_cfg"][window]

        ui_section(
            "ACTIVE SCENARIO",
            f"Window {window} → 1",
            "Prediksi one-step-ahead pada data testing menggunakan konfigurasi Hybrid SARIMA–FTS terpilih.",
            "bi-calendar2-week" if window == 6 else "bi-calendar2-range",
        )

        st.markdown(
            f"<div class='method-band'><div class='method-band-title'><i class='bi bi-sliders2'></i> Ringkasan Skenario</div>"
            f"<div class='method-band-items'><span class='method-chip blue'><b>Testing</b> N={len(out)}</span><span class='method-arrow'>→</span>"
            f"<span class='method-chip gold'><b>MAPE</b> {metric['MAPE (%)']:.2f}%</span><span class='method-arrow'>→</span>"
            f"<span class='method-chip green'><b>α</b> {cfg['alpha']:.2f}</span><span class='method-note'>FTS intervals = {cfg['n_intervals']}</span></div></div>",
            unsafe_allow_html=True,
        )

        ui_section(
            "VISUALIZATION",
            "Aktual vs Prediksi Hybrid",
            "Jarak visualisasi dibuat lebih longgar agar grafik dapat dibaca tanpa berhimpitan dengan teks.",
            "bi-graph-up-arrow",
        )
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=out["Tanggal"], y=out["Aktual"], mode="lines+markers", name="Aktual",
            line=dict(color="#52667b", width=2.2), marker=dict(size=5),
        ))
        fig.add_trace(go.Scatter(
            x=out["Tanggal"], y=out["Hybrid_SARIMA_FTS"], mode="lines+markers", name="Prediksi Hybrid",
            line=dict(color="#b89962", width=2.5), marker=dict(size=5),
        ))
        fig = ui_chart(fig, 470)
        fig.update_layout(
            title=f"Testing · Window {window} → 1",
            title_font=dict(size=13, color="#1b2738"),
        )
        fig.update_xaxes(title="Tanggal")
        fig.update_yaxes(title="Jumlah Kunjungan")
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        render_chart(fig, use_container_width=True, gap=6)
        st.markdown("</div>", unsafe_allow_html=True)

        ui_section(
            "DETAIL",
            "Tabel Prediksi",
            "Output lengkap untuk dokumentasi dan ekspor hasil skenario aktif.",
            "bi-table",
        )
        display_cols = ["Tanggal", "Aktual", "SARIMA", "FTS_raw", "FTS_calibrated", "Hybrid_SARIMA_FTS", "Error_Hybrid", "alpha", "n_intervals"]
        st.dataframe(out[display_cols], use_container_width=True, hide_index=True)
        st.download_button(
            f"Download Hasil Window {window}",
            out.to_csv(index=False).encode("utf-8"),
            f"hasil_hybrid_window_{window}.csv",
            "text/csv",
            key=f"download_window_{window}",
        )

    with tab6:
        render_window_detail(6)
    with tab12:
        render_window_detail(12)

# 06 — EVALUASI MODEL
# ============================================================
elif menu.startswith("06"):
    ui_header("MODEL EVALUATION", "Evaluasi Model", "Ringkasan performa testing, kontribusi Hybrid terhadap baseline, dan diagnosis error untuk SARIMA, FTS terkalibrasi, dan Hybrid SARIMA–FTS.", 6, "bi-speedometer2")
    if bundle is None:
        empty_state("Dataset belum diproses", "Jalankan pipeline terlebih dahulu.", "bi-speedometer2")
        st.stop()

    metrics = bundle["metrics"].copy()
    hybrid_metrics = metrics[metrics["Model"] == "Hybrid_SARIMA_FTS"].copy()

    ui_section("Hybrid Performance", "Metrik Testing", "MAE, RMSE, MAPE, dan R² untuk dua skenario window. N menunjukkan jumlah observasi testing.", "bi-bar-chart-line")
    c1,c2,c3,c4 = st.columns(4)
    with c1: ui_metric("W6 · MAE", fmt_num(hybrid_metrics.loc[hybrid_metrics["Window"]==6,"MAE"].iloc[0],2), f"Testing · N={bundle['test_n']}", "bi-rulers")
    with c2: ui_metric("W6 · MAPE", f"{hybrid_metrics.loc[hybrid_metrics['Window']==6,'MAPE (%)'].iloc[0]:.2f}%", f"Valid N={np.count_nonzero(bundle['results'][6]['Aktual'].to_numpy()!=0)}", "bi-percent")
    with c3: ui_metric("W12 · MAE", fmt_num(hybrid_metrics.loc[hybrid_metrics["Window"]==12,"MAE"].iloc[0],2), f"Testing · N={bundle['test_n']}", "bi-rulers")
    with c4: ui_metric("W12 · MAPE", f"{hybrid_metrics.loc[hybrid_metrics['Window']==12,'MAPE (%)'].iloc[0]:.2f}%", f"Valid N={np.count_nonzero(bundle['results'][12]['Aktual'].to_numpy()!=0)}", "bi-percent")

    ui_section("Baseline Comparison", "Kontribusi Hybrid Terhadap SARIMA", "Perubahan error dihitung terhadap SARIMA pada testing untuk window yang sama; nilai positif berarti error Hybrid lebih rendah daripada baseline.", "bi-bar-chart")
    improvement = evaluation_comparison_frame(metrics)
    if not improvement.empty:
        st.dataframe(improvement.round(4), use_container_width=True, hide_index=True)

        vis1, vis2 = st.columns(2)
        with vis1:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=[f"W{int(w)}" for w in improvement["Window"]], y=improvement["SARIMA MAPE (%)"], name="SARIMA"))
            fig.add_trace(go.Bar(x=[f"W{int(w)}" for w in improvement["Window"]], y=improvement["Hybrid MAPE (%)"], name="Hybrid"))
            fig = ui_chart(fig, 370)
            fig.update_layout(barmode="group", title="MAPE Testing · SARIMA vs Hybrid", title_font=dict(size=12,color="#1b2738"))
            fig.update_yaxes(title="MAPE (%)")
            render_chart(fig,use_container_width=True)
        with vis2:
            fig = go.Figure(go.Bar(x=[f"W{int(w)}" for w in improvement["Window"]], y=improvement["Δ MAPE vs SARIMA (%)"], name="Δ MAPE"))
            fig = ui_chart(fig, 370)
            fig.update_layout(title="Perubahan MAPE terhadap SARIMA", title_font=dict(size=12,color="#1b2738"))
            fig.update_yaxes(title="Perubahan (%)", zeroline=True)
            render_chart(fig,use_container_width=True)

    ui_section("Model Comparison", "Perbandingan Seluruh Komponen", "Gunakan tabel ini untuk melihat MAE, RMSE, MAPE, dan R² tanpa membuat ranking otomatis.", "bi-columns-gap")
    comparison = model_comparison_frame(metrics).copy()
    comparison["Skenario"] = comparison.apply(
        lambda r: "SARIMA · Baseline" if r["Model"] == "SARIMA" else (f"FTS Chen · W{int(r['Window'])}" if r["Model"] == "FTS_calibrated" else f"Hybrid · W{int(r['Window'])}"),
        axis=1,
    )
    st.dataframe(comparison.round(4), use_container_width=True, hide_index=True)

    col1,col2 = st.columns(2)
    with col1:
        fig=go.Figure()
        fig.add_trace(go.Bar(x=["W6","W12"],y=hybrid_metrics["MAE"],name="MAE"))
        fig.add_trace(go.Bar(x=["W6","W12"],y=hybrid_metrics["RMSE"],name="RMSE"))
        fig=ui_chart(fig,410); fig.update_layout(barmode="group",title="MAE dan RMSE Hybrid",title_font=dict(size=12,color="#1b2738")); fig.update_yaxes(title="Nilai Error")
        render_chart(fig,use_container_width=True)
    with col2:
        fig=go.Figure(go.Bar(x=["W6","W12"],y=hybrid_metrics["MAPE (%)"],name="MAPE"))
        fig=ui_chart(fig,410); fig.update_layout(title="MAPE Hybrid",title_font=dict(size=12,color="#1b2738")); fig.update_yaxes(title="MAPE (%)")
        render_chart(fig,use_container_width=True)

    ui_section("Error Diagnostics", "Diagnosis Residual/Error", "Periksa pola residual, distribusi error, aktual vs prediksi, serta uji Ljung–Box pada residual untuk skenario terpilih.", "bi-activity")
    diag_window = st.radio("Skenario diagnostik", [6,12], format_func=lambda x: f"Window {x} → 1", horizontal=True, key="diag_window")
    diag = bundle["results"][diag_window].copy()
    diag["Residual"] = diag["Error_Hybrid"]
    d1,d2 = st.columns(2)
    with d1:
        fig=go.Figure(go.Bar(x=diag["Tanggal"], y=diag["Residual"], name="Residual"))
        fig=ui_chart(fig,360); fig.update_layout(title=f"Residual per Periode · W{diag_window}", title_font=dict(size=12,color="#1b2738")); fig.update_xaxes(title="Tanggal"); fig.update_yaxes(title="Aktual − Prediksi")
        render_chart(fig,use_container_width=True)
    with d2:
        fig=go.Figure(go.Histogram(x=diag["Residual"], nbinsx=12, name="Residual"))
        fig=ui_chart(fig,360); fig.update_layout(title=f"Distribusi Residual · W{diag_window}", title_font=dict(size=12,color="#1b2738")); fig.update_xaxes(title="Residual"); fig.update_yaxes(title="Frekuensi")
        render_chart(fig,use_container_width=True)

    d3,d4,d5,d6 = st.columns(4)
    mean_err = float(diag["Residual"].mean())
    std_err = float(diag["Residual"].std(ddof=0))
    min_err = float(diag["Residual"].min())
    max_err = float(diag["Residual"].max())
    with d3: ui_metric("Mean Error", fmt_num(mean_err,2), "Rata-rata residual", "bi-calculator")
    with d4: ui_metric("Std. Error", fmt_num(std_err,2), "Simpangan residual", "bi-distribute-vertical")
    with d5: ui_metric("Min Error", fmt_num(min_err,2), "Residual minimum", "bi-arrow-down-circle")
    with d6: ui_metric("Max Error", fmt_num(max_err,2), "Residual maksimum", "bi-arrow-up-circle")

    fig=go.Figure(go.Scatter(x=diag["Aktual"], y=diag["Hybrid_SARIMA_FTS"], mode="markers", name="Testing"))
    fig=ui_chart(fig,380); fig.update_layout(title=f"Aktual vs Prediksi · W{diag_window}",title_font=dict(size=12,color="#1b2738")); fig.update_xaxes(title="Aktual"); fig.update_yaxes(title="Prediksi Hybrid")
    render_chart(fig,use_container_width=True)

    # Ljung-Box is only shown when enough residual observations are available for the requested lags.
    n_resid = len(diag["Residual"])
    if n_resid >= 6:
        max_lag = min(6, n_resid - 2)
        try:
            lb = acorr_ljungbox(diag["Residual"], lags=max_lag, return_df=True)
            lb = lb.reset_index().rename(columns={"index":"Lag", "lb_stat":"Q-Stat", "lb_pvalue":"p-value"})
            ui_section("Residual Test", "Uji Ljung–Box", f"Pemeriksaan autokorelasi residual hingga lag {max_lag}. Hasil digunakan sebagai diagnosis pola yang tersisa, bukan sebagai satu-satunya dasar pemilihan model.", "bi-bezier2")
            st.dataframe(lb.round(4), use_container_width=True, hide_index=True)
        except Exception as exc:
            st.caption(f"Uji Ljung–Box tidak dapat dihitung: {exc}")

    ui_section("Evaluation Table", "Seluruh Hasil Testing", "Tabel lengkap SARIMA, FTS terkalibrasi, dan Hybrid pada masing-masing window.", "bi-table")
    st.dataframe(metrics.round(4),use_container_width=True,hide_index=True)

    ui_section("Research Export", "Export Hasil Penelitian", "Gabungkan dataset, preprocessing, tuning, metrik, prediksi, konfigurasi, dan future forecast (jika sudah dibuat) dalam satu ZIP.", "bi-file-earmark-zip")
    future_export = st.session_state.get("future_result")
    zip_bytes = make_research_zip(bundle, future=future_export)
    st.download_button(
        "Download Complete Research Package",
        zip_bytes,
        "hybrid_sarima_fts_research_package.zip",
        "application/zip",
        use_container_width=True,
    )

# ============================================================
# 07 — FUTURE FORECAST
# ============================================================
elif menu.startswith("07"):
    ui_header("FUTURE FORECAST", "Forecast Masa Depan", "Bangun proyeksi setelah observasi historis terakhir berdasarkan konfigurasi Hybrid terpilih.", 7, "bi-calendar2-range")
    if bundle is None:
        empty_state("Dataset belum diproses", "Jalankan pipeline terlebih dahulu.", "bi-calendar2-range")
        st.stop()
    st.markdown("<div class='research-note'><i class='bi bi-info-circle'></i> &nbsp;<b>Catatan Interpretasi:</b> forecast masa depan merupakan proyeksi model berdasarkan pola historis dan konfigurasi aktif; hasil ini bukan nilai aktual yang telah terjadi. Komponen FTS diperbarui secara rekursif menggunakan prediksi Hybrid pada periode sebelumnya.</div>", unsafe_allow_html=True)
    ui_section("Forecast Setup", "Konfigurasi Proyeksi", "Pilih window dan horizon forecast.", "bi-sliders2")
    a,b,c=st.columns([1,1,1])
    with a: window=st.selectbox("Window FTS",WINDOWS,format_func=lambda x:f"Window {x} → 1")
    with b: periods=st.slider("Horizon",1,24,12)
    with c:
        cfg=bundle["best_cfg"][window]
        st.markdown(f"<div class='surface surface-tight'><div class='metric-label'>Konfigurasi Aktif</div><div style='font-size:12px;font-weight:800;color:#182436;margin-top:5px;'>α = {cfg['alpha']:.2f} · {cfg['n_intervals']} intervals</div><div style='font-size:9.5px;color:#8491a1;margin-top:3px;'>SARIMA{SARIMA_ORDER}{SARIMA_SEASONAL_ORDER} · FTS Chen Weighted</div></div>",unsafe_allow_html=True)
    run_future=st.button("Generate Forecast",type="primary",use_container_width=True)
    signature=(id(bundle),window,periods,bundle.get("split_ratio"))
    if run_future:
        with st.spinner("Menghitung proyeksi masa depan..."):
            try:
                future=make_future_forecast(bundle,periods,window)
                st.session_state.future_result=future
                st.session_state.future_signature=signature
            except Exception as exc:
                st.error(f"Forecast gagal dihitung: {exc}")
                st.stop()
    future=st.session_state.get("future_result")
    if future is not None and st.session_state.get("future_signature") == signature:
        start_date=future["Tanggal"].iloc[0]; end_date=future["Tanggal"].iloc[-1]
        avg_val=future["Prediksi_Hybrid"].mean(); min_val=future["Prediksi_Hybrid"].min(); max_val=future["Prediksi_Hybrid"].max()
        ui_section("Forecast Summary", "Ringkasan Proyeksi", "Statistik forecast dari horizon yang dipilih.", "bi-clipboard2-pulse")
        c1,c2,c3,c4=st.columns(4)
        with c1: ui_metric("Mulai",fmt_month(start_date),"Periode forecast","bi-calendar3")
        with c2: ui_metric("Selesai",fmt_month(end_date),"Periode forecast","bi-calendar-check")
        with c3: ui_metric("Rata-rata",fmt_num(avg_val,1),"Prediksi hybrid","bi-bar-chart")
        with c4: ui_metric("Rentang",f"{fmt_num(min_val,0)} – {fmt_num(max_val,0)}","Minimum – maksimum","bi-arrows-expand")
        ui_section("Projection", "Historis dan Forecast", "Batas vertikal menandai awal periode forecast.", "bi-graph-up-arrow")
        render_chart(make_forecast_chart(bundle["series"],future),use_container_width=True)
        ui_section("Output", "Tabel Forecast", "Data proyeksi dapat digunakan sebagai bahan dokumentasi penelitian.", "bi-table")
        st.dataframe(future,use_container_width=True,hide_index=True)
        st.download_button("Download Forecast",future.to_csv(index=False).encode("utf-8"),"forecast_hybrid_masa_depan.csv","text/csv")
    else:
        empty_state("Belum ada forecast baru", "Pilih window dan horizon, lalu klik Generate Forecast untuk membuat proyeksi.", "bi-graph-up-arrow")

# ============================================================
# 08 — KONFIGURASI MODEL
# ============================================================
elif menu.startswith("08"):
    ui_header("MODEL GOVERNANCE", "Konfigurasi Model", "Parameter model, konfigurasi window, alpha, dan ringkasan tuning validation yang digunakan.", 8, "bi-sliders2-vertical")
    if bundle is None:
        empty_state("Dataset belum diproses", "Jalankan pipeline terlebih dahulu.", "bi-sliders2")
        st.stop()
    ui_section("Method", "Arsitektur Hybrid SARIMA–FTS", "Komponen model yang digunakan pada pipeline aktif.", "bi-diagram-3")
    st.markdown(
        """
        <div class='row g-3'>
            <div class='col-12 col-md-3'><div class='feature'><div class='feature-icon'><i class='bi bi-diagram-2'></i></div><div class='feature-title'>SARIMA</div><div class='feature-text'>Model seasonal time series dengan konfigurasi SARIMA yang ditetapkan pada pipeline.</div></div></div>
            <div class='col-12 col-md-3'><div class='feature'><div class='feature-icon'><i class='bi bi-grid-3x3-gap'></i></div><div class='feature-title'>FTS Chen</div><div class='feature-text'>Fuzzy time series dengan weighted consequent untuk prediksi level.</div></div></div>
            <div class='col-12 col-md-3'><div class='feature'><div class='feature-icon'><i class='bi bi-sliders'></i></div><div class='feature-title'>Calibration</div><div class='feature-text'>Linear calibration dipelajari dari pasangan prediksi FTS dan aktual.</div></div></div>
            <div class='col-12 col-md-3'><div class='feature'><div class='feature-icon'><i class='bi bi-intersect'></i></div><div class='feature-title'>Hybrid</div><div class='feature-text'>Ensemble berbobot antara prediksi SARIMA dan FTS terkalibrasi.</div></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    ui_section("Methodology", "Metodologi Eksperimen", "Ringkasan parameter dan urutan eksperimen yang sedang aktif.", "bi-diagram-3")
    mcol1,mcol2,mcol3 = st.columns(3)
    with mcol1:
        st.markdown("<div class='feature'><div class='feature-icon'><i class='bi bi-graph-up'></i></div><div class='feature-title'>Training → Validation → Testing</div><div class='feature-text'>Pembagian data dipertahankan secara kronologis; 12 observasi terakhir training digunakan untuk validation/tuning.</div></div>",unsafe_allow_html=True)
    with mcol2:
        st.markdown("<div class='feature'><div class='feature-icon'><i class='bi bi-sliders'></i></div><div class='feature-title'>Selection Criterion</div><div class='feature-text'>Konfigurasi dipilih menggunakan urutan MAPE, kemudian MAE, lalu RMSE pada validation.</div></div>",unsafe_allow_html=True)
    with mcol3:
        st.markdown("<div class='feature'><div class='feature-icon'><i class='bi bi-intersect'></i></div><div class='feature-title'>Hybrid Ensemble</div><div class='feature-text'>Prediksi akhir menggabungkan SARIMA dan FTS terkalibrasi dengan bobot alpha yang dipelajari dari validation.</div></div>",unsafe_allow_html=True)
    render_data_split_band(bundle)

    ui_section("Configuration", "Parameter Aktif", "Nilai berikut dibaca langsung dari bundle hasil pipeline.", "bi-sliders2")
    cfg_rows=[
        {"Parameter":"Model SARIMA","Nilai":f"SARIMA{SARIMA_ORDER}{SARIMA_SEASONAL_ORDER}","Keterangan":"Konfigurasi pipeline"},
        {"Parameter":"Periode musiman","Nilai":SEASONAL_PERIOD,"Keterangan":"Data bulanan"},
        {"Parameter":"Validation","Nilai":VALIDATION_SIZE,"Keterangan":"Observasi terakhir training"},
        {"Parameter":"Split aktif","Nilai":f"{int(bundle['split_ratio']*100)}:{100-int(bundle['split_ratio']*100)}","Keterangan":"Kronologis"},
        {"Parameter":"FTS","Nilai":"Chen Weighted","Keterangan":"Weighted consequent"},
        {"Parameter":"Window 6","Nilai":f"n_intervals={bundle['best_cfg'][6]['n_intervals']} · α={bundle['best_cfg'][6]['alpha']:.2f}","Keterangan":"Hasil tuning validation"},
        {"Parameter":"Window 12","Nilai":f"n_intervals={bundle['best_cfg'][12]['n_intervals']} · α={bundle['best_cfg'][12]['alpha']:.2f}","Keterangan":"Hasil tuning validation"},
    ]
    st.dataframe(pd.DataFrame(cfg_rows),use_container_width=True,hide_index=True)
    ui_section("Tuning", "Validation Search Space", "Preview konfigurasi yang dievaluasi sebelum testing.", "bi-search")
    tuning_preview=(bundle["v7_tuning"].sort_values(["Window","MAPE (%)","MAE","RMSE"]).groupby("Window").head(12))
    st.dataframe(tuning_preview.round(4),use_container_width=True,hide_index=True)
    render_tuning_charts(bundle)
    ui_section("Formula", "Ensemble Hybrid", "Persamaan gabungan yang digunakan oleh pipeline.", "bi-function")
    cfg6 = bundle["best_cfg"][6]; cfg12 = bundle["best_cfg"][12]
    st.markdown(
        f"<div class='surface'><div class='row g-3'><div class='col-12 col-md-6'><div class='tag tag-blue'>W6 · α={cfg6['alpha']:.2f} · {cfg6['n_intervals']} intervals</div><div class='method-note' style='display:block;margin-top:7px;'>Alpha menjadi bobot SARIMA; {1-cfg6['alpha']:.2f} menjadi bobot FTS terkalibrasi.</div></div><div class='col-12 col-md-6'><div class='tag tag-gold'>W12 · α={cfg12['alpha']:.2f} · {cfg12['n_intervals']} intervals</div><div class='method-note' style='display:block;margin-top:7px;'>Alpha menjadi bobot SARIMA; {1-cfg12['alpha']:.2f} menjadi bobot FTS terkalibrasi.</div></div></div></div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='surface'><div style='font-size:17px;font-weight:800;color:#182436;letter-spacing:-.02em;'>ŷ<sub>t</sub><sup>Hybrid</sup> = α × ŷ<sub>t</sub><sup>SARIMA</sup> + (1 − α) × ŷ<sub>t</sub><sup>FTS</sup></div><div style='margin-top:8px;color:#7d8b9b;font-size:10px;line-height:1.6;'>Alpha merupakan bobot SARIMA pada ensemble dan konfigurasi window/interval dipilih melalui validation.</div></div>",unsafe_allow_html=True)

# ============================================================
# 09 — PANDUAN APLIKASI
# ============================================================
elif menu.startswith("09"):
    ui_header(
        "USER GUIDE",
        "Panduan Aplikasi",
        "Panduan penggunaan dashboard dan informasi program penelitian Hybrid SARIMA–FTS Waterpark Sumenep.",
        9,
        "bi-book",
    )

    st.markdown(
        """
        <div class='guide-hero'>
            <div class='guide-hero-title'>Selamat datang di Hybrid SARIMA–FTS Forecast Dashboard</div>
            <div class='guide-hero-text'>
                Halaman ini berfungsi sebagai buku panduan mini yang membantu pengguna memahami
                tujuan aplikasi, persiapan dataset, urutan penggunaan setiap menu, keluaran model,
                serta cara membaca hasil forecasting tanpa harus membuka kode program.
            </div>
            <span class='guide-badge'><i class='bi bi-info-circle'></i> Forecast Dashboard · User Guide</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Mulai Cepat",
        "Panduan Menu",
        "Informasi Program",
        "Metodologi",
        "Interpretasi",
    ])

    # --------------------------------------------------------
    # TAB 1 — MULAI CEPAT
    # --------------------------------------------------------
    with tab1:
        ui_section(
            "START HERE",
            "Cara Menggunakan Aplikasi",
            "Ikuti urutan berikut untuk menjalankan forecasting dari dataset hingga proyeksi masa depan.",
            "bi-play-circle",
        )

        steps = [
            ("01", "Input Dataset", "Buka menu Input Dataset, upload CSV/Excel, lalu pilih pembagian Train : Test."),
            ("02", "Jalankan Pipeline", "Klik Jalankan Hybrid SARIMA–FTS untuk menjalankan preprocessing, tuning, validation, testing, dan ensemble."),
            ("03", "Periksa Data", "Gunakan Data Historis dan audit preprocessing untuk memastikan periode dan nilai telah terbaca dengan benar."),
            ("04", "Baca Forecasting", "Gunakan Hasil Forecasting atau Windowing untuk melihat prediksi one-step-ahead pada Window 6 dan Window 12."),
            ("05", "Evaluasi Model", "Gunakan Evaluasi Model untuk membaca MAE, RMSE, MAPE, R² serta membandingkan komponen model."),
            ("06", "Forecast Masa Depan", "Pilih window dan horizon 1–24 periode, kemudian Generate Forecast untuk memperoleh proyeksi setelah data historis terakhir."),
        ]

        step_html = "<div class='row g-3'>"
        for no, title, desc in steps:
            step_html += (
                f"<div class='col-12 col-md-6 col-lg-4'>"
                f"<div class='guide-card'>"
                f"<div class='guide-number'>Tahap {no}</div>"
                f"<div class='guide-card-title'>{title}</div>"
                f"<div class='guide-card-text'>{desc}</div>"
                f"</div></div>"
            )
        step_html += "</div>"
        st.markdown(step_html, unsafe_allow_html=True)

        ui_section(
            "DATA REQUIREMENT",
            "Format Dataset",
            "Dataset harus dapat dikenali sebagai data periode dan jumlah kunjungan bulanan.",
            "bi-file-earmark-spreadsheet",
        )

        st.markdown(
            """
            <div class='surface'>
                <table class='guide-table'>
                    <tr>
                        <th>Komponen</th>
                        <th>Ketentuan</th>
                        <th>Contoh</th>
                    </tr>
                    <tr>
                        <td>Format file</td>
                        <td>CSV, XLS, atau XLSX</td>
                        <td><span class='guide-code'>data_historis.csv</span></td>
                    </tr>
                    <tr>
                        <td>Periode</td>
                        <td>Kolom tanggal, periode, bulan, atau format tanggal yang dapat dikenali</td>
                        <td>Januari 2020 / 2020-01</td>
                    </tr>
                    <tr>
                        <td>Nilai</td>
                        <td>Kolom jumlah kunjungan/pengunjung/visitor/visits/jumlah</td>
                        <td>12.540</td>
                    </tr>
                    <tr>
                        <td>Minimum</td>
                        <td>Minimal 36 observasi valid agar model musiman periode 12 dapat dievaluasi</td>
                        <td>≥ 36 bulan</td>
                    </tr>
                    <tr>
                        <td>Split</td>
                        <td>Pilihan pembagian data secara kronologis</td>
                        <td>70:30 atau 80:20</td>
                    </tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class='guide-callout' style='margin-top:14px;'>
                <i class='bi bi-lightbulb'></i>
                <div>
                    <b style='color:#30455d;'>Tips:</b>
                    gunakan data bulanan yang memiliki periode berurutan. Nilai 0 atau kosong
                    diproses oleh aturan preprocessing aplikasi melalui interpolasi linear pada
                    posisi yang dapat diinterpolasi.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # TAB 2 — PANDUAN MENU
    # --------------------------------------------------------
    with tab2:
        ui_section(
            "NAVIGATION",
            "Penjelasan Setiap Menu",
            "Setiap halaman memiliki fungsi berbeda dalam rangkaian analisis.",
            "bi-signpost-split",
        )

        menu_rows = [
            ("01", "Beranda", "Ringkasan dataset aktif, workflow, snapshot performa, dan grafik aktual vs prediksi."),
            ("02", "Input Dataset", "Upload dataset, pilih split 70:30/80:20, lalu jalankan pipeline."),
            ("03", "Data Historis", "Melihat tren historis, statistik dasar, tabel observasi, dan ekspor data."),
            ("04", "Hasil Forecasting", "Melihat prediksi one-step-ahead dan perbandingan Aktual, SARIMA, FTS, serta Hybrid."),
            ("05", "Windowing", "Membaca hasil Window 6 → 1 dan Window 12 → 1 secara terpisah."),
            ("06", "Evaluasi Model", "Membandingkan MAE, RMSE, MAPE, dan R² serta melihat diagnostic error."),
            ("07", "Forecast Masa Depan", "Menghasilkan proyeksi setelah observasi historis terakhir untuk horizon 1–24 periode."),
            ("08", "Konfigurasi Model", "Melihat arsitektur model, parameter aktif, tuning validation, dan formula ensemble."),
            ("09", "Panduan Aplikasi", "Mempelajari cara penggunaan aplikasi, struktur penelitian, metodologi, dan interpretasi hasil."),
        ]

        menu_html = "<div class='row g-3'>"
        for no, title, desc in menu_rows:
            menu_html += (
                f"<div class='col-12 col-md-6'>"
                f"<div class='guide-card'>"
                f"<div class='guide-number'>Menu {no}</div>"
                f"<div class='guide-card-title'>{title}</div>"
                f"<div class='guide-card-text'>{desc}</div>"
                f"</div></div>"
            )
        menu_html += "</div>"
        st.markdown(menu_html, unsafe_allow_html=True)

        ui_section(
            "OUTPUT",
            "Jenis Keluaran Aplikasi",
            "Output dapat digunakan untuk analisis, dokumentasi, dan bahan laporan penelitian.",
            "bi-file-earmark-arrow-down",
        )
        outputs = [
            ("Data Historis", "CSV data observasi pada rentang yang dipilih."),
            ("Forecast Testing", "CSV hasil prediksi one-step-ahead beserta komponen model dan error."),
            ("Future Forecast", "CSV proyeksi masa depan berdasarkan konfigurasi Hybrid aktif."),
            ("Research Package", "Paket ZIP yang menggabungkan data, preprocessing, tuning, metrics, prediksi, konfigurasi, dan future forecast bila tersedia."),
        ]
        out_html = "<div class='row g-3'>"
        for title, desc in outputs:
            out_html += (
                f"<div class='col-12 col-md-6'>"
                f"<div class='feature'>"
                f"<div class='feature-title'>{title}</div>"
                f"<div class='feature-text'>{desc}</div>"
                f"</div></div>"
            )
        out_html += "</div>"
        st.markdown(out_html, unsafe_allow_html=True)

    # --------------------------------------------------------
    # TAB 3 — INFORMASI PROGRAM
    # --------------------------------------------------------
    with tab3:
        ui_section(
            "ABOUT THE PROGRAM",
            "Informasi Dashboard",
            "Ringkasan fungsi dan komponen program berdasarkan implementasi yang digunakan.",
            "bi-window-stack",
        )

        info_cards = [
            ("Tujuan", "Menyediakan antarmuka penelitian untuk pengolahan data kunjungan, forecasting hybrid, evaluasi performa, dan proyeksi masa depan."),
            ("Platform", "Dashboard interaktif berbasis Streamlit dengan visualisasi Plotly."),
            ("Data", "Data historis kunjungan bulanan Waterpark Sumenep yang dapat berasal dari dataset bawaan atau file pengguna."),
            ("Model", "Hybrid SARIMA–FTS dengan SARIMA seasonal, FTS Chen Weighted, linear calibration, dan ensemble berbobot."),
            ("Evaluasi", "MAE, RMSE, MAPE, dan R² pada data testing."),
            ("Eksperimen", "Dua skenario window: 6 → 1 dan 12 → 1 dengan validation 12 observasi."),
        ]

        info_html = "<div class='row g-3'>"
        for title, desc in info_cards:
            info_html += (
                f"<div class='col-12 col-md-6 col-lg-4'>"
                f"<div class='guide-card'>"
                f"<div class='guide-card-title'>{title}</div>"
                f"<div class='guide-card-text'>{desc}</div>"
                f"</div></div>"
            )
        info_html += "</div>"
        st.markdown(info_html, unsafe_allow_html=True)

        ui_section(
            "SYSTEM FLOW",
            "Alur Kerja Program",
            "Gambaran sederhana hubungan antara input, model, evaluasi, dan output.",
            "bi-diagram-3",
        )

        st.markdown(
            """
            <div class='surface'>
                <div class='stepper'>
                    <div class='step on'><span class='step-dot'>01</span> Dataset</div>
                    <span class='step-arrow'>→</span>
                    <div class='step on'><span class='step-dot'>02</span> Preprocessing</div>
                    <span class='step-arrow'>→</span>
                    <div class='step on'><span class='step-dot'>03</span> SARIMA + FTS</div>
                    <span class='step-arrow'>→</span>
                    <div class='step on'><span class='step-dot'>04</span> Validation</div>
                    <span class='step-arrow'>→</span>
                    <div class='step on'><span class='step-dot'>05</span> Testing</div>
                    <span class='step-arrow'>→</span>
                    <div class='step on'><span class='step-dot'>06</span> Forecast</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        ui_section(
            "TECH STACK",
            "Teknologi Yang Digunakan",
            "Library utama yang terlihat langsung pada implementasi program.",
            "bi-code-slash",
        )

        tech = [
            ("Python", "Bahasa pemrograman utama."),
            ("Streamlit", "Antarmuka dashboard interaktif."),
            ("Pandas & NumPy", "Pengolahan dan transformasi data."),
            ("Statsmodels", "Implementasi SARIMAX/SARIMA."),
            ("Scikit-learn", "Linear calibration dan evaluation metrics."),
            ("Plotly", "Grafik time series dan diagnostic."),
        ]
        tech_html = "<div class='row g-3'>"
        for title, desc in tech:
            tech_html += (
                f"<div class='col-12 col-sm-6 col-lg-4'>"
                f"<div class='feature'>"
                f"<div class='feature-title'>{title}</div>"
                f"<div class='feature-text'>{desc}</div>"
                f"</div></div>"
            )
        tech_html += "</div>"
        st.markdown(tech_html, unsafe_allow_html=True)

    # --------------------------------------------------------
    # TAB 4 — METODOLOGI
    # --------------------------------------------------------
    with tab4:
        ui_section(
            "METHODOLOGY",
            "Alur Metodologi",
            "Penjelasan konseptual singkat. Parameter numerik eksperimen tersedia di Konfigurasi Model.",
            "bi-diagram-3",
        )

        pipeline_cards = [
            ("01", "Data & preprocessing", "Data bulanan dibaca, periode dinormalisasi, dan observasi yang memenuhi aturan preprocessing ditangani sebelum pemodelan."),
            ("02", "SARIMA", "SARIMA digunakan sebagai komponen seasonal time-series untuk menghasilkan prediksi one-step-ahead."),
            ("03", "FTS Chen Weighted", "FTS digunakan sebagai komponen fuzzy dengan weighted consequent pada Window 6 dan Window 12."),
            ("04", "Calibration", "Prediksi FTS dikalibrasi menggunakan regresi linear sebelum digabungkan dengan SARIMA."),
            ("05", "Hybrid ensemble", "Prediksi akhir merupakan kombinasi berbobot SARIMA dan FTS terkalibrasi berdasarkan alpha yang dipilih melalui validation."),
            ("06", "Evaluation", "Performa diuji pada data testing menggunakan MAE, RMSE, MAPE, dan R²."),
        ]
        pipe_html = "<div class='row g-3'>"
        for no, title, desc in pipeline_cards:
            pipe_html += (
                f"<div class='col-12 col-md-6'><div class='guide-card'>"
                f"<div class='guide-number'>Tahap {no}</div>"
                f"<div class='guide-card-title'>{title}</div>"
                f"<div class='guide-card-text'>{desc}</div>"
                f"</div></div>"
            )
        pipe_html += "</div>"
        st.markdown(pipe_html, unsafe_allow_html=True)

        ui_section(
            "PARAMETER LOCATION",
            "Dimana Melihat Parameter?",
            "Gunakan halaman Konfigurasi Model untuk angka parameter aktif dan hasil tuning validation.",
            "bi-sliders2",
        )
        st.markdown(
            """
            <div class='guide-callout'>
                <i class='bi bi-arrow-right-circle'></i>
                <div>
                    <b style='color:#30455d;'>Konfigurasi Model</b> menjadi sumber utama untuk
                    SARIMA order, seasonal order, validation size, Window, interval FTS, alpha,
                    selection criterion, search space, dan formula ensemble.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # TAB 5 — INTERPRETASI
    # --------------------------------------------------------
    with tab5:
        ui_section(
            "READING THE RESULTS",
            "Cara Membaca Metrik Evaluasi",
            "Gunakan penjelasan berikut sebagai panduan interpretasi hasil testing.",
            "bi-info-circle",
        )

        metric_cards = [
            ("MAE", "Mean Absolute Error", "Mengukur rata-rata besar kesalahan absolut antara nilai aktual dan prediksi."),
            ("RMSE", "Root Mean Squared Error", "Memberikan penalti lebih besar terhadap kesalahan yang besar."),
            ("MAPE", "Mean Absolute Percentage Error", "Mengukur kesalahan absolut relatif dalam bentuk persentase; pada fungsi evaluasi aplikasi, nilai aktual nol diabaikan dalam perhitungan MAPE."),
            ("R²", "Coefficient of Determination", "Menggambarkan proporsi variasi data aktual yang dijelaskan oleh prediksi pada evaluasi."),
        ]
        metric_html = "<div class='row g-3'>"
        for code, title, desc in metric_cards:
            metric_html += (
                f"<div class='col-12 col-md-6'>"
                f"<div class='guide-card'>"
                f"<div class='guide-number'>{code}</div>"
                f"<div class='guide-card-title'>{title}</div>"
                f"<div class='guide-card-text'>{desc}</div>"
                f"</div></div>"
            )
        metric_html += "</div>"
        st.markdown(metric_html, unsafe_allow_html=True)

        ui_section(
            "INTERPRETATION",
            "Hal Yang Perlu Diperhatikan",
            "Informasi berikut membantu menjaga interpretasi tetap sesuai konteks penelitian.",
            "bi-shield-check",
        )

        st.markdown(
            """
            <div class='row g-3'>
                <div class='col-12 col-md-6'>
                    <div class='guide-callout'>
                        <i class='bi bi-check2-circle'></i>
                        <div>
                            <b style='color:#30455d;'>Validation dan testing berbeda.</b>
                            Validation digunakan untuk tuning konfigurasi; testing digunakan untuk
                            mengevaluasi hasil pada data yang tidak dipakai dalam proses tuning.
                        </div>
                    </div>
                </div>
                <div class='col-12 col-md-6'>
                    <div class='guide-callout'>
                        <i class='bi bi-calendar2-range'></i>
                        <div>
                            <b style='color:#30455d;'>Future forecast adalah proyeksi.</b>
                            Nilai yang dihasilkan untuk periode setelah data historis terakhir
                            merupakan hasil model, bukan nilai aktual yang telah terjadi.
                        </div>
                    </div>
                </div>
                <div class='col-12'>
                    <div class='guide-callout'>
                        <i class='bi bi-database-check'></i>
                        <div>
                            <b style='color:#30455d;'>Periksa preprocessing sebelum analisis.</b>
                            Gunakan audit preprocessing untuk melihat observasi yang mengalami
                            perubahan sebelum model dijalankan.
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        ui_section(
            "DOCUMENTATION",
            "Untuk Kebutuhan Laporan Penelitian",
            "Dashboard menyediakan output yang dapat dipakai sebagai bahan dokumentasi.",
            "bi-file-earmark-text",
        )

        st.markdown(
            """
            <div class='surface'>
                <ul class='guide-list'>
                    <li>Gunakan screenshot atau grafik dari halaman Data Historis, Hasil Forecasting, dan Evaluasi Model sebagai ilustrasi hasil.</li>
                    <li>Gunakan halaman Konfigurasi Model untuk mendokumentasikan parameter SARIMA, FTS, window, alpha, validation, dan formula ensemble.</li>
                    <li>Gunakan Complete Research Package untuk menyimpan dataset, audit preprocessing, tuning, metrics, prediksi, konfigurasi, dan future forecast.</li>
                    <li>Sertakan keterangan bahwa dashboard merupakan media implementasi dan visualisasi dari pipeline penelitian.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
    <div class='footer'>
        <b style='color:#5d6c7d;'>Hybrid SARIMA–FTS Forecast Dashboard</b> · Waterpark Sumenep Asta Tinggi<br>
    </div>
    """,
    unsafe_allow_html=True,
)