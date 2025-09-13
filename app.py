# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Flight Delay Predictor", page_icon="🛫", layout="wide")

# --- sedikit CSS biar rapet & tombol full-width ---
st.markdown("""
<style>
div.block-container {padding-top: 1rem; padding-bottom: 1rem;}
.stButton>button {width: 100%;}
</style>
""", unsafe_allow_html=True)

# ============== Load model bundle (.joblib) ==============
@st.cache_resource(show_spinner=False)
def load_bundle(path: str = "delay_pipeline.joblib"):
    bundle = joblib.load(path)
    pipe = bundle["pipeline"]
    num_cols = bundle["num_cols"]
    cat_cols = bundle["cat_cols"]
    thr = float(bundle.get("threshold", 0.35))  # fallback kalau tidak ada
    return pipe, num_cols, cat_cols, thr

try:
    pipe, NUM_COLS, CAT_COLS, THR_DEFAULT = load_bundle()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# ============== TOP-10 options (isi hasil EDA-mu; ada fallback) ==============
TOP10 = {
    "AIRLINE": [],
    "ORIGIN_AIRPORT": [],
    "DESTINATION_AIRPORT": [],
}
FALLBACK = {
    "AIRLINE": ["WN","DL","AA","UA","US","B6","EV","AS","F9","HA"],
    "ORIGIN_AIRPORT": ["ATL","ORD","DFW","DEN","LAX","PHX","IAH","SFO","LAS","CLT"],
    "DESTINATION_AIRPORT": ["ATL","ORD","DFW","DEN","LAX","PHX","IAH","SFO","LAS","CLT"],
}
for k in TOP10:
    if not TOP10[k]:
        TOP10[k] = FALLBACK[k]

# ============== Header ==============
st.title("🛫 Flight Delay Predictor")

# ============== Layout 2 kolom: kiri (form+hasil), kanan (threshold+fitur) ==============
left, right = st.columns([2, 1], gap="large")

# ------- Panel kanan: threshold & tabel fitur (kosong dulu, isi setelah prediksi) -------
with right:
    st.subheader("Inference Settings")
    thr = st.slider("Decision threshold (DELAY jika ≥)", 0.10, 0.90, float(THR_DEFAULT), 0.01,
                    help="Jika probabilitas delay ≥ threshold, prediksi = DELAY.")
    feat_placeholder = st.empty()  # nanti diisi tabel fitur

# ------- Panel kiri: form input -------
with left:
    with st.form("delay_form"):
        c1, c2 = st.columns(2)
        with c1:
            month  = st.number_input("MONTH (1-12)", min_value=1, max_value=12, value=6, step=1)
            day    = st.number_input("DAY (1-31)", min_value=1, max_value=31, value=15, step=1)
            dow    = st.number_input("DAY_OF_WEEK (1-7)", min_value=1, max_value=7, value=5, step=1)
            dist   = st.number_input("DISTANCE (miles)", min_value=0, max_value=5000, value=500, step=1)
        with c2:
            dep_hhmm = st.number_input("SCHEDULED_DEPARTURE (HHMM)", min_value=0, max_value=2359, value=1830, step=1,
                                       help="Format HHMM. Contoh: 1830 = 18:30")
            sch_time = st.number_input("SCHEDULED_TIME (minutes)", min_value=10, max_value=600, value=120, step=5)

        air = st.selectbox("AIRLINE (Top-10)", TOP10["AIRLINE"])
        org = st.selectbox("ORIGIN_AIRPORT (Top-10)", TOP10["ORIGIN_AIRPORT"])
        dst = st.selectbox("DESTINATION_AIRPORT (Top-10)", TOP10["DESTINATION_AIRPORT"])

        c3, c4 = st.columns([1,1])
        submitted = c3.form_submit_button("Predict", type="primary")
        reset     = c4.form_submit_button("Reset")

# reset = reload halaman
if reset:
    st.experimental_rerun()

# ============== Prediksi ==============
if submitted:
    # validasi HHMM
    hh, mm = int(dep_hhmm // 100), int(dep_hhmm % 100)
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        left.error("Format SCHEDULED_DEPARTURE tidak valid. Gunakan HHMM (contoh: 1830).")
        st.stop()

    # data sesuai urutan training
    data = {
        "AIRLINE": air,
        "ORIGIN_AIRPORT": org,
        "DESTINATION_AIRPORT": dst,
        "MONTH": int(month),
        "DAY": int(day),
        "DAY_OF_WEEK": int(dow),
        "SCHEDULED_DEPARTURE": int(dep_hhmm),
        "SCHEDULED_TIME": int(sch_time),
        "DISTANCE": int(dist),
    }

    X = pd.DataFrame([data], columns=CAT_COLS + NUM_COLS)
    for c in CAT_COLS:
        X[c] = X[c].astype(str)

    try:
        proba_delay = float(pipe.predict_proba(X)[0, 1])
    except Exception as e:
        left.error(f"Gagal melakukan prediksi: {e}")
        st.stop()

    label = "DELAY" if proba_delay >= thr else "ON-TIME"

    # Hasil di panel kiri (padat & tanpa ruang kosong)
    left.caption(f"p(delay) = **{proba_delay:.3f}** • threshold = **{thr:.2f}**")
    if label == "DELAY":
        left.error("Prediction: DELAY")
    else:
        left.success("Prediction: ON-TIME")

    # Tabel fitur di panel kanan (bukan di bawah)
    feat_df = pd.DataFrame([data]).T.reset_index()
    feat_df.columns = ["Feature", "Value"]
    with right:
        st.subheader("Fitur yang dipakai model")
        feat_placeholder.dataframe(feat_df, use_container_width=True, height=360)
