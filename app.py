"""
app.py — Streamlit Frontend for EEG Depression Detection
---------------------------------------------------------
Two operating modes:
  1. Upload a raw EEG .txt file (3-column, tab/space separated) → predict HC or MDD
  2. Train the model from scratch using the MODMA ZIP dataset

Run with:
    streamlit run app.py
"""

import os
import io
import zipfile
import time

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import joblib

from scipy.signal import butter, sosfiltfilt, welch
from scipy.stats import skew, kurtosis
from scipy.integrate import trapezoid

# Optional sklearn imports — only needed for training
try:
    from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, classification_report,
        confusion_matrix, roc_auc_score
    )
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

# ─── Constants ───────────────────────────────────────────────────────────────
SAMPLING_RATE = 250
CHANNEL_NAMES = ["Fp1", "Fpz", "Fp2"]
CHANNEL_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]
FREQ_BANDS = {
    "Delta (1–4 Hz)": (1, 4),
    "Theta (4–8 Hz)": (4, 8),
    "Alpha (8–13 Hz)": (8, 13),
    "Beta (13–30 Hz)": (13, 30),
    "Gamma (30–40 Hz)": (30, 40),
}
MODEL_FILE = os.path.join(os.path.dirname(__file__), "adaboost_eeg_model.joblib")
if not os.path.exists(MODEL_FILE):
    alt_model = os.path.join(os.path.dirname(__file__), "adaboost_71pct_experiment.joblib")
    if os.path.exists(alt_model):
        MODEL_FILE = alt_model

# ─── EEG Processing Functions ────────────────────────────────────────────────

def preprocess_eeg(eeg_data, sampling_rate=SAMPLING_RATE):
    eeg_data = np.asarray(eeg_data, dtype=float)
    if eeg_data.ndim != 2 or eeg_data.shape[1] != 3:
        raise ValueError("EEG data must have exactly 3 columns (Fp1, Fpz, Fp2).")
    if len(eeg_data) < 20:
        raise ValueError("EEG recording is too short to filter safely (need ≥ 20 samples).")
    if not np.isfinite(eeg_data).all():
        raise ValueError("EEG data contains NaN or infinite values.")
    sos = butter(N=4, Wn=[1, 40], btype="bandpass", fs=sampling_rate, output="sos")
    return sosfiltfilt(sos, eeg_data, axis=0)


def extract_eeg_features(eeg_data, sampling_rate=SAMPLING_RATE):
    filtered = preprocess_eeg(eeg_data, sampling_rate)
    features = {}
    band_dict = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 40)}
    for ch_idx, ch in enumerate(CHANNEL_NAMES):
        sig = filtered[:, ch_idx]
        features[f"{ch}_mean"] = np.mean(sig)
        features[f"{ch}_std"] = np.std(sig)
        features[f"{ch}_variance"] = np.var(sig)
        features[f"{ch}_rms"] = np.sqrt(np.mean(sig ** 2))
        features[f"{ch}_peak_to_peak"] = np.ptp(sig)
        features[f"{ch}_skewness"] = skew(sig)
        features[f"{ch}_kurtosis"] = kurtosis(sig)
        nperseg = min(len(sig), sampling_rate * 4)
        freqs, psd = welch(sig, fs=sampling_rate, nperseg=nperseg)
        bps = {}
        for band, (lo, hi) in band_dict.items():
            mask = (freqs >= lo) & (freqs < hi)
            bps[band] = trapezoid(psd[mask], freqs[mask]) if mask.sum() >= 2 else 0.0
            features[f"{ch}_{band}_power"] = bps[band]
        total = sum(bps.values()) + 1e-12
        for band, bp in bps.items():
            features[f"{ch}_{band}_relative_power"] = bp / total
    return features, filtered


# ─── UI Helpers ──────────────────────────────────────────────────────────────

def plot_raw_eeg(eeg_data, title="Raw EEG Signal (uploaded)"):
    t = np.arange(len(eeg_data)) / SAMPLING_RATE
    fig = go.Figure()
    for i, (ch, color) in enumerate(zip(CHANNEL_NAMES, CHANNEL_COLORS)):
        fig.add_trace(go.Scatter(x=t, y=eeg_data[:, i], name=ch,
                                  line=dict(color=color, width=1), opacity=0.85))
    fig.update_layout(
        title=title, xaxis_title="Time (s)", yaxis_title="Amplitude (µV)",
        legend=dict(orientation="h"), height=350,
        margin=dict(l=40, r=20, t=50, b=40)
    )
    return fig


def plot_psd(filtered, title="Power Spectral Density (filtered)"):
    fig = go.Figure()
    for i, (ch, color) in enumerate(zip(CHANNEL_NAMES, CHANNEL_COLORS)):
        nperseg = min(len(filtered[:, i]), SAMPLING_RATE * 4)
        freqs, psd = welch(filtered[:, i], fs=SAMPLING_RATE, nperseg=nperseg)
        mask = freqs <= 45
        fig.add_trace(go.Scatter(x=freqs[mask], y=10 * np.log10(psd[mask] + 1e-12),
                                  name=ch, line=dict(color=color, width=2)))
    # Band annotations
    band_colors = ["rgba(173,216,230,0.25)", "rgba(144,238,144,0.25)",
                   "rgba(255,255,153,0.25)", "rgba(255,165,0,0.2)", "rgba(255,100,100,0.2)"]
    band_ranges = [(1,4), (4,8), (8,13), (13,30), (30,40)]
    band_labels = ["δ", "θ", "α", "β", "γ"]
    for (lo, hi), label, col in zip(band_ranges, band_labels, band_colors):
        fig.add_vrect(x0=lo, x1=hi, fillcolor=col, layer="below", line_width=0,
                      annotation_text=label, annotation_position="top left")
    fig.update_layout(
        title=title, xaxis_title="Frequency (Hz)", yaxis_title="Power (dB)",
        legend=dict(orientation="h"), height=350,
        margin=dict(l=40, r=20, t=50, b=40)
    )
    return fig


def plot_band_power_radar(features, channel="Fp1"):
    bands = ["delta", "theta", "alpha", "beta", "gamma"]
    labels = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
    values = [features.get(f"{channel}_{b}_relative_power", 0) for b in bands]
    values += [values[0]]  # close the loop
    labels += [labels[0]]
    fig = go.Figure(go.Scatterpolar(r=values, theta=labels, fill="toself",
                                     line_color=CHANNEL_COLORS[CHANNEL_NAMES.index(channel)]))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                      showlegend=False, height=300,
                      title=f"{channel} Band Power Distribution",
                      margin=dict(l=40, r=40, t=50, b=40))
    return fig


def render_prediction_banner(label, probability):
    is_mdd = (label == 1)
    color = "#c0392b" if is_mdd else "#27ae60"
    icon = "🔴" if is_mdd else "🟢"
    diagnosis = "Major Depressive Disorder (MDD)" if is_mdd else "Healthy Control (HC)"
    pct = f"{probability * 100:.1f}%"
    st.markdown(f"""
    <div style="background:{color};border-radius:12px;padding:20px 28px;color:white;margin:16px 0;">
        <h2 style="margin:0;font-size:1.6rem;">{icon} Prediction: {diagnosis}</h2>
        <p style="margin:6px 0 0;font-size:1.1rem;opacity:0.9;">Model confidence: <strong>{pct}</strong></p>
    </div>
    """, unsafe_allow_html=True)


# ─── Training helper (for Train tab) ─────────────────────────────────────────

def train_from_zip(zip_bytes, progress_bar, status_text):
    if not SKLEARN_OK:
        st.error("scikit-learn is not installed. Cannot train.")
        return None

    zip_file = zipfile.ZipFile(io.BytesIO(zip_bytes), "r")

    status_text.text("📋 Reading participant metadata…")
    excel_files = [n for n in zip_file.namelist() if n.endswith(".xlsx")]
    if not excel_files:
        st.error("No Excel metadata file found inside the ZIP.")
        return None
    metadata = pd.read_excel(io.BytesIO(zip_file.read(excel_files[0])))
    metadata = metadata[["subject id", "type"]].dropna(subset=["subject id", "type"]).copy()
    metadata["subject id"] = metadata["subject id"].astype(int)
    metadata["type"] = metadata["type"].astype(str).str.strip().str.upper()
    metadata["label"] = metadata["type"].map({"HC": 0, "MDD": 1})
    metadata = metadata.dropna(subset=["label"])
    metadata["label"] = metadata["label"].astype(int)

    eeg_files = [n for n in zip_file.namelist() if n.lower().endswith("_still.txt")]
    eeg_file_by_id = {int(os.path.basename(n).split("_")[0]): n for n in eeg_files}

    feature_rows = []
    total = len(eeg_file_by_id)
    status_text.text(f"🔬 Extracting features from {total} participants…")

    for i, (pid, fname) in enumerate(eeg_file_by_id.items()):
        row = metadata[metadata["subject id"] == pid]
        if row.empty:
            continue
        eeg = np.loadtxt(io.BytesIO(zip_file.read(fname)))
        feats, _ = extract_eeg_features(eeg)
        feats["subject_id"] = pid
        feats["group"] = row.iloc[0]["type"]
        feats["label"] = int(row.iloc[0]["label"])
        feature_rows.append(feats)
        progress_bar.progress((i + 1) / total * 0.6)

    df = pd.DataFrame(feature_rows)
    first_cols = ["subject_id", "group", "label"]
    other_cols = [c for c in df.columns if c not in first_cols]
    df = df[first_cols + other_cols]
    df = df.replace([np.inf, -np.inf], np.nan)
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    X = df.drop(columns=["subject_id", "group", "label"])
    y = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    status_text.text("⚙️ Running GridSearchCV (AdaBoost, 40 combos × 5-fold CV)…")
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", AdaBoostClassifier(estimator=DecisionTreeClassifier(random_state=42), random_state=42)),
    ])
    param_grid = {
        "model__n_estimators": [25, 50, 100, 200],
        "model__learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
        "model__estimator__max_depth": [1, 2],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(pipeline, param_grid, scoring="balanced_accuracy", cv=cv, n_jobs=-1, refit=True)
    gs.fit(X_train, y_train)
    progress_bar.progress(0.9)

    best = gs.best_estimator_
    y_pred = best.predict(X_test)
    y_prob = best.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    report = classification_report(y_test, y_pred, labels=[0, 1],
                                   target_names=["HC", "MDD"], output_dict=True)

    model_package = {
        "model": best,
        "model_name": "AdaBoost",
        "feature_columns": X.columns.tolist(),
        "sampling_rate": SAMPLING_RATE,
        "channel_names": CHANNEL_NAMES,
        "filter_band_hz": [1, 40],
        "label_meaning": {0: "Healthy Control (HC)", 1: "Major Depressive Disorder (MDD)"},
        "test_accuracy": acc,
        "balanced_accuracy": bal_acc,
        "roc_auc": auc,
        "best_params": gs.best_params_,
        "confusion_matrix": cm,
        "classification_report": report,
    }
    joblib.dump(model_package, MODEL_FILE)
    progress_bar.progress(1.0)
    status_text.text("✅ Training complete! Model saved.")
    return model_package


# ─── Main App ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="EEG Depression Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/24701-illustration-of-a-human-brain.svg/640px-24701-illustration-of-a-human-brain.svg.png", width=120)
    st.title("🧠 EEG Depression Detector")
    st.markdown("""
**Model:** AdaBoost + Decision Trees  
**Dataset:** MODMA 3-channel resting EEG  
**Channels:** Fp1, Fpz, Fp2  
**Sampling rate:** 250 Hz  
**Features:** 51 EEG features (time-domain + frequency-domain)  
**Train accuracy:** ~71.4% (14-participant holdout)
    """)
    st.divider()
    st.caption("⚠️ **Research use only.** Not a clinical diagnostic tool.")

    model_loaded = os.path.exists(MODEL_FILE)
    if model_loaded:
        st.success("✅ Model file found")
    else:
        st.warning("⚠️ No model file found. Train the model first.")

# Tabs
tab_predict, tab_train, tab_about = st.tabs(["🔮 Predict", "🏋️ Train Model", "ℹ️ About"])

# ─── PREDICT TAB ─────────────────────────────────────────────────────────────
with tab_predict:
    st.header("EEG-Based Depression Screening")
    st.markdown("""
Upload a **3-channel resting EEG** recording in plain-text format (tab or space separated).  
Each row = one time point; columns = **Fp1**, **Fpz**, **Fp2** (in order).  
The model applies a 1–40 Hz Butterworth bandpass filter, extracts 51 features, and classifies.
    """)

    if not os.path.exists(MODEL_FILE):
        st.error("No trained model found. Please go to the **Train Model** tab first.")
    else:
        pkg = joblib.load(MODEL_FILE)
        model = pkg["model"]
        feature_columns = pkg["feature_columns"]

        uploaded_file = st.file_uploader(
            "Upload EEG .txt file (3 columns: Fp1, Fpz, Fp2)",
            type=["txt", "csv"],
            help="Plain-text, whitespace-separated, 3 columns, 250 Hz sampling rate"
        )

        if uploaded_file is not None:
            with st.spinner("Processing EEG signal…"):
                try:
                    raw_bytes = uploaded_file.read()
                    eeg_data = np.loadtxt(io.BytesIO(raw_bytes))

                    if eeg_data.ndim == 1:
                        st.error("File appears to be single-channel. Need 3 columns.")
                        st.stop()
                    if eeg_data.shape[1] != 3:
                        st.error(f"Expected 3 columns, found {eeg_data.shape[1]}.")
                        st.stop()

                    n_samples = len(eeg_data)
                    duration = n_samples / SAMPLING_RATE

                    # Extract features
                    feats, filtered = extract_eeg_features(eeg_data)
                    feat_df = pd.DataFrame([feats])

                    # Align columns with trained model
                    for col in feature_columns:
                        if col not in feat_df.columns:
                            feat_df[col] = 0.0
                    feat_df = feat_df[feature_columns]

                    # Predict
                    pred_label = model.predict(feat_df)[0]
                    pred_proba = model.predict_proba(feat_df)[0]
                    pred_confidence = pred_proba[pred_label]

                except Exception as e:
                    st.error(f"Error processing file: {e}")
                    st.stop()

            # ── Signal info
            col1, col2, col3 = st.columns(3)
            col1.metric("Duration", f"{duration:.1f} s")
            col2.metric("Samples", f"{n_samples:,}")
            col3.metric("Sampling Rate", f"{SAMPLING_RATE} Hz")

            st.divider()

            # ── Prediction result
            render_prediction_banner(pred_label, pred_confidence)

            # Probability gauge
            prob_hc = pred_proba[0]
            prob_mdd = pred_proba[1]
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_mdd * 100,
                number={"suffix": "%", "font": {"size": 36}},
                title={"text": "MDD Probability"},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": "#c0392b" if prob_mdd > 0.5 else "#27ae60"},
                    "steps": [
                        {"range": [0, 40], "color": "#d5f5e3"},
                        {"range": [40, 60], "color": "#fdebd0"},
                        {"range": [60, 100], "color": "#fadbd8"},
                    ],
                    "threshold": {"line": {"color": "black", "width": 3}, "thickness": 0.75, "value": 50},
                },
            ))
            fig_gauge.update_layout(height=280, margin=dict(l=30, r=30, t=50, b=20))

            prob_col, gauge_col = st.columns([1, 1])
            with prob_col:
                st.markdown("#### Class Probabilities")
                st.markdown(f"🟢 **Healthy Control (HC):** {prob_hc*100:.1f}%")
                st.markdown(f"🔴 **Major Depressive Disorder (MDD):** {prob_mdd*100:.1f}%")
                st.progress(prob_mdd, text="MDD probability")
                st.markdown("---")
                st.markdown("**51 Extracted Features**")
                feat_display = pd.DataFrame(feats, index=["value"]).T
                st.dataframe(feat_display.style.format("{:.4f}"), height=200)
            with gauge_col:
                st.plotly_chart(fig_gauge, use_container_width=True)

            st.divider()

            # ── EEG Visualizations
            st.markdown("### EEG Signal Visualization")
            viz_tabs = st.tabs(["📈 Raw Signal", "🌊 Filtered Signal", "📊 PSD", "🕸️ Band Power"])

            with viz_tabs[0]:
                st.plotly_chart(plot_raw_eeg(eeg_data, "Raw EEG (before filtering)"), use_container_width=True)
            with viz_tabs[1]:
                st.plotly_chart(plot_raw_eeg(filtered, "Filtered EEG (1–40 Hz Butterworth)"), use_container_width=True)
            with viz_tabs[2]:
                st.plotly_chart(plot_psd(filtered), use_container_width=True)
            with viz_tabs[3]:
                r1, r2, r3 = st.columns(3)
                with r1:
                    st.plotly_chart(plot_band_power_radar(feats, "Fp1"), use_container_width=True)
                with r2:
                    st.plotly_chart(plot_band_power_radar(feats, "Fpz"), use_container_width=True)
                with r3:
                    st.plotly_chart(plot_band_power_radar(feats, "Fp2"), use_container_width=True)

# ─── TRAIN TAB ───────────────────────────────────────────────────────────────
with tab_train:
    st.header("Train the AdaBoost Model")
    st.markdown("""
Upload the **MODMA ZIP dataset** (`854301_EEG_3Channels_Resting_Lanzhou_2015.zip`) to train (or retrain) the model.  
Training runs GridSearchCV over 40 hyperparameter combinations with 5-fold stratified cross-validation.  
This typically takes **1–3 minutes** depending on your CPU.
    """)

    zip_uploader = st.file_uploader(
        "Upload MODMA ZIP dataset", type=["zip"],
        help="The original ZIP file containing the EEG _still.txt files and an Excel metadata file."
    )

    if zip_uploader is not None:
        col_start, _ = st.columns([1, 3])
        with col_start:
            start_btn = st.button("🚀 Start Training", type="primary", use_container_width=True)

        if start_btn:
            progress_bar = st.progress(0)
            status_text = st.empty()
            t0 = time.time()

            result = train_from_zip(zip_uploader.read(), progress_bar, status_text)

            elapsed = time.time() - t0
            if result:
                st.success(f"Training finished in {elapsed:.1f} s")

                # Show results
                st.markdown("### Model Performance on Holdout Set")
                m1, m2, m3 = st.columns(3)
                m1.metric("Accuracy", f"{result['test_accuracy']*100:.1f}%")
                m2.metric("Balanced Accuracy", f"{result['balanced_accuracy']*100:.1f}%")
                m3.metric("ROC-AUC", f"{result['roc_auc']:.3f}")

                st.markdown("**Best Hyperparameters:**")
                st.json(result["best_params"])

                # Confusion matrix
                cm = result["confusion_matrix"]
                fig_cm, ax = plt.subplots(figsize=(4, 3))
                im = ax.imshow(cm, cmap="Blues")
                ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
                ax.set_xticklabels(["HC", "MDD"]); ax.set_yticklabels(["HC", "MDD"])
                ax.set_xlabel("Predicted"); ax.set_ylabel("True")
                ax.set_title("Confusion Matrix")
                for i in range(2):
                    for j in range(2):
                        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
                plt.tight_layout()
                st.pyplot(fig_cm)

                st.info(f"Model saved to `{MODEL_FILE}`. Switch to the **Predict** tab to use it.")

    elif os.path.exists(MODEL_FILE):
        pkg = joblib.load(MODEL_FILE)
        st.success("A trained model already exists. Stats from last training run:")
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{pkg.get('test_accuracy', 0)*100:.1f}%")
        c2.metric("Balanced Accuracy", f"{pkg.get('balanced_accuracy', pkg.get('test_accuracy', 0))*100:.1f}%")
        c3.metric("ROC-AUC", f"{pkg.get('roc_auc', 0):.3f}")
        if "best_params" in pkg:
            st.markdown("**Best Hyperparameters:**")
            st.json(pkg["best_params"])
        st.markdown(f"**Features:** {len(pkg['feature_columns'])} EEG features")

# ─── ABOUT TAB ───────────────────────────────────────────────────────────────
with tab_about:
    st.header("About This Application")
    st.markdown("""
### EEG Depression Detection System
This application uses **EEG (Electroencephalography)** brain signals to screen for 
**Major Depressive Disorder (MDD)** using machine learning.

---

### Dataset
**MODMA** – 3-channel resting EEG dataset (Lanzhou, 2015)
- **55 participants:** 29 Healthy Controls (HC) + 26 MDD patients
- **Channels:** Fp1, Fpz, Fp2 (frontal lobe)
- **Sampling Rate:** 250 Hz
- **Recording:** Eyes-closed resting state

---

### Feature Extraction (51 features total — 17 per channel)

| Category | Features |
|----------|----------|
| **Time-domain** | Mean, Std, Variance, RMS, Peak-to-Peak, Skewness, Kurtosis |
| **Frequency-domain (absolute)** | Delta, Theta, Alpha, Beta, Gamma band power |
| **Frequency-domain (relative)** | Relative power in each of the 5 bands |

**Preprocessing:** 4th-order Butterworth 1–40 Hz bandpass filter

---

### Model: AdaBoost
- **Base estimator:** Decision Tree (depth 1 or 2)
- **Hyperparameter search:** GridSearchCV (40 combos, 5-fold stratified CV)
- **Best params:** `n_estimators=100`, `learning_rate=0.01`, `max_depth=1`
- **Holdout accuracy:** 71.4% (7 HC + 7 MDD test set)

---

### ⚠️ Limitations
> This model is for **academic research only**.  
> The 71.4% result comes from a very small test set (14 participants).  
> It is **not a validated clinical diagnostic tool**.  
> Do not use for medical diagnosis.
    """)
