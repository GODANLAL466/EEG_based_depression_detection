"""
train_model.py
Run this script ONCE to train the AdaBoost model and save it as a .joblib file.
The Streamlit app will then load the saved model for fast predictions.

Usage:
    py train_model.py
"""

import os
import io
import zipfile
import numpy as np
import pandas as pd
import joblib

from scipy.signal import butter, sosfiltfilt, welch
from scipy.stats import skew, kurtosis
from scipy.integrate import trapezoid

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report

RANDOM_STATE = 42
SAMPLING_RATE = 250
CHANNEL_NAMES = ["Fp1", "Fpz", "Fp2"]

# ─── Path to the ZIP file ───────────────────────────────────────────────────
ZIP_PATH = os.path.join(os.path.dirname(__file__), "854301_EEG_3Channels_Resting_Lanzhou_2015 (1).zip")
MODEL_FILE = os.path.join(os.path.dirname(__file__), "adaboost_eeg_model.joblib")


def preprocess_eeg(eeg_data, sampling_rate=SAMPLING_RATE):
    eeg_data = np.asarray(eeg_data, dtype=float)
    if eeg_data.ndim != 2 or eeg_data.shape[1] != 3:
        raise ValueError("EEG data must have 3 columns (Fp1, Fpz, Fp2).")
    if len(eeg_data) < 20:
        raise ValueError("EEG recording is too short to filter safely.")
    if not np.isfinite(eeg_data).all():
        raise ValueError("EEG data contains missing or infinite values.")
    filter_sos = butter(N=4, Wn=[1, 40], btype="bandpass", fs=sampling_rate, output="sos")
    return sosfiltfilt(filter_sos, eeg_data, axis=0)


def extract_eeg_features(eeg_data, sampling_rate=SAMPLING_RATE):
    filtered_signal = preprocess_eeg(eeg_data, sampling_rate)
    features = {}
    frequency_bands = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 40),
    }
    for channel_number, channel_name in enumerate(CHANNEL_NAMES):
        channel_signal = filtered_signal[:, channel_number]
        features[channel_name + "_mean"] = np.mean(channel_signal)
        features[channel_name + "_std"] = np.std(channel_signal)
        features[channel_name + "_variance"] = np.var(channel_signal)
        features[channel_name + "_rms"] = np.sqrt(np.mean(channel_signal ** 2))
        features[channel_name + "_peak_to_peak"] = np.ptp(channel_signal)
        features[channel_name + "_skewness"] = skew(channel_signal)
        features[channel_name + "_kurtosis"] = kurtosis(channel_signal)
        samples_per_segment = min(len(channel_signal), sampling_rate * 4)
        frequencies, power_spectrum = welch(channel_signal, fs=sampling_rate, nperseg=samples_per_segment)
        band_powers = {}
        for band_name, (lo, hi) in frequency_bands.items():
            mask = (frequencies >= lo) & (frequencies < hi)
            if np.sum(mask) >= 2:
                bp = trapezoid(power_spectrum[mask], frequencies[mask])
            else:
                bp = 0.0
            band_powers[band_name] = bp
            features[channel_name + "_" + band_name + "_power"] = bp
        total = sum(band_powers.values()) + 1e-12
        for band_name, bp in band_powers.items():
            features[channel_name + "_" + band_name + "_relative_power"] = bp / total
    return features


def main():
    print(f"Loading ZIP: {ZIP_PATH}")
    zip_file = zipfile.ZipFile(ZIP_PATH, "r")

    # Load metadata
    excel_files = [n for n in zip_file.namelist() if n.endswith(".xlsx")]
    if not excel_files:
        raise FileNotFoundError("No Excel metadata file found inside the ZIP.")
    metadata = pd.read_excel(io.BytesIO(zip_file.read(excel_files[0])))
    metadata = metadata[["subject id", "type"]].dropna(subset=["subject id", "type"]).copy()
    metadata["subject id"] = metadata["subject id"].astype(int)
    metadata["type"] = metadata["type"].astype(str).str.strip().str.upper()
    metadata["label"] = metadata["type"].map({"HC": 0, "MDD": 1})
    metadata = metadata.dropna(subset=["label"])
    metadata["label"] = metadata["label"].astype(int)
    print(f"Participants in metadata: {len(metadata)}")

    # Match EEG files
    eeg_files = [n for n in zip_file.namelist() if n.lower().endswith("_still.txt")]
    eeg_file_by_id = {}
    for file_name in eeg_files:
        pid = int(os.path.basename(file_name).split("_")[0])
        eeg_file_by_id[pid] = file_name

    # Extract features
    feature_rows = []
    for i, (participant_id, file_name) in enumerate(eeg_file_by_id.items()):
        row = metadata[metadata["subject id"] == participant_id]
        if row.empty:
            continue
        group = row.iloc[0]["type"]
        label = int(row.iloc[0]["label"])
        eeg_data = np.loadtxt(io.BytesIO(zip_file.read(file_name)))
        feats = extract_eeg_features(eeg_data)
        feats["subject_id"] = participant_id
        feats["group"] = group
        feats["label"] = label
        feature_rows.append(feats)
        print(f"  [{i+1}/{len(eeg_file_by_id)}] Processed participant {participant_id} ({group})", flush=True)

    feature_df = pd.DataFrame(feature_rows)
    first_cols = ["subject_id", "group", "label"]
    other_cols = [c for c in feature_df.columns if c not in first_cols]
    feature_df = feature_df[first_cols + other_cols]

    # Handle missing/infinite
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    if feature_df.isnull().sum().sum() > 0:
        num_cols = feature_df.select_dtypes(include=[np.number]).columns
        feature_df[num_cols] = feature_df[num_cols].fillna(feature_df[num_cols].median())

    X = feature_df.drop(columns=["subject_id", "group", "label"])
    y = feature_df["label"].astype(int)
    print(f"\nFeature shape: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Build pipeline
    ada_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", AdaBoostClassifier(
            estimator=DecisionTreeClassifier(random_state=42),
            random_state=42
        )),
    ])

    ada_param_grid = {
        "model__n_estimators": [25, 50, 100, 200],
        "model__learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
        "model__estimator__max_depth": [1, 2],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\nRunning GridSearchCV (this may take a minute)...")
    grid_search = GridSearchCV(
        estimator=ada_pipeline,
        param_grid=ada_param_grid,
        scoring="balanced_accuracy",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    print(f"\nBest params: {grid_search.best_params_}")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test Balanced Accuracy: {bal_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, labels=[0, 1],
                                target_names=["Healthy Control (HC)", "MDD"]))

    # Save model package
    model_package = {
        "model": best_model,
        "model_name": "AdaBoost",
        "feature_columns": X.columns.tolist(),
        "sampling_rate": SAMPLING_RATE,
        "channel_names": CHANNEL_NAMES,
        "filter_band_hz": [1, 40],
        "label_meaning": {0: "Healthy Control (HC)", 1: "Major Depressive Disorder (MDD)"},
        "test_accuracy": acc,
        "best_params": grid_search.best_params_,
    }
    joblib.dump(model_package, MODEL_FILE)
    print(f"\nModel saved to: {MODEL_FILE}")


if __name__ == "__main__":
    main()
