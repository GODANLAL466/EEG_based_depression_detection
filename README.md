# 🧠 EEG Depression Detection — Streamlit App

An interactive web application for EEG-based Major Depressive Disorder (MDD) screening using an **AdaBoost classifier** trained on the MODMA 3-channel resting EEG dataset.

> ⚠️ **Research use only. Not a clinical diagnostic tool.**

---

## 📸 Features

- **Upload** a raw 3-channel EEG `.txt` file → get an instant HC / MDD prediction
- **Visualise** the raw signal, filtered signal, power spectral density, and per-channel band-power radar charts
- **Train** the model from scratch inside the app by uploading the MODMA ZIP dataset
- **Gauge chart** showing MDD probability with confidence score

---

## 🗂️ Project Structure

```
eecg_project_anti/
├── app.py                          # Streamlit frontend
├── train_model.py                  # Standalone training script
├── requirements.txt                # Python dependencies
├── .gitignore
└── EEG_AdaBoost_71pct_Reproduction (1).ipynb   # Original research notebook
```

> The dataset ZIP (`854301_EEG_3Channels_Resting_Lanzhou_2015 (1).zip`) and trained model  
> (`adaboost_eeg_model.joblib`) are **not committed** (see `.gitignore`).

---

## ⚙️ Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model (first time only)

**Option A — command line:**
```bash
py train_model.py
```
Place the dataset ZIP in the same folder first.

**Option B — inside the app:**  
Go to the **Train Model** tab and upload the ZIP.

### 3. Run the app

```bash
py -m streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## 🧪 Model Details

| Property | Value |
|----------|-------|
| **Algorithm** | AdaBoost + Decision Tree |
| **Dataset** | MODMA 3-channel resting EEG (55 participants) |
| **Channels** | Fp1, Fpz, Fp2 (frontal lobe) |
| **Sampling rate** | 250 Hz |
| **Preprocessing** | 4th-order Butterworth 1–40 Hz bandpass filter |
| **Features** | 51 (17 per channel: 7 time-domain + 10 frequency-domain) |
| **Split** | 75% train / 25% test, stratified, `random_state=42` |
| **Best params** | `n_estimators=100`, `learning_rate=0.01`, `max_depth=1` |
| **Holdout accuracy** | **71.43%** (14-participant test set) |
| **ROC-AUC** | 0.673 |

### Feature categories (17 per channel × 3 channels = 51)

**Time-domain (7):** Mean, Std, Variance, RMS, Peak-to-Peak, Skewness, Kurtosis  
**Frequency-domain absolute (5):** Delta, Theta, Alpha, Beta, Gamma band power  
**Frequency-domain relative (5):** Relative power in each band  

---

## 📊 Input Format

Upload a plain-text EEG file (`.txt` or `.csv`):
- **Whitespace or comma separated**
- **3 columns** in order: `Fp1`, `Fpz`, `Fp2`
- **Sampling rate:** 250 Hz (resting-state recording)
- No header row

Example first few rows:
```
-12.5   3.1   8.4
-11.2   2.9   7.6
...
```

---

## 📚 Dataset Reference

**MODMA Dataset** – 3-channel EEG:  
Mumtaz, W., et al. (2015). *A Review of EEG and MEG Research Paradigms for Mental Health Applications.*  
Available at: [figshare.com](https://figshare.com)

- 55 participants: 29 Healthy Controls (HC) + 26 MDD patients
- Labels: HC = 0, MDD = 1

---

## ⚠️ Limitations

- The 71.43% accuracy comes from a **very small** 14-person test set
- The test set was examined during development → not an independent final estimate
- Pre-split imputation may introduce minor data leakage
- **Not validated for clinical use**
