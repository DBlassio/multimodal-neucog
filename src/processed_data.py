"""
processed_data.py

Produces two CSV files in data/processed/:
    eeg_features_pre.csv  - EEG band power (PRE session, ses-1)
    psycho_features_pre.csv - Psychometric scores (PRE)

Both share `subject_id` and `label` (RD / MD) for future merging.
Only PRE-intervention session (ses-1) is used as our scope is focused.
"""

import warnings
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import mne
from mne.time_frequency import psd_array_welch
from tqdm import tqdm
warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("WARNING")
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# PATHS
ROOT      = Path(__file__).resolve().parent.parent
DATA_RAW  = ROOT / "data" / "raw" / "ds006260"
DATA_PROC = ROOT / "data" / "processed"
DATA_PROC.mkdir(parents=True, exist_ok=True)

PSYCHO_XLSX = DATA_RAW / "01_Psychometric_Data.xlsx"


# EEG CONFIG
SESSION = "ses-1"   # PRE-intervention only

# Frequency bands standard for cognitive EEG studies
FREQ_BANDS = {
    "delta": (1.0,  4.0),
    "theta": (4.0,  8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0)}

# 7 regions covering 28 / 32 channels.
# Excluded: FT9, FT10, TP9, TP10 (border electrodes, high noise, rarely used).

REGIONS = {
    "frontal":["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"],# 7 ch
    "frontocentral":["FC1", "FC2", "FC5", "FC6"],# 4 ch
    "central":["C3", "C4", "Cz"],# 3 ch
    "centroparietal":["CP1", "CP2", "CP5", "CP6"],# 4 ch
    "temporal":["T7", "T8", "P7", "P8"],# 4 ch
    "parietal":["P3", "P4", "Pz"],# 3 ch
    "occipital":["O1", "O2", "Oz"],# 3 ch
    }
# Total: 28 / 32 channels
# Features per subject: 7 regions × 4 bands × 2 conditions = 56

# Subjects with known data issues
EXCLUDE_EEG   = {"sub-Mc22"}                              # no ses-1 at all
NO_TASK_RUN2  = {"sub-Rc05", "sub-Rc13", "sub-Rc14",
                 "sub-Re02", "sub-Re06"}                  # task_* will be NaN


# Helpers
def find_channel_indices(ch_names_upper: list[str], target_list: list[str]) -> list[int]:
    """Return indices of target channels (case-insensitive). May be empty."""
    targets_upper = {t.upper() for t in target_list}
    return [i for i, ch in enumerate(ch_names_upper) if ch in targets_upper]


def compute_band_power(raw: mne.io.BaseRaw, prefix: str) -> dict:
    """
    Compute log-band power per region × frequency band.
    Returns flat dict: {'baseline_frontal_theta': float, ...}

    Log-transform (log1p) stabilises variance, standard in EEG ML pipelines.
    Welch PSD with 2-second FFT window and 50 % overlap (MNE default).
    """
    data= raw.get_data()                   # (n_ch, n_times)
    sfreq = raw.info["sfreq"]
    ch_upper = [ch.upper() for ch in raw.ch_names]

    psds, freqs = psd_array_welch(data,sfreq=sfreq,fmin=1.0,fmax=30.0,
                                  n_fft=int(sfreq * 2),
                                  verbose=False)   # psds → (n_channels, n_freqs)

    features = {}
    for region, ch_list in REGIONS.items():
        idx = find_channel_indices(ch_upper, ch_list)
        for band, (fmin, fmax) in FREQ_BANDS.items():
            key = f"{prefix}_{region}_{band}"
            if not idx:
                features[key] = np.nan
                continue
            freq_mask  = (freqs >= fmin) & (freqs < fmax)
            band_power = psds[idx, :][:, freq_mask].mean()
            features[key] = float(np.log10(max(band_power, 1e-30)))

    return features


def load_subject_eeg(sub_id: str, label: str) -> dict:
    """
    Load baseline (run-1) + task (run-2) EEG for ses-1.
    Missing files → NaN (logged at DEBUG level).
    """
    eeg_dir = DATA_RAW / sub_id / SESSION / "eeg"
    row     = {"subject_id": sub_id, "label": label}

    for run, prefix in [("run-1", "baseline"), ("run-2", "task")]:
        fname = f"{sub_id}_{SESSION}_task-SmartickDataset_{run}_eeg.set"
        fpath = eeg_dir / fname

        if not fpath.exists():
            if sub_id in NO_TASK_RUN2 and run == "run-2":
                log.debug(f"  [{sub_id}] No task run-2 (known missing) → NaN")
            else:
                log.debug(f"  [{sub_id}] {fname} not found → NaN")
            for region in REGIONS:
                for band in FREQ_BANDS:
                    row[f"{prefix}_{region}_{band}"] = np.nan
            continue

        try:
            raw = mne.io.read_raw_eeglab(str(fpath), preload=True, verbose=False)
            row.update(compute_band_power(raw, prefix))
        except Exception as exc:
            log.warning(f"  [{sub_id}] Error reading {run}: {exc}")
            for region in REGIONS:
                for band in FREQ_BANDS:
                    row[f"{prefix}_{region}_{band}"] = np.nan

    return row


# EEG PIPELINE
def build_eeg_features() -> pd.DataFrame:
    log.info("\n── EEG Feature Extraction ──────────────────────────────────────")
    log.info("   Session    : ses-1 (PRE-intervention)")
    log.info("   Conditions : baseline (run-1) + task (run-2)")
    log.info("   Regions    : 7  |  Bands : 4  |  Features : 56 per subject")
    log.info(f"  Excluded   : {EXCLUDE_EEG} (no ses-1 data)")

    subjects = sorted([
        d.name for d in DATA_RAW.iterdir()
        if d.is_dir() and d.name.startswith("sub-") and d.name not in EXCLUDE_EEG
    ])

    rows = []
    for sub_id in tqdm(subjects, desc="  EEG"):
        group_char = sub_id.split("-")[1][0].upper()   # R → RD, M → MD
        label = "RD" if group_char == "R" else "MD"
        rows.append(load_subject_eeg(sub_id, label))

    df = pd.DataFrame(rows)

    # Summary
    n_full     = df["task_frontal_alpha"].notna().sum()
    n_baseline = df["baseline_frontal_alpha"].notna().sum()
    log.info(f"\n   Subjects processed      : {len(df)}")
    log.info(f"   With full baseline data : {n_baseline}")
    log.info(f"   With full task data     : {n_full}")
    log.info(f"   Task NaN (known gaps)   : {len(df) - n_full}")

    out = DATA_PROC / "eeg_features_pre.csv"
    df.to_csv(out, index=False)
    log.info(f"\n     {out.name}  →  {df.shape[0]} rows × {df.shape[1]} cols")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PSYCHOMETRIC PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def build_psycho_features() -> pd.DataFrame:
    log.info("\n── Psychometric Feature Extraction ─────────────────────────────")
    log.info("   Source : 01_Psychometric_Data.xlsx  (PRE columns only)")

    # ── READING sheet ─────────────────────────────────────────────────────────
    df_r = pd.read_excel(PSYCHO_XLSX, sheet_name="READING").rename(columns={
        "Subject":                 "subject_id",
        "AGE":                     "age",
        "GENDER":                  "gender",
        "IQ":                      "IQ",
        "Selective_Attention_PRE": "selective_attention",
        "Orthographic_ERR_PRE":    "ortho_errors",
        "Phonological_ERR_PRE":    "phonol_errors",
        "Reading_Speed_PRE":       "reading_speed",
        "Comprehension_PRE":       "comprehension",
    })
    df_r["subject_id"] = df_r["subject_id"].str.strip()
    df_r["label"]      = "RD"
    df_r["math_wrat4"] = np.nan
    df_r["math_hits"]  = np.nan
    df_r["math_rt"]    = np.nan

    # ── MATH sheet ────────────────────────────────────────────────────────────
    df_m = pd.read_excel(PSYCHO_XLSX, sheet_name="MATH").rename(columns={
        "Subject":                 "subject_id",
        "AGE":                     "age",
        "GENDER":                  "gender",
        "IQ":                      "IQ",
        "Selective_Attention_PRE": "selective_attention",
        "WRAT4_PRE":               "math_wrat4",
        "hits_PRE":                "math_hits",
        "RT_PRE":                  "math_rt",
    })
    df_m["subject_id"]   = df_m["subject_id"].str.strip()
    df_m["label"]        = "MD"
    df_m["ortho_errors"] = np.nan
    df_m["phonol_errors"]= np.nan
    df_m["reading_speed"]= np.nan
    df_m["comprehension"]= np.nan

    # ── Stack ─────────────────────────────────────────────────────────────────
    col_order = [
        "subject_id", "label",
        "age", "gender", "IQ",
        "selective_attention",
        "ortho_errors", "phonol_errors", "reading_speed", "comprehension",  # RD
        "math_wrat4", "math_hits", "math_rt",                               # MD
    ]
    df = pd.concat([df_r, df_m], ignore_index=True)
    df = df[[c for c in col_order if c in df.columns]]

    log.info(f"   RD subjects : {(df['label']=='RD').sum()}")
    log.info(f"   MD subjects : {(df['label']=='MD').sum()}")

    out = DATA_PROC / "psycho_features_pre.csv"
    df.to_csv(out, index=False)
    log.info(f"\n     {out.name}  →  {df.shape[0]} rows × {df.shape[1]} cols")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

def sanity_check(df_psycho: pd.DataFrame, df_eeg: pd.DataFrame) -> None:
    log.info("\n── Sanity Check ─────────────────────────────────────────────────")

    ids_p   = set(df_psycho["subject_id"].dropna())
    ids_e   = set(df_eeg["subject_id"].dropna())
    common  = ids_p & ids_e

    log.info(f"   Psycho subjects         : {len(ids_p)}")
    log.info(f"   EEG subjects            : {len(ids_e)}")
    log.info(f"   Match on subject_id     : {len(common)}")

    only_p = ids_p - ids_e
    only_e = ids_e - ids_p
    if only_p:
        log.info(f"   Only in psycho CSV      : {sorted(only_p)}")
    if only_e:
        log.info(f"   Only in EEG CSV         : {sorted(only_e)}")

    # NaN summary
    for prefix in ["baseline", "task"]:
        col  = f"{prefix}_frontal_alpha"
        n_ok = df_eeg[col].notna().sum() if col in df_eeg.columns else 0
        log.info(f"   {prefix:10s} valid rows  : {n_ok}/{len(df_eeg)}")

    log.info("\n   Merge snippet for notebooks:")
    log.info("   df = pd.merge(df_eeg, df_psycho, on=['subject_id', 'label'])")
    log.info("   # Drop sub-Mc22 if needed: df = df[df.label.notna()]")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("=" * 60)
    log.info("  multimodal-neucog · Preprocessing Pipeline")
    log.info("  PRE-intervention session (ses-1) only")
    log.info("=" * 60)

    df_psycho = build_psycho_features()
    df_eeg    = build_eeg_features()

    sanity_check(df_psycho, df_eeg)

    log.info("\n  Done. Files ready for GitHub:")
    log.info(f"   {DATA_PROC / 'psycho_features_pre.csv'}")
    log.info(f"   {DATA_PROC / 'eeg_features_pre.csv'}")
