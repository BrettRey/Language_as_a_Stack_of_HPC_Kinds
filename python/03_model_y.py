#!/usr/bin/env python3

"""
03_model_y.py
--------------

Fit logistic regression models to examine how the probability of the
close front rounded vowel /y/ depends on vowel inventory size.  The
script also fits a comparison model for /i/, computes a 10-fold
cross-validated AUC for the /y/ model, and produces a plot showing
observed presence/absence, the fitted probability curves with 95 %
confidence intervals, and a lighter comparison curve for /i/.

"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import unicodedata
import re

from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

RAW_DIR = os.path.join('data', 'raw')
PHOIBLE_FILE = os.path.join(RAW_DIR, 'phoible.csv')
LANGUOID_FILE = os.path.join(RAW_DIR, 'languoid.csv')
FIG_DIR = 'figs'
OUT_DIR = 'out'

RANDOM_REPS = 10
RANDOM_SEED = 20250101


def normalize_phoneme_strip_marks(s: str) -> str:
    """Normalise a phoneme string to NFKC and strip length/tone/stress marks.

    This function removes combining characters that encode length (long, half-long),
    stress (primary stress, secondary stress) and common tone diacritics (acute, grave, circumflex, tilde,
    macron, breve, diaeresis) while preserving base characters that reflect
    rounding and frontness.  The result is trimmed of surrounding
    whitespace.
    """
    if pd.isnull(s):
        return ''
    # Normalise to NFKC
    s = unicodedata.normalize('NFKC', str(s))
    # Remove designated combining marks
    s = re.sub(r'[\u02D0\u02D1\u02C8\u02CC\u0300\u0301\u0302\u0303\u0304\u0306\u0308]', '', s)
    return s.strip()


def compute_inventory_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Compute inventory sizes and presence flags for /y/ and /i/ per inventory."""
    df = df.copy()
    df['Phoneme_norm'] = df['Phoneme'].apply(normalize_phoneme_strip_marks)
    def agg(group):
        total = group['Phoneme_norm'].nunique()
        vowels = group.loc[group['SegmentClass'] == 'vowel', 'Phoneme_norm'].nunique()
        y_present = (group['Phoneme_norm'] == 'y').any()
        i_present = (group['Phoneme_norm'] == 'i').any()
        vowels_excl_y = vowels - int(y_present)
        vowels_excl_i = vowels - int(i_present)
        return pd.Series({'total_inventory_size': total,
                          'vowel_inventory_size': vowels,
                          'vowel_inv_excl_y': vowels_excl_y,
                          'vowel_inv_excl_i': vowels_excl_i,
                          'y_present': int(y_present),
                          'i_present': int(i_present)})
    return df.groupby(['Glottocode', 'InventoryID']).apply(agg).reset_index()


def select_largest_inventory(sizes: pd.DataFrame) -> pd.DataFrame:
    sizes_sorted = sizes.sort_values(by=['Glottocode', 'total_inventory_size', 'vowel_inventory_size'], ascending=[True, False, False])
    return sizes_sorted.groupby('Glottocode').head(1).reset_index(drop=True)

def select_smallest_inventory(sizes: pd.DataFrame) -> pd.DataFrame:
    sizes_sorted = sizes.sort_values(by=['Glottocode', 'total_inventory_size', 'vowel_inventory_size'], ascending=[True, True, True])
    return sizes_sorted.groupby('Glottocode').head(1).reset_index(drop=True)


def select_random_inventory(sizes: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    return sizes.groupby('Glottocode').apply(lambda g: g.sample(1, random_state=rng.integers(0, 2**31 - 1))).reset_index(drop=True)


def compute_family_mapping(languoid: pd.DataFrame) -> pd.DataFrame:
    families = languoid[languoid['level'] == 'family'][['id', 'name']].rename(columns={'id': 'family_id', 'name': 'family_name'})
    langs = languoid[['id', 'family_id']]
    mapping = langs.merge(families, on='family_id', how='left')
    mapping['family_name'] = mapping['family_name'].fillna('(Unknown)')
    return mapping[['id', 'family_name']]


def prepare_dataset(sizes: pd.DataFrame, phoible: pd.DataFrame, languoid: pd.DataFrame,
                    selection: str, rng: np.random.Generator | None = None) -> pd.DataFrame:
    if selection == 'largest':
        selected = select_largest_inventory(sizes)
    elif selection == 'smallest':
        selected = select_smallest_inventory(sizes)
    elif selection == 'random':
        if rng is None:
            rng = np.random.default_rng(RANDOM_SEED)
        selected = select_random_inventory(sizes, rng)
    else:
        raise ValueError(f"Unknown selection: {selection}")
    language_meta = phoible[['Glottocode', 'LanguageName']].drop_duplicates()
    data = selected.merge(language_meta, on='Glottocode', how='left')
    mapping = compute_family_mapping(languoid)
    data = data.merge(mapping, left_on='Glottocode', right_on='id', how='left')
    data['family_name'] = data['family_name'].fillna('(Unknown)')
    return data


def design_matrix(data: pd.DataFrame, vowel_col: str) -> pd.DataFrame:
    """Create design matrix with centred vowel size and dummy codes for family."""
    X = pd.DataFrame({
        'vowel_inv_c': data[vowel_col] - data[vowel_col].mean()
    })
    fam_dummies = pd.get_dummies(data['family_name'], drop_first=True)
    X = pd.concat([X, fam_dummies], axis=1)
    return X


def fit_models(data: pd.DataFrame):
    """Fit logistic regression models for /y/ and /i/ using sklearn."""
    X = design_matrix(data, 'vowel_inv_excl_y')
    model_y = LogisticRegression(max_iter=1000)
    model_y.fit(X, data['y_present'])
    X_i = design_matrix(data, 'vowel_inv_excl_i')
    model_i = LogisticRegression(max_iter=1000)
    model_i.fit(X_i, data['i_present'])
    return model_y, model_i


def cross_validated_auc(data: pd.DataFrame, target_col: str, vowel_col: str, group_col: str = "family_name") -> float:
    """Compute mean AUC over 10 grouped folds for the logistic model."""
    X = design_matrix(data, vowel_col)
    y = data[target_col].values
    groups = data[group_col].values
    gkf = GroupKFold(n_splits=10)
    aucs: list[float] = []
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train, y_train)
        # compute predicted probabilities
        probs = lr.predict_proba(X_test)[:, 1]
        # skip if only one class in test set
        if len(np.unique(y_test)) < 2:
            continue
        aucs.append(roc_auc_score(y_test, probs))
    return float(np.mean(aucs)) if aucs else float('nan')


def generate_predictions(model: LogisticRegression, data: pd.DataFrame,
                         vowel_range: np.ndarray, vowel_col: str) -> np.ndarray:
    """Generate predicted probabilities for a sequence of vowel sizes.

    This helper ensures that the feature columns used to train the model
    are reproduced exactly for the new data.  All dummy variables for
    families are set to zero so that predictions correspond to the
    reference family.
    """
    # Build design matrix once to capture column order
    X_base = design_matrix(data, vowel_col)
    col_names = X_base.columns
    mean_vowel = data[vowel_col].mean()
    # Construct new design matrix with same columns
    X_new = pd.DataFrame(0, index=np.arange(len(vowel_range)), columns=col_names)
    X_new['vowel_inv_c'] = vowel_range - mean_vowel
    # Predict probabilities
    preds = model.predict_proba(X_new)[:, 1]
    return preds


def bootstrap_prediction_intervals(data: pd.DataFrame, vowel_range: np.ndarray, vowel_col: str,
                                   n_boot: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute bootstrap mean and 95% prediction interval for P(/y/) curves."""
    preds_list: list[np.ndarray] = []
    rng = np.random.default_rng(20250101)
    # Full design matrix and response
    X_full = design_matrix(data, vowel_col)
    y_full = data['y_present'].values
    # New design matrix for prediction (all family dummies = 0)
    col_names = X_full.columns
    mean_vowel = data[vowel_col].mean()
    X_new = pd.DataFrame(0, index=np.arange(len(vowel_range)), columns=col_names)
    X_new['vowel_inv_c'] = vowel_range - mean_vowel
    # Bootstrap loop
    for _ in range(n_boot):
        idx = rng.integers(0, len(data), len(data))
        X_boot = X_full.iloc[idx]
        y_boot = y_full[idx]
        if len(np.unique(y_boot)) < 2:
            continue
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_boot, y_boot)
        preds_list.append(lr.predict_proba(X_new)[:, 1])
    preds_arr = np.array(preds_list)
    mean_pred = preds_arr.mean(axis=0)
    lower = np.percentile(preds_arr, 2.5, axis=0)
    upper = np.percentile(preds_arr, 97.5, axis=0)
    return mean_pred, lower, upper


def main() -> None:
    if not os.path.exists(PHOIBLE_FILE) or not os.path.exists(LANGUOID_FILE):
        raise FileNotFoundError('Required raw files missing; run 01_download_phoible.py first')
    phoible = pd.read_csv(PHOIBLE_FILE)
    languoid = pd.read_csv(LANGUOID_FILE)
    sizes = compute_inventory_flags(phoible)

    rows = []
    rng = np.random.default_rng(RANDOM_SEED)
    selections = [("largest", 0), ("smallest", 0)]
    selections += [("random", i + 1) for i in range(RANDOM_REPS)]

    primary_data = None
    primary_models = None

    for selection, rep in selections:
        data = prepare_dataset(sizes, phoible, languoid, selection=selection, rng=rng)
        model_y, model_i = fit_models(data)
        cv_auc_y = cross_validated_auc(data, "y_present", "vowel_inv_excl_y")
        cv_auc_i = cross_validated_auc(data, "i_present", "vowel_inv_excl_i")
        coef_y = float(model_y.coef_[0][0])
        coef_i = float(model_i.coef_[0][0])
        rows.append({
            "selection": selection,
            "replicate": rep,
            "n_languages": int(len(data)),
            "y_prevalence": float(data["y_present"].mean()),
            "i_prevalence": float(data["i_present"].mean()),
            "y_coef": coef_y,
            "y_or": float(np.exp(coef_y)),
            "y_cv_auc": float(cv_auc_y),
            "i_coef": coef_i,
            "i_or": float(np.exp(coef_i)),
            "i_cv_auc": float(cv_auc_i),
        })
        if selection == "largest" and rep == 0:
            primary_data = data
            primary_models = (model_y, model_i, cv_auc_y)

    os.makedirs(OUT_DIR, exist_ok=True)
    sens_df = pd.DataFrame(rows)
    sens_path = os.path.join(OUT_DIR, "y_model_sensitivity.csv")
    sens_df.to_csv(sens_path, index=False)
    metrics_path = os.path.join(OUT_DIR, "y_model_metrics.csv")
    sens_df[(sens_df["selection"] == "largest") & (sens_df["replicate"] == 0)].to_csv(metrics_path, index=False)
    print(f"[model] sensitivity metrics written to {sens_path}")
    print(f"[model] primary metrics written to {metrics_path}")

    if primary_data is None or primary_models is None:
        raise RuntimeError("Primary selection missing")

    data = primary_data
    model_y, model_i, cv_auc = primary_models
    vmin = min(data["vowel_inv_excl_y"].min(), data["vowel_inv_excl_i"].min())
    vmax = max(data["vowel_inv_excl_y"].max(), data["vowel_inv_excl_i"].max())
    vowel_seq = np.linspace(vmin, vmax, 100)
    y_mean, y_lower, y_upper = bootstrap_prediction_intervals(data, vowel_seq, "vowel_inv_excl_y", n_boot=200)
    i_pred = generate_predictions(model_i, data, vowel_seq, "vowel_inv_excl_i")
    fig, ax = plt.subplots(figsize=(8, 5))
    jitter = np.random.default_rng(RANDOM_SEED).uniform(-0.02, 0.02, size=len(data))
    ax.scatter(data["vowel_inv_excl_y"], data["y_present"] + jitter,
               color="black", alpha=0.5, s=10, label="Languages (/y/ present = 1)")
    ax.plot(vowel_seq, y_mean, color="#0072B2", lw=2, label="Predicted P(/y/)")
    ax.fill_between(vowel_seq, y_lower, y_upper, color="#0072B2", alpha=0.2)
    ax.plot(vowel_seq, i_pred, color="#D55E00", lw=1.5, ls="--", label="Predicted P(/i/)", alpha=0.8)
    ax.set_xlabel("Vowel inventory size (excluding target segment)")
    ax.set_ylabel("Probability of vowel in inventory")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Probability of /y/ vs vowel inventory size")
    ax.legend(loc="upper left")
    ax.text(vmax * 0.97, 0.9, f"10-fold CV AUC (/y/): {cv_auc:.3f}",
            ha="right", va="center", fontsize=9)
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, "y_vs_vowel_inventory.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "y_vs_vowel_inventory.png"), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, "y_vs_vowel_inventory.svg"))
    plt.close(fig)
    print("[model] probability plot saved to figs/y_vs_vowel_inventory.{pdf,png,svg}")


if __name__ == '__main__':
    main()
