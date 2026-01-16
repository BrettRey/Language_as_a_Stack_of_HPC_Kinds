#!/usr/bin/env python3

"""
02_make_ridgelines.py
---------------------

Generate ridgeline density plots of total phoneme inventory size by
language family and compute per-family summary statistics.  The
script reads the raw PHOIBLE and Glottolog languoid tables, cleans
and deduplicates the data, summarises inventory sizes, and writes a
CSV of counts, medians and IQRs to `out/summary_tables.csv`.  The
ridgeline plot is saved as PDF, PNG and SVG in the `figs/` directory.

Because the environment lacks the `joypy` package, ridgelines are
constructed manually using Gaussian kernel density estimates from
SciPy.  Families with fewer than `min_n` languages are excluded from
the plot.

"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # ensure non-interactive backend
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from plot_style import RIDGELINE_CMAP, PALETTE, set_plot_style

RAW_DIR = os.path.join('data', 'raw')
OUT_DIR = 'out'
FIG_DIR = 'figs'

PHOIBLE_FILE = os.path.join(RAW_DIR, 'phoible.csv')
LANGUOID_FILE = os.path.join(RAW_DIR, 'languoid.csv')

MIN_N = 10  # minimum number of languages per family to include in plot
RANDOM_REPS = 10
RANDOM_SEED = 20250101

def normalize_phoneme(x: pd.Series) -> pd.Series:
    """Normalise phoneme strings using NFC and strip whitespace.

    We use unicode normalization from Python's unicodedata.  No
    diacritic stripping is performed here because counts are based on
    exact segments.
    """
    import unicodedata
    return x.apply(lambda s: unicodedata.normalize('NFC', str(s)).strip())


def compute_inventory_sizes(phoible: pd.DataFrame) -> pd.DataFrame:
    """Compute total and vowel inventory sizes per (Glottocode, InventoryID).

    Returns a DataFrame with columns: Glottocode, InventoryID,
    total_inventory_size, vowel_inventory_size.
    """
    df = phoible.copy()
    df['Phoneme_norm'] = normalize_phoneme(df['Phoneme'])
    def agg_fn(group):
        total = group['Phoneme_norm'].nunique()
        vowels = group.loc[group['SegmentClass'] == 'vowel', 'Phoneme_norm'].nunique()
        return pd.Series({'total_inventory_size': total, 'vowel_inventory_size': vowels})
    sizes = df.groupby(['Glottocode', 'InventoryID']).apply(agg_fn).reset_index()
    return sizes


def select_largest_inventory(sizes: pd.DataFrame) -> pd.DataFrame:
    """Within each Glottocode, select the inventory with the largest total size.

    Ties are broken by the largest vowel inventory size, then the first.
    """
    sizes_sorted = sizes.sort_values(by=['Glottocode', 'total_inventory_size', 'vowel_inventory_size'], ascending=[True, False, False])
    largest = sizes_sorted.groupby('Glottocode').head(1).reset_index(drop=True)
    return largest


def select_smallest_inventory(sizes: pd.DataFrame) -> pd.DataFrame:
    """Within each Glottocode, select the inventory with the smallest total size."""
    sizes_sorted = sizes.sort_values(by=['Glottocode', 'total_inventory_size', 'vowel_inventory_size'], ascending=[True, True, True])
    smallest = sizes_sorted.groupby('Glottocode').head(1).reset_index(drop=True)
    return smallest


def select_random_inventory(sizes: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Within each Glottocode, randomly select one inventory."""
    return sizes.groupby('Glottocode').apply(lambda g: g.sample(1, random_state=rng.integers(0, 2**31 - 1))).reset_index(drop=True)


def compute_family_mapping(languoid: pd.DataFrame) -> pd.DataFrame:
    """Create a mapping from Glottocode to family name using the languoid table."""
    families = languoid[languoid['level'] == 'family'][['id', 'name']].rename(columns={'id': 'family_id', 'name': 'family_name'})
    langs = languoid[['id', 'family_id']]
    mapping = langs.merge(families, on='family_id', how='left')
    mapping['family_name'].fillna('(Unknown)', inplace=True)
    return mapping[['id', 'family_name']]


def compute_summary(language_data: pd.DataFrame) -> pd.DataFrame:
    """Compute per-family counts, medians and IQRs of total inventory size."""
    summary = (language_data.groupby('family_name')['total_inventory_size']
               .agg(N='count', median_inventory='median',
                    q25=lambda x: np.percentile(x, 25),
                    q75=lambda x: np.percentile(x, 75))
               .reset_index())
    summary['iqr_inventory'] = summary['q75'] - summary['q25']
    summary.drop(columns=['q25', 'q75'], inplace=True)
    return summary


def bootstrap_family_medians(df: pd.DataFrame, n_boot: int = 2000, random_state: int = 20250101) -> pd.DataFrame:
    """Bootstrap family medians and 95% CIs for total inventory size."""
    rng = np.random.default_rng(random_state)
    records = []
    for fam, sub in df.groupby('family_name'):
        values = sub['total_inventory_size'].to_numpy()
        if len(values) == 0:
            continue
        med = float(np.median(values))
        boots = []
        for _ in range(n_boot):
            sample = rng.choice(values, size=len(values), replace=True)
            boots.append(np.median(sample))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        records.append({'family_name': fam, 'median': med, 'ci_lower': float(lo), 'ci_upper': float(hi), 'n': int(len(values))})
    return pd.DataFrame.from_records(records)


def compute_band_mass(df: pd.DataFrame, low: int = 20, high: int = 50) -> pd.DataFrame:
    """Compute the fraction of KDE mass within [low, high] for each family."""
    records = []
    grid = np.linspace(df['total_inventory_size'].min(), df['total_inventory_size'].max(), 1000)
    for fam, sub in df.groupby('family_name'):
        values = sub['total_inventory_size'].to_numpy()
        if len(values) < 2 or np.unique(values).size < 2:
            mass = float('nan')
        else:
            try:
                kde = gaussian_kde(values, bw_method='scott')
                density = kde(grid)
                density /= np.trapz(density, grid)
                mask = (grid >= low) & (grid <= high)
                mass = float(np.trapz(density[mask], grid[mask]))
            except Exception:
                mass = float('nan')
        records.append({'family_name': fam, 'mass_in_band': mass, 'n': int(len(values))})
    return pd.DataFrame.from_records(records)


def prepare_language_data(selected: pd.DataFrame, phoible: pd.DataFrame, languoid: pd.DataFrame) -> pd.DataFrame:
    language_meta = phoible[['Glottocode', 'LanguageName']].drop_duplicates()
    language_data = selected.merge(language_meta, on='Glottocode', how='left')
    mapping = compute_family_mapping(languoid)
    language_data = language_data.merge(mapping, left_on='Glottocode', right_on='id', how='left')
    language_data['family_name'] = language_data['family_name'].fillna('(Unknown)')
    return language_data

def make_ridgeline_plot(language_data: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Create and save the ridgeline density plot."""
    set_plot_style()
    # Filter to families with at least MIN_N languages
    valid_fams = summary[summary['N'] >= MIN_N]['family_name']
    df = language_data[language_data['family_name'].isin(valid_fams)].copy()
    # Determine order by median inventory size (descending)
    fam_order = (df.groupby('family_name')['total_inventory_size']
                 .median().sort_values(ascending=False).index.tolist())
    # Global IQR across all languages
    q25 = np.percentile(language_data['total_inventory_size'], 25)
    q75 = np.percentile(language_data['total_inventory_size'], 75)
    print(f"[ridgelines] global IQR: {q25:.1f}--{q75:.1f}")
    # Set up figure
    fig, ax = plt.subplots(figsize=(8, 9))
    # Colour map
    cmap = RIDGELINE_CMAP
    # Determine x range
    x_min = language_data['total_inventory_size'].min() - 5
    x_max = language_data['total_inventory_size'].max() + 5
    xs = np.linspace(x_min, x_max, 512)
    # Plot each family
    for idx, fam in enumerate(fam_order):
        data = df[df['family_name'] == fam]['total_inventory_size']
        if len(data) < 2:
            continue
        kde = gaussian_kde(data)
        ys = kde(xs)
        ys = ys / ys.max()  # normalize density height
        offset = idx  # vertical offset for ridgeline
        ax.fill_between(xs, offset, ys + offset, color=cmap(idx / max(len(fam_order)-1, 1)), alpha=0.7)
        ax.text(x_min + 1, offset + 0.1, fam, va='bottom', ha='left', fontsize=9)
    # Draw global IQR band
    ax.axvspan(q25, q75, ymin=0, ymax=1, color=PALETTE["gray"], alpha=0.12)
    # Formatting
    ax.set_xlabel('Total phoneme inventory size')
    ax.set_ylabel('Language family (ordered by median size)')
    ax.set_yticks([])
    ax.set_title('Phoneme inventory sizes by family')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-1, len(fam_order) + 0.5)
    fig.tight_layout()
    # Save figures
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, 'inventory_ridgelines.pdf'))
    fig.savefig(os.path.join(FIG_DIR, 'inventory_ridgelines.png'), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, 'inventory_ridgelines.svg'))
    plt.close(fig)
    print('[ridgelines] plot saved to figs/inventory_ridgelines.{pdf,png,svg}')


def main() -> None:
    if not os.path.exists(PHOIBLE_FILE) or not os.path.exists(LANGUOID_FILE):
        raise FileNotFoundError('Required raw files missing; run 01_download_phoible.py first')
    # Read data
    print('[ridgelines] reading PHOIBLE ...')
    phoible = pd.read_csv(PHOIBLE_FILE)
    print('[ridgelines] reading languoid ...')
    languoid = pd.read_csv(LANGUOID_FILE)
    # Compute inventory sizes and select largest inventory per language
    print('[ridgelines] computing inventory sizes ...')
    sizes = compute_inventory_sizes(phoible)
    rng = np.random.default_rng(RANDOM_SEED)
    selection_specs = [("largest", 0), ("smallest", 0)]
    selection_specs += [("random", i + 1) for i in range(RANDOM_REPS)]

    all_metrics = []
    all_summaries = []

    for selection, rep in selection_specs:
        if selection == "largest":
            selected = select_largest_inventory(sizes)
        elif selection == "smallest":
            selected = select_smallest_inventory(sizes)
        elif selection == "random":
            selected = select_random_inventory(sizes, rng)
        else:
            raise ValueError(f"Unknown selection: {selection}")

        language_data = prepare_language_data(selected, phoible, languoid)
        summary = compute_summary(language_data)
        summary_sel = summary.copy()
        summary_sel["selection"] = selection
        summary_sel["replicate"] = rep
        all_summaries.append(summary_sel)

        fam_medians = bootstrap_family_medians(language_data)
        band_mass = compute_band_mass(language_data, low=20, high=50)
        metrics = fam_medians.merge(band_mass[['family_name', 'mass_in_band']], on='family_name', how='left')
        metrics["selection"] = selection
        metrics["replicate"] = rep
        all_metrics.append(metrics)

        if selection == "largest" and rep == 0:
            os.makedirs(OUT_DIR, exist_ok=True)
            summary.to_csv(os.path.join(OUT_DIR, 'summary_tables.csv'), index=False)
            print(f"[ridgelines] summary written to {os.path.join(OUT_DIR, 'summary_tables.csv')}")
            make_ridgeline_plot(language_data, summary)

    os.makedirs(OUT_DIR, exist_ok=True)
    metrics_df = pd.concat(all_metrics, ignore_index=True)
    metrics_path = os.path.join(OUT_DIR, 'ridgeline_band_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[ridgelines] band metrics written to {metrics_path}")

    summaries_df = pd.concat(all_summaries, ignore_index=True)
    summaries_path = os.path.join(OUT_DIR, 'summary_tables_selection.csv')
    summaries_df.to_csv(summaries_path, index=False)
    print(f"[ridgelines] selection summaries written to {summaries_path}")


if __name__ == '__main__':
    main()
