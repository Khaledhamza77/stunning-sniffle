"""Classify each clip's congestion level and score it against the ground truth.

This is the project's one genuinely validatable accuracy metric. The dataset has
no per-vehicle ground truth - no boxes, counts, speeds or lane labels - so the
only thing our pipeline can be scored against is the single light/medium/heavy
label each clip carries.

Method
------
The labels are defined in README_TRAFFICDB relative to free flow ("free-flowing",
"traffic at reduced speed", "stopped or very slow"), so the classifier works in
those terms: each clip's median track speed is divided by a free-flow reference,
and two thresholds split the resulting ratio into three classes.

Two properties make this honest rather than convenient:

  - The speed ratio is independent of the unknown focal length f (see
    scripts/scene.py), so no absolute-speed anchor is involved anywhere here.
  - The free-flow reference and both thresholds are fitted on the training half
    of the dataset's own published train/test splits (EvalSet.mat, the same 4
    folds the original papers used) and scored on the held-out half.

A count-only classifier is fitted the same way as a baseline, to test whether the
speed machinery earns its place.

Usage:
    uv run python scripts/evaluate_congestion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibrate_scene as cal  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = REPO_ROOT / "outputs" / "analysis"
OUT_DIR = REPO_ROOT / "outputs" / "evaluation"

LABELS = ["light", "medium", "heavy"]
FREE_FLOW_PERCENTILE = 85  # a robust stand-in for "free-flowing" speed


def load_clip_table() -> pd.DataFrame:
    """Per-clip metrics joined onto the full clip list, so clips that produced no
    detections at all appear as rows with missing features rather than vanishing."""
    info = cal.load_info_table()[["filename", "congestion", "weather", "date", "timestamp"]]
    metrics = pd.read_csv(ANALYSIS_DIR / "clips.csv")
    df = info.merge(metrics.drop(columns=["congestion_true", "weather", "date"], errors="ignore"),
                    left_on="filename", right_on="clip", how="left")
    df["n_tracks"] = df["n_tracks"].fillna(0)
    df["mean_vehicles_in_roi"] = df["mean_vehicles_in_roi"].fillna(0.0)
    return df


def load_splits() -> list[tuple[list[str], list[str]]]:
    """The dataset's own 4 train/test folds, mapped from 1-based indices to names."""
    master = sio.loadmat(cal.DATASET_DIR / "ImageMaster.mat")["imagemaster"][0]
    names = [str(e[0, 0]["root"][0]) for e in master]
    ev = sio.loadmat(cal.DATASET_DIR / "EvalSet.mat")
    folds = []
    for tr, te in zip(ev["trainind"][0], ev["testind"][0]):
        folds.append((
            [names[i - 1] for i in tr.ravel()],
            [names[i - 1] for i in te.ravel()],
        ))
    return folds


def fit_thresholds(x: np.ndarray, y: np.ndarray, higher_is_lighter: bool) -> tuple[float, float]:
    """Pick the two cut points that maximise training accuracy.

    Candidates are drawn from the observed feature values, so the search needs no
    assumption about the feature's scale.
    """
    cands = np.unique(np.quantile(x[~np.isnan(x)], np.linspace(0.02, 0.98, 60)))
    best, best_acc = (cands[0], cands[-1]), -1.0
    for i, t_lo in enumerate(cands):
        for t_hi in cands[i + 1:]:
            acc = (predict(x, (t_lo, t_hi), higher_is_lighter) == y).mean()
            if acc > best_acc:
                best, best_acc = (float(t_lo), float(t_hi)), acc
    return best


def predict(x: np.ndarray, thresholds: tuple[float, float], higher_is_lighter: bool) -> np.ndarray:
    """Map a feature to labels. `higher_is_lighter` is True for speed (fast =
    free-flowing) and False for vehicle count (busy = congested)."""
    t_lo, t_hi = thresholds
    out = np.full(len(x), "medium", dtype=object)
    if higher_is_lighter:
        out[x >= t_hi] = "light"
        out[x < t_lo] = "heavy"
    else:
        out[x < t_lo] = "light"
        out[x >= t_hi] = "heavy"
    return out


def confusion(y_true, y_pred) -> pd.DataFrame:
    m = pd.DataFrame(0, index=LABELS, columns=LABELS, dtype=int)
    for t, p in zip(y_true, y_pred):
        m.loc[t, p] += 1
    m.index.name, m.columns.name = "true", "predicted"
    return m


def run_fold(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Fit the free-flow reference and thresholds on train, score on test."""
    # free-flow reference from TRAIN ONLY - using all clips would leak test data
    ref = float(np.nanpercentile(train["median_speed_raw"], FREE_FLOW_PERCENTILE))

    for d in (train, test):
        d["speed_ratio"] = d["median_speed_raw"] / ref

    # count-only baseline, fitted the same way
    t_count = fit_thresholds(train["mean_vehicles_in_roi"].to_numpy(),
                             train["congestion"].to_numpy(), higher_is_lighter=False)
    pred_count = predict(test["mean_vehicles_in_roi"].to_numpy(), t_count, higher_is_lighter=False)

    # speed classifier, with the count baseline as fallback where speed is missing
    has_speed = train["speed_ratio"].notna()
    t_speed = fit_thresholds(train.loc[has_speed, "speed_ratio"].to_numpy(),
                             train.loc[has_speed, "congestion"].to_numpy(), higher_is_lighter=True)
    pred_speed = predict(test["speed_ratio"].to_numpy(), t_speed, higher_is_lighter=True)
    missing = test["speed_ratio"].isna().to_numpy()
    pred_speed[missing] = pred_count[missing]

    y = test["congestion"].to_numpy()
    return {
        "free_flow_ref": ref,
        "thresholds_speed": t_speed,
        "thresholds_count": t_count,
        "acc_speed": float((pred_speed == y).mean()),
        "acc_count": float((pred_count == y).mean()),
        "n_test": len(test),
        "n_fallback": int(missing.sum()),
        "confusion_speed": confusion(y, pred_speed),
        "y_true": y,
        "pred_speed": pred_speed,
        "pred_count": pred_count,
        "test_clips": test["filename"].tolist(),
    }


# Congestion level is ORDERED (light -> heavy), so it gets an ordinal ramp: one
# hue, monotonically darkening, rather than unrelated categorical hues. Steps 250 /
# 450 / 650 of the blue ramp; validated with the dataviz palette validator in
# --ordinal mode (all checks pass), and their pairwise separation (CVD dE 19.0,
# normal-vision 19.5) clears the categorical floors of 8 and 15 as well. The
# lightest step sits at 2.06:1 on the light surface, which obliges visible labels -
# supplied here as direct annotations plus predictions.csv as the table view.
CLASS_COLOR = {"light": "#86b6ef", "medium": "#2a78d6", "heavy": "#104281"}
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SURFACE, INK, INK_MUTED = "#fcfcfb", "#0b0b0b", "#52514e"


def _style(ax):
    """Recessive axes and grid; text in ink tokens, never a series colour."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d8d7d2")
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=3)
    ax.xaxis.label.set_color(INK_MUTED)
    ax.yaxis.label.set_color(INK_MUTED)
    ax.title.set_color(INK)


def make_plots(df: pd.DataFrame, cm: pd.DataFrame, cmc: pd.DataFrame,
               acc_speed: float, acc_count: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    blues = LinearSegmentedColormap.from_list("seqblue", SEQ_BLUE)

    # --- 1. confusion matrices, side by side --------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), facecolor=SURFACE)
    for ax, m, title in (
        (axes[0], cm, f"Speed classifier — {acc_speed:.1%}"),
        (axes[1], cmc, f"Count-only baseline — {acc_count:.1%}"),
    ):
        frac = m.div(m.sum(axis=1), axis=0)  # colour by recall so the big light row cannot dominate
        ax.imshow(frac.to_numpy(), cmap=blues, vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                v, f = m.iloc[i, j], frac.iloc[i, j]
                ax.text(j, i, f"{v}\n{f:.0%}", ha="center", va="center", fontsize=9,
                        color="#ffffff" if f > 0.5 else INK)
        ax.set_xticks(range(3), LABELS)
        ax.set_yticks(range(3), LABELS)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(title, fontsize=10, pad=8)
        _style(ax)
        ax.grid(False)
    fig.suptitle("Held-out congestion classification, pooled over the dataset's 4 folds",
                 fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrices.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # --- 2. why speed works and counting does not ---------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), facecolor=SURFACE)
    rng = np.random.default_rng(0)
    for ax, col, name in (
        (axes[0], "speed_ratio", "Median speed (fraction of free-flow)"),
        (axes[1], "mean_vehicles_in_roi", "Mean vehicles in region"),
    ):
        for i, lab in enumerate(LABELS):
            vals = df.loc[df["congestion"] == lab, col].dropna()
            ax.scatter(vals, i + rng.uniform(-0.16, 0.16, len(vals)), s=14,
                       color=CLASS_COLOR[lab], alpha=0.55, linewidths=0, zorder=2)
            med = vals.median()
            ax.plot([med, med], [i - 0.30, i + 0.30], color=INK, lw=2, zorder=3)
            # direct label: relief for the lightest step, and saves a legend lookup
            ax.text(med, i + 0.40, f"median {med:.2f}", ha="center", fontsize=8, color=INK_MUTED)
        ax.set_yticks(range(3), LABELS)
        ax.set_xlabel(name)
        ax.set_ylim(-0.6, 2.7)
        ax.grid(axis="x", color="#eceae5", lw=1)
        ax.set_axisbelow(True)
        _style(ax)
    axes[0].set_title("Speed separates all three classes", fontsize=10, pad=8)
    axes[1].set_title("Counting cannot separate medium from heavy", fontsize=10, pad=8)
    fig.suptitle("Each point is one clip, positioned by feature and grouped by its true label",
                 fontsize=10, color=INK_MUTED)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "feature_separation.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    df = load_clip_table()
    folds = load_splits()
    by_name = df.set_index("filename")

    print(f"{len(df)} clips; {df['median_speed_raw'].isna().sum()} without a speed estimate "
          f"({df['n_tracks'].eq(0).sum()} with no tracks at all)")
    print(f"Evaluating on the dataset's own {len(folds)} train/test folds (EvalSet.mat)\n")

    results, all_true, all_pred, all_pred_count = [], [], [], []
    for i, (tr_names, te_names) in enumerate(folds):
        train = by_name.loc[tr_names].reset_index()
        test = by_name.loc[te_names].reset_index()
        r = run_fold(train, test)
        results.append(r)
        all_true.extend(r["y_true"]); all_pred.extend(r["pred_speed"]); all_pred_count.extend(r["pred_count"])
        print(f"fold {i}: speed {r['acc_speed']:.1%}  |  count-only {r['acc_count']:.1%}   "
              f"(n={r['n_test']}, {r['n_fallback']} fell back to count)")

    acc_s = np.array([r["acc_speed"] for r in results])
    acc_c = np.array([r["acc_count"] for r in results])
    print(f"\nmean across folds:  speed {acc_s.mean():.1%} (sd {acc_s.std():.1%})  "
          f"|  count-only {acc_c.mean():.1%} (sd {acc_c.std():.1%})")

    print("\nPooled confusion matrix (speed classifier, held-out predictions only):")
    cm = confusion(all_true, all_pred)
    print(cm.to_string())
    print("\nPer-class recall:")
    for lab in LABELS:
        n = cm.loc[lab].sum()
        print(f"  {lab:7s} {cm.loc[lab, lab] / n:.1%}  ({cm.loc[lab, lab]}/{n})")

    print("\nCount-only baseline, same pooling:")
    cmc = confusion(all_true, all_pred_count)
    print(cmc.to_string())
    for lab in LABELS:
        n = cmc.loc[lab].sum()
        print(f"  {lab:7s} recall {cmc.loc[lab, lab] / n:.1%}")

    # For display only: a single global free-flow reference, so all clips are on one
    # axis. The evaluated model above uses per-fold, train-only references.
    df["speed_ratio"] = df["median_speed_raw"] / np.nanpercentile(
        df["median_speed_raw"], FREE_FLOW_PERCENTILE)
    make_plots(df, cm, cmc, acc_s.mean(), acc_c.mean())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "clip": [c for r in results for c in r["test_clips"]],
        "congestion_true": all_true,
        "congestion_pred": all_pred,
        "congestion_pred_count_only": all_pred_count,
    }).to_csv(OUT_DIR / "predictions.csv", index=False)
    cm.to_csv(OUT_DIR / "confusion_matrix.csv")
    print(f"\nWrote predictions and confusion matrix to {OUT_DIR}")


if __name__ == "__main__":
    main()
