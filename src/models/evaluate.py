"""
src/models/evaluate.py
=============================================================================
Model Evaluation — Metrics, Feature Importance & Residual Diagnostics
=============================================================================
Responsibilities:
  - Score a trained ModelArtifact against its held-out test set
  - Compute MAE, RMSE, MAPE, MedAE, R² and a bias / coverage summary
  - Produce publication-quality plots saved to outputs/evaluation/:
      1. Feature importance bar chart (top-N by gain)
      2. Actual vs predicted scatter / time-series overlay
      3. Residual distribution histogram + Q-Q plot
      4. Rolling MAE over the test window (temporal error profile)
  - Emit a per-zone JSON metrics report alongside each plot
  - Support batch evaluation of all zones from a training summary

Architecture position: MODEL FACTORY → EVALUATION → outputs/evaluation/
Upstream:  src/models/train.py  (ModelArtifact, load_model_artifact)
=============================================================================
"""

from __future__ import annotations

import json
import logging
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")    # non-interactive backend — safe in headless CI environments
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

from .train import ModelArtifact, TrainingConfig, load_model_artifact

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logger = logging.getLogger("model_factory.evaluate")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(name)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    logger.addHandler(_handler)

# ---------------------------------------------------------------------------
# Plot Style Constants
# ---------------------------------------------------------------------------
_PALETTE = {
    "primary":   "#1B4F72",   # deep navy — primary bars / lines
    "accent":    "#E74C3C",   # Swedish red — highlights / errors
    "neutral":   "#85929E",   # mid-grey — secondary elements
    "grid":      "#ECF0F1",   # very light grey — background grid
    "text":      "#2C3E50",   # dark slate — all text
    "positive":  "#27AE60",   # green — positive residuals
    "negative":  "#E74C3C",   # red — negative residuals
}
_FIG_DPI   = 150
_FIG_WIDTH = 14   # inches

# Number of top features shown in the importance chart
_TOP_N_FEATURES = 30


# ===========================================================================
# Metric Computation
# ===========================================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    zone: str,
) -> dict:
    """
    Compute a comprehensive suite of regression metrics for one test fold.

    Metrics returned
    ────────────────
    mae:        Mean Absolute Error — primary operational metric.
                Directly interpretable as average MWh forecast error.
    rmse:       Root Mean Squared Error — penalises large errors more.
                Critical for grid operators: a 500 MWh spike matters more
                than five 100 MWh errors.
    mape:       Mean Absolute Percentage Error.  Computed only where
                |y_true| > 1 MWh to avoid ÷0 instability near zero.
                Reported as a percentage (e.g. 12.3 = 12.3%).
    medae:      Median Absolute Error — robust to outliers.
    r2:         Coefficient of determination.  1.0 = perfect; 0.0 = mean
                baseline; negative = worse than predicting the mean.
    bias:       Mean signed error (y_pred - y_true).  Systematic over/
                under-prediction indicator.
    max_error:  Largest single-hour absolute error in the test set.
    p90_error:  90th-percentile absolute error.  Useful for setting
                operational reserve margins.

    Args:
        y_true: Observed imbalance_mwh values (test set).
        y_pred: Model predictions aligned to y_true.
        zone:   Zone label for the results dict.

    Returns:
        Dict of metric name → float value.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    abs_errors = np.abs(y_true - y_pred)

    # MAPE: only compute on hours where |actual| > 1 MWh
    mape_mask = np.abs(y_true) > 1.0
    if mape_mask.sum() > 0:
        mape = float(
            np.mean(abs_errors[mape_mask] / np.abs(y_true[mape_mask])) * 100
        )
    else:
        mape = float("nan")

    metrics = {
        "zone":       zone,
        "n_test":     int(len(y_true)),
        "mae":        float(mean_absolute_error(y_true, y_pred)),
        "rmse":       float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape_pct":   mape,
        "medae":      float(median_absolute_error(y_true, y_pred)),
        "r2":         float(r2_score(y_true, y_pred)),
        "bias":       float(np.mean(y_pred - y_true)),
        "max_error":  float(np.max(abs_errors)),
        "p90_error":  float(np.percentile(abs_errors, 90)),
        "p50_error":  float(np.percentile(abs_errors, 50)),
    }

    logger.info(
        "Metrics | zone=%s | MAE=%.2f | RMSE=%.2f | MAPE=%.1f%% | "
        "R²=%.4f | bias=%.2f | max_err=%.1f | p90_err=%.1f",
        zone,
        metrics["mae"], metrics["rmse"], metrics["mape_pct"],
        metrics["r2"], metrics["bias"],
        metrics["max_error"], metrics["p90_error"],
    )
    return metrics


def baseline_naive_metrics(y_true: np.ndarray, zone: str) -> dict:
    """
    Compute metrics for the naïve lag-24h baseline (persistence forecast).

    The lag-24h baseline predicts that next hour's imbalance equals the
    same-hour yesterday.  It is the natural benchmark for hourly energy
    time-series: a model that cannot beat lag-24h is not useful.

    Returns the same metric dict structure as compute_metrics() but
    prefixed with 'baseline_'.
    """
    y_true = np.asarray(y_true, dtype=float)

    # Lag-24h: shift the test series forward by 24 rows
    y_baseline = np.roll(y_true, 24)
    y_baseline[:24] = np.nan                 # first 24 hours are undefined

    valid = ~np.isnan(y_baseline)
    if valid.sum() == 0:
        return {"zone": zone, "baseline_note": "insufficient test data for baseline"}

    y_t = y_true[valid]
    y_b = y_baseline[valid]
    abs_err = np.abs(y_t - y_b)

    return {
        "zone":            zone,
        "baseline_mae":    float(mean_absolute_error(y_t, y_b)),
        "baseline_rmse":   float(np.sqrt(mean_squared_error(y_t, y_b))),
        "baseline_medae":  float(median_absolute_error(y_t, y_b)),
        "baseline_r2":     float(r2_score(y_t, y_b)),
        "baseline_p90":    float(np.percentile(abs_err, 90)),
    }


# ===========================================================================
# Feature Importance Plot
# ===========================================================================

def plot_feature_importance(
    artifact: ModelArtifact,
    output_dir: Path,
    top_n: int = _TOP_N_FEATURES,
) -> Path:
    """
    Horizontal bar chart of the top-N features by LightGBM gain importance.

    Gain importance (sum of split gains across all trees for a feature) is
    preferred over split-count importance because it is proportional to the
    actual reduction in the MAE loss function — a feature that makes many
    low-quality splits is not as useful as one that makes few high-quality ones.

    Features are colour-coded by their feature class:
      - CLASS 1 (lag/rolling): primary blue
      - CLASS 2 (housing):     teal
      - CLASS 3 (rate):        amber
      - Calendar:              slate

    The chart is annotated with the gain value and cumulative importance
    percentage for rapid at-a-glance interpretation.

    Args:
        artifact:   ModelArtifact with fitted model and feature_cols list.
        output_dir: Directory to write the PNG.
        top_n:      Number of features to display (default: 30).

    Returns:
        Path to the saved PNG file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = artifact.model
    feature_cols = artifact.feature_cols
    zone = artifact.zone

    # LightGBM stores gain importance as a dict or array
    importance_gain = model.booster_.feature_importance(importance_type="gain")
    importance_split = model.booster_.feature_importance(importance_type="split")

    df_imp = pd.DataFrame({
        "feature": feature_cols,
        "gain":    importance_gain,
        "split":   importance_split,
    }).sort_values("gain", ascending=False).reset_index(drop=True)

    df_top = df_imp.head(top_n).copy()
    df_top["gain_pct"]    = df_top["gain"] / df_top["gain"].sum() * 100
    df_top["cum_pct"]     = df_top["gain_pct"].cumsum()

    # Colour by feature class
    def _class_colour(feat: str) -> str:
        if any(feat.startswith(p) for p in [
            "load_mw_lag", "imbalance_mwh_lag", "price_eur_mwh_lag",
            "load_mw_roll", "imbalance_mwh_roll", "price_eur_mwh_roll",
            "imbalance_mwh_diff", "load_mw_diff",
            "imbalance_weekly", "load_mw_cv", "imbalance_mwh_zscore",
            "price_eur_mwh_dev",
        ]):
            return _PALETTE["primary"]    # CLASS 1
        if any(feat.startswith(p) for p in [
            "smahus", "winter_x", "seasonal_temp", "construction_x",
            "peak_tariff_x", "load_mw_yoy", "rate_x_construction",
        ]):
            return "#17A589"              # CLASS 2 — teal
        if any(feat.startswith(p) for p in [
            "riksbank_rate", "riksbank_policy",
        ]):
            return "#D4AC0D"              # CLASS 3 — amber
        return _PALETTE["neutral"]        # Calendar / other

    colours = [_class_colour(f) for f in df_top["feature"]]

    fig, ax = plt.subplots(figsize=(_FIG_WIDTH, max(8, top_n * 0.38)), dpi=_FIG_DPI)
    bars = ax.barh(
        y=range(len(df_top)),
        width=df_top["gain"],
        color=colours,
        edgecolor="white",
        linewidth=0.5,
        height=0.75,
    )

    # Annotate each bar with gain % and cumulative %
    for i, (_, row) in enumerate(df_top.iterrows()):
        ax.text(
            row["gain"] * 1.01, i,
            f"{row['gain_pct']:.1f}%  (cum {row['cum_pct']:.0f}%)",
            va="center", ha="left",
            fontsize=7.5, color=_PALETTE["text"],
        )

    ax.set_yticks(range(len(df_top)))
    ax.set_yticklabels(df_top["feature"], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance (Gain)", fontsize=10, color=_PALETTE["text"])
    ax.set_title(
        f"Top {top_n} Feature Importances by Gain — Zone {zone}\n"
        f"LightGBM Imbalance Regressor  |  "
        f"Trained on {artifact.train_metadata.get('train_rows', '?')} rows",
        fontsize=12, color=_PALETTE["text"], pad=14,
    )

    # Legend for feature classes
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=_PALETTE["primary"], label="CLASS 1 — Lag / Rolling"),
        Patch(facecolor="#17A589",           label="CLASS 2 — Housing / Seasonal"),
        Patch(facecolor="#D4AC0D",           label="CLASS 3 — Riksbank Rate"),
        Patch(facecolor=_PALETTE["neutral"], label="Calendar"),
    ]
    ax.legend(
        handles=legend_elements, loc="lower right",
        fontsize=8, framealpha=0.9,
    )

    ax.set_facecolor(_PALETTE["grid"])
    ax.grid(axis="x", color="white", linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()

    out_path = output_dir / f"{zone}_feature_importance.png"
    fig.savefig(out_path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Feature importance plot saved → %s", out_path)
    return out_path


# ===========================================================================
# Actual vs Predicted Plot
# ===========================================================================

def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: Optional[pd.Series],
    zone: str,
    output_dir: Path,
    n_sample_hours: int = 336,   # 2 weeks for the time-series panel
) -> Path:
    """
    Two-panel figure: scatter (all test points) + time-series overlay (sample).

    Panel 1 — Scatter:
        Each point = one test hour.  Perfect prediction = diagonal line.
        Point density is shown via transparency.  Includes the 45° reference
        line and ±1 MAE bands.

    Panel 2 — Time series:
        First `n_sample_hours` of the test set, actual vs predicted.
        Shows where the model tracks well and where it struggles (e.g.
        during Midsommar load drops or post-rate-decision industrial ramps).

    Args:
        y_true:          Observed imbalance_mwh (test set, numpy array).
        y_pred:          Model predictions (aligned).
        timestamps:      pd.Series of UTC timestamps aligned to y_true.
        zone:            Zone label.
        output_dir:      Directory for output PNG.
        n_sample_hours:  Hours to show in time-series panel.

    Returns:
        Path to saved PNG.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mae = float(mean_absolute_error(y_true, y_pred))

    fig, (ax_scatter, ax_ts) = plt.subplots(
        1, 2,
        figsize=(_FIG_WIDTH, 6),
        dpi=_FIG_DPI,
        gridspec_kw={"width_ratios": [1, 1.8]},
    )

    # ── Panel 1: Scatter ─────────────────────────────────────────────────
    lim = max(np.abs(y_true).max(), np.abs(y_pred).max()) * 1.05
    ax_scatter.scatter(
        y_true, y_pred,
        alpha=0.18, s=6,
        color=_PALETTE["primary"],
        rasterized=True,
    )
    ax_scatter.plot([-lim, lim], [-lim, lim], "--", color=_PALETTE["accent"],
                    lw=1.2, label="Perfect prediction")
    ax_scatter.fill_between(
        [-lim, lim],
        [-lim - mae, lim - mae],
        [-lim + mae, lim + mae],
        alpha=0.12, color=_PALETTE["accent"],
        label=f"±MAE ({mae:.1f} MWh)",
    )
    ax_scatter.set_xlim(-lim, lim)
    ax_scatter.set_ylim(-lim, lim)
    ax_scatter.set_xlabel("Actual imbalance_mwh", fontsize=10)
    ax_scatter.set_ylabel("Predicted imbalance_mwh", fontsize=10)
    ax_scatter.set_title(f"Actual vs Predicted — {zone}\n({len(y_true):,} test hours)",
                          fontsize=11)
    ax_scatter.legend(fontsize=8)
    ax_scatter.set_facecolor(_PALETTE["grid"])
    ax_scatter.grid(color="white", linewidth=0.8)
    ax_scatter.spines[["top", "right"]].set_visible(False)

    # ── Panel 2: Time series ─────────────────────────────────────────────
    n = min(n_sample_hours, len(y_true))
    x_axis = (
        timestamps.iloc[:n].values
        if timestamps is not None and len(timestamps) >= n
        else np.arange(n)
    )
    ax_ts.fill_between(
        x_axis, y_true[:n], y_pred[:n],
        alpha=0.15, color=_PALETTE["accent"], label="Forecast error",
    )
    ax_ts.plot(x_axis, y_true[:n],  color=_PALETTE["primary"],  lw=1.2, label="Actual")
    ax_ts.plot(x_axis, y_pred[:n],  color=_PALETTE["accent"],   lw=1.0,
               linestyle="--", label="Predicted", alpha=0.85)
    ax_ts.axhline(0, color=_PALETTE["neutral"], lw=0.8, linestyle=":")
    ax_ts.set_xlabel("Time (UTC)", fontsize=10)
    ax_ts.set_ylabel("Imbalance (MWh)", fontsize=10)
    ax_ts.set_title(
        f"Test Window — First {n}h  |  MAE={mae:.1f} MWh",
        fontsize=11,
    )
    ax_ts.legend(fontsize=8)
    ax_ts.set_facecolor(_PALETTE["grid"])
    ax_ts.grid(color="white", linewidth=0.8)
    ax_ts.spines[["top", "right"]].set_visible(False)
    if timestamps is not None:
        fig.autofmt_xdate(rotation=30)

    fig.suptitle(
        f"Model Evaluation — SE Grid Imbalance Regressor | Zone {zone}",
        fontsize=13, y=1.02, color=_PALETTE["text"],
    )
    fig.tight_layout()

    out_path = output_dir / f"{zone}_actual_vs_predicted.png"
    fig.savefig(out_path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Actual vs Predicted plot saved → %s", out_path)
    return out_path


# ===========================================================================
# Residual Diagnostics Plot
# ===========================================================================

def plot_residual_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    zone: str,
    output_dir: Path,
) -> Path:
    """
    Two-panel residual diagnostic figure.

    Panel 1 — Residual histogram + KDE:
        Distribution of (y_pred - y_true) with a normal reference curve.
        Useful for detecting systematic bias, heavy tails, or multimodality.
        Annotated with mean, std, and skewness.

    Panel 2 — Normal Q-Q plot:
        Quantiles of the residuals vs theoretical normal quantiles.
        Deviations from the diagonal reveal non-normality: fat tails (LightGBM
        tends to under-predict extreme imbalance events) appear as S-shaped
        curves above/below the diagonal.

    Args:
        y_true:     Observed values.
        y_pred:     Predicted values.
        zone:       Zone label.
        output_dir: Output directory.

    Returns:
        Path to saved PNG.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    residuals = y_pred - y_true
    res_mean  = float(np.mean(residuals))
    res_std   = float(np.std(residuals))
    res_skew  = float(scipy_stats.skew(residuals))
    res_kurt  = float(scipy_stats.kurtosis(residuals))

    fig, (ax_hist, ax_qq) = plt.subplots(1, 2, figsize=(12, 5), dpi=_FIG_DPI)

    # ── Panel 1: Residual distribution ───────────────────────────────────
    n_bins = min(80, max(20, len(residuals) // 200))
    ax_hist.hist(
        residuals, bins=n_bins,
        color=_PALETTE["primary"], alpha=0.75,
        density=True, edgecolor="white", linewidth=0.4,
        label="Residuals",
    )
    # Normal reference
    x_ref = np.linspace(residuals.min(), residuals.max(), 300)
    ax_hist.plot(
        x_ref,
        scipy_stats.norm.pdf(x_ref, res_mean, res_std),
        color=_PALETTE["accent"], lw=1.8, linestyle="--",
        label=f"N({res_mean:.1f}, {res_std:.1f}²)",
    )
    ax_hist.axvline(0, color=_PALETTE["neutral"], lw=1.0, linestyle=":")
    ax_hist.axvline(res_mean, color=_PALETTE["accent"], lw=1.2,
                    linestyle="-", label=f"Mean bias = {res_mean:.2f}")
    ax_hist.set_xlabel("Residual  (Predicted − Actual)  [MWh]", fontsize=10)
    ax_hist.set_ylabel("Density", fontsize=10)
    ax_hist.set_title(
        f"Residual Distribution — Zone {zone}\n"
        f"skew={res_skew:.2f}  kurt={res_kurt:.2f}  "
        f"std={res_std:.1f} MWh",
        fontsize=11,
    )
    ax_hist.legend(fontsize=8)
    ax_hist.set_facecolor(_PALETTE["grid"])
    ax_hist.grid(color="white", linewidth=0.8)
    ax_hist.spines[["top", "right"]].set_visible(False)

    # ── Panel 2: Q-Q plot ─────────────────────────────────────────────────
    # Sample quantiles to avoid O(n²) on large test sets
    n_qq = min(len(residuals), 2000)
    sample_idx = np.random.default_rng(42).choice(len(residuals), n_qq, replace=False)
    res_sample = np.sort(residuals[sample_idx])

    (osm, osr), (slope, intercept, r_qq) = scipy_stats.probplot(
        res_sample, dist="norm", fit=True
    )

    ax_qq.scatter(osm, osr, s=6, alpha=0.4, color=_PALETTE["primary"], rasterized=True)
    x_line = np.linspace(osm.min(), osm.max(), 200)
    ax_qq.plot(
        x_line, slope * x_line + intercept,
        color=_PALETTE["accent"], lw=1.5, linestyle="--",
        label=f"Fit  R={r_qq:.4f}",
    )
    ax_qq.set_xlabel("Theoretical Quantiles", fontsize=10)
    ax_qq.set_ylabel("Sample Quantiles  [MWh]", fontsize=10)
    ax_qq.set_title(
        f"Normal Q-Q Plot — Zone {zone}\n"
        f"Deviations from diagonal indicate heavy tails / skew",
        fontsize=11,
    )
    ax_qq.legend(fontsize=8)
    ax_qq.set_facecolor(_PALETTE["grid"])
    ax_qq.grid(color="white", linewidth=0.8)
    ax_qq.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out_path = output_dir / f"{zone}_residual_diagnostics.png"
    fig.savefig(out_path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Residual diagnostics plot saved → %s", out_path)
    return out_path


# ===========================================================================
# Rolling MAE Plot
# ===========================================================================

def plot_rolling_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: Optional[pd.Series],
    zone: str,
    output_dir: Path,
    window_hours: int = 168,   # 7-day rolling window
) -> Path:
    """
    Rolling MAE over the test window — temporal error profile.

    A flat rolling MAE suggests consistent model performance.  Spikes reveal
    specific periods where the model struggles:
      - Winter storm events (sudden load surges)
      - Riksbank rate decision days (industrial load behavioural shift)
      - Holiday periods (pattern breakdowns)

    The chart overlays the actual imbalance time-series (scaled to the
    secondary axis) so you can correlate error spikes with market events.

    Args:
        y_true:        Observed values.
        y_pred:        Predicted values.
        timestamps:    UTC timestamps aligned to y_true (optional).
        zone:          Zone label.
        output_dir:    Output directory.
        window_hours:  Rolling window size in hours (default 168 = 7 days).

    Returns:
        Path to saved PNG.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    abs_errors = np.abs(y_true - y_pred)
    rolling_mae = (
        pd.Series(abs_errors)
        .rolling(window=window_hours, min_periods=window_hours // 2)
        .mean()
        .values
    )
    overall_mae = float(np.mean(abs_errors))

    x_axis = (
        timestamps.values
        if timestamps is not None and len(timestamps) == len(y_true)
        else np.arange(len(y_true))
    )

    fig, ax_mae = plt.subplots(figsize=(_FIG_WIDTH, 5), dpi=_FIG_DPI)

    ax_mae.plot(
        x_axis, rolling_mae,
        color=_PALETTE["primary"], lw=1.5,
        label=f"Rolling MAE ({window_hours}h window)",
    )
    ax_mae.axhline(
        overall_mae, color=_PALETTE["accent"], lw=1.2, linestyle="--",
        label=f"Overall MAE = {overall_mae:.1f} MWh",
    )
    ax_mae.fill_between(x_axis, 0, rolling_mae, alpha=0.12, color=_PALETTE["primary"])

    # Secondary axis: actual imbalance (for event correlation)
    ax_actual = ax_mae.twinx()
    ax_actual.plot(
        x_axis, y_true,
        color=_PALETTE["neutral"], lw=0.6, alpha=0.5,
        label="Actual imbalance",
    )
    ax_actual.set_ylabel("Actual Imbalance (MWh)", fontsize=9, color=_PALETTE["neutral"])
    ax_actual.tick_params(axis="y", labelcolor=_PALETTE["neutral"])

    ax_mae.set_xlabel("Time (UTC)", fontsize=10)
    ax_mae.set_ylabel(f"Rolling MAE ({window_hours}h) [MWh]", fontsize=10)
    ax_mae.set_title(
        f"Temporal Error Profile — Zone {zone}  |  "
        f"Rolling {window_hours}h MAE over Test Window",
        fontsize=12, color=_PALETTE["text"],
    )

    # Unified legend
    lines_mae,   labels_mae   = ax_mae.get_legend_handles_labels()
    lines_act,   labels_act   = ax_actual.get_legend_handles_labels()
    ax_mae.legend(lines_mae + lines_act, labels_mae + labels_act, fontsize=8, loc="upper left")

    ax_mae.set_facecolor(_PALETTE["grid"])
    ax_mae.grid(color="white", linewidth=0.8)
    ax_mae.spines[["top", "right"]].set_visible(False)

    if timestamps is not None:
        fig.autofmt_xdate(rotation=30)

    fig.tight_layout()
    out_path = output_dir / f"{zone}_rolling_mae.png"
    fig.savefig(out_path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Rolling MAE plot saved → %s", out_path)
    return out_path


# ===========================================================================
# Zone Evaluation Orchestrator
# ===========================================================================

def evaluate_zone(
    artifact: ModelArtifact,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
    timestamps: Optional[pd.Series] = None,
) -> dict:
    """
    Full evaluation pipeline for one bidding zone.

    Steps:
      1. Generate predictions on the held-out test set.
      2. Compute all regression metrics.
      3. Compute naïve lag-24h baseline for comparison.
      4. Produce all four diagnostic plots.
      5. Write a JSON metrics report.

    Args:
        artifact:   ModelArtifact loaded from disk or returned by train.py.
        X_test:     Feature matrix for the held-out test set.
        y_test:     True imbalance_mwh values for the held-out test set.
        output_dir: Root output directory (plots go into a zone sub-folder).
        timestamps: UTC timestamps aligned to X_test / y_test (optional,
                    used for time-axis labelling in plots).

    Returns:
        Evaluation results dict with metrics, baseline, and plot paths.
    """
    zone = artifact.zone
    zone_out = output_dir / zone
    zone_out.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Evaluating zone=%s | test_rows=%d", zone, len(X_test))
    logger.info("=" * 60)

    # Align feature columns: model expects exact feature list from training
    missing_features = [f for f in artifact.feature_cols if f not in X_test.columns]
    if missing_features:
        raise ValueError(
            f"Zone {zone}: test set missing {len(missing_features)} feature(s) "
            f"expected by the model: {missing_features[:10]}..."
        )
    X_aligned = X_test[artifact.feature_cols]

    # Generate predictions
    y_true = y_test.values.astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_pred = artifact.model.predict(X_aligned).astype(float)

    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, zone)
    baseline = baseline_naive_metrics(y_true, zone)

    # Improvement over baseline
    if "baseline_mae" in baseline and baseline["baseline_mae"] > 0:
        metrics["mae_improvement_vs_baseline_pct"] = float(
            (baseline["baseline_mae"] - metrics["mae"]) / baseline["baseline_mae"] * 100
        )

    # All four plots
    plot_paths: dict[str, str] = {}

    p = plot_feature_importance(artifact, zone_out)
    plot_paths["feature_importance"] = str(p)

    p = plot_actual_vs_predicted(y_true, y_pred, timestamps, zone, zone_out)
    plot_paths["actual_vs_predicted"] = str(p)

    p = plot_residual_diagnostics(y_true, y_pred, zone, zone_out)
    plot_paths["residual_diagnostics"] = str(p)

    p = plot_rolling_mae(y_true, y_pred, timestamps, zone, zone_out)
    plot_paths["rolling_mae"] = str(p)

    # Combine into results dict
    results = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "zone": zone,
        "model_metadata": artifact.train_metadata,
        "metrics": metrics,
        "baseline": baseline,
        "plots": plot_paths,
    }

    # Write JSON metrics report
    report_path = zone_out / f"{zone}_evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)
    logger.info("Evaluation report saved → %s", report_path)

    # Human-readable console summary
    _print_metrics_table(metrics, baseline, zone)

    return results


def _print_metrics_table(metrics: dict, baseline: dict, zone: str) -> None:
    """Print a formatted metrics table to the logger."""
    sep = "─" * 56
    logger.info(sep)
    logger.info("  EVALUATION SUMMARY — Zone %-6s", zone)
    logger.info(sep)
    logger.info("  %-28s %12s %12s", "Metric", "LightGBM", "Lag-24h baseline")
    logger.info(sep)

    rows = [
        ("MAE  (MWh)",    "mae",   "baseline_mae"),
        ("RMSE (MWh)",    "rmse",  "baseline_rmse"),
        ("MedAE (MWh)",   "medae", "baseline_medae"),
        ("R²",            "r2",    "baseline_r2"),
        ("P90 Error",     "p90_error", "baseline_p90"),
    ]
    for label, mk, bk in rows:
        mv = metrics.get(mk, float("nan"))
        bv = baseline.get(bk, float("nan"))
        logger.info("  %-28s %12.3f %12.3f", label, mv, bv)

    logger.info(sep)
    if "mae_improvement_vs_baseline_pct" in metrics:
        logger.info(
            "  MAE improvement vs lag-24h baseline: %.1f%%",
            metrics["mae_improvement_vs_baseline_pct"],
        )
    if "mape_pct" in metrics:
        logger.info("  MAPE: %.1f%%  |  Bias: %.2f MWh", metrics["mape_pct"], metrics["bias"])
    logger.info(sep)


# ===========================================================================
# Batch Evaluation Entry Point
# ===========================================================================

def evaluate_all_zones(
    model_dir: Path,
    gold_parquet_dir: Path,
    output_dir: Path,
    zone_filter: Optional[list[str]] = None,
    test_hours: int = 720,
) -> dict[str, dict]:
    """
    Load all zone model artifacts and evaluate against held-out test sets.

    This function reconstructs the exact same test set that train.py held
    out by applying the same temporal_train_test_split logic.  It loads the
    Gold layer, takes the last `test_hours` rows per zone, and scores each
    model against them.

    Args:
        model_dir:         Directory containing {zone}_lgbm_imbalance.joblib files.
        gold_parquet_dir:  Root Gold layer directory.
        output_dir:        Root output directory for evaluation results.
        zone_filter:       Optional list of zones to evaluate.
        test_hours:        Must match the value used during training.

    Returns:
        Dict mapping zone → evaluation results dict.
    """
    from .train import load_gold_parquet, resolve_feature_columns

    model_dir  = Path(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_gold = load_gold_parquet(gold_parquet_dir, zone_filter)
    zones = zone_filter or sorted(df_gold["zone"].unique().tolist())

    all_results: dict[str, dict] = {}

    for zone in zones:
        model_path = model_dir / f"{zone}_lgbm_imbalance.joblib"
        if not model_path.exists():
            logger.warning("No model artifact for zone %s at %s — skipping.", zone, model_path)
            continue

        try:
            artifact = load_model_artifact(model_path)
        except Exception as exc:
            logger.error("Failed to load model for zone %s: %s", zone, exc, exc_info=True)
            continue

        df_zone = df_gold[df_gold["zone"] == zone].copy()
        df_ready = df_zone[df_zone["feature_ready"]].sort_values("timestamp_utc")

        if len(df_ready) < test_hours:
            logger.warning(
                "Zone %s: only %d feature-ready rows, cannot reconstruct %dh test set.",
                zone, len(df_ready), test_hours,
            )
            continue

        df_test = df_ready.iloc[-test_hours:].copy()
        feature_cols = resolve_feature_columns(df_test, "imbalance_mwh")
        X_test = df_test[feature_cols]
        y_test = df_test["imbalance_mwh"]
        timestamps = df_test["timestamp_utc"].reset_index(drop=True)

        try:
            results = evaluate_zone(artifact, X_test, y_test, output_dir, timestamps)
            all_results[zone] = results
        except Exception as exc:
            logger.error("Evaluation failed for zone %s: %s", zone, exc, exc_info=True)

    # Write cross-zone summary
    summary_path = output_dir / "evaluation_summary.json"
    summary = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "zones": {
            zone: res.get("metrics", {})
            for zone, res in all_results.items()
        },
    }
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    logger.info("Cross-zone evaluation summary → %s", summary_path)

    return all_results


# ===========================================================================
# CLI Entry Point
# ===========================================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Evaluate LightGBM grid imbalance models and produce diagnostic plots."
    )
    parser.add_argument(
        "--model-dir",
        default="outputs/models",
        help="Directory containing trained .joblib model artifacts.",
    )
    parser.add_argument(
        "--gold-dir",
        default="data/gold",
        help="Root directory of the Gold layer Parquet store.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/evaluation",
        help="Directory for evaluation plots and JSON reports.",
    )
    parser.add_argument(
        "--zones",
        nargs="+",
        default=None,
        help="Zones to evaluate (default: all). E.g. --zones SE1 SE3",
    )
    parser.add_argument(
        "--test-hours",
        type=int,
        default=720,
        help="Hold-out test window size in hours (must match training value).",
    )
    args = parser.parse_args()

    evaluate_all_zones(
        model_dir=Path(args.model_dir),
        gold_parquet_dir=Path(args.gold_dir),
        output_dir=Path(args.output_dir),
        zone_filter=args.zones,
        test_hours=args.test_hours,
    )
