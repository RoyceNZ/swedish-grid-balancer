"""
src/models/train.py  — MARKET FUNDAMENTALS TRAINING ADDITIONS
=============================================================================
Additive patch for train.py.
=============================================================================

This file documents all changes required to enable LightGBM to train over
the three new CLASS 4 market fundamental feature arrays.

There are three distinct integration points:

  1. _NON_FEATURE_COLS — add the raw source columns that must NOT be fed
     directly to the model (features use derived/ratio/lag versions).

  2. resolve_feature_columns() — _exclude set update to match point 1.

  3. TrainingConfig — update cv_gap_hours documentation to explain why
     168h remains the correct gap for the new features.

The 5-fold TimeSeriesSplit with cv_gap_hours=168 requires NO change because:
  - CLASS 4A (outage ratio lags): max lag is 168h → 168h gap is exactly right.
  - CLASS 4B (net position diffs): max diff is 24h → fully covered by 168h gap.
  - CLASS 4C (hydro depletion velocity): max diff is 2,016h (12 weeks) → the
    gap must be ≥ max_lag used in features.  However, the hydro depletion
    velocity at hour T is computed from reservoir_fill_ratio[T] −
    reservoir_fill_ratio[T − 12w].  Since reservoir data has a 1-week
    publication lag (enforced at Silver), the effective information horizon
    is already conservative.  The 168h CV gap prevents the 1h and 24h lag
    features (which dominate autocorrelation) from leaking; the 12w hydro
    diff feature is insensitive to a 168h gap because its minimum step is
    168h by construction.
=============================================================================
"""

from __future__ import annotations

# ===========================================================================
# PATCH 1 — _NON_FEATURE_COLS update
# ===========================================================================
#
# In grid_balancing.py, replace the existing _NON_FEATURE_COLS definition with:
#
# _NON_FEATURE_COLS: frozenset[str] = frozenset({
#     # ── Identifiers ─────────────────────────────────────────────────────
#     "timestamp_utc", "zone", "year",
#     # ── Targets ─────────────────────────────────────────────────────────
#     "imbalance_mwh",
#     # ── Raw grid metrics (features use lag/rolling versions) ─────────────
#     "load_mw", "price_eur_mwh",
#     "load_mw_clean", "imbalance_mwh_clean",
#     # ── SCB / Riksbank raw levels ─────────────────────────────────────────
#     "smahus_construction_index", "smahus_price_index",
#     "riksbank_policy_rate_pct",
#     # ── Weather raw observations ──────────────────────────────────────────
#     "temperature_c", "wind_speed_ms",
#     "quality_flag_temp", "quality_flag_wind",
#     # ── Quality / metadata flags ─────────────────────────────────────────
#     "is_anomaly", "is_anomaly_weather", "is_quarantined",
#     "has_imputed_grid", "feature_ready",
#     "direction", "resolution_minutes",
#     # ── CLASS 4 raw market fundamental sources ────────────────────────────
#     # These raw level columns are excluded; CLASS 4 features (ratios, diffs,
#     # rolling stats, interaction terms) are the model inputs.
#     "outage_mw",                      # → nuclear_outage_vs_demand_ratio,
#                                       #   outage_mw_roll_mean_*h, outage_mw_diff_*h
#     "active_unit_count",              # → is_major_nuclear_outage
#     "scheduled_net_position_mw",      # → scheduled_flow_stress_metric,
#                                       #   net_position_roll_mean_*h,
#                                       #   net_position_diff_*h
#     "reservoir_fill_ratio",           # → hydro_depletion_velocity,
#                                       #   reservoir_fill_diff_*,
#                                       #   hydro_depletion_x_imbalance_lag*h,
#                                       #   is_hydro_crunch
# })
#
# In train.py, update the _exclude set inside resolve_feature_columns() to
# mirror the same set (both must stay in sync):

_TRAIN_EXCLUDE_COLS: frozenset[str] = frozenset({
    # Identifiers
    "timestamp_utc", "zone", "year",
    # Targets
    "imbalance_mwh",
    # Raw grid metrics
    "load_mw", "price_eur_mwh",
    "load_mw_clean", "imbalance_mwh_clean",
    # SCB / Riksbank raw levels
    "smahus_construction_index", "smahus_price_index",
    "riksbank_policy_rate_pct",
    # Weather raw observations
    "temperature_c", "wind_speed_ms",
    "quality_flag_temp", "quality_flag_wind",
    # Quality / metadata flags
    "is_anomaly", "is_anomaly_weather", "is_quarantined",
    "has_imputed_grid", "feature_ready",
    "direction", "resolution_minutes",
    # CLASS 4 raw market fundamental sources (excluded — use derived features)
    "outage_mw",
    "active_unit_count",
    "scheduled_net_position_mw",
    "reservoir_fill_ratio",
})


# ===========================================================================
# PATCH 2 — resolve_feature_columns() update
# ===========================================================================
#
# Replace the existing _exclude set literal inside resolve_feature_columns()
# with _TRAIN_EXCLUDE_COLS (defined above), or equivalently copy the expanded
# set directly.  The function body is otherwise unchanged:
#
# def resolve_feature_columns(df: pd.DataFrame, target_col: str) -> list[str]:
#     _exclude = _TRAIN_EXCLUDE_COLS | {target_col}
#     feature_cols = sorted([c for c in df.columns if c not in _exclude])
#     logger.info(
#         "Feature set resolved: %d columns (excluded %d non-feature cols)",
#         len(feature_cols), len(_exclude),
#     )
#     return feature_cols


# ===========================================================================
# PATCH 3 — TrainingConfig docstring update
# ===========================================================================
#
# Extend the cv_gap_hours docstring to document CLASS 4 gap requirements.
# Replace the existing cv_gap_hours field with:
#
#     cv_gap_hours: int = 168          # 7-day gap between CV folds
#     #
#     # Gap justification for CLASS 4 market fundamental features:
#     #
#     # CLASS 4A (outage ratio):
#     #   Max lag feature: outage_mw_diff_24h (24h lookback)
#     #   168h gap > 24h → no adjacent-hour leakage possible.
#     #
#     # CLASS 4B (net position / flow stress):
#     #   Max lag feature: net_position_diff_24h (24h lookback)
#     #   168h gap > 24h → safe.
#     #   Gate-closure lag (12h UTC D-1) further insulates pre-gate rows.
#     #
#     # CLASS 4C (hydro depletion velocity):
#     #   Max diff feature: reservoir_fill_diff_12w (2,016h = 12 weeks).
#     #   The 168h gap does NOT fully cover the 12w lookback window.
#     #   However, the hydro fill ratio already carries a 1-week publication
#     #   lag from the Silver layer, so no future reservoir state can appear
#     #   in training features.  The 168h gap covers the short-horizon lag
#     #   features (imbalance_mwh_lag_1h, lag_24h) that are multiplied into
#     #   the hydro interaction terms; the hydro signal itself is already
#     #   causally bounded by the publication-lag shift in Silver.
#     #
#     # Conclusion: cv_gap_hours=168 is sufficient and correct for all
#     # CLASS 1–4 features given the Silver-layer publication-lag enforcement.


# ===========================================================================
# PATCH 4 — Full updated TrainingConfig (drop-in replacement)
# ===========================================================================
#
# This is the complete updated TrainingConfig dataclass.  Copy it verbatim
# into train.py, replacing the existing definition.

UPDATED_TRAINING_CONFIG_SOURCE = '''
@dataclass
class TrainingConfig:
    """
    Immutable configuration for one training run.

    Serialised alongside the model artifact so any future inference session
    can reconstruct the exact feature set and hyperparameters used.

    Attributes
    ──────────
    gold_parquet_dir:
        Root directory of the Gold layer Parquet store.
    zone_filter:
        Subset of bidding zones to train on.  None = all available zones.
    target_col:
        Target variable column name (must exist in Gold layer).
    test_hours:
        Number of hours reserved as the final hold-out test set.
        Never touched during cross-validation or fitting.
    n_cv_splits:
        Number of TimeSeriesSplit folds for cross-validation.
    cv_gap_hours:
        Gap inserted between each CV train/test fold to prevent adjacent-
        hour leakage.  168h (7 days) covers CLASS 1–4 feature lookbacks:
          - CLASS 1:  max lag = 168h            → exactly covered
          - CLASS 2:  max shift = 2,190h (QoQ)  → Silver-layer causal ffill
          - CLASS 3:  max lookback = 8,760h      → Silver-layer causal ffill
          - CLASS 4A: max lag = 24h outage diff  → covered by 168h gap
          - CLASS 4B: max lag = 24h pos diff     → covered; gate-close enforced
          - CLASS 4C: max diff = 2,016h hydro    → publication lag in Silver
    lgbm_params:
        LightGBM hyperparameters passed to lgb.LGBMRegressor().
        Note on new market fundamental features:
          - nuclear_outage_vs_demand_ratio can spike to large values during
            simultaneous multi-unit outages; the lambda_l1/l2 regularisation
            and min_gain_to_split act as natural guards.
          - is_major_nuclear_outage and is_hydro_crunch are binary (0/1);
            LightGBM handles these natively without special encoding.
          - Hydro depletion interaction columns (SE1/SE2 only) will be NaN
            for SE3/SE4 rows; use_missing=True handles this correctly.
    """
    gold_parquet_dir:       str = "data/gold"
    zone_filter:            Optional[list[str]] = None
    target_col:             str = "imbalance_mwh"
    test_hours:             int = 168
    n_cv_splits:            int = 5
    cv_gap_hours:           int = 168
    early_stopping_rounds:  int = 50
    random_state:           int = 42
    output_dir:             str = "outputs/models"
    lgbm_params: dict = field(default_factory=lambda: {
        "objective":          "regression_l1",
        "metric":             ["mae", "rmse"],
        "n_estimators":       2000,
        "learning_rate":      0.05,
        "num_leaves":         127,
        "max_depth":          -1,
        "min_child_samples":  50,
        "feature_fraction":   0.8,
        "bagging_fraction":   0.85,
        "bagging_freq":       5,
        "lambda_l1":          0.1,
        "lambda_l2":          0.1,
        "min_gain_to_split":  0.01,
        "use_missing":        True,   # critical: NaN market-fundamental features
                                      # in SE3/SE4 hydro interaction columns
        "verbose":            -1,
        "n_jobs":             -1,
    })
'''


# ===========================================================================
# Validation helper: enumerate expected CLASS 4 feature columns
# ===========================================================================

def expected_class4_feature_columns(zone: str) -> list[str]:
    """
    Return the full list of CLASS 4 column names expected in the Gold layer
    for a given zone.  Used in pipeline validation / smoke tests to confirm
    that build_market_fundamental_features() produced all expected outputs.

    Args:
        zone: Bidding zone string ('SE1', 'SE2', 'SE3', 'SE4').

    Returns:
        Sorted list of column name strings.
    """
    cols: list[str] = [
        # CLASS 4A — Outage vs demand
        "nuclear_outage_vs_demand_ratio",
        "outage_mw_roll_mean_24h",
        "outage_mw_roll_mean_168h",
        "is_major_nuclear_outage",
        "outage_mw_diff_1h",
        "outage_mw_diff_24h",
        # CLASS 4B — Scheduled flow stress
        "scheduled_flow_stress_abs",
        "net_position_roll_mean_24h",
        "net_position_roll_mean_168h",
        "net_position_roll_std_24h",
        "net_position_roll_std_168h",
        "scheduled_flow_stress_metric",
        "scheduled_flow_stress_x_seasonal",
        "net_position_diff_1h",
        "net_position_diff_24h",
        # CLASS 4C — Hydro depletion (all zones)
        "reservoir_fill_diff_1w",
        "reservoir_fill_diff_4w",
        "reservoir_fill_diff_12w",
        "hydro_depletion_velocity",
        "reservoir_fill_roll_mean_4w",
        "reservoir_fill_roll_mean_12w",
    ]
    # SE1/SE2-specific interaction columns
    if zone in {"SE1", "SE2"}:
        cols += [
            "hydro_depletion_x_imbalance_lag1h",
            "hydro_depletion_x_imbalance_lag24h",
            "is_hydro_crunch",
        ]
    return sorted(cols)
