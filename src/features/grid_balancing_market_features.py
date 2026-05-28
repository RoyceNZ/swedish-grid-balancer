"""
src/features/grid_balancing.py  — CLASS 4 MARKET FUNDAMENTALS ADDITIONS
=============================================================================
Additive patch for the existing GridBalancingFeatureEngineer.
=============================================================================

Merge instructions:
  1. Add the CLASS 4 constants (below) to the module-level constants block
     in grid_balancing.py after the CLASS 3 Riksbank lookback constants.
  2. Add _NON_FEATURE_COLS_MARKET_FUNDAMENTALS to the _NON_FEATURE_COLS set.
  3. Insert build_market_fundamental_features() into the module after
     build_rate_features().
  4. Call build_market_fundamental_features() from
     GridBalancingFeatureEngineer.build_feature_matrix() as a new stage
     after the CLASS 3 block.

All three engineered composite features pass through the existing
_sanitise_numeric_columns() framework immediately after computation
so that Inf/NaN values from division, ratio, or interaction operations
are caught before they reach the model training pipeline.
=============================================================================
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from src.features.grid_balancing import _rolling_safe, _sanitise_numeric_columns

logger = logging.getLogger("grid_balancing_features")


# ===========================================================================
# CLASS 4 — Market Fundamental Feature Constants
# ===========================================================================

# Rolling windows used for REMIT and net-position features (in hours)
_OUTAGE_ROLL_WINDOWS: list[int] = [24, 168]        # 1-day, 7-day
_NET_POS_ROLL_WINDOWS: list[int] = [24, 168]

# Hydro depletion velocity: short autocorrelation lookbacks (weeks expressed in hours)
_HYDRO_LOOKBACK_HOURS_1W:  int = 168    # 1 week
_HYDRO_LOOKBACK_HOURS_4W:  int = 672    # 4 weeks
_HYDRO_LOOKBACK_HOURS_12W: int = 2_016  # 12 weeks (~3 months)

# Zones where the hydro buffer depletion feature is physically meaningful.
# SE1/SE2 hold the northern reservoir basin; SE3/SE4 are thermal-dominant.
_HYDRO_RELEVANT_ZONES = frozenset({"SE1", "SE2"})

# Raw source columns that must not be fed directly to the model
# (features use derived/lagged/ratio versions instead)
_NON_FEATURE_COLS_MARKET_FUNDAMENTALS: frozenset[str] = frozenset({
    "outage_mw",
    "active_unit_count",
    "scheduled_net_position_mw",
    "reservoir_fill_ratio",
})

# Wind-speed and temperature columns produced by the weather stream.
# Referenced in the flow-stress interaction feature.
_WIND_COL  = "wind_speed_ms"
_TEMP_COL  = "temperature_c"


# ===========================================================================
# CLASS 4 — build_market_fundamental_features()
# ===========================================================================

def build_market_fundamental_features(
    df: pd.DataFrame,
    config,            # FeatureConfig instance (passed through from orchestrator)
    zone: str,
) -> tuple[pd.DataFrame, list[str]]:
    """
    CLASS 4 — Market Fundamental Feature Engineering.

    Computes three composite market intelligence features:

      4A  nuclear_outage_vs_demand_ratio
      4B  scheduled_flow_stress_metric
      4C  hydro_buffer_depletion_velocity   (SE1/SE2 only)

    All new columns are passed through _sanitise_numeric_columns() before
    return so that division-by-zero, Inf, and extreme clip violations are
    neutralised before reaching LightGBM.

    Strictly causal:
        All lag and rolling operations use shift() and rolling() which are
        inherently backward-looking.  No future values are used at any point.

    Args:
        df:     Silver/Gold DataFrame for a single bidding zone.
                Must contain: load_mw_clean, outage_mw,
                              scheduled_net_position_mw, reservoir_fill_ratio.
                Weather columns (wind_speed_ms, temperature_c) are used when
                present but are not required.
        config: FeatureConfig instance (provides clip_lower, clip_upper,
                nan_sentinel, min_periods_fraction).
        zone:   Bidding zone string (used for zone-specific feature gating).

    Returns:
        (df, new_cols): DataFrame with CLASS 4 columns appended, and the
                        sorted list of newly added column names.
    """
    new_cols: list[str] = []

    # ── Guard: require load_mw_clean at minimum ────────────────────────────
    if "load_mw_clean" not in df.columns:
        logger.warning(
            "CLASS4 | zone=%s | load_mw_clean not found — skipping.", zone
        )
        return df, new_cols

    # ==================================================================
    # 4A — Nuclear / Base-load Outage vs Demand Ratio
    # ==================================================================
    #
    # Formula:  nuclear_outage_vs_demand_ratio = outage_mw / load_mw_clean
    #
    # Interpretation:
    #   - Near zero: no significant base-load capacity offline; grid is
    #     operating from its full generation stack.
    #   - > 0.05 (5%): meaningful forced reduction in dispatchable base-load;
    #     the system must substitute thermal, import, or reduce through
    #     demand response to maintain frequency.
    #   - > 0.10 (10%): severe stress; historically correlated with large
    #     positive imbalance events in SE3/SE4 where nuclear is concentrated.
    #
    # Causality:
    #   outage_mw at each hour is the ACTIVE offline MW at that instant
    #   (derived from REMIT UMM real-time messages, not forward-looking).
    #   load_mw_clean is the contemporaneous observed load.  The ratio is
    #   therefore a point-in-time measurement, not a forecast.
    #
    # Lagged versions capture the persistence of an outage event: a trip
    # that began 24h or 168h ago is still weighing on the system and is
    # a strong predictor of sustained imbalance stress.

    if "outage_mw" in df.columns:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            denom = df["load_mw_clean"].replace(0, np.nan)
            df["nuclear_outage_vs_demand_ratio"] = df["outage_mw"] / denom
        new_cols.append("nuclear_outage_vs_demand_ratio")

        # Rolling outage severity (sum of offline MW over trailing windows)
        for w in _OUTAGE_ROLL_WINDOWS:
            min_p = max(1, int(w * config.min_periods_fraction))
            col = f"outage_mw_roll_mean_{w}h"
            df[col] = _rolling_safe(df["outage_mw"], w, min_p, "mean")
            new_cols.append(col)

        # Binary flag: is there currently a "major" outage (> threshold)?
        from src.ingestion.market_fundamentals import NUCLEAR_ALERT_THRESHOLD_MW
        df["is_major_nuclear_outage"] = (
            df["outage_mw"] > NUCLEAR_ALERT_THRESHOLD_MW
        ).astype(np.int8)
        new_cols.append("is_major_nuclear_outage")

        # Outage onset: step-change (positive when a new trip event starts)
        df["outage_mw_diff_1h"]  = df["outage_mw"].diff(1)
        df["outage_mw_diff_24h"] = df["outage_mw"].diff(24)
        new_cols += ["outage_mw_diff_1h", "outage_mw_diff_24h"]

    else:
        logger.warning("CLASS4A | zone=%s | outage_mw not in DataFrame.", zone)

    # ==================================================================
    # 4B — Scheduled Flow Stress Metric
    # ==================================================================
    #
    # Formula:
    #   Base:      |scheduled_net_position_mw|
    #   With wind: |scheduled_net_position_mw| / (1 + wind_speed_ms)
    #   With temp: |scheduled_net_position_mw| * seasonal_temp_proxy
    #              where seasonal_temp_proxy = -cos(2π × day_of_year / 365)
    #              (peaks in winter when heating demand amplifies flow stress)
    #
    # Interpretation:
    #   High absolute net position = the zone is either a large exporter or
    #   a large importer relative to its neighbours.  Both extremes represent
    #   stress: a large export position leaves less local margin for
    #   unexpected demand spikes; a large import position creates vulnerability
    #   to transmission outages.
    #
    #   Wind interaction:
    #     Strong winds (SE2/SE4 coastal) suppress the flow-stress impact
    #     because wind generation can absorb or release imbalances quickly.
    #     We therefore divide by (1 + wind_speed) to shrink the stress signal
    #     when wind is available as a flexible balancing resource.
    #
    #   Seasonal temperature interaction:
    #     Winter cold snaps drive demand surges that amplify the operational
    #     stress of any given net export position.  The cosine seasonal
    #     proxy (same shape as CLASS 2's seasonal_temp_proxy) scales the
    #     stress metric up in winter and down in summer.
    #
    # Gate-closure causality:
    #   scheduled_net_position_mw is only non-null AFTER 12:00 UTC on D-1
    #   (enforced by align_net_positions_to_hourly in the Silver layer).
    #   All features derived from it inherit this causal constraint.

    if "scheduled_net_position_mw" in df.columns:
        abs_pos = df["scheduled_net_position_mw"].abs()

        # Raw absolute position and its rolling stats
        df["scheduled_flow_stress_abs"] = abs_pos
        new_cols.append("scheduled_flow_stress_abs")

        for w in _NET_POS_ROLL_WINDOWS:
            min_p = max(1, int(w * config.min_periods_fraction))
            col_mean = f"net_position_roll_mean_{w}h"
            col_std  = f"net_position_roll_std_{w}h"
            df[col_mean] = _rolling_safe(df["scheduled_net_position_mw"], w, min_p, "mean")
            df[col_std]  = _rolling_safe(df["scheduled_net_position_mw"], w, min_p, "std")
            new_cols += [col_mean, col_std]

        # Wind-speed interaction: stress relief from local wind generation
        if _WIND_COL in df.columns:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                # (1 + wind) ensures denominator ≥ 1; NaN wind → NaN metric
                wind_denom = (1.0 + df[_WIND_COL].clip(lower=0)).replace(0, np.nan)
                df["scheduled_flow_stress_metric"] = abs_pos / wind_denom
            new_cols.append("scheduled_flow_stress_metric")
            logger.debug(
                "CLASS4B | zone=%s | flow_stress×wind engineered", zone
            )
        else:
            # Fallback: use raw absolute position when wind is unavailable
            df["scheduled_flow_stress_metric"] = abs_pos.copy()
            new_cols.append("scheduled_flow_stress_metric")
            logger.debug(
                "CLASS4B | zone=%s | wind_speed_ms absent — using raw abs position.",
                zone,
            )

        # Seasonal temperature amplification
        if "day_of_year" in df.columns:
            # Seasonal proxy: -cos peaks on Jan 1 (winter maximum)
            seasonal_amp = -np.cos(
                2 * np.pi * (df["day_of_year"] - 1) / 365
            ).rename("_seasonal_amp")
            df["scheduled_flow_stress_x_seasonal"] = (
                df["scheduled_flow_stress_metric"] * (1.0 + 0.5 * seasonal_amp)
            )
            new_cols.append("scheduled_flow_stress_x_seasonal")
        elif _TEMP_COL in df.columns:
            # Alternative: use actual temperature as the seasonal amplifier
            # Cold temperatures → high heating demand → higher stress
            # Normalise temperature so −20°C → +1 boost, +20°C → −1 dampen
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                temp_norm = (-df[_TEMP_COL] / 20.0).clip(-2, 2)
                df["scheduled_flow_stress_x_seasonal"] = (
                    df["scheduled_flow_stress_metric"] * (1.0 + 0.4 * temp_norm)
                )
            new_cols.append("scheduled_flow_stress_x_seasonal")

        # Scheduled position step-change (identifies intra-day re-dispatch)
        df["net_position_diff_1h"]  = df["scheduled_net_position_mw"].diff(1)
        df["net_position_diff_24h"] = df["scheduled_net_position_mw"].diff(24)
        new_cols += ["net_position_diff_1h", "net_position_diff_24h"]

    else:
        logger.warning(
            "CLASS4B | zone=%s | scheduled_net_position_mw not in DataFrame.", zone
        )

    # ==================================================================
    # 4C — Hydro Buffer Depletion Velocity (SE1/SE2 primary; all zones secondary)
    # ==================================================================
    #
    # Formula:
    #   reservoir_fill_ratio_diff_1w  = fill[t] − fill[t − 168h]
    #   reservoir_fill_ratio_diff_4w  = fill[t] − fill[t − 672h]
    #   hydro_depletion_velocity      = diff_1w (primary rate of change)
    #
    # Interaction with Class 1 autocorrelation lags:
    #   hydro_depletion_x_imbalance_lag1h  =
    #       reservoir_fill_ratio_diff_1w × imbalance_mwh_lag_1h
    #
    #   Interpretation: A fast-draining reservoir (negative 1w diff) paired
    #   with a recent large negative imbalance (system short) signals a
    #   compounding flexibility crunch: the hydro buffer is depleting AND
    #   the grid is already in deficit.  This interaction is the most
    #   predictive signal for SE1/SE2 flexibility crunch events.
    #
    #   For SE3/SE4: we retain the raw reservoir features as macro context
    #   (spill-down signal from the north), but suppress the interaction
    #   terms which have no physical meaning for thermal-dominant zones.
    #
    # Causality:
    #   reservoir_fill_ratio is stepped from weekly published data with a
    #   one-week publication lag (enforced in align_hydro_reservoir_to_hourly).
    #   diff() is backward-looking by construction.

    if "reservoir_fill_ratio" in df.columns:
        fill = df["reservoir_fill_ratio"]

        # Multi-horizon depletion velocity
        df["reservoir_fill_diff_1w"]  = fill.diff(_HYDRO_LOOKBACK_HOURS_1W)
        df["reservoir_fill_diff_4w"]  = fill.diff(_HYDRO_LOOKBACK_HOURS_4W)
        df["reservoir_fill_diff_12w"] = fill.diff(_HYDRO_LOOKBACK_HOURS_12W)
        new_cols += [
            "reservoir_fill_diff_1w",
            "reservoir_fill_diff_4w",
            "reservoir_fill_diff_12w",
        ]

        # Primary depletion velocity (1-week horizon is the most actionable)
        df["hydro_depletion_velocity"] = df["reservoir_fill_diff_1w"]
        new_cols.append("hydro_depletion_velocity")

        # Rolling mean fill (contextual level)
        for w_weeks, w_hours in [(4, 672), (12, 2_016)]:
            min_p = max(1, int(w_hours * config.min_periods_fraction))
            col   = f"reservoir_fill_roll_mean_{w_weeks}w"
            df[col] = _rolling_safe(fill, w_hours, min_p, "mean")
            new_cols.append(col)

        # ── Zone-specific interaction: SE1/SE2 flexibility crunch signal ──
        if zone in _HYDRO_RELEVANT_ZONES:
            if "imbalance_mwh_lag_1h" in df.columns:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    df["hydro_depletion_x_imbalance_lag1h"] = (
                        df["hydro_depletion_velocity"] * df["imbalance_mwh_lag_1h"]
                    )
                new_cols.append("hydro_depletion_x_imbalance_lag1h")

            if "imbalance_mwh_lag_24h" in df.columns:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    df["hydro_depletion_x_imbalance_lag24h"] = (
                        df["hydro_depletion_velocity"] * df["imbalance_mwh_lag_24h"]
                    )
                new_cols.append("hydro_depletion_x_imbalance_lag24h")

            # Crunch flag: reservoir below 25% AND actively depleting
            df["is_hydro_crunch"] = (
                (fill < 0.25) & (df["hydro_depletion_velocity"] < -0.005)
            ).astype(np.int8)
            new_cols.append("is_hydro_crunch")

            logger.debug(
                "CLASS4C | zone=%s (hydro-relevant) | crunch_hours=%d",
                zone, int(df["is_hydro_crunch"].sum()),
            )
        else:
            logger.debug(
                "CLASS4C | zone=%s | hydro interaction terms suppressed "
                "(thermal-dominant zone).", zone,
            )

    else:
        logger.warning(
            "CLASS4C | zone=%s | reservoir_fill_ratio not in DataFrame.", zone
        )

    # ==================================================================
    # Sanitise all CLASS 4 columns via the existing framework
    # ==================================================================
    df = _sanitise_numeric_columns(
        df, new_cols,
        clip_lower=config.clip_lower,
        clip_upper=config.clip_upper,
        nan_sentinel=config.nan_sentinel,
        context="CLASS4 market_fundamentals",
    )

    logger.info(
        "CLASS4 market fundamentals | zone=%s | %d features engineered",
        zone, len(new_cols),
    )
    return df, new_cols



# ===========================================================================
# GridBalancingFeatureEngineer.build_feature_matrix() DIFF
# ===========================================================================
#
# Apply these changes inside build_feature_matrix() in GridBalancingFeatureEngineer.
#
# ── After the CLASS 3 block, add CLASS 4 ───────────────────────────────────
#
#      # CLASS 3 — Riksbank rate features
#      df_zone, class3_cols = build_rate_features(df_zone, self.config, zone)
#      all_feature_cols.extend(class3_cols)
#
# +    # CLASS 4 — Market Fundamental features (REMIT, net positions, hydro)
# +    df_zone, class4_cols = build_market_fundamental_features(
# +        df_zone, self.config, zone
# +    )
# +    all_feature_cols.extend(class4_cols)
#
#      # Feature readiness flag
#      df_zone = mark_feature_ready(df_zone, self.config, zone)
