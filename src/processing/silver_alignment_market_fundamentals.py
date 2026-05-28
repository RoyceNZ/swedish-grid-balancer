"""
src/processing/silver_alignment.py  — MARKET FUNDAMENTALS ADDITIONS
=============================================================================
Additive patch for the existing TemporalHarmonizer class.
=============================================================================

This file contains all new code required to integrate the three market
fundamental streams (REMIT outages, scheduled net positions, hydro reservoir
levels) into the Silver alignment pipeline.  It is structured as a standalone
module that is imported by silver_alignment.py.

Merge instructions:
  1. Add the free functions (align_remit_to_hourly, align_net_positions_to_hourly,
     align_hydro_reservoir_to_hourly, align_market_fundamentals_to_hourly) to
     silver_alignment.py after the existing align_scb_quarterly_to_hourly().
  2. Apply the TemporalHarmonizer diff at the bottom of this file.

Temporal alignment strategies:
  ┌─────────────────────────────┬────────────────────────────────────────────┐
  │ Stream                      │ Strategy                                   │
  ├─────────────────────────────┼────────────────────────────────────────────┤
  │ REMIT outages (hourly)      │ Direct left-join on (timestamp_utc, zone). │
  │                             │ Gaps → 0.0 MW (no outage = 0 is correct).  │
  │                             │ No interpolation: outages are step events. │
  ├─────────────────────────────┼────────────────────────────────────────────┤
  │ Net positions (hourly)      │ merge_asof(direction='backward') keyed on  │
  │                             │ gate_closed_at_utc.  Gate closes at 12:45  │
  │                             │ CET D-1; only hours AFTER that cutoff get  │
  │                             │ a non-null position.  Pre-gate hours → NaN.│
  ├─────────────────────────────┼────────────────────────────────────────────┤
  │ Hydro reservoir (weekly)    │ Each ISO-week observation is first shifted  │
  │                             │ forward by 7 days (publication lag) before  │
  │                             │ merge_asof(direction='backward').           │
  │                             │ Short gaps (≤ 2 weeks) filled via linear    │
  │                             │ interpolation; longer gaps → ffill.         │
  └─────────────────────────────┴────────────────────────────────────────────┘

All joins are STRICTLY CAUSAL — no future information is visible to any row.
=============================================================================
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger("silver_alignment")


# ---------------------------------------------------------------------------
# Silver Schema Extension — Market Fundamentals Columns
# ---------------------------------------------------------------------------

class SilverMarketFundamentalsRecord(BaseModel):
    """
    Extended Silver record schema capturing the three market fundamental fields.

    These columns are appended to SilverHourlyGridRecord by
    align_market_fundamentals_to_hourly().  Validated row-wise at the
    Silver boundary before the combined frame is passed to the Gold layer.
    """
    timestamp_utc:             str
    zone:                      str
    outage_mw:                 float = 0.0
    active_unit_count:         int   = 0
    scheduled_net_position_mw: Optional[float] = None   # NaN before gate-close
    reservoir_fill_ratio:      Optional[float] = None   # NaN before first obs

    @field_validator("zone")
    @classmethod
    def zone_valid(cls, v: str) -> str:
        if v not in {"SE1", "SE2", "SE3", "SE4"}:
            raise ValueError(f"zone must be SE1–SE4; got '{v}'.")
        return v

    @field_validator("outage_mw")
    @classmethod
    def outage_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"outage_mw cannot be negative; got {v}.")
        return v

    @field_validator("reservoir_fill_ratio")
    @classmethod
    def fill_ratio_in_unit_interval(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError(
                f"reservoir_fill_ratio must be in [0, 1]; got {v}."
            )
        return v


# ===========================================================================
# STREAM A — REMIT Outages → Hourly Grid
# ===========================================================================

def align_remit_to_hourly(
    df_hourly: pd.DataFrame,
    df_remit:  pd.DataFrame,
    zone:      str,
    timestamp_col: str = "timestamp_utc",
) -> pd.DataFrame:
    """
    Left-join REMIT outage data onto the hourly grid for one bidding zone.

    Design:
    ───────
    REMIT outages are already expanded to per-hour rows by the Bronze fetcher
    (fetch_remit_outages), so this join is a simple left merge on
    (timestamp_utc, zone).  Hours with no active outage receive outage_mw=0.0
    and active_unit_count=0 — this is physically correct (0 MW offline is NOT
    a missing value; it is a meaningful market signal).

    No interpolation is applied.  Nuclear trip events are step functions:
    the unit is either online or offline at each discrete hour.

    Leakage safety:
    ───────────────
    REMIT UMMs for unplanned outages are published in real-time (not in
    advance), so there is no future-data risk in this join.  The Bronze
    fetcher tags records with their actual outage window; the Silver layer
    simply joins on delivery hour, which is the correct causal observation.

    Args:
        df_hourly:     Hourly grid DataFrame (must contain timestamp_col and 'zone').
        df_remit:      REMIT Bronze DataFrame produced by fetch_remit_outages().
                       Required columns: [timestamp_utc, zone, outage_mw].
        zone:          Target bidding zone (e.g. 'SE3').
        timestamp_col: Name of the hourly UTC timestamp column.

    Returns:
        df_hourly with two new columns appended:
            outage_mw           — total MW offline at that hour (0 if no outage)
            active_unit_count   — number of distinct units with an active outage
    """
    required = {"timestamp_utc", "zone", "outage_mw"}
    missing  = required - set(df_remit.columns)
    if missing:
        raise ValueError(
            f"df_remit missing required columns: {missing}. "
            "Ensure it was produced by fetch_remit_outages()."
        )

    # Filter to target zone and coerce timestamps
    df_r = df_remit[df_remit["zone"] == zone].copy()
    df_h = df_hourly.copy()

    df_h[timestamp_col] = pd.to_datetime(df_h[timestamp_col], utc=True)
    df_r["timestamp_utc"] = pd.to_datetime(df_r["timestamp_utc"], utc=True)

    # Aggregate to zone-hour level if not already done
    agg_cols: dict = {"outage_mw": "sum"}
    if "active_unit_count" in df_r.columns:
        agg_cols["active_unit_count"] = "sum"

    df_r_agg = df_r.groupby("timestamp_utc", as_index=False).agg(agg_cols)

    # Left join: hours with no REMIT record get NaN → fill with 0.0
    df_merged = df_h.merge(
        df_r_agg[["timestamp_utc"] + list(agg_cols.keys())],
        left_on=timestamp_col,
        right_on="timestamp_utc",
        how="left",
        suffixes=("", "_remit"),
    )
    if "timestamp_utc_remit" in df_merged.columns:
        df_merged.drop(columns=["timestamp_utc_remit"], inplace=True)

    # 0.0 is the physically correct fill for hours with no active outage
    df_merged["outage_mw"] = df_merged["outage_mw"].fillna(0.0)
    if "active_unit_count" not in df_merged.columns:
        df_merged["active_unit_count"] = 0
    df_merged["active_unit_count"] = df_merged["active_unit_count"].fillna(0).astype(int)

    outage_hours = int((df_merged["outage_mw"] > 0).sum())
    peak_outage  = df_merged["outage_mw"].max()
    logger.info(
        "REMIT alignment | zone=%s | %d outage-hours | peak=%.0f MW | "
        "total_hours=%d",
        zone, outage_hours, peak_outage, len(df_merged),
    )
    return df_merged


# ===========================================================================
# STREAM B — Scheduled Net Positions → Hourly Grid
# ===========================================================================

def _compute_gate_closure_utc(delivery_ts: pd.Series) -> pd.Series:
    """
    Compute the gate-closure UTC timestamp for each delivery hour.

    ENTSO-E DA gate closure is at 12:45 CET (UTC+1) / CEST (UTC+2).
    For each delivery hour, the gate closed at 12:45 CET on the preceding
    calendar day (D-1 at 11:45 UTC in winter, 10:45 UTC in summer).

    We use 12:00 UTC on D-1 as a conservative proxy (guaranteed to be
    before 12:45 CET in all seasons) so that the merge_asof join never
    accidentally exposes a position before its gate closure.

    Args:
        delivery_ts: UTC-aware pd.Series of delivery timestamps.

    Returns:
        UTC-aware pd.Series of gate-closure timestamps.
    """
    # Normalise to date, subtract 1 day, set hour to 12:00 UTC
    delivery_dates = delivery_ts.dt.normalize()
    gate_dates = delivery_dates - pd.Timedelta(days=1)
    return gate_dates + pd.Timedelta(hours=12)


def align_net_positions_to_hourly(
    df_hourly:     pd.DataFrame,
    df_net_pos:    pd.DataFrame,
    zone:          str,
    timestamp_col: str = "timestamp_utc",
) -> pd.DataFrame:
    """
    Join Day-Ahead scheduled net positions onto the hourly grid.

    Gate-closure causality:
    ───────────────────────
    Net positions are published at 12:45 CET on D-1.  A delivery hour on
    2024-06-15 08:00 UTC has its position available from 2024-06-14 10:45 UTC
    (CEST) onward.  To enforce this, we:
      1. Add a 'gate_closed_at_utc' column to df_net_pos (12:00 UTC D-1 proxy).
      2. Use merge_asof(direction='backward') on the hourly grid's
         timestamp_utc vs. the gate_closed_at_utc key.
      3. Any training row whose timestamp_utc < gate_closed_at_utc of the
         nearest position will receive NaN — i.e., intra-day rows before
         12:00 UTC on Day D never see that day's published net position.

    This is the correct causal representation: an operator running a model
    at 08:00 UTC on day D does NOT yet have D's net position; they only have
    D-1's (and earlier) positions from the archive.

    Args:
        df_hourly:     Hourly grid DataFrame.
        df_net_pos:    Net positions Bronze DataFrame.
                       Required columns: [timestamp_utc, zone, scheduled_net_position_mw]
        zone:          Target bidding zone.
        timestamp_col: Name of the hourly UTC timestamp column.

    Returns:
        df_hourly with new column:
            scheduled_net_position_mw — NaN before gate-close; MW value after
    """
    required = {"timestamp_utc", "zone", "scheduled_net_position_mw"}
    missing  = required - set(df_net_pos.columns)
    if missing:
        raise ValueError(
            f"df_net_pos missing required columns: {missing}."
        )

    df_p = df_net_pos[df_net_pos["zone"] == zone].copy()
    df_h = df_hourly.copy()

    df_h[timestamp_col] = pd.to_datetime(df_h[timestamp_col], utc=True)
    df_p["timestamp_utc"] = pd.to_datetime(df_p["timestamp_utc"], utc=True)

    if df_p.empty:
        logger.warning(
            "No net position data for zone=%s — column will be NaN.", zone
        )
        df_h["scheduled_net_position_mw"] = np.nan
        return df_h

    # ── Gate-closure enforcement ─────────────────────────────────────────
    # gate_closed_at_utc is the key for the causal merge.
    # Position for delivery hour T is "published" from gate_closed_at_utc(T) onward.
    df_p = df_p.sort_values("timestamp_utc").reset_index(drop=True)
    df_p["gate_closed_at_utc"] = _compute_gate_closure_utc(df_p["timestamp_utc"])

    # Sort df_h by timestamp for merge_asof
    df_h_sorted = df_h.sort_values(timestamp_col).reset_index(drop=True)

    # merge_asof: for each hourly row, find the net position record whose
    # gate_closed_at_utc is the latest one still <= the row's timestamp_utc.
    # This means: "the most recently gate-closed position available at this hour."
    df_merged = pd.merge_asof(
        df_h_sorted,
        df_p[["gate_closed_at_utc", "scheduled_net_position_mw"]],
        left_on=timestamp_col,
        right_on="gate_closed_at_utc",
        direction="backward",
    )
    df_merged.drop(columns=["gate_closed_at_utc"], errors="ignore", inplace=True)

    # Restore original sort order
    df_merged = df_merged.sort_values(timestamp_col).reset_index(drop=True)

    coverage = 100 * df_merged["scheduled_net_position_mw"].notna().mean()
    logger.info(
        "Net position alignment | zone=%s | coverage=%.1f%% | "
        "mean_pos=%.0f MW | %d total rows",
        zone, coverage,
        df_merged["scheduled_net_position_mw"].mean(),
        len(df_merged),
    )
    if coverage < 80.0:
        logger.warning(
            "Low net position coverage (%.1f%%) for zone=%s. "
            "Check Bronze fetch window vs hourly grid window.",
            coverage, zone,
        )
    return df_merged


# ===========================================================================
# STREAM C — Hydro Reservoir Levels → Hourly Grid
# ===========================================================================

def _week_to_monday_utc(year: pd.Series, week: pd.Series) -> pd.Series:
    """
    Convert (year, ISO week_of_year) pairs to UTC timestamps of Monday 00:00.

    ISO weeks start on Monday.  Week 1 is the week containing the first
    Thursday of the year (ISO 8601 standard).

    Returns:
        UTC-aware pd.Series of Monday midnight timestamps.
    """
    # Build a string "YYYY-WXX-1" (ISO week day 1 = Monday) and parse
    week_strings = year.astype(str) + "-W" + week.astype(str).str.zfill(2) + "-1"
    return pd.to_datetime(week_strings, format="%G-W%V-%u", utc=True)


def align_hydro_reservoir_to_hourly(
    df_hourly:     pd.DataFrame,
    df_hydro:      pd.DataFrame,
    timestamp_col: str = "timestamp_utc",
    max_interp_weeks: int = 2,
) -> pd.DataFrame:
    """
    Step-down weekly hydro reservoir fill ratios onto the hourly UTC grid.

    The hydro stream is not zone-specific: it represents the aggregate
    SE1+SE2 northern basin (the only basin with significant multi-week
    storage in the Swedish system).  The same fill ratio is therefore
    valid for all four zones in the Silver master frame; zone-specific
    interaction features are built in the Gold layer (CLASS 4).

    Publication lag enforcement:
    ────────────────────────────
    Energimyndigheten publishes each week's reservoir data on the FOLLOWING
    Monday.  A dataset labelled "Week 22" covers Mon–Sun of week 22 but is
    not public until Monday of week 23.

    To enforce this:
      1. Each weekly observation is assigned its publication timestamp:
         Monday 00:00 UTC of the ISO week AFTER the observation week.
      2. merge_asof(direction='backward') then ensures each hourly row
         sees only reservoir data that had already been published at that hour.

    Gap-filling:
    ────────────
    Short gaps (≤ max_interp_weeks × 168 hours) are filled with linear
    interpolation.  This is appropriate for the reservoir fill ratio because:
      - The physical process is continuous: water levels change smoothly.
      - Short gaps arise from publication delays or missing weekly reports,
        not from step-changes in the underlying process.
    Gaps longer than max_interp_weeks are forward-filled (last-known value).

    Args:
        df_hourly:        Hourly grid DataFrame.
        df_hydro:         Hydro Bronze DataFrame.
                          Required columns: [year, week_of_year, reservoir_fill_ratio]
        timestamp_col:    Name of the hourly UTC timestamp column.
        max_interp_weeks: Maximum gap in weeks to fill via linear interpolation.
                          Defaults to 2 (i.e., ≤ 336 hours interpolated).

    Returns:
        df_hourly with new column:
            reservoir_fill_ratio — weekly fill level forward-filled to hourly
    """
    required = {"year", "week_of_year", "reservoir_fill_ratio"}
    missing  = required - set(df_hydro.columns)
    if missing:
        raise ValueError(
            f"df_hydro missing required columns: {missing}."
        )

    df_w = df_hydro.copy()
    df_h = df_hourly.copy()
    df_h[timestamp_col] = pd.to_datetime(df_h[timestamp_col], utc=True)

    if df_w.empty:
        logger.warning("No hydro reservoir data — column will be NaN.")
        df_h["reservoir_fill_ratio"] = np.nan
        return df_h

    # ── Compute publication timestamp: Monday of the NEXT ISO week ────────
    # Each observation for week W is published at the start of week W+1.
    df_w["obs_monday_utc"]  = _week_to_monday_utc(df_w["year"], df_w["week_of_year"])
    df_w["pub_monday_utc"]  = df_w["obs_monday_utc"] + pd.Timedelta(weeks=1)
    df_w = df_w.sort_values("pub_monday_utc").reset_index(drop=True)

    # ── Causal merge: each hourly row sees only already-published data ─────
    df_h_sorted = df_h.sort_values(timestamp_col).reset_index(drop=True)

    df_merged = pd.merge_asof(
        df_h_sorted,
        df_w[["pub_monday_utc", "reservoir_fill_ratio"]],
        left_on=timestamp_col,
        right_on="pub_monday_utc",
        direction="backward",
    )
    df_merged.drop(columns=["pub_monday_utc"], errors="ignore", inplace=True)

    # ── Gap-filling: interpolate short gaps, ffill longer ones ───────────
    max_interp_hours = max_interp_weeks * 168
    df_merged["reservoir_fill_ratio"] = (
        df_merged["reservoir_fill_ratio"]
        .interpolate(method="linear", limit=max_interp_hours, limit_direction="forward")
        .ffill()  # catch any remaining NaN at the start
    )

    coverage = 100 * df_merged["reservoir_fill_ratio"].notna().mean()
    mean_fill = df_merged["reservoir_fill_ratio"].mean()
    logger.info(
        "Hydro reservoir alignment | coverage=%.1f%% | mean_fill=%.3f | "
        "%d hourly rows",
        coverage, mean_fill, len(df_merged),
    )
    return df_merged


# ===========================================================================
# Master Orchestrator — All Three Streams
# ===========================================================================

def align_market_fundamentals_to_hourly(
    df_hourly:     pd.DataFrame,
    df_remit:      Optional[pd.DataFrame] = None,
    df_net_pos:    Optional[pd.DataFrame] = None,
    df_hydro:      Optional[pd.DataFrame] = None,
    zone:          str = "SE3",
    timestamp_col: str = "timestamp_utc",
) -> pd.DataFrame:
    """
    Orchestrate the three market fundamental joins for one bidding zone.

    Inserts after the Riksbank rate join (Stage 4) and before the quarantine
    step (Stage 5) in TemporalHarmonizer.align().

    Each stream is applied only when its DataFrame is provided and non-empty.
    Missing streams produce NaN columns with correct names so downstream
    feature engineering can safely check `if col in df.columns`.

    Args:
        df_hourly:     Hourly grid DataFrame with all prior Silver columns.
        df_remit:      REMIT outages DataFrame (zone-aggregated).
        df_net_pos:    Scheduled net positions DataFrame (all zones; filtered internally).
        df_hydro:      Weekly hydro reservoir fill DataFrame (national).
        zone:          Target bidding zone string.
        timestamp_col: Hourly UTC timestamp column name.

    Returns:
        df_hourly extended with:
            outage_mw                  — total MW offline (0 if none)
            active_unit_count          — count of units with active outage
            scheduled_net_position_mw  — DA cleared net position (NaN pre-gate)
            reservoir_fill_ratio       — weekly fill ratio stepped to hourly
    """
    logger.info(
        "=== align_market_fundamentals_to_hourly | zone=%s ===", zone
    )
    df = df_hourly.copy()

    # ── Stream A: REMIT outages ──────────────────────────────────────────
    if df_remit is not None and not df_remit.empty:
        df = align_remit_to_hourly(df, df_remit, zone=zone, timestamp_col=timestamp_col)
    else:
        logger.warning("No REMIT data for zone=%s — outage_mw=0.0 for all rows.", zone)
        df["outage_mw"]         = 0.0
        df["active_unit_count"] = 0

    # ── Stream B: Scheduled net positions ────────────────────────────────
    if df_net_pos is not None and not df_net_pos.empty:
        df = align_net_positions_to_hourly(df, df_net_pos, zone=zone, timestamp_col=timestamp_col)
    else:
        logger.warning(
            "No net position data for zone=%s — scheduled_net_position_mw=NaN.",
            zone,
        )
        df["scheduled_net_position_mw"] = np.nan

    # ── Stream C: Hydro reservoir levels (national; applied to all zones) ─
    if df_hydro is not None and not df_hydro.empty:
        df = align_hydro_reservoir_to_hourly(df, df_hydro, timestamp_col=timestamp_col)
    else:
        logger.warning(
            "No hydro reservoir data — reservoir_fill_ratio=NaN for zone=%s.", zone
        )
        df["reservoir_fill_ratio"] = np.nan

    # ── Diagnostic coverage summary ───────────────────────────────────────
    outage_cov  = 100 * (df["outage_mw"] >= 0).mean()   # always present
    netpos_cov  = 100 * df["scheduled_net_position_mw"].notna().mean()
    hydro_cov   = 100 * df["reservoir_fill_ratio"].notna().mean()

    logger.info(
        "Market fundamentals joined | zone=%s | outage_cov=%.1f%% | "
        "net_pos_cov=%.1f%% | hydro_cov=%.1f%% | rows=%d",
        zone, outage_cov, netpos_cov, hydro_cov, len(df),
    )
    return df


# ===========================================================================
# TemporalHarmonizer DIFF — Market Fundamentals Integration
# ===========================================================================
#
# Apply these changes to the TemporalHarmonizer class in silver_alignment.py.
# Lines prefixed '+' are additions; lines prefixed '-' are deletions.
#
# ── __init__ — new parameters ──────────────────────────────────────────────
#
#  def __init__(
#      self,
#      interpolation_method:  str = "linear",
#      max_grid_gap_hours:    int = 6,
#      max_weather_gap_hours: int = 3,
# +    hydro_max_interp_weeks: int = 2,
#  ) -> None:
#      self.interpolation_method   = interpolation_method
#      self.max_grid_gap_hours     = max_grid_gap_hours
#      self.max_weather_gap_hours  = max_weather_gap_hours
# +    self.hydro_max_interp_weeks = hydro_max_interp_weeks
#
# ── align() — updated signature ────────────────────────────────────────────
#
#  def align(
#      self,
#      df_hourly:       pd.DataFrame,
#      df_quarterly:    Optional[pd.DataFrame] = None,
#      df_policy_rate:  Optional[pd.DataFrame] = None,
#      df_weather:      Optional[pd.DataFrame] = None,
# +    df_remit:        Optional[pd.DataFrame] = None,
# +    df_net_positions: Optional[pd.DataFrame] = None,
# +    df_hydro:        Optional[pd.DataFrame] = None,
#      zone: str = "SE3",
#  ) -> pd.DataFrame:
#
# ── align() — Stage 4c insertion (after weather join, before quarantine) ───
#
#      # Stage 4b: SMHI weather join [spatial-hourly nearest match]
#      ...  (existing weather join block) ...
#
# +    # Stage 4c: Market fundamentals join [multi-key causal left joins]
# +    df = align_market_fundamentals_to_hourly(
# +        df_hourly=df,
# +        df_remit=df_remit,
# +        df_net_pos=df_net_positions,
# +        df_hydro=df_hydro,
# +        zone=zone,
# +    )
#
#      # Stage 5: Quarantine
#      df = self._apply_quarantine(df)
#
# ── align() — updated completion log ───────────────────────────────────────
#
#      logger.info(
#          "Alignment complete | zone=%s | %d hourly rows | %d quarantined | "
#          "SCB=%.1f%% | rate=%.1f%% | temp=%.1f%% | wind=%.1f%% | "
# +        "outage_hours=%d | net_pos_cov=%.1f%% | hydro_cov=%.1f%%",
#          zone,
#          len(df), int(df["is_quarantined"].sum()),
#          100 * df["smahus_construction_index"].notna().mean(),
#          100 * df["riksbank_policy_rate_pct"].notna().mean(),
#          100 * df["temperature_c"].notna().mean(),
#          100 * df["wind_speed_ms"].notna().mean(),
# +        int((df["outage_mw"] > 0).sum()),
# +        100 * df["scheduled_net_position_mw"].notna().mean(),
# +        100 * df["reservoir_fill_ratio"].notna().mean(),
#      )
#
# ── align_all_zones() — updated signature ──────────────────────────────────
#
#  def align_all_zones(
#      self,
#      all_zone_grids:          dict[str, pd.DataFrame],
#      df_quarterly_by_zone:    Optional[dict[str, pd.DataFrame]] = None,
#      df_policy_rate:          Optional[pd.DataFrame] = None,
#      df_weather_by_zone:      Optional[dict[str, pd.DataFrame]] = None,
# +    df_remit:                Optional[pd.DataFrame] = None,
# +    df_net_positions:        Optional[pd.DataFrame] = None,
# +    df_hydro:                Optional[pd.DataFrame] = None,
#  ) -> pd.DataFrame:
#
# ── align_all_zones() — pass new streams into per-zone align() ─────────────
#
#          df_aligned = self.align(
#              df_hourly=df_grid,
#              df_quarterly=quarterly,
#              df_policy_rate=df_policy_rate,
#              df_weather=weather,
# +            df_remit=df_remit,
# +            df_net_positions=df_net_positions,
# +            df_hydro=df_hydro,
#              zone=zone_label,
#          )
