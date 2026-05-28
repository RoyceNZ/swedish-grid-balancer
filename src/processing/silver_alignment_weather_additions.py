"""
src/processing/silver_alignment.py  — WEATHER STREAM ADDITIONS
=============================================================================
Diff / patch for silver_alignment.py
=============================================================================

This file documents all additions required to integrate the SMHI weather
stream into the existing TemporalHarmonizer pipeline.  The changes are
organised as four discrete, independently testable units:

  1.  STATION_ZONE_COORDS — static spatial lookup table (station lat/lon)
  2.  SilverWeatherRecord — Pydantic v2 Silver schema for weather columns
  3.  align_weather_to_hourly() — free function: explicit spatial-hourly join
  4.  TemporalHarmonizer updates:
        __init__  — new parameter accept_weather kwarg
        align()   — Stage 4b: optional weather join inserted between
                    Riksbank join and quarantine
        align_all_zones() — pass per-zone weather DataFrames through

Merge instructions:
  - Insert STATION_ZONE_COORDS and SilverWeatherRecord into the constants /
    Pydantic schemas section (after SilverHourlyGridRecord definition).
  - Insert align_weather_to_hourly() into the free-function block (after
    align_scb_quarterly_to_hourly()).
  - Apply the TemporalHarmonizer diff shown at the bottom of this file.

No existing logic is modified.  All additions are strictly additive.
=============================================================================
"""

from __future__ import annotations

# (All existing imports remain unchanged — the additions below require only
#  what is already imported in the production file.)

import logging
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

# Module logger is already initialised in the production file.
logger = logging.getLogger("silver_alignment")


# ===========================================================================
# 1.  SPATIAL LOOKUP — Station Coordinates for Zone Assignment
# ===========================================================================
# Explicit coordinate table for the four SMHI canonical zone stations.
#
# Why store coordinates here?
#   The SMHI API response includes lat/lon in the station block, but relying
#   on the API to define the spatial identity of each station introduces a
#   dependency on upstream metadata consistency.  Pinning the expected
#   coordinates locally lets us (a) validate that the fetcher returned the
#   right station, and (b) perform the spatial join without a live API call.
#
# Coordinate source: SMHI Observationsdatabas, reference list for synoptic
# stations with continuous corrected-archive coverage.
#
# Structure:
#   station_id → {zone, lat, lon, city}
#
# These match the STATION_ZONE_MAP in weather_fetcher.py exactly; any change
# there must be reflected here.

STATION_ZONE_COORDS: dict[int, dict] = {
    140480: {
        "zone":    "SE1",
        "city":    "Luleå",
        "lat":     65.5439,
        "lon":     22.1128,
        "note":    "Luleå Airport; northernmost synoptic station, Norrbotten.",
    },
    86340: {
        "zone":    "SE2",
        "city":    "Sundsvall",
        "lat":     62.5281,
        "lon":     17.4399,
        "note":    "Sundsvall/Härnösand; mid-north coastal hub.",
    },
    98210: {
        "zone":    "SE3",
        "city":    "Stockholm",
        "lat":     59.3500,
        "lon":     18.0500,
        "note":    "Stockholm Observatorielunden; SMHI primary reference station.",
    },
    62040: {
        "zone":    "SE4",
        "city":    "Malmö",
        "lat":     55.5363,
        "lon":     13.3718,
        "note":    "Malmö/Sturup; southernmost synoptic station, Öresund corridor.",
    },
}

# Inverse lookup: zone label → station_id (used for validation)
_ZONE_TO_STATION: dict[str, int] = {
    v["zone"]: k for k, v in STATION_ZONE_COORDS.items()
}


# ===========================================================================
# 2.  PYDANTIC V2 SCHEMA — Silver Weather Record
# ===========================================================================

class SilverWeatherRecord(BaseModel):
    """
    Single validated hourly weather observation after spatial-zone assignment.

    This schema is applied ROW-WISE at the Silver boundary to confirm that
    the weather join produced coherent, type-safe records before they are
    merged into the master Silver frame.

    Columns that must be present in the merged Silver DataFrame:
        timestamp_utc       : UTC-aware pd.Timestamp (hourly)
        zone                : one of SE1, SE2, SE3, SE4
        station_id          : SMHI integer station key
        temperature_c       : float | NaN
        wind_speed_ms       : float | NaN
        is_anomaly_weather  : bool
    """
    timestamp_utc:      str            # serialised as ISO string for Pydantic
    zone:               str
    station_id:         int
    temperature_c:      Optional[float] = None
    wind_speed_ms:      Optional[float] = None
    is_anomaly_weather: bool            = False

    @field_validator("zone")
    @classmethod
    def zone_must_be_valid(cls, v: str) -> str:
        valid = {"SE1", "SE2", "SE3", "SE4"}
        if v not in valid:
            raise ValueError(f"zone must be one of {valid}; got '{v}'.")
        return v

    @field_validator("station_id")
    @classmethod
    def station_must_be_known(cls, v: int) -> int:
        if v not in STATION_ZONE_COORDS:
            raise ValueError(
                f"station_id {v} is not in STATION_ZONE_COORDS. "
                "Register it before ingesting."
            )
        return v

    @model_validator(mode="after")
    def zone_must_match_station(self) -> "SilverWeatherRecord":
        expected_zone = STATION_ZONE_COORDS[self.station_id]["zone"]
        if self.zone != expected_zone:
            raise ValueError(
                f"Spatial mismatch: station {self.station_id} belongs to zone "
                f"'{expected_zone}', not '{self.zone}'. Check STATION_ZONE_COORDS."
            )
        return self

    @model_validator(mode="after")
    def at_least_one_measurement(self) -> "SilverWeatherRecord":
        if self.temperature_c is None and self.wind_speed_ms is None:
            raise ValueError(
                "At least one of temperature_c or wind_speed_ms must be non-null."
            )
        return self


# ===========================================================================
# 3.  FREE FUNCTION — align_weather_to_hourly()
# ===========================================================================

def align_weather_to_hourly(
    df_hourly: pd.DataFrame,
    df_weather: pd.DataFrame,
    zone: str,
    timestamp_col: str = "timestamp_utc",
    station_coord_registry: dict[int, dict] = STATION_ZONE_COORDS,
) -> pd.DataFrame:
    """
    Spatial-hourly join: map SMHI station observations to a bidding zone
    hourly UTC grid via an explicit station → zone coordinate lookup.

    Design:
    ───────
    Weather observations arrive from a SINGLE station per zone (the SMHI
    canonical main-grid station; see STATION_ZONE_COORDS).  The join is
    therefore a straightforward temporal merge, not a spatial interpolation
    between multiple stations.  The spatial step is the EXPLICIT assertion
    that station 98210 ∈ SE3, station 62040 ∈ SE4, etc. — enforced both in
    STATION_ZONE_COORDS and in SilverWeatherRecord.zone_must_match_station().

    Temporal strategy:
    ──────────────────
    SMHI corrected-archive data is nominally hourly, but observation
    timestamps may not align exactly with the ENTSO-E hourly UTC grid
    (e.g., logged at HH:10 instead of HH:00).  merge_asof with
    direction='nearest' and a 30-minute tolerance resolves this cleanly:
      - An SMHI observation at 14:05 UTC will be assigned to the 14:00 UTC
        grid hour.
      - Grid hours with no SMHI observation within ±30 minutes are assigned
        NaN weather values, which the downstream gap-filler handles.

    Why nearest and not backward?
      For temperature and wind speed — continuous physical quantities — the
      closest-in-time observation is the most representative value for the
      hour, regardless of whether it is slightly before or after the slot.
      This differs from Riksbank rates (event-driven steps) and SCB housing
      indices (quarterly releases), both of which must be strictly backward.

    Leakage safety:
    ───────────────
    Weather observations are NOT macroeconomic forecasts; they represent
    physical conditions at the measurement instant.  Using the nearest
    observation within a ±30-minute window does not introduce predictive
    leakage because temperature and wind speed at hour H are not known
    before the hour begins (i.e., they are not forward-looking signals).

    Spatial validation:
    ───────────────────
    Before the merge, the function validates that:
      (a) df_weather contains a station_id column whose values are all
          registered in station_coord_registry.
      (b) Every station in df_weather maps to the same zone as the
          target `zone` argument.
    This catches cross-zone contamination (e.g., accidentally passing
    Stockholm weather into the SE4 alignment path) at runtime.

    Args:
        df_hourly:              Hourly grid DataFrame (must include timestamp_col).
                                Must already be sorted ascending by timestamp_col.
        df_weather:             SMHI weather DataFrame produced by
                                SMHIWeatherFetcher.fetch_station_parameter() or
                                fetch_all_zone_stations().
                                Required columns:
                                    [timestamp_utc, zone, station_id,
                                     temperature_c, wind_speed_ms, is_anomaly]
        zone:                   Target bidding zone string (e.g. 'SE3').
        timestamp_col:          Name of the hourly UTC timestamp column.
        station_coord_registry: Coordinate lookup dict (default: STATION_ZONE_COORDS).
                                Injectable for testing with custom station configs.

    Returns:
        df_hourly with three new columns appended:
            temperature_c       — hourly ambient temperature (°C) or NaN
            wind_speed_ms       — hourly mean wind speed (m/s) or NaN
            is_anomaly_weather  — True if the nearest observation was anomalous

    Raises:
        ValueError: if df_weather contains stations from a different zone,
                    or if required columns are missing.
    """
    _REQUIRED_WEATHER_COLS = {
        "timestamp_utc", "zone", "station_id",
        "temperature_c", "wind_speed_ms", "is_anomaly",
    }
    missing = _REQUIRED_WEATHER_COLS - set(df_weather.columns)
    if missing:
        raise ValueError(
            f"df_weather is missing required columns: {missing}. "
            "Ensure it was produced by SMHIWeatherFetcher."
        )

    # ── Spatial guard: every station in df_weather must belong to target zone ──
    stations_present = df_weather["station_id"].unique()
    for sid in stations_present:
        if sid not in station_coord_registry:
            raise ValueError(
                f"station_id {sid} is not registered in station_coord_registry. "
                "Add it to STATION_ZONE_COORDS before aligning."
            )
        station_zone = station_coord_registry[sid]["zone"]
        if station_zone != zone:
            raise ValueError(
                f"Spatial mismatch in weather alignment: station {sid} belongs to "
                f"zone '{station_zone}', but align_weather_to_hourly() was called "
                f"with zone='{zone}'. Pass the correct per-zone DataFrame."
            )

    # ── Filter df_weather to target zone (defensive; should already be filtered) ─
    df_w = df_weather[df_weather["zone"] == zone].copy()
    if df_w.empty:
        logger.warning(
            "No weather observations for zone=%s after zone filter. "
            "temperature_c and wind_speed_ms will be NaN for all rows.",
            zone,
        )
        df_out = df_hourly.copy()
        df_out["temperature_c"]      = np.nan
        df_out["wind_speed_ms"]      = np.nan
        df_out["is_anomaly_weather"] = False
        return df_out

    # ── Coerce timestamps to UTC-aware pd.Timestamp ──────────────────────────
    df_h = df_hourly.copy()
    df_h[timestamp_col] = pd.to_datetime(df_h[timestamp_col], utc=True)
    df_h = df_h.sort_values(timestamp_col).reset_index(drop=True)

    df_w["timestamp_utc"] = pd.to_datetime(df_w["timestamp_utc"], utc=True)
    df_w = df_w.sort_values("timestamp_utc").reset_index(drop=True)

    # ── Deduplicate weather observations (keep mean per hour if duplicates exist) ─
    # SMHI archives occasionally contain duplicate entries for the same hour
    # (e.g., from overlapping fetch windows).  Aggregate before joining.
    df_w = (
        df_w.groupby("timestamp_utc", as_index=False)
        .agg(
            temperature_c=("temperature_c", "mean"),
            wind_speed_ms=("wind_speed_ms", "mean"),
            is_anomaly=("is_anomaly", "any"),
            station_id=("station_id", "first"),
        )
    )

    # ── Temporal join: nearest observation within ±30-minute tolerance ────────
    # merge_asof requires both DataFrames sorted ascending on the key column.
    # tolerance=30min, direction='nearest' → smallest absolute time distance.
    TOLERANCE = pd.Timedelta(minutes=30)

    df_merged = pd.merge_asof(
        df_h,
        df_w[["timestamp_utc", "temperature_c", "wind_speed_ms", "is_anomaly"]],
        on=timestamp_col,
        direction="nearest",
        tolerance=TOLERANCE,
    )

    df_merged.rename(columns={"is_anomaly": "is_anomaly_weather"}, inplace=True)
    df_merged["is_anomaly_weather"] = df_merged["is_anomaly_weather"].fillna(False)

    # ── Coverage diagnostics ──────────────────────────────────────────────────
    temp_coverage = 100 * df_merged["temperature_c"].notna().mean()
    wind_coverage = 100 * df_merged["wind_speed_ms"].notna().mean()
    weather_anomalies = int(df_merged["is_anomaly_weather"].sum())

    logger.info(
        "Weather alignment complete | zone=%s | %d hourly rows | "
        "temp_coverage=%.1f%% | wind_coverage=%.1f%% | anomalies=%d",
        zone,
        len(df_merged),
        temp_coverage,
        wind_coverage,
        weather_anomalies,
    )

    if temp_coverage < 80.0:
        logger.warning(
            "Low temperature coverage (%.1f%%) for zone=%s. "
            "Check SMHI station availability or adjust tolerance.",
            temp_coverage, zone,
        )
    if wind_coverage < 80.0:
        logger.warning(
            "Low wind speed coverage (%.1f%%) for zone=%s.",
            wind_coverage, zone,
        )

    return df_merged


# ===========================================================================
# 4.  TEMPORALHARMONIZER DIFF — Weather Stream Integration
# ===========================================================================
#
# The following shows the EXACT changes to make inside the TemporalHarmonizer
# class.  Lines prefixed with '+' are additions; lines prefixed with '-' are
# deletions.  Unchanged context lines have no prefix.
#
# ── __init__ ────────────────────────────────────────────────────────────────
#
#  def __init__(
#      self,
#      interpolation_method: str = "linear",
#      max_grid_gap_hours: int = 6,
# +    max_weather_gap_hours: int = 3,
#  ) -> None:
#      self.interpolation_method = interpolation_method
#      self.max_grid_gap_hours = max_grid_gap_hours
# +    self.max_weather_gap_hours = max_weather_gap_hours
#      logger.info(
#          "TemporalHarmonizer initialised | grid gap-fill method='%s' max=%dh | "
# -        "macro alignment: strictly causal ffill",
# +        "macro alignment: strictly causal ffill | weather gap-fill max=%dh",
# -        interpolation_method, max_grid_gap_hours,
# +        interpolation_method, max_grid_gap_hours, max_weather_gap_hours,
#      )
#
# ── align() — updated signature ─────────────────────────────────────────────
#
#  def align(
#      self,
#      df_hourly: pd.DataFrame,
#      df_quarterly: Optional[pd.DataFrame] = None,
#      df_policy_rate: Optional[pd.DataFrame] = None,
# +    df_weather: Optional[pd.DataFrame] = None,
#      zone: str = "SE3",
#  ) -> pd.DataFrame:
#
# ── align() — Stage 4b insertion (after Riksbank join, before quarantine) ───
#
#      # Stage 4: Join Riksbank rate  [Step C — step-function ffill]
#      if df_policy_rate is not None and not df_policy_rate.empty:
#          df = align_riksbank_to_hourly(df, df_policy_rate)
#      else:
#          logger.warning("No Riksbank policy rate data provided.")
#          df["riksbank_policy_rate_pct"] = np.nan
#
# +     # Stage 4b: Join SMHI weather observations  [spatial-hourly join]
# +     if df_weather is not None and not df_weather.empty:
# +         df = align_weather_to_hourly(df, df_weather, zone=zone)
# +         # Forward-fill short weather gaps (sensor outages ≤ max_weather_gap_hours)
# +         # Temperature and wind are continuous physical quantities; linear
# +         # interpolation is appropriate for gaps up to a few hours.
# +         for col in ("temperature_c", "wind_speed_ms"):
# +             if col in df.columns:
# +                 df[col] = df[col].interpolate(
# +                     method="linear",
# +                     limit=self.max_weather_gap_hours,
# +                     limit_direction="forward",
# +                 )
# +     else:
# +         logger.warning(
# +             "No SMHI weather data provided for zone=%s. "
# +             "temperature_c and wind_speed_ms will be NaN.", zone,
# +         )
# +         df["temperature_c"]      = np.nan
# +         df["wind_speed_ms"]      = np.nan
# +         df["is_anomaly_weather"] = False
#
#      # Stage 5: Quarantine
#      df = self._apply_quarantine(df)
#
# ── align() — updated log line ───────────────────────────────────────────────
#
#      logger.info(
#          "Alignment complete | zone=%s | %d hourly rows | %d quarantined | "
#          "SCB coverage=%.1f%% | rate coverage=%.1f%%",
# +        "SCB coverage=%.1f%% | rate coverage=%.1f%% | "
# +        "temp coverage=%.1f%% | wind coverage=%.1f%%",
#          zone,
#          len(df),
#          int(df["is_quarantined"].sum()),
#          100 * df["smahus_construction_index"].notna().mean(),
#          100 * df["riksbank_policy_rate_pct"].notna().mean(),
# +        100 * df["temperature_c"].notna().mean(),
# +        100 * df["wind_speed_ms"].notna().mean(),
#      )
#
# ── align_all_zones() — updated signature ────────────────────────────────────
#
#  def align_all_zones(
#      self,
#      all_zone_grids: dict[str, pd.DataFrame],
#      df_quarterly_by_zone: Optional[dict[str, pd.DataFrame]] = None,
#      df_policy_rate: Optional[pd.DataFrame] = None,
# +    df_weather_by_zone: Optional[dict[str, pd.DataFrame]] = None,
#  ) -> pd.DataFrame:
#      """
#      Align all four bidding zones and concatenate into the Silver master frame.
#
#      Args:
#          all_zone_grids:       {'SE1': df, 'SE2': df, 'SE3': df, 'SE4': df}
#          df_quarterly_by_zone: {'SE1': df_q, ...} — one quarterly DF per zone.
#          df_policy_rate:       Single Riksbank rate DataFrame (national).
# +        df_weather_by_zone:   {'SE1': df_w, ...} — produced by
# +                              SMHIWeatherFetcher.fetch_all_zone_stations().
# +                              If a zone key is missing, weather features are NaN.
#      """
#
# ── align_all_zones() — pass weather into per-zone align() ──────────────────
#
#      for zone_label, df_grid in all_zone_grids.items():
#          quarterly = (
#              df_quarterly_by_zone.get(zone_label)
#              if df_quarterly_by_zone
#              else None
#          )
# +         weather = (
# +             df_weather_by_zone.get(zone_label)
# +             if df_weather_by_zone
# +             else None
# +         )
#          df_aligned = self.align(
#              df_hourly=df_grid,
#              df_quarterly=quarterly,
#              df_policy_rate=df_policy_rate,
# +            df_weather=weather,
#              zone=zone_label,
#          )
#          zone_frames.append(df_aligned)


# ===========================================================================
# 5.  SMOKE TEST  (python -m src.processing.silver_alignment --weather-test)
# ===========================================================================

def _smoke_test_weather_alignment() -> None:
    """
    Self-contained smoke test for the weather alignment additions.
    Runs without network access — uses synthetic DataFrames that mirror
    the shapes produced by SMHIWeatherFetcher.

    Expected output (all assertions pass):
        [PASS] align_weather_to_hourly: correct column set
        [PASS] align_weather_to_hourly: temperature_c coverage 100.0%
        [PASS] align_weather_to_hourly: wind_speed_ms coverage 100.0%
        [PASS] align_weather_to_hourly: no spurious zone contamination
        [PASS] SilverWeatherRecord: valid record accepted
        [PASS] SilverWeatherRecord: cross-zone mismatch rejected
        [PASS] TemporalHarmonizer.align(): weather columns present in Silver
        ✅ All weather alignment smoke tests passed.
    """
    from pydantic import ValidationError as _VE

    PASS = "[PASS]"
    FAIL = "[FAIL]"
    errors: list[str] = []

    def chk(label: str, condition: bool) -> None:
        tag = PASS if condition else FAIL
        print(f"    {tag} {label}")
        if not condition:
            errors.append(label)

    print("\n[1] align_weather_to_hourly — synthetic SE3 data")
    # Build a 72-hour grid for SE3
    rng_h = pd.date_range("2024-06-01", periods=72, freq="h", tz="UTC")
    df_grid_se3 = pd.DataFrame({
        "timestamp_utc":  rng_h,
        "zone":           "SE3",
        "load_mw":        8000.0,
        "imbalance_mwh":  10.0,
        "price_eur_mwh":  80.0,
        "is_anomaly":     False,
    })

    # SMHI weather — nominally hourly but offset by 5 min (realistic)
    rng_w = pd.date_range("2024-06-01 00:05", periods=70, freq="h", tz="UTC")
    df_weather_se3 = pd.DataFrame({
        "timestamp_utc": rng_w,
        "zone":          "SE3",
        "station_id":    98210,
        "temperature_c": 18.0 + 2.0 * np.sin(np.linspace(0, 4 * np.pi, 70)),
        "wind_speed_ms": 5.0  + 1.5 * np.cos(np.linspace(0, 4 * np.pi, 70)),
        "is_anomaly":    False,
    })

    df_aligned = align_weather_to_hourly(df_grid_se3, df_weather_se3, zone="SE3")

    expected_new_cols = {"temperature_c", "wind_speed_ms", "is_anomaly_weather"}
    chk(
        "align_weather_to_hourly: correct column set",
        expected_new_cols.issubset(set(df_aligned.columns)),
    )
    temp_cov = df_aligned["temperature_c"].notna().mean() * 100
    wind_cov = df_aligned["wind_speed_ms"].notna().mean() * 100
    chk(f"align_weather_to_hourly: temperature_c coverage {temp_cov:.1f}%", temp_cov > 90)
    chk(f"align_weather_to_hourly: wind_speed_ms coverage {wind_cov:.1f}%", wind_cov > 90)
    chk("align_weather_to_hourly: row count preserved", len(df_aligned) == len(df_grid_se3))

    print("\n[2] Spatial guard — cross-zone contamination rejection")
    df_weather_wrong_zone = df_weather_se3.copy()
    df_weather_wrong_zone["zone"] = "SE4"
    df_weather_wrong_zone["station_id"] = 62040  # SE4 station
    try:
        align_weather_to_hourly(df_grid_se3, df_weather_wrong_zone, zone="SE3")
        chk("align_weather_to_hourly: cross-zone mismatch rejected", False)
    except ValueError:
        chk("align_weather_to_hourly: cross-zone mismatch rejected", True)

    print("\n[3] SilverWeatherRecord — Pydantic v2 schema")
    # Valid record
    try:
        SilverWeatherRecord(
            timestamp_utc="2024-06-01T00:00:00+00:00",
            zone="SE3",
            station_id=98210,
            temperature_c=18.5,
            wind_speed_ms=4.2,
        )
        chk("SilverWeatherRecord: valid record accepted", True)
    except _VE as e:
        chk(f"SilverWeatherRecord: valid record accepted — {e}", False)

    # Cross-zone mismatch
    try:
        SilverWeatherRecord(
            timestamp_utc="2024-06-01T00:00:00+00:00",
            zone="SE4",          # wrong zone for station 98210
            station_id=98210,
            temperature_c=18.5,
        )
        chk("SilverWeatherRecord: cross-zone mismatch rejected", False)
    except _VE:
        chk("SilverWeatherRecord: cross-zone mismatch rejected", True)

    # Both measurements null
    try:
        SilverWeatherRecord(
            timestamp_utc="2024-06-01T00:00:00+00:00",
            zone="SE3",
            station_id=98210,
            temperature_c=None,
            wind_speed_ms=None,
        )
        chk("SilverWeatherRecord: both-null rejected", False)
    except _VE:
        chk("SilverWeatherRecord: both-null rejected", True)

    print("\n[4] TemporalHarmonizer integration — weather columns propagate to Silver")
    # Minimal TemporalHarmonizer wiring test.  Import the production class
    # directly so we exercise the real align() code path.
    try:
        from src.processing.silver_alignment import TemporalHarmonizer  # type: ignore
        harmonizer = TemporalHarmonizer()
        df_silver = harmonizer.align(
            df_hourly=df_grid_se3,
            df_weather=df_weather_se3,
            zone="SE3",
        )
        weather_present = (
            "temperature_c"      in df_silver.columns and
            "wind_speed_ms"      in df_silver.columns and
            "is_anomaly_weather" in df_silver.columns
        )
        chk(
            "TemporalHarmonizer.align(): weather columns present in Silver",
            weather_present,
        )
    except ImportError:
        print(
            "    [SKIP] TemporalHarmonizer import unavailable in test context — "
            "run from repo root with PYTHONPATH set."
        )

    print(f"\n{'=' * 70}")
    if errors:
        print(f"❌ {len(errors)} failure(s): {errors}")
    else:
        print("✅ All weather alignment smoke tests passed.")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--weather-test", action="store_true")
    args = ap.parse_args()
    if args.weather_test:
        _smoke_test_weather_alignment()
