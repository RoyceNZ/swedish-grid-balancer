"""
src/ingestion/market_fundamentals.py
=============================================================================
Market Fundamentals — Bronze Stream Ingestion
=============================================================================
Responsibilities:
  - Fetch three forward-looking market intelligence streams:
      A. REMIT Urgent Market Messages (UMM) — unplanned nuclear/thermal outages
      B. ENTSO-E 12.1.F Scheduled Commercial Net Positions — DA cleared flows
      C. Nord Pool / Energimyndigheten Hydro Reservoir Fill Levels — weekly
  - Enforce strict Pydantic v2 schema validation at the ingestion boundary
  - Write immutable Bronze layer Parquet snapshots to data/bronze/market/
  - Provide production-ready mock generators that faithfully replicate
    real market mechanics when live API credentials are unavailable

Stream A — REMIT UMM Notes:
  - Source: ENTSO-E REMIT RSS endpoint or Nord Pool UMM portal (JSON/XML)
  - Endpoint (production): https://umm.nordpoolgroup.com/api/messages
  - Authentication: Nord Pool API key (header: Authorization: Bearer <token>)
  - Key fields: messageId, eventStart, eventStop, affectedUnit, unavailableCapacity
  - Output: hourly expanded rows per [zone, timestamp_utc, outage_mw]
  - Nuclear base-load units mapped to zones via NUCLEAR_UNIT_ZONE_MAP

Stream B — ENTSO-E 12.1.F Net Positions:
  - Source: ENTSO-E Transparency Platform
  - Endpoint: /api?documentType=A25&contractMarketAgreement.Type=A01
  - Day-Ahead clearing locked at 12:45 CET the preceding day
  - Output: hourly [zone, timestamp_utc, scheduled_net_position_mw]
  - Positive = net exporter, negative = net importer

Stream C — Hydro Reservoir Levels:
  - Source: Energimyndigheten / Nord Pool weekly state matrix
  - Published each Monday covering the ISO week just closed
  - Output: weekly [year, week_of_year, reservoir_fill_ratio] (0.0–1.0)
  - Covers aggregate SE1+SE2 hydro basin (northern reservoir system)

Architecture position: LIVE API LAYER → BRONZE LAYER
Downstream: src/processing/silver_alignment.py (TemporalHarmonizer)
=============================================================================
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from ..utils.pipeline_logging import get_pipeline_logger
from ..utils.bronze import BronzeWriterBase

# ---------------------------------------------------------------------------
# Module Logger
# ---------------------------------------------------------------------------
logger = get_pipeline_logger("ingestion.market_fundamentals")


# ---------------------------------------------------------------------------
# Constants & Domain Maps
# ---------------------------------------------------------------------------

PIPELINE_VERSION = "1.0.0"
BRONZE_MARKET_DIR = Path("data/bronze/market")

# Nuclear and large thermal generation units mapped to bidding zones.
# Source: ENTSO-E EIC register + SVK operational network topology.
# Forsmark (1,2,3) + Ringhals (3,4) = active Swedish nuclear fleet as of 2025.
# Thermal base-load: Stenungsund (SE3), Karlshamn (SE4) oil/gas peakers.
NUCLEAR_UNIT_ZONE_MAP: dict[str, str] = {
    # Forsmark — Uppland, SE3 transmission node
    "FORSMARK_1": "SE3",
    "FORSMARK_2": "SE3",
    "FORSMARK_3": "SE3",
    # Ringhals — Halland, SE3 transmission node
    "RINGHALS_3": "SE3",
    "RINGHALS_4": "SE3",
    # Thermal base-load
    "STENUNGSUND_B3": "SE3",
    "KARLSHAMN_G1":   "SE4",
    "KARLSHAMN_G2":   "SE4",
    "KARLSHAMN_G3":   "SE4",
}

# Approximate installed capacity per unit (MW) — used to bound mock generation
UNIT_INSTALLED_CAPACITY_MW: dict[str, float] = {
    "FORSMARK_1": 1008.0,
    "FORSMARK_2": 1120.0,
    "FORSMARK_3": 1170.0,
    "RINGHALS_3":  1063.0,
    "RINGHALS_4":  1130.0,
    "STENUNGSUND_B3": 630.0,
    "KARLSHAMN_G1":   200.0,
    "KARLSHAMN_G2":   200.0,
    "KARLSHAMN_G3":   200.0,
}

# Physics bounds
OUTAGE_MW_MIN: float = 0.0
OUTAGE_MW_MAX: float = 5000.0          # upper bound: entire Swedish nuclear fleet
NET_POSITION_MW_MIN: float = -5000.0   # Sweden rarely exceeds ±4 GW net
NET_POSITION_MW_MAX: float =  5000.0
RESERVOIR_FILL_RATIO_MIN: float = 0.0
RESERVOIR_FILL_RATIO_MAX: float = 1.0

# 12:45 CET is the gate closure for ENTSO-E DA net positions (UTC offset varies
# by season, but we store everything in UTC and note the CET convention here)
DA_GATE_CLOSURE_HOUR_UTC_WINTER = 11   # 12:45 CET (UTC+1)
DA_GATE_CLOSURE_HOUR_UTC_SUMMER = 10   # 12:45 CEST (UTC+2)

# Nuclear outage alert threshold used by dashboard and validation checks
NUCLEAR_ALERT_THRESHOLD_MW: float = 1_000.0


# ---------------------------------------------------------------------------
# Pydantic v2 Schemas — Ingestion Boundary Contracts
# ---------------------------------------------------------------------------

class RemitOutageRecord(BaseModel):
    """
    Single validated REMIT UMM outage event after zone-hour expansion.

    The REMIT API returns event-level records with a start/stop window.
    We expand these into per-hour rows (one row per UTC hour that falls
    within [event_start, event_stop]) at the Bronze boundary so that
    downstream alignment code always sees a uniform hourly time index.
    """
    timestamp_utc: datetime = Field(..., description="UTC hour slot start")
    zone:          str       = Field(..., description="Bidding zone (SE1–SE4)")
    unit_name:     str       = Field(..., description="EIC generating unit name")
    outage_mw:     float     = Field(..., ge=0.0, description="Unavailable capacity MW")
    outage_type:   str       = Field(default="UNPLANNED", description="PLANNED|UNPLANNED")

    @field_validator("zone")
    @classmethod
    def zone_must_be_valid(cls, v: str) -> str:
        if v not in {"SE1", "SE2", "SE3", "SE4"}:
            raise ValueError(f"zone must be SE1–SE4; got '{v}'.")
        return v

    @field_validator("outage_mw")
    @classmethod
    def outage_within_physics_bounds(cls, v: float) -> float:
        if not (OUTAGE_MW_MIN <= v <= OUTAGE_MW_MAX):
            raise ValueError(
                f"outage_mw={v} outside physics bounds "
                f"[{OUTAGE_MW_MIN}, {OUTAGE_MW_MAX}]."
            )
        return v


class ScheduledNetPositionRecord(BaseModel):
    """
    Single validated hourly scheduled commercial net position.

    Positive = Sweden is a net exporter in that zone-hour.
    Negative = Sweden is a net importer.
    Gate-closure constraint: net positions are only available for delivery
    hours AFTER the 12:45 CET gate closure of the preceding day.
    """
    timestamp_utc:             datetime = Field(..., description="UTC delivery hour")
    zone:                      str      = Field(..., description="Bidding zone (SE1–SE4)")
    scheduled_net_position_mw: float    = Field(
        ..., description="Net contractual position (MW), positive=export"
    )

    @field_validator("zone")
    @classmethod
    def zone_valid(cls, v: str) -> str:
        if v not in {"SE1", "SE2", "SE3", "SE4"}:
            raise ValueError(f"zone must be SE1–SE4; got '{v}'.")
        return v

    @field_validator("scheduled_net_position_mw")
    @classmethod
    def within_physical_range(cls, v: float) -> float:
        if not (NET_POSITION_MW_MIN <= v <= NET_POSITION_MW_MAX):
            raise ValueError(
                f"scheduled_net_position_mw={v} outside range "
                f"[{NET_POSITION_MW_MIN}, {NET_POSITION_MW_MAX}]."
            )
        return v


class HydroReservoirRecord(BaseModel):
    """
    Single validated weekly hydro reservoir fill observation.

    reservoir_fill_ratio is the aggregate energy equivalent fill level
    expressed as a fraction of maximum theoretical capacity (0.0–1.0).
    Published Monday mornings for the ISO week that ended the preceding Sunday.
    """
    year:                int   = Field(..., ge=1990, le=2100)
    week_of_year:        int   = Field(..., ge=1, le=53)
    reservoir_fill_ratio: float = Field(
        ..., ge=0.0, le=1.0,
        description="Aggregate fill as fraction of max capacity"
    )

    @model_validator(mode="after")
    def week_valid_for_year(self) -> "HydroReservoirRecord":
        # ISO weeks run 1–52 for most years, 1–53 for long years
        max_week = 53 if _iso_year_has_53_weeks(self.year) else 52
        if self.week_of_year > max_week:
            raise ValueError(
                f"week_of_year={self.week_of_year} invalid for year {self.year} "
                f"(max ISO week = {max_week})."
            )
        return self


class MarketFundamentalsBronzeManifest(BaseModel):
    """Metadata manifest written alongside every Bronze Parquet snapshot."""
    pipeline_version:    str = PIPELINE_VERSION
    ingested_at_utc:     str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    stream_name:         str
    records_total:       int
    records_validation_failed: int
    period_start_utc:    Optional[str] = None
    period_end_utc:      Optional[str] = None
    bronze_path:         str
    mock_data:           bool = False


# ---------------------------------------------------------------------------
# Helper Utilities
# ---------------------------------------------------------------------------

def _iso_year_has_53_weeks(year: int) -> bool:
    """Return True if the given year contains ISO week 53."""
    # A year has 53 ISO weeks iff Jan 1 or Dec 31 is a Thursday
    import calendar
    jan1 = datetime(year, 1, 1)
    dec31 = datetime(year, 12, 31)
    return jan1.weekday() == 3 or dec31.weekday() == 3


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _build_hourly_range(
    start_utc: datetime,
    end_utc: datetime,
) -> pd.DatetimeIndex:
    """Return a UTC-aware hourly DatetimeIndex from start (inclusive) to end (exclusive)."""
    return pd.date_range(start=start_utc, end=end_utc, freq="h", tz="UTC", inclusive="left")


# ---------------------------------------------------------------------------
# Bronze Writer
# ---------------------------------------------------------------------------

class MarketBronzeWriter(BronzeWriterBase):
    """
    Persists Bronze market fundamental snapshots as Parquet files.

    Naming convention:
        {stream_name}_{ingested_UTC}.parquet

    Parquet is used (not gzip JSON) because these streams are tabular
    from the start and downstream alignment code reads them directly via
    pd.read_parquet() without a parse step.
    """

    def __init__(self, bronze_dir: Path = BRONZE_MARKET_DIR) -> None:
        super().__init__(bronze_dir)

    def write(
        self,
        df: pd.DataFrame,
        stream_name: str,
        fixed_filename: Optional[str] = None,
    ) -> Path:
        """Write DataFrame to a timestamped Parquet file."""
        if fixed_filename:
            fname = fixed_filename
        else:
            ts_str = _now_utc().strftime("%Y%m%dT%H%M%SZ")
            fname = f"{stream_name}_{ts_str}.parquet"
        out_path = self.bronze_dir / fname
        df.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")
        size_kb = out_path.stat().st_size / 1024
        logger.info(
            "Bronze snapshot written → %s (%.1f KB | %d rows)",
            out_path, size_kb, len(df),
        )
        return out_path



# ===========================================================================
# STREAM A — REMIT Unplanned Outages
# ===========================================================================

def _generate_mock_remit_outages(
    period_start: datetime,
    period_end: datetime,
    rng_seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Generate realistic mock REMIT UMM event records.

    Mechanics:
      - One to three unplanned trips are seeded per month of the window.
      - Trip duration is sampled from a log-normal distribution (median ~36h)
        reflecting real forced-outage recovery times for nuclear and large
        thermal units.
      - Unavailable capacity is sampled between 50% and 100% of the unit's
        installed capacity (partial and full trips are both realistic).
      - Event timestamps are aligned to full UTC hours before expansion.

    Returns:
        List of raw event dicts (pre-validation, mirrors REMIT API shape).
    """
    rng = np.random.default_rng(rng_seed)
    units = list(NUCLEAR_UNIT_ZONE_MAP.keys())
    events: list[dict[str, Any]] = []

    window_days = max(1, (period_end - period_start).days)
    # ~2.5 unplanned trips per month across the entire fleet is historically plausible
    n_events = max(1, int(window_days / 30 * 2.5))

    for i in range(n_events):
        unit = rng.choice(units)
        installed_mw = UNIT_INSTALLED_CAPACITY_MW[unit]

        # Random start within the window
        offset_hours = rng.integers(0, max(1, int(window_days * 24)))
        event_start = period_start + timedelta(hours=int(offset_hours))
        event_start = event_start.replace(minute=0, second=0, microsecond=0)

        # Log-normal duration: median ~36 h, σ=0.8
        duration_h = int(np.clip(rng.lognormal(mean=3.58, sigma=0.8), 2, 720))
        event_stop = event_start + timedelta(hours=duration_h)

        # Partial or full trip
        trip_fraction = rng.uniform(0.5, 1.0)
        unavailable_mw = round(installed_mw * trip_fraction, 1)

        events.append({
            "messageId":           f"UMM-MOCK-{i:04d}",
            "unit":                unit,
            "zone":                NUCLEAR_UNIT_ZONE_MAP[unit],
            "eventStart":          event_start.isoformat(),
            "eventStop":           event_stop.isoformat(),
            "unavailableCapacity": unavailable_mw,
            "messageType":         "UNPLANNED",
        })
        logger.debug(
            "Mock REMIT event | %s zone=%s | %.0f MW | %s → %s",
            unit, NUCLEAR_UNIT_ZONE_MAP[unit], unavailable_mw,
            event_start.strftime("%Y-%m-%dT%H"), event_stop.strftime("%Y-%m-%dT%H"),
        )

    return events


def _expand_remit_events_to_hourly(
    events: list[dict[str, Any]],
    period_start: datetime,
    period_end: datetime,
) -> tuple[pd.DataFrame, int]:
    """
    Expand raw REMIT event records into per-hour rows with Pydantic validation.

    Each event covers [eventStart, eventStop).  We iterate every UTC hour
    in that window that also falls within [period_start, period_end) and
    emit one row per (zone, hour, unit).  Multiple simultaneous outages on
    the same zone are kept as separate rows so downstream can choose to
    sum, max, or count them independently.

    Returns:
        (validated_df, validation_failures)
    """
    rows: list[dict[str, Any]] = []
    failures = 0

    for ev in events:
        try:
            ev_start = pd.Timestamp(ev["eventStart"]).tz_localize("UTC") \
                if pd.Timestamp(ev["eventStart"]).tzinfo is None \
                else pd.Timestamp(ev["eventStart"]).tz_convert("UTC")
            ev_stop  = pd.Timestamp(ev["eventStop"]).tz_localize("UTC") \
                if pd.Timestamp(ev["eventStop"]).tzinfo is None \
                else pd.Timestamp(ev["eventStop"]).tz_convert("UTC")
        except Exception as exc:
            logger.warning("Cannot parse REMIT event timestamps: %s — skipping.", exc)
            failures += 1
            continue

        hourly_slots = pd.date_range(
            start=max(ev_start, pd.Timestamp(period_start, tz="UTC")),
            end=min(ev_stop, pd.Timestamp(period_end, tz="UTC")),
            freq="h", inclusive="left",
        )
        if hourly_slots.empty:
            continue  # Event outside the requested window

        for slot in hourly_slots:
            try:
                record = RemitOutageRecord(
                    timestamp_utc=slot.to_pydatetime(),
                    zone=ev["zone"],
                    unit_name=ev["unit"],
                    outage_mw=float(ev["unavailableCapacity"]),
                    outage_type=ev.get("messageType", "UNPLANNED"),
                )
                rows.append({
                    "timestamp_utc": slot,
                    "zone":          record.zone,
                    "unit_name":     record.unit_name,
                    "outage_mw":     record.outage_mw,
                    "outage_type":   record.outage_type,
                })
            except (ValidationError, ValueError) as exc:
                logger.warning("REMIT record validation failure: %s", exc)
                failures += 1

    if not rows:
        return _empty_remit_df(), failures

    df = pd.DataFrame(rows)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df.sort_values(["timestamp_utc", "zone", "unit_name"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df, failures


def _empty_remit_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "timestamp_utc", "zone", "unit_name", "outage_mw", "outage_type"
    ])


def fetch_remit_outages(
    period_start: Optional[datetime] = None,
    period_end:   Optional[datetime] = None,
    bronze_dir:   Path = BRONZE_MARKET_DIR,
    use_mock:     bool = True,
    api_token:    Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch REMIT Urgent Market Messages for unplanned generation outages.

    Production path (use_mock=False):
        Calls the Nord Pool UMM API (requires api_token) and parses the
        returned event list.  The raw response is written to Bronze before
        any parsing so raw payloads are always preserved.

    Mock path (use_mock=True, default):
        Generates statistically realistic synthetic outage events using
        _generate_mock_remit_outages().  The mock data faithfully replicates
        the per-hour expansion schema so all downstream code is exercised
        identically regardless of which path is used.

    Temporal causality guarantee:
        outage_mw in each row represents the MW unavailable AT that UTC hour.
        Downstream alignment uses this directly as a point-in-time observation;
        no forward-fill is applied here because the Silver layer owns that
        responsibility (align_market_fundamentals_to_hourly).

    Args:
        period_start: UTC start of the desired window. Defaults to 90 days ago.
        period_end:   UTC end of the desired window. Defaults to now.
        bronze_dir:   Output directory for Bronze Parquet snapshot.
        use_mock:     If True, generate mock data instead of calling live API.
        api_token:    Nord Pool API bearer token (required if use_mock=False).

    Returns:
        pd.DataFrame with columns:
            [timestamp_utc, zone, unit_name, outage_mw, outage_type]
        One row per (zone, hour, unit) for every active outage slot.

    Output file:
        data/bronze/market/remit_unplanned_outages.parquet
    """
    if period_start is None:
        period_start = _now_utc() - timedelta(days=90)
    if period_end is None:
        period_end = _now_utc()

    period_start = period_start.replace(tzinfo=timezone.utc) \
        if period_start.tzinfo is None else period_start
    period_end = period_end.replace(tzinfo=timezone.utc) \
        if period_end.tzinfo is None else period_end

    logger.info(
        "fetch_remit_outages | window %s → %s | mock=%s",
        period_start.strftime("%Y-%m-%d"), period_end.strftime("%Y-%m-%d"), use_mock,
    )

    writer = MarketBronzeWriter(bronze_dir)
    validation_failures = 0

    if use_mock:
        raw_events = _generate_mock_remit_outages(period_start, period_end)
    else:
        # Production: call Nord Pool UMM API
        if not api_token:
            raise EnvironmentError(
                "Nord Pool API token required for live REMIT fetch. "
                "Set api_token= or use use_mock=True for development."
            )
        # NOTE: In production, replace this block with BaseHTTPClient.request()
        # following the same pattern as ENTSOEClient._fetch_load_chunk().
        raise NotImplementedError(
            "Live Nord Pool UMM API integration not yet wired. "
            "Set use_mock=True to use the production-schema mock path."
        )

    df, validation_failures = _expand_remit_events_to_hourly(
        raw_events, period_start, period_end
    )

    # Aggregate to zone-hour level: sum all simultaneous unit outages per zone-slot
    # This produces the canonical outage_mw column expected by silver_alignment.
    df_agg = (
        df.groupby(["timestamp_utc", "zone"], as_index=False)
        .agg(
            outage_mw=("outage_mw", "sum"),
            active_unit_count=("unit_name", "nunique"),
        )
    )
    df_agg["timestamp_utc"] = pd.to_datetime(df_agg["timestamp_utc"], utc=True)

    # Persist to Bronze
    bronze_path = writer.write(
        df_agg,
        stream_name="remit_unplanned_outages",
        fixed_filename="remit_unplanned_outages.parquet",
    )
    manifest = MarketFundamentalsBronzeManifest(
        stream_name="remit_unplanned_outages",
        records_total=len(df_agg),
        records_validation_failed=validation_failures,
        period_start_utc=period_start.isoformat(),
        period_end_utc=period_end.isoformat(),
        bronze_path=str(bronze_path),
        mock_data=use_mock,
    )
    writer.write_manifest(manifest, bronze_path)

    logger.info(
        "fetch_remit_outages complete | %d zone-hour rows | "
        "%d validation failures | mock=%s",
        len(df_agg), validation_failures, use_mock,
    )
    return df_agg


# ===========================================================================
# STREAM B — Scheduled Commercial Net Positions (ENTSO-E 12.1.F)
# ===========================================================================

def _generate_mock_net_positions(
    period_start: datetime,
    period_end: datetime,
    rng_seed: int = 7,
) -> list[dict[str, Any]]:
    """
    Generate realistic mock scheduled net position records.

    Physical mechanics embedded in the mock:
      - SE1/SE2: chronically net exporting due to surplus hydro in the north;
        base position is positive (export) with seasonal drawdown in summer.
      - SE3: near-balanced; oscillates around zero with a small import bias
        in winter (residential heating demand).
      - SE4: chronically net importing; relies on SE3 and continental imports.
      - All zones have 24-h sinusoidal intra-day shape (grid is tightest
        during morning/evening peaks).
      - Gate-closure rule: positions are only available for hours strictly
        after the 12:45 UTC-equivalent of the preceding day (i.e. 36-hour
        forward horizon at most, 12-hour minimum).

    Returns:
        List of hourly position dicts, one per (zone, delivery_hour).
    """
    rng = np.random.default_rng(rng_seed)
    hours = _build_hourly_range(period_start, period_end)
    records: list[dict[str, Any]] = []

    # Base export/import bias per zone (MW)
    zone_base_mw = {"SE1": +800.0, "SE2": +600.0, "SE3": -50.0, "SE4": -900.0}
    # Seasonal amplitude (winter draws SE1/SE2 down toward zero)
    zone_seasonal_amp = {"SE1": 400.0, "SE2": 300.0, "SE3": 200.0, "SE4": 150.0}

    for ts in hours:
        doy = ts.day_of_year
        seasonal_factor = np.cos(2 * np.pi * (doy - 355) / 365)  # peak Jan 1

        for zone in ["SE1", "SE2", "SE3", "SE4"]:
            base = zone_base_mw[zone]
            amp  = zone_seasonal_amp[zone]
            # Intra-day sine: peak exports/imports at 07:00 and 19:00
            intraday = 150.0 * np.sin(2 * np.pi * (ts.hour - 7) / 24)
            noise = rng.normal(0, 30)
            position_mw = base - amp * seasonal_factor + intraday + noise
            position_mw = float(np.clip(position_mw,
                                        NET_POSITION_MW_MIN, NET_POSITION_MW_MAX))
            records.append({
                "timestamp_utc":             ts,
                "zone":                      zone,
                "scheduled_net_position_mw": round(position_mw, 1),
            })
    return records


def fetch_scheduled_net_positions(
    period_start: Optional[datetime] = None,
    period_end:   Optional[datetime] = None,
    bronze_dir:   Path = BRONZE_MARKET_DIR,
    use_mock:     bool = True,
    api_token:    Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch ENTSO-E 12.1.F Scheduled Commercial Net Positions.

    Each row represents the net contractual hourly energy position for a
    bidding zone, locked in at the Day-Ahead gate closure (12:45 CET D-1).
    Positive values = net exporter; negative = net importer.

    Gate-closure causality:
        Positions for delivery day D are only known AFTER 12:45 CET on D-1.
        The Silver layer alignment function (align_market_fundamentals_to_hourly)
        enforces this by computing a 'gate_closed_at_utc' column and using
        merge_asof(direction='backward') keyed on it — ensuring no delivery-day
        position bleeds into training rows before its gate-closure time.

    Args:
        period_start: UTC start of delivery window. Defaults to 30 days ago.
        period_end:   UTC end of delivery window. Defaults to now.
        bronze_dir:   Output directory for Bronze Parquet snapshot.
        use_mock:     If True, generate mock data instead of calling live API.
        api_token:    ENTSO-E API token (required if use_mock=False).

    Returns:
        pd.DataFrame with columns:
            [timestamp_utc, zone, scheduled_net_position_mw]

    Output file:
        data/bronze/market/scheduled_net_positions.parquet
    """
    if period_start is None:
        period_start = _now_utc() - timedelta(days=30)
    if period_end is None:
        period_end = _now_utc()

    period_start = period_start.replace(tzinfo=timezone.utc) \
        if period_start.tzinfo is None else period_start
    period_end = period_end.replace(tzinfo=timezone.utc) \
        if period_end.tzinfo is None else period_end

    logger.info(
        "fetch_scheduled_net_positions | window %s → %s | mock=%s",
        period_start.strftime("%Y-%m-%d"), period_end.strftime("%Y-%m-%d"), use_mock,
    )

    writer = MarketBronzeWriter(bronze_dir)
    validation_failures = 0

    if use_mock:
        raw_records = _generate_mock_net_positions(period_start, period_end)
    else:
        if not api_token:
            raise EnvironmentError(
                "ENTSO-E API token required for live net-position fetch."
            )
        raise NotImplementedError(
            "Live ENTSO-E 12.1.F API integration not yet wired. "
            "Set use_mock=True to use the production-schema mock path."
        )

    # Pydantic v2 validation at ingestion boundary
    validated_rows: list[dict[str, Any]] = []
    for raw in raw_records:
        try:
            rec = ScheduledNetPositionRecord(
                timestamp_utc=raw["timestamp_utc"],
                zone=raw["zone"],
                scheduled_net_position_mw=raw["scheduled_net_position_mw"],
            )
            validated_rows.append({
                "timestamp_utc":             pd.Timestamp(rec.timestamp_utc, tz="UTC"),
                "zone":                      rec.zone,
                "scheduled_net_position_mw": rec.scheduled_net_position_mw,
            })
        except (ValidationError, ValueError) as exc:
            validation_failures += 1
            logger.warning("Net position validation failure: %s", exc)

    if not validated_rows:
        logger.error("All net position records failed validation.")
        df = pd.DataFrame(columns=["timestamp_utc", "zone", "scheduled_net_position_mw"])
    else:
        df = pd.DataFrame(validated_rows)
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        df.sort_values(["timestamp_utc", "zone"], inplace=True)
        df.reset_index(drop=True, inplace=True)

    bronze_path = writer.write(
        df,
        stream_name="scheduled_net_positions",
        fixed_filename="scheduled_net_positions.parquet",
    )
    manifest = MarketFundamentalsBronzeManifest(
        stream_name="scheduled_net_positions",
        records_total=len(df),
        records_validation_failed=validation_failures,
        period_start_utc=period_start.isoformat(),
        period_end_utc=period_end.isoformat(),
        bronze_path=str(bronze_path),
        mock_data=use_mock,
    )
    writer.write_manifest(manifest, bronze_path)

    logger.info(
        "fetch_scheduled_net_positions complete | %d rows | %d failures | mock=%s",
        len(df), validation_failures, use_mock,
    )
    return df


# ===========================================================================
# STREAM C — Hydro Reservoir Fill Levels
# ===========================================================================

def _generate_mock_hydro_reservoir_levels(
    year_start: int = 2020,
    year_end:   int = 2025,
    rng_seed:   int = 99,
) -> list[dict[str, Any]]:
    """
    Generate realistic mock weekly hydro reservoir fill ratio observations.

    Physical mechanics:
      - Swedish hydro reservoirs (SE1 + SE2 northern basin) follow a strong
        annual cycle: fill during spring snowmelt (weeks 18–28), drain through
        autumn/winter production (weeks 28–14 of the next year).
      - Typical annual range: 20–95% fill; long-term average ~65%.
      - Year-to-year variation is modelled as a correlated random walk with
        mean-reversion (reservoirs physically cannot go below ~10% without
        regulatory intervention or above 100%).
      - ISO week 1 of each year starts at the prior December level.

    Returns:
        List of weekly dicts [year, week_of_year, reservoir_fill_ratio].
    """
    rng = np.random.default_rng(rng_seed)
    records: list[dict[str, Any]] = []

    # Multi-year correlated annual cycle
    level = 0.60   # starting fill ratio
    for year in range(year_start, year_end + 1):
        max_week = 53 if _iso_year_has_53_weeks(year) else 52
        for week in range(1, max_week + 1):
            # Seasonal driver: snowmelt refill peaks around week 22 (late May)
            seasonal_push = 0.008 * np.sin(2 * np.pi * (week - 3) / 52)
            # Mean-reversion pull toward 0.65 target fill
            mean_rev = 0.005 * (0.65 - level)
            noise = rng.normal(0, 0.008)
            level = float(np.clip(level + seasonal_push + mean_rev + noise, 0.08, 0.99))
            records.append({
                "year":                 year,
                "week_of_year":         week,
                "reservoir_fill_ratio": round(level, 4),
            })
    return records


def fetch_hydro_reservoir_levels(
    year_start: int = 2020,
    year_end:   Optional[int] = None,
    bronze_dir: Path = BRONZE_MARKET_DIR,
    use_mock:   bool = True,
) -> pd.DataFrame:
    """
    Fetch weekly hydro reservoir fill levels for the northern SE1/SE2 basin.

    The reservoir fill ratio captures the physical flexibility buffer in
    the hydro-generation system.  A declining ratio (depletion velocity)
    signals an approaching flexibility crunch; a high ratio gives operators
    room to absorb imbalances by spilling to thermal dispatch instead of
    hydro draw.

    Publication lag:
        Each week's data is published the following Monday morning by
        Energimyndigheten / Nord Pool.  The Silver layer alignment function
        enforces this one-week publication lag via a date-offset shift before
        the merge_asof join, preventing any look-ahead into future reservoir
        states.

    Args:
        year_start: First year to fetch (inclusive). Defaults to 2020.
        year_end:   Last year to fetch (inclusive). Defaults to current year.
        bronze_dir: Output directory for Bronze Parquet snapshot.
        use_mock:   If True, generate mock data instead of calling live source.

    Returns:
        pd.DataFrame with columns:
            [year, week_of_year, reservoir_fill_ratio]

    Output file:
        data/bronze/market/hydro_reservoir_weekly.parquet
    """
    if year_end is None:
        year_end = _now_utc().year

    logger.info(
        "fetch_hydro_reservoir_levels | years %d–%d | mock=%s",
        year_start, year_end, use_mock,
    )

    writer = MarketBronzeWriter(bronze_dir)
    validation_failures = 0

    if use_mock:
        raw_records = _generate_mock_hydro_reservoir_levels(year_start, year_end)
    else:
        raise NotImplementedError(
            "Live Energimyndigheten / Nord Pool hydro API integration not yet wired. "
            "Set use_mock=True to use the production-schema mock path."
        )

    validated_rows: list[dict[str, Any]] = []
    for raw in raw_records:
        try:
            rec = HydroReservoirRecord(**raw)
            validated_rows.append({
                "year":                 rec.year,
                "week_of_year":         rec.week_of_year,
                "reservoir_fill_ratio": rec.reservoir_fill_ratio,
            })
        except (ValidationError, ValueError) as exc:
            validation_failures += 1
            logger.warning("Hydro reservoir validation failure: %s", exc)

    if not validated_rows:
        logger.error("All hydro reservoir records failed validation.")
        df = pd.DataFrame(columns=["year", "week_of_year", "reservoir_fill_ratio"])
    else:
        df = pd.DataFrame(validated_rows)
        df.sort_values(["year", "week_of_year"], inplace=True)
        df.reset_index(drop=True, inplace=True)

    bronze_path = writer.write(
        df,
        stream_name="hydro_reservoir_weekly",
        fixed_filename="hydro_reservoir_weekly.parquet",
    )
    manifest = MarketFundamentalsBronzeManifest(
        stream_name="hydro_reservoir_weekly",
        records_total=len(df),
        records_validation_failed=validation_failures,
        period_start_utc=f"{year_start}-W01",
        period_end_utc=f"{year_end}-W52",
        bronze_path=str(bronze_path),
        mock_data=use_mock,
    )
    writer.write_manifest(manifest, bronze_path)

    logger.info(
        "fetch_hydro_reservoir_levels complete | %d weekly rows | %d failures",
        len(df), validation_failures,
    )
    return df


# ===========================================================================
# Convenience: Fetch All Three Streams
# ===========================================================================

def fetch_all_market_fundamentals(
    period_start:  Optional[datetime] = None,
    period_end:    Optional[datetime] = None,
    year_start:    int = 2020,
    bronze_dir:    Path = BRONZE_MARKET_DIR,
    use_mock:      bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Fetch all three market fundamental streams in one call.

    Returns:
        {
          "remit_outages":        pd.DataFrame,
          "net_positions":        pd.DataFrame,
          "hydro_reservoirs":     pd.DataFrame,
        }
    """
    logger.info("=== Fetching all market fundamentals streams ===")
    return {
        "remit_outages":    fetch_remit_outages(
            period_start, period_end, bronze_dir, use_mock
        ),
        "net_positions":    fetch_scheduled_net_positions(
            period_start, period_end, bronze_dir, use_mock
        ),
        "hydro_reservoirs": fetch_hydro_reservoir_levels(
            year_start=year_start, bronze_dir=bronze_dir, use_mock=use_mock
        ),
    }


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Standalone execution:  python -m src.ingestion.market_fundamentals
    Fetches all three streams using mock data and prints summary.
    """
    from datetime import timezone as _tz

    start = datetime(2024, 1, 1, tzinfo=_tz.utc)
    end   = datetime(2024, 12, 31, 23, tzinfo=_tz.utc)

    streams = fetch_all_market_fundamentals(
        period_start=start, period_end=end, year_start=2020
    )

    print("\n" + "=" * 70)
    print("Market Fundamentals Bronze Ingestion — Summary")
    print("=" * 70)
    for name, df in streams.items():
        print(f"  {name:<28}: {len(df):>6} rows | columns: {list(df.columns)}")
    print("=" * 70)
    print(f"Bronze output: {BRONZE_MARKET_DIR.resolve()}")
