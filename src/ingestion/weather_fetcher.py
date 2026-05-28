"""
src/ingestion/weather_fetcher.py
=============================================================================
SMHI Open Data Meteorologiska Observationer — Weather Ingestion Client
=============================================================================
Responsibilities:
  - Fetch hourly ambient temperature (Parameter 1) and wind speed (Parameter 4)
    from SMHI's Open Data Meteorologiska Observationer REST API
  - Target the four canonical grid observation stations anchoring SE1–SE4:
      Luleå      (SE1) → station 140480
      Sundsvall  (SE2) → station  86340
      Stockholm  (SE3) → station  98210
      Malmö      (SE4) → station  62040
  - Validate raw observation records with Pydantic v2 at the ingestion boundary
  - Write immutable, gzip-compressed Bronze layer JSON snapshots with
    execution timestamps and metadata manifests to data/bronze/weather/
  - Inherit exponential backoff + circuit breaker from BaseHTTPClient

SMHI Open Data API Notes:
  - Base URL: https://opendata-download-metobs.smhi.se/api/version/latest
  - Authentication: None required (fully public API)
  - Rate limit: Not formally published; conservative 1.0 s inter-request delay
  - Response format: JSON (application/json)
  - Endpoint pattern:
      /parameter/{parameter_id}/station/{station_id}/period/
        corrected-archive/data.json
      Periods available: latest-hour, latest-day, latest-months, corrected-archive
  - Parameter IDs used:
      1  → Lufttemperatur (ambient temperature, °C), 1-hourly
      4  → Vindhastighet (mean wind speed m/s), 1-hourly
  - Station coordinate reference: SWEREF99 TM / WGS84 lat-lon
  - Response geometry: each observation carries a 'date' (epoch ms) + 'value' string

Architecture position: LIVE API LAYER → BRONZE LAYER
Upstream:  SMHI Open Data REST API
Downstream: src/processing/silver_alignment.py (TemporalHarmonizer)
=============================================================================
"""

from __future__ import annotations

import gzip
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx
import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from .client import (
    BaseHTTPClient,
    NonRetryableHTTPError,
    RetryExhaustedError,
    retry_with_backoff,
    DEFAULT_MAX_RETRIES,
    DEFAULT_BASE_BACKOFF_SECONDS,
    DEFAULT_MAX_BACKOFF_SECONDS,
)
from ..utils.pipeline_logging import get_pipeline_logger
from ..utils.bronze import BronzeWriterBase

# ---------------------------------------------------------------------------
# Module Logger
# ---------------------------------------------------------------------------
logger = get_pipeline_logger("ingestion.weather_fetcher")


# ---------------------------------------------------------------------------
# SMHI API Constants
# ---------------------------------------------------------------------------
SMHI_BASE_URL = (
    "https://opendata-download-metobs.smhi.se/api/version/latest"
)
SMHI_REQUEST_DELAY_SECONDS = 1.0   # conservative; API has no published limit

# SMHI parameter IDs
PARAM_TEMPERATURE   = 1   # Lufttemperatur (°C), hourly
PARAM_WIND_SPEED    = 4   # Vindhastighet medelvind (m/s), hourly

# Canonical main-grid observation stations, one per bidding zone.
# Station IDs sourced from SMHI's published reference list for high-priority
# synoptic stations with continuous corrected-archive coverage.
#
# Mapping rationale:
#   SE1 → Luleå Airport (140480): northernmost population centre in Norrbotten;
#         representative of the high-latitude hydro-dominant zone.
#   SE2 → Sundsvall / Härnösand (86340): mid-north coastal hub; Ångermanland
#         border matches the SE1/SE2 transmission constraint.
#   SE3 → Stockholm / Observatorielunden (98210): SMHI's primary reference
#         station; captures the dominant load centre for the largest zone.
#   SE4 → Malmö / Sturup (62040): southernmost SMHI synoptic station covering
#         Skåne load region and Öresund wind corridor.
STATION_ZONE_MAP: dict[int, str] = {
    140480: "SE1",   # Luleå
    86340:  "SE2",   # Sundsvall
    98210:  "SE3",   # Stockholm
    62040:  "SE4",   # Malmö
}

# Physics-bound sanity limits for Sweden
TEMPERATURE_MIN_C   = -50.0   # below Kvikkjokk record low
TEMPERATURE_MAX_C   =  40.0   # above Uppsala record high
WIND_SPEED_MIN_MS   =   0.0   # cannot be negative
WIND_SPEED_MAX_MS   =  70.0   # above any measured Swedish hourly mean

PIPELINE_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Pydantic v2 Schemas — SMHI Ingestion Boundary Contracts
# ---------------------------------------------------------------------------

class SMHIObservationPoint(BaseModel):
    """
    Single validated raw observation from SMHI.

    The SMHI API returns date as epoch-milliseconds and value as a string.
    Both are coerced and validated here at the ingestion boundary so that
    downstream code can work with typed Python objects without defensive
    casting.
    """
    date_epoch_ms: int = Field(..., description="Epoch milliseconds (UTC)")
    value_raw: str     = Field(..., description="Raw string value from API")
    quality: str       = Field(default="G", description="SMHI quality flag")

    @field_validator("value_raw")
    @classmethod
    def value_must_be_parseable_float(cls, v: str) -> str:
        try:
            float(v.replace(",", "."))
        except ValueError:
            raise ValueError(f"Observation value '{v}' cannot be cast to float.")
        return v

    @field_validator("date_epoch_ms")
    @classmethod
    def epoch_must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"Epoch ms must be positive; got {v}.")
        return v

    @property
    def timestamp_utc(self) -> pd.Timestamp:
        return pd.Timestamp(self.date_epoch_ms, unit="ms", tz="UTC")

    @property
    def value_float(self) -> float:
        return float(self.value_raw.replace(",", "."))


class SMHIStationRecord(BaseModel):
    """
    Validated top-level station metadata returned by the SMHI API.
    Used to confirm we have the expected station and parameter before
    persisting to Bronze.
    """
    station_id: int         = Field(..., description="SMHI station key")
    station_name: str       = Field(..., description="Human-readable station name")
    parameter_id: int       = Field(..., description="SMHI parameter key (1=temp, 4=wind)")
    parameter_name: str     = Field(default="", description="SMHI parameter label")
    latitude: float         = Field(..., ge=-90.0, le=90.0)
    longitude: float        = Field(..., ge=-180.0, le=180.0)
    records_total: int      = Field(..., ge=0)
    records_validation_failed: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def failed_must_not_exceed_total(self) -> "SMHIStationRecord":
        if self.records_validation_failed > self.records_total:
            raise ValueError(
                "records_validation_failed cannot exceed records_total: "
                f"{self.records_validation_failed} > {self.records_total}"
            )
        return self


class SMHIBronzeManifest(BaseModel):
    """
    Metadata manifest written alongside each Bronze gzip snapshot.

    Mirrors the BronzeManifestRecord pattern from energy_client.py so that
    downstream audit tooling can treat all Bronze manifests uniformly.
    """
    pipeline_version: str   = PIPELINE_VERSION
    ingested_at_utc: str    = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    station_id: int
    station_name: str
    bidding_zone: str
    parameter_id: int
    parameter_name: str
    latitude: float
    longitude: float
    records_total: int
    records_validation_failed: int
    anomalies_flagged: int
    obs_start_utc: Optional[str]
    obs_end_utc: Optional[str]
    bronze_path: str


# ---------------------------------------------------------------------------
# Anomaly Detection Helpers
# ---------------------------------------------------------------------------

def _flag_temperature_anomalies(series: pd.Series) -> pd.Series:
    """
    Return boolean Series marking observations outside physics bounds.
    NaN values are NOT flagged here — they indicate missing data, not
    erroneous readings; the gap-filler in silver_alignment handles them.
    """
    return (series < TEMPERATURE_MIN_C) | (series > TEMPERATURE_MAX_C)


def _flag_wind_anomalies(series: pd.Series) -> pd.Series:
    return (series < WIND_SPEED_MIN_MS) | (series > WIND_SPEED_MAX_MS)


# ---------------------------------------------------------------------------
# Bronze Layer Writer
# ---------------------------------------------------------------------------

class WeatherBronzeWriter(BronzeWriterBase):
    """
    Writes immutable Bronze layer weather snapshots.

    Naming convention:
        weather_{param_label}_{zone}_{station_id}_{period}_{ingested_UTC}.json.gz

    The period token (e.g. 'corrected-archive') is included so that
    future fetchers targeting 'latest-months' or 'latest-day' produce
    non-colliding filenames without risking overwrites.
    """

    def __init__(self, bronze_dir: Path) -> None:
        super().__init__(bronze_dir)
        logger.debug("WeatherBronzeWriter initialised | dir=%s", self.bronze_dir)

    def write(
        self,
        payload: dict[str, Any],
        station_id: int,
        zone: str,
        parameter_id: int,
        period: str = "corrected-archive",
    ) -> Path:
        """Gzip-compress and persist raw API JSON payload to Bronze."""
        param_label = "temperature" if parameter_id == PARAM_TEMPERATURE else "wind_speed"
        ingested_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        filename = (
            f"weather_{param_label}_{zone}_{station_id}_{period}_{ingested_str}.json.gz"
        )
        out_path = self.bronze_dir / filename

        with gzip.open(out_path, "wt", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)

        size_kb = out_path.stat().st_size / 1024
        logger.info(
            "Bronze snapshot written → %s (%.1f KB compressed)",
            out_path, size_kb,
        )
        return out_path



# ---------------------------------------------------------------------------
# Response Parser
# ---------------------------------------------------------------------------

class SMHIResponseParser:
    """
    Parses a raw SMHI Open Data JSON response into validated observation records.

    The SMHI corrected-archive response structure:
        {
          "station": {"key": "...", "name": "...", "latitude": ..., "longitude": ...},
          "parameter": {"key": "...", "name": "..."},
          "value": [
            {"date": <epoch_ms>, "value": "...", "quality": "G"},
            ...
          ]
        }

    Returns:
        (station_meta: dict, validated_records: list[SMHIObservationPoint],
         validation_failures: int)
    """

    def parse(
        self,
        raw: dict[str, Any],
        station_id: int,
        parameter_id: int,
    ) -> tuple[dict[str, Any], list[SMHIObservationPoint], int]:
        station_block  = raw.get("station",   {})
        parameter_block = raw.get("parameter", {})
        value_list     = raw.get("value",      [])

        station_meta = {
            "station_id":     station_id,
            "station_name":   station_block.get("name", ""),
            "parameter_id":   parameter_id,
            "parameter_name": parameter_block.get("name", ""),
            "latitude":       float(station_block.get("latitude",  0.0)),
            "longitude":      float(station_block.get("longitude", 0.0)),
        }

        validated: list[SMHIObservationPoint] = []
        failures = 0

        for raw_point in value_list:
            try:
                point = SMHIObservationPoint(
                    date_epoch_ms=int(raw_point["date"]),
                    value_raw=str(raw_point["value"]),
                    quality=str(raw_point.get("quality", "G")),
                )
                validated.append(point)
            except (ValidationError, KeyError, TypeError, ValueError) as exc:
                failures += 1
                logger.warning(
                    "Observation validation failure | station=%d param=%d | %s",
                    station_id, parameter_id, exc,
                )

        logger.info(
            "Parsed station=%d param=%d | total=%d validated=%d failures=%d",
            station_id, parameter_id,
            len(value_list), len(validated), failures,
        )
        return station_meta, validated, failures


# ---------------------------------------------------------------------------
# DataFrame Builder
# ---------------------------------------------------------------------------

def _build_observation_dataframe(
    records: list[SMHIObservationPoint],
    station_id: int,
    zone: str,
    parameter_id: int,
) -> pd.DataFrame:
    """
    Convert validated observation records into a typed, anomaly-tagged DataFrame.

    Output schema:
        timestamp_utc    : pd.Timestamp (UTC-aware)
        zone             : str          (e.g. 'SE3')
        station_id       : int
        temperature_c    : float | NaN  (present iff parameter_id == 1)
        wind_speed_ms    : float | NaN  (present iff parameter_id == 4)
        quality_flag     : str
        is_anomaly       : bool
    """
    if not records:
        logger.warning(
            "No validated records to build DataFrame for station=%d param=%d.",
            station_id, parameter_id,
        )
        return _empty_weather_df(parameter_id)

    rows = [
        {
            "timestamp_utc": r.timestamp_utc,
            "zone":          zone,
            "station_id":    station_id,
            "value":         r.value_float,
            "quality_flag":  r.quality,
        }
        for r in records
    ]
    df = pd.DataFrame(rows)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df.sort_values("timestamp_utc", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Rename value column to semantically typed name and flag anomalies
    if parameter_id == PARAM_TEMPERATURE:
        df.rename(columns={"value": "temperature_c"}, inplace=True)
        df["wind_speed_ms"] = float("nan")
        df["is_anomaly"] = _flag_temperature_anomalies(df["temperature_c"])
    else:
        df.rename(columns={"value": "wind_speed_ms"}, inplace=True)
        df["temperature_c"] = float("nan")
        df["is_anomaly"] = _flag_wind_anomalies(df["wind_speed_ms"])

    anomaly_count = int(df["is_anomaly"].sum())
    if anomaly_count:
        logger.warning(
            "⚡ %d / %d weather records flagged as anomalous "
            "(station=%d zone=%s param=%d)",
            anomaly_count, len(df), station_id, zone, parameter_id,
        )
    return df[
        ["timestamp_utc", "zone", "station_id",
         "temperature_c", "wind_speed_ms", "quality_flag", "is_anomaly"]
    ]


def _empty_weather_df(parameter_id: int) -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "timestamp_utc", "zone", "station_id",
        "temperature_c", "wind_speed_ms", "quality_flag", "is_anomaly",
    ])


# ---------------------------------------------------------------------------
# Main Fetcher Client
# ---------------------------------------------------------------------------

class SMHIWeatherFetcher(BaseHTTPClient):
    """
    Production SMHI Open Data Meteorologiska Observationer ingestion client.

    Inherits:
      - Resilient async/sync HTTP session from BaseHTTPClient
      - Per-host circuit breaker
      - Exponential backoff with full jitter

    Adds:
      - Station-parameter routing with explicit zone assignment
      - Pydantic v2 validation at the ingestion boundary
      - Physics-bound anomaly tagging (without record loss)
      - Immutable gzip-compressed Bronze snapshot + manifest on every fetch
      - Conservative 1.0 s inter-request delay (SMHI has no published limit)

    Output contract (what downstream silver_alignment.py expects):
        fetch_station_parameter() →
            pd.DataFrame[timestamp_utc, zone, station_id,
                         temperature_c, wind_speed_ms, quality_flag, is_anomaly]

        fetch_all_zone_stations() →
            dict[str, pd.DataFrame]  (keyed by zone label, e.g. 'SE3')

    Usage:
        with SMHIWeatherFetcher() as fetcher:
            df_temp = fetcher.fetch_station_parameter(
                station_id=98210,
                parameter_id=PARAM_TEMPERATURE,
            )
            all_weather = fetcher.fetch_all_zone_stations()
    """

    def __init__(
        self,
        bronze_dir: Path = Path("data/bronze/weather"),
        request_timeout_seconds: float = 60.0,   # archive responses can be large
        connect_timeout_seconds: float = 10.0,
        max_retries: int = DEFAULT_MAX_RETRIES,
        inter_request_delay: float = SMHI_REQUEST_DELAY_SECONDS,
    ) -> None:
        super().__init__(
            base_url=SMHI_BASE_URL,
            max_retries=max_retries,
            request_timeout_seconds=request_timeout_seconds,
            connect_timeout_seconds=connect_timeout_seconds,
            circuit_breaker_threshold=3,          # public API — open quickly
            circuit_breaker_recovery_seconds=60.0,
            http2=False,                          # SMHI does not support HTTP/2
        )
        self._bronze = WeatherBronzeWriter(bronze_dir)
        self._parser = SMHIResponseParser()
        self._inter_request_delay = inter_request_delay

        logger.info(
            "SMHIWeatherFetcher initialised | bronze=%s | delay=%.1fs | "
            "stations=%s",
            bronze_dir,
            inter_request_delay,
            list(STATION_ZONE_MAP.keys()),
        )

    # ------------------------------------------------------------------
    # Public Sync API
    # ------------------------------------------------------------------

    def fetch_station_parameter(
        self,
        station_id: int,
        parameter_id: int,
        period: str = "corrected-archive",
    ) -> pd.DataFrame:
        """
        Fetch and persist one parameter for one station.

        Args:
            station_id:   SMHI integer station key (e.g. 98210 for Stockholm).
            parameter_id: SMHI parameter ID — use PARAM_TEMPERATURE or
                          PARAM_WIND_SPEED module constants.
            period:       SMHI period string.  'corrected-archive' returns the
                          full quality-controlled historical record.

        Returns:
            pd.DataFrame with columns:
                [timestamp_utc, zone, station_id, temperature_c,
                 wind_speed_ms, quality_flag, is_anomaly]
            Empty DataFrame with correct schema on total failure.

        Side effects:
            Writes one .json.gz Bronze snapshot + .manifest.json to bronze_dir.
        """
        zone = STATION_ZONE_MAP.get(station_id)
        if zone is None:
            raise ValueError(
                f"Station ID {station_id} is not in the canonical zone map. "
                f"Known stations: {list(STATION_ZONE_MAP.keys())}. "
                "Add it to STATION_ZONE_MAP before fetching."
            )

        endpoint = (
            f"/parameter/{parameter_id}"
            f"/station/{station_id}"
            f"/period/{period}"
            "/data.json"
        )
        param_label = (
            "temperature" if parameter_id == PARAM_TEMPERATURE else "wind_speed"
        )

        logger.info(
            "Fetching SMHI | station=%d zone=%s param=%d (%s) period=%s",
            station_id, zone, parameter_id, param_label, period,
        )

        # Step 1: HTTP fetch (retry + circuit-breaker via BaseHTTPClient)
        try:
            raw_response: dict[str, Any] = self._fetch_json(endpoint)
        except (RetryExhaustedError, NonRetryableHTTPError) as exc:
            logger.error(
                "SMHI fetch failed | station=%d param=%d | %s",
                station_id, parameter_id, exc,
            )
            return _empty_weather_df(parameter_id)

        # Step 2: Bronze write — ALWAYS before validation so raw payload is preserved
        bronze_path = self._bronze.write(
            payload=raw_response,
            station_id=station_id,
            zone=zone,
            parameter_id=parameter_id,
            period=period,
        )

        # Step 3: Parse + Pydantic v2 validation at the ingestion boundary
        station_meta, validated_records, validation_failures = self._parser.parse(
            raw=raw_response,
            station_id=station_id,
            parameter_id=parameter_id,
        )

        # Step 4: Build typed, anomaly-tagged DataFrame
        df = _build_observation_dataframe(
            records=validated_records,
            station_id=station_id,
            zone=zone,
            parameter_id=parameter_id,
        )

        # Step 5: Write manifest (after DataFrame build so we have row counts)
        obs_start = (
            str(df["timestamp_utc"].min()) if not df.empty else None
        )
        obs_end = (
            str(df["timestamp_utc"].max()) if not df.empty else None
        )
        manifest = SMHIBronzeManifest(
            station_id=station_id,
            station_name=station_meta.get("station_name", ""),
            bidding_zone=zone,
            parameter_id=parameter_id,
            parameter_name=station_meta.get("parameter_name", ""),
            latitude=station_meta.get("latitude", 0.0),
            longitude=station_meta.get("longitude", 0.0),
            records_total=len(validated_records) + validation_failures,
            records_validation_failed=validation_failures,
            anomalies_flagged=int(df["is_anomaly"].sum()) if not df.empty else 0,
            obs_start_utc=obs_start,
            obs_end_utc=obs_end,
            bronze_path=str(bronze_path),
        )
        self._bronze.write_manifest(manifest, bronze_path)

        logger.info(
            "Fetch complete | station=%d zone=%s param=%d | "
            "%d rows | %d anomalies | %d validation failures",
            station_id, zone, parameter_id,
            len(df), manifest.anomalies_flagged, validation_failures,
        )
        return df

    def fetch_all_zone_stations(
        self,
        parameter_ids: Optional[list[int]] = None,
        period: str = "corrected-archive",
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch temperature and wind speed for all four canonical zone stations.

        Iterates STATION_ZONE_MAP with a conservative inter-request delay to
        avoid hammering the SMHI endpoint.  Each (station, parameter) pair
        produces one Bronze snapshot.  The two parameters per station are then
        merged on timestamp_utc so that downstream silver_alignment receives a
        single weather DataFrame per zone.

        Args:
            parameter_ids: List of SMHI parameter IDs to fetch.
                           Defaults to [PARAM_TEMPERATURE, PARAM_WIND_SPEED].
            period:        SMHI period string (default: 'corrected-archive').

        Returns:
            dict keyed by zone label (e.g. 'SE3') mapping to a merged DataFrame:
                [timestamp_utc, zone, station_id, temperature_c,
                 wind_speed_ms, quality_flag_temp, quality_flag_wind, is_anomaly]
            Zones with total fetch failures are absent from the dict.
        """
        if parameter_ids is None:
            parameter_ids = [PARAM_TEMPERATURE, PARAM_WIND_SPEED]

        results: dict[str, list[pd.DataFrame]] = {}

        for station_id, zone in STATION_ZONE_MAP.items():
            results[zone] = []
            for param_id in parameter_ids:
                df = self.fetch_station_parameter(
                    station_id=station_id,
                    parameter_id=param_id,
                    period=period,
                )
                if not df.empty:
                    results[zone].append(df)
                # Conservative inter-request delay regardless of success/failure
                time.sleep(self._inter_request_delay)

        merged: dict[str, pd.DataFrame] = {}
        for zone, frames in results.items():
            if not frames:
                logger.error(
                    "All fetches failed for zone=%s — excluding from output.", zone
                )
                continue
            if len(frames) == 1:
                merged[zone] = frames[0]
            else:
                # Merge temperature + wind on timestamp; outer join preserves
                # all hourly rows even when one parameter has gaps.
                df_temp = next(
                    (f for f in frames if f["temperature_c"].notna().any()), None
                )
                df_wind = next(
                    (f for f in frames if f["wind_speed_ms"].notna().any()), None
                )
                if df_temp is not None and df_wind is not None:
                    merged[zone] = _merge_parameter_frames(df_temp, df_wind, zone)
                else:
                    # Fallback: return the single available frame
                    merged[zone] = frames[0]

        logger.info(
            "fetch_all_zone_stations complete | zones_fetched=%s",
            list(merged.keys()),
        )
        return merged

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    @retry_with_backoff(
        max_retries=DEFAULT_MAX_RETRIES,
        base_backoff=DEFAULT_BASE_BACKOFF_SECONDS,
        max_backoff=DEFAULT_MAX_BACKOFF_SECONDS,
    )
    def _fetch_json(self, endpoint: str) -> dict[str, Any]:
        """
        Execute a single GET request and return the parsed JSON body.

        The @retry_with_backoff decorator (inherited from client.py) handles
        429 / 5xx retries with exponential backoff + full jitter.
        The circuit breaker is checked inside BaseHTTPClient.request().
        """
        response = self.request("GET", endpoint)
        return response.json()


# ---------------------------------------------------------------------------
# Parameter Frame Merger
# ---------------------------------------------------------------------------

def _merge_parameter_frames(
    df_temp: pd.DataFrame,
    df_wind: pd.DataFrame,
    zone: str,
) -> pd.DataFrame:
    """
    Outer-merge temperature and wind DataFrames on timestamp_utc.

    Quality flag columns are disambiguated to quality_flag_temp /
    quality_flag_wind so that downstream code can distinguish the source
    quality assessment for each measurement axis.

    is_anomaly is True if EITHER parameter observation is anomalous,
    propagating maximum conservatism into the Silver anomaly flag.
    """
    df_t = df_temp[[
        "timestamp_utc", "zone", "station_id", "temperature_c", "quality_flag", "is_anomaly"
    ]].rename(columns={
        "quality_flag": "quality_flag_temp",
        "is_anomaly":   "is_anomaly_temp",
    })

    df_w = df_wind[[
        "timestamp_utc", "wind_speed_ms", "quality_flag", "is_anomaly"
    ]].rename(columns={
        "quality_flag": "quality_flag_wind",
        "is_anomaly":   "is_anomaly_wind",
    })

    df_merged = pd.merge(df_t, df_w, on="timestamp_utc", how="outer")

    # Propagate zone / station_id into rows that came only from one side
    df_merged["zone"]       = df_merged["zone"].ffill().bfill()
    df_merged["station_id"] = df_merged["station_id"].ffill().bfill()

    # Unified anomaly flag: anomalous if either parameter is flagged
    df_merged["is_anomaly"] = (
        df_merged["is_anomaly_temp"].fillna(False) |
        df_merged["is_anomaly_wind"].fillna(False)
    )

    df_merged.drop(columns=["is_anomaly_temp", "is_anomaly_wind"], inplace=True)
    df_merged.sort_values("timestamp_utc", inplace=True)
    df_merged.reset_index(drop=True, inplace=True)

    logger.debug(
        "Merged temperature+wind for zone=%s | %d total hourly rows",
        zone, len(df_merged),
    )
    return df_merged


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Standalone execution:  python -m src.ingestion.weather_fetcher

    Fetches corrected-archive temperature and wind speed for all four
    canonical zone stations and prints summary statistics to stdout.
    Outputs are written to data/bronze/weather/.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch SMHI weather observations for Swedish bidding zones."
    )
    parser.add_argument(
        "--station-id",
        type=int,
        default=None,
        help=(
            "Single SMHI station ID to fetch (optional). "
            "If omitted, all four zone stations are fetched."
        ),
    )
    parser.add_argument(
        "--parameter-id",
        type=int,
        choices=[PARAM_TEMPERATURE, PARAM_WIND_SPEED],
        default=None,
        help="SMHI parameter ID: 1=temperature, 4=wind speed (default: both).",
    )
    parser.add_argument(
        "--period",
        type=str,
        default="corrected-archive",
        help="SMHI period string (default: corrected-archive).",
    )
    parser.add_argument(
        "--bronze-dir",
        type=Path,
        default=Path("data/bronze/weather"),
        help="Bronze output directory (default: data/bronze/weather).",
    )
    args = parser.parse_args()

    with SMHIWeatherFetcher(bronze_dir=args.bronze_dir) as fetcher:
        if args.station_id is not None:
            param_ids = (
                [args.parameter_id]
                if args.parameter_id
                else [PARAM_TEMPERATURE, PARAM_WIND_SPEED]
            )
            for pid in param_ids:
                df = fetcher.fetch_station_parameter(
                    station_id=args.station_id,
                    parameter_id=pid,
                    period=args.period,
                )
                print(f"\nStation {args.station_id} | param={pid} | rows={len(df)}")
                if not df.empty:
                    print(df.tail(5).to_string(index=False))
        else:
            all_weather = fetcher.fetch_all_zone_stations(period=args.period)
            print("\n======================================================================")
            print("SMHI Weather Fetch Summary")
            print("======================================================================")
            for zone, df in all_weather.items():
                temp_cov = 100 * df["temperature_c"].notna().mean() if not df.empty else 0
                wind_cov = 100 * df["wind_speed_ms"].notna().mean() if not df.empty else 0
                print(
                    f"  {zone}: {len(df):>6} rows | "
                    f"temp_coverage={temp_cov:.1f}% | "
                    f"wind_coverage={wind_cov:.1f}%"
                )
            print("======================================================================")
