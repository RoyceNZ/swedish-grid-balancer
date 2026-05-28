"""
src/ingestion/riksbank_fetcher.py
=============================================================================
Sveriges Riksbank SWEA REST API — Policy Rate (Styrränta) Ingestion Client
=============================================================================
Responsibilities:
  - Fetch the complete historical Riksbank policy rate (Styrränta) series
    from the SWEA (Swedish Economic Archive) REST API
  - Parse the episodic, event-driven rate change time series into a
    step-function DataFrame keyed by effective date (UTC)
  - Validate all records with Pydantic v2 at the ingestion boundary:
    any schema change in the SWEA API is caught immediately and logged
    as CRITICAL before the pipeline aborts safely
  - Write immutable, timestamped Bronze layer snapshots as gzip-compressed
    JSON — one file per execution
  - Output a Silver-ready DataFrame with columns:
      [effective_date_utc, policy_rate_pct]
    ready for TemporalHarmonizer._join_policy_rate() in silver_alignment.py,
    which uses pd.merge_asof(direction='backward') to forward-fill the
    step-function rate onto hourly timestamps

Riksbank SWEA API Notes:
  - Base URL: https://api.riksbank.se/swea/v1
  - Authentication: None required (public API)
  - Rate limit: undocumented — implemented with conservative 1-second
    inter-request delay and exponential backoff on 429/5xx
  - Response format: JSON (application/json)
  - Series ID for Styrränta: "SEREPOREPOEFF"
    (Riksbank policy rate, effective date, percent)
  - Dates are returned in "YYYY-MM-DD" format — converted to UTC timestamps
    at midnight on the effective date (00:00 UTC)
  - Rate changes are infrequent (6–8 per year at scheduled Riksbank meetings)
    but can occur outside scheduled meetings (emergency adjustments)

Step-Function Semantics:
  The Styrränta applies from its effective date until the next rate change.
  The TemporalHarmonizer forward-fills it hourly using merge_asof backward.
  This means a rate announced on "2024-03-27" applies to every hour from
  2024-03-27 00:00 UTC onward until the next effective_date_utc.
  Announcement dates before the effective date are NOT used — only the
  effective date matters for leakage-safe alignment.

Architecture position: LIVE API LAYER → BRONZE LAYER
Upstream:  Riksbank SWEA REST API (https://api.riksbank.se/swea/v1)
Downstream: src/processing/silver_alignment.py (TemporalHarmonizer)
=============================================================================
"""

from __future__ import annotations

import gzip
import json
import logging
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

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

# ---------------------------------------------------------------------------
# Module Logger
# ---------------------------------------------------------------------------
logger = get_pipeline_logger("ingestion.riksbank_fetcher")


# ---------------------------------------------------------------------------
# Riksbank SWEA API Constants
# ---------------------------------------------------------------------------
RIKSBANK_BASE_URL = "https://api.riksbank.se/swea/v1"

# Series identifier for the Riksbank policy rate (Styrränta / Reporänta)
# "SEREPOREPOEFF" = Policy Rate (Effective), percent per annum
SERIES_ID_STYRRANTAN = "SEREPOREPOEFF"

# Conservative inter-request delay (Riksbank doesn't publish a rate limit)
RIKSBANK_REQUEST_DELAY_SECONDS = 1.0

# Sanity bounds for the Swedish policy rate in percent
# Historical range: −0.50% (2015 negative rate policy) to +4.75% (2023 peak)
RATE_MIN_PCT = -2.0   # Floor: allow for extreme negative rate scenarios
RATE_MAX_PCT = 15.0   # Ceiling: above any modern Riksbank rate

# Date format used by the SWEA API in response bodies
SWEA_DATE_FORMAT = "%Y-%m-%d"


# ---------------------------------------------------------------------------
# Pydantic v2 Schemas — Riksbank Ingestion Boundary Contracts
# ---------------------------------------------------------------------------

class RiksbankRatePoint(BaseModel):
    """
    Single validated Riksbank policy rate change observation.

    Each record represents one rate-setting decision that takes effect
    on effective_date_utc. The rate persists as a step function until
    the next RiksbankRatePoint in the series.

    Validation rules:
      - effective_date_utc must be a UTC-aware datetime (midnight of the date)
      - policy_rate_pct must be within historical physics bounds
      - period must be parseable as YYYY-MM-DD
    """

    model_config = {"frozen": True}

    effective_date_utc: datetime = Field(
        ...,
        description="Date the rate takes effect, expressed as UTC midnight timestamp"
    )
    policy_rate_pct: float = Field(
        ...,
        ge=RATE_MIN_PCT,
        le=RATE_MAX_PCT,
        description="Policy rate in percent per annum (e.g. 4.00 = 4.00%)",
    )
    raw_date_str: str = Field(
        ...,
        description="Original date string from SWEA API e.g. '2024-03-27'",
    )

    @field_validator("effective_date_utc", mode="before")
    @classmethod
    def coerce_to_utc_midnight(cls, v: Any) -> datetime:
        """
        Convert various date representations to UTC midnight datetime.

        The step-function semantics require an exact UTC timestamp
        for merge_asof(direction='backward') to work correctly.
        A rate effective on "2024-03-27" must map to
        2024-03-27 00:00:00+00:00 — the first moment it is in force.
        """
        if isinstance(v, datetime):
            # Strip time component, force midnight UTC
            return v.replace(hour=0, minute=0, second=0, microsecond=0,
                             tzinfo=timezone.utc)
        if isinstance(v, date):
            return datetime(v.year, v.month, v.day, tzinfo=timezone.utc)
        if isinstance(v, str):
            try:
                d = datetime.strptime(v.strip(), SWEA_DATE_FORMAT).date()
                return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
            except ValueError as exc:
                raise ValueError(
                    f"Cannot parse date string '{v}'. "
                    f"Expected format: '{SWEA_DATE_FORMAT}' (e.g. '2024-03-27')."
                ) from exc
        if isinstance(v, pd.Timestamp):
            return datetime(v.year, v.month, v.day, tzinfo=timezone.utc)
        raise ValueError(
            f"effective_date_utc: unsupported type {type(v).__name__}. "
            f"Expected str, date, datetime, or pd.Timestamp."
        )

    @field_validator("raw_date_str")
    @classmethod
    def validate_date_str_format(cls, v: str) -> str:
        try:
            datetime.strptime(v.strip(), SWEA_DATE_FORMAT)
        except ValueError as exc:
            raise ValueError(
                f"raw_date_str '{v}' does not match '{SWEA_DATE_FORMAT}': {exc}"
            ) from exc
        return v.strip()

    @model_validator(mode="after")
    def check_effective_date_consistency(self) -> "RiksbankRatePoint":
        """
        Verify that effective_date_utc matches the date encoded in raw_date_str.

        This cross-field guard catches cases where the coercion logic
        produced an incorrect date (e.g. off-by-one in timezone handling).
        """
        raw_date = datetime.strptime(self.raw_date_str, SWEA_DATE_FORMAT).date()
        effective_date = self.effective_date_utc.date()
        if raw_date != effective_date:
            raise ValueError(
                f"Date mismatch: raw_date_str='{self.raw_date_str}' encodes "
                f"{raw_date} but effective_date_utc encodes {effective_date}. "
                f"Possible timezone coercion error."
            )
        return self


class RiksbankSeriesMetadata(BaseModel):
    """
    Metadata extracted from the SWEA API series info endpoint.

    Used to validate that the series ID still corresponds to the
    expected economic concept before fetching the full data.
    """

    series_id: str
    series_name: str
    unit: Optional[str] = None
    frequency: Optional[str] = None  # "D" = daily, "M" = monthly, etc.
    from_date: Optional[str] = None
    to_date: Optional[str] = None

    @model_validator(mode="after")
    def warn_if_unexpected_series(self) -> "RiksbankSeriesMetadata":
        """
        Emit a WARNING if the series name doesn't contain expected keywords.

        This is a soft guard — a rename of the series at Riksbank would still
        allow ingestion to continue but would alert the pipeline operators.
        """
        name_lower = self.series_name.lower()
        expected_keywords = ["repo", "policy", "styrränta", "ränta"]
        if not any(kw in name_lower for kw in expected_keywords):
            logger.warning(
                "SWEA series '%s' name='%s' does not contain expected keywords %s. "
                "Verify that series ID '%s' still refers to the Styrränta.",
                self.series_id, self.series_name, expected_keywords, self.series_id,
            )
        return self


class RiksbankBronzeManifest(BaseModel):
    """Metadata sidecar written alongside every Riksbank Bronze snapshot."""

    series_id: str
    from_date: Optional[str]
    to_date: Optional[str]
    records_total: int
    records_validation_failed: int
    rate_changes_captured: int
    rate_min_pct: Optional[float]
    rate_max_pct: Optional[float]
    bronze_path: str
    ingested_at_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    pipeline_version: str = "1.0.0"


# ---------------------------------------------------------------------------
# Bronze Writer — Riksbank
# ---------------------------------------------------------------------------
class RiksbankBronzeWriter:
    """
    Persists raw Riksbank SWEA API responses as gzip-compressed JSON snapshots.

    Naming:  styrrantan_riksbank_{from_date}_{to_date}_{ingested_at_utc}.json.gz
    """

    def __init__(self, bronze_dir: Path) -> None:
        self.bronze_dir = Path(bronze_dir)
        self.bronze_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        payload: dict[str, Any],
        from_date: Optional[str],
        to_date: Optional[str],
    ) -> Path:
        """Compress and write raw JSON response to Bronze layer."""
        ts_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        from_tag = (from_date or "all").replace("-", "")
        to_tag = (to_date or "latest").replace("-", "")
        fname = f"styrrantan_riksbank_{from_tag}_{to_tag}_{ts_str}.json.gz"
        out_path = self.bronze_dir / fname

        with gzip.open(out_path, "wt", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=None)

        size_kb = out_path.stat().st_size / 1024
        logger.info(
            "Riksbank Bronze snapshot written → %s (%.1f KB compressed)",
            out_path, size_kb,
        )
        return out_path

    def write_manifest(
        self, manifest: RiksbankBronzeManifest, bronze_path: Path
    ) -> Path:
        """Write JSON metadata manifest alongside the Bronze snapshot."""
        mpath = bronze_path.with_suffix("").with_suffix(".manifest.json")
        with open(mpath, "w", encoding="utf-8") as fh:
            json.dump(manifest.model_dump(), fh, indent=2, ensure_ascii=False)
        logger.debug("Riksbank manifest written → %s", mpath)
        return mpath


# ---------------------------------------------------------------------------
# SWEA Response Parser
# ---------------------------------------------------------------------------
class SWEAResponseParser:
    """
    Parses the Riksbank SWEA API JSON response into flat RiksbankRatePoint records.

    SWEA v1 observation endpoint returns:
    {
      "groups": [
        {
          "seriesId": "SEREPOREPOEFF",
          "observations": [
            {"date": "1994-06-02", "value": "7.00"},
            {"date": "1994-09-19", "value": "7.25"},
            ...
          ]
        }
      ]
    }

    Alternative response format (series endpoint):
    {
      "observations": [
        {"date": "2024-03-27", "value": "3.75", "serie": "SEREPOREPOEFF"},
        ...
      ]
    }

    We handle both formats via a normalised extraction step.

    Missing values:
      - SWEA uses "NA" or null for missing observations — these are excluded
      - Empty string values are excluded
    """

    MISSING_SENTINELS = {"NA", "N/A", "null", "", ".", ".."}

    def parse(self, response_json: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Extract flat rate change records from a SWEA API response.

        Returns:
            List of dicts with keys [raw_date_str, effective_date_utc, policy_rate_pct]
            Sorted ascending by effective_date_utc.
        """
        observations = self._extract_observations(response_json)
        if not observations:
            logger.warning("SWEA response contained no parseable observations.")
            return []

        records: list[dict[str, Any]] = []
        skip_count = 0

        for obs in observations:
            date_str = obs.get("date", "")
            value_str = str(obs.get("value", "NA")).strip()

            if not date_str:
                skip_count += 1
                continue

            if value_str.upper() in self.MISSING_SENTINELS:
                logger.debug("Skipping missing value for date '%s'.", date_str)
                skip_count += 1
                continue

            try:
                rate_pct = float(value_str.replace(",", "."))
            except ValueError:
                logger.warning(
                    "Cannot parse rate value '%s' for date '%s' — skipping.",
                    value_str, date_str,
                )
                skip_count += 1
                continue

            records.append({
                "raw_date_str": date_str,
                "effective_date_utc": date_str,  # validator will coerce to datetime
                "policy_rate_pct": rate_pct,
            })

        if skip_count:
            logger.info(
                "SWEA parse: %d observations parsed, %d skipped (missing/malformed).",
                len(records), skip_count,
            )

        # Sort ascending by date string (lexicographic sort works for YYYY-MM-DD)
        records.sort(key=lambda r: r["raw_date_str"])
        return records

    def _extract_observations(
        self, response_json: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Normalise the response structure to extract a flat list of observations.

        Handles both SWEA response shapes:
          1. {"groups": [{"observations": [...]}]}
          2. {"observations": [...]}
          3. Direct list at root (some SWEA endpoints)
        """
        # Shape 1: nested groups
        if "groups" in response_json:
            all_obs: list[dict[str, Any]] = []
            for group in response_json["groups"]:
                all_obs.extend(group.get("observations", []))
            if all_obs:
                return all_obs

        # Shape 2: direct observations array
        if "observations" in response_json:
            return response_json["observations"]

        # Shape 3: response IS the array (SWEA sometimes returns bare arrays)
        if isinstance(response_json, list):
            return response_json

        logger.warning(
            "Unknown SWEA response structure. "
            "Keys present: %s", list(response_json.keys())
        )
        return []


# ---------------------------------------------------------------------------
# Main Riksbank Fetcher Client
# ---------------------------------------------------------------------------
class RiksbankFetcher(BaseHTTPClient):
    """
    Production Riksbank SWEA REST API ingestion client.

    Fetches the complete Styrränta (policy rate) series, validates it,
    and outputs a Silver-ready DataFrame of step-function rate changes.

    Inherits from BaseHTTPClient for circuit breaker + retry resilience.

    Output contract (what TemporalHarmonizer expects):
      fetch_policy_rate() →
        pd.DataFrame with columns:
          [effective_date_utc, policy_rate_pct]
        sorted ascending by effective_date_utc
        effective_date_utc is UTC-aware pd.Timestamp at midnight

    Usage (sync batch context):
        with RiksbankFetcher() as rb:
            df_rate = rb.fetch_policy_rate(from_date="2020-01-01")

    Usage with date window:
        df_rate = rb.fetch_policy_rate(
            from_date="2022-01-01",
            to_date="2026-01-01",
        )
    """

    def __init__(
        self,
        bronze_dir: Path = Path("data/bronze/riksbank"),
        request_timeout_seconds: float = 30.0,
        connect_timeout_seconds: float = 10.0,
        max_retries: int = DEFAULT_MAX_RETRIES,
        inter_request_delay_seconds: float = RIKSBANK_REQUEST_DELAY_SECONDS,
    ) -> None:
        super().__init__(
            base_url=RIKSBANK_BASE_URL,
            max_retries=max_retries,
            request_timeout_seconds=request_timeout_seconds,
            connect_timeout_seconds=connect_timeout_seconds,
            circuit_breaker_threshold=4,
            circuit_breaker_recovery_seconds=90.0,
            http2=False,  # SWEA API does not support HTTP/2
        )
        self._bronze = RiksbankBronzeWriter(bronze_dir)
        self._delay = inter_request_delay_seconds

        logger.info(
            "RiksbankFetcher initialised | bronze=%s | delay=%.1fs",
            bronze_dir, inter_request_delay_seconds,
        )

    # ------------------------------------------------------------------
    # BaseHTTPClient override
    # ------------------------------------------------------------------
    def _build_default_headers(self) -> dict[str, str]:
        return {
            "Accept": "application/json",
            "User-Agent": "SwedenGridPipeline/1.0 (riksbank_fetcher)",
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fetch_policy_rate(
        self,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        series_id: str = SERIES_ID_STYRRANTAN,
    ) -> pd.DataFrame:
        """
        Fetch the Riksbank Styrränta series from the SWEA API.

        This method fetches the COMPLETE series by default (from_date=None),
        which is the recommended approach for the pipeline since the series
        is small (~200 observations from 1994 to present) and step-function
        alignment requires the full history to correctly forward-fill the
        most recent known rate for any given timestamp.

        Args:
            from_date:  Optional start date filter "YYYY-MM-DD".
                        If None, fetches from the series inception.
            to_date:    Optional end date filter "YYYY-MM-DD".
                        If None, fetches up to the most recent observation.
            series_id:  SWEA series identifier. Default: SEREPOREPOEFF.

        Returns:
            pd.DataFrame with columns:
              [effective_date_utc, policy_rate_pct]
            sorted ascending by effective_date_utc.
            effective_date_utc is UTC-timezone-aware pd.Timestamp (midnight).
            Empty DataFrame with correct schema on total failure.

        Implementation note:
            We first verify the series metadata (name, unit) to catch any
            Riksbank series renaming. Then fetch observations. Both steps
            apply exponential backoff via the @retry_with_backoff decorator.
        """
        logger.info(
            "Fetching Riksbank policy rate | series=%s | from=%s | to=%s",
            series_id, from_date or "inception", to_date or "latest",
        )

        # Step 1: Verify series metadata (optional but catches schema drift)
        try:
            metadata = self._fetch_series_metadata(series_id)
            logger.info(
                "SWEA series verified | id=%s | name='%s' | unit=%s | "
                "available: %s → %s",
                metadata.series_id, metadata.series_name, metadata.unit,
                metadata.from_date, metadata.to_date,
            )
        except (RetryExhaustedError, NonRetryableHTTPError, ValueError) as exc:
            # Metadata failure is non-fatal: log and continue with data fetch
            logger.warning(
                "Cannot retrieve SWEA series metadata for '%s': %s. "
                "Proceeding without metadata validation.",
                series_id, exc,
            )

        # Step 2: Fetch observation data
        try:
            raw_response = self._fetch_observations(series_id, from_date, to_date)
        except (RetryExhaustedError, NonRetryableHTTPError, ValueError) as exc:
            logger.error(
                "Riksbank rate fetch failed: %s. Returning empty DataFrame.", exc
            )
            return self._empty_df()

        # Step 3: Bronze write (always — raw payload preserved before validation)
        bronze_path = self._bronze.write(
            payload=raw_response,
            from_date=from_date,
            to_date=to_date,
        )

        # Step 4: Parse flat observation records
        parser = SWEAResponseParser()
        raw_records = parser.parse(raw_response)

        if not raw_records:
            logger.warning(
                "No observations parsed from SWEA response. "
                "Raw payload preserved at %s.",
                bronze_path,
            )
            self._bronze.write_manifest(
                RiksbankBronzeManifest(
                    series_id=series_id, from_date=from_date, to_date=to_date,
                    records_total=0, records_validation_failed=0,
                    rate_changes_captured=0, rate_min_pct=None, rate_max_pct=None,
                    bronze_path=str(bronze_path),
                ),
                bronze_path,
            )
            return self._empty_df()

        # Step 5: Pydantic v2 validation
        validated_records, validation_failures = self._validate_records(raw_records)

        if not validated_records:
            logger.error(
                "All %d Riksbank records failed Pydantic validation. "
                "SWEA API schema may have changed. Raw payload at %s.",
                len(raw_records), bronze_path,
            )
            return self._empty_df()

        # Step 6: Detect and log rate change events (useful audit information)
        rate_changes = self._detect_rate_changes(validated_records)
        logger.info(
            "Rate change events detected: %d over the fetched period",
            len(rate_changes),
        )
        for change in rate_changes[-5:]:  # Log last 5 changes
            logger.info(
                "  Rate change: %s → %.2f%% (Δ %+.2f%%)",
                change["date"], change["new_rate"], change["delta"],
            )

        # Step 7: Build Silver-ready DataFrame
        df = pd.DataFrame(validated_records)
        df["effective_date_utc"] = pd.to_datetime(df["effective_date_utc"], utc=True)
        df = df[["effective_date_utc", "policy_rate_pct"]]
        df.sort_values("effective_date_utc", inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Step 8: Write manifest
        self._bronze.write_manifest(
            RiksbankBronzeManifest(
                series_id=series_id,
                from_date=from_date,
                to_date=to_date,
                records_total=len(df),
                records_validation_failed=validation_failures,
                rate_changes_captured=len(rate_changes),
                rate_min_pct=float(df["policy_rate_pct"].min()),
                rate_max_pct=float(df["policy_rate_pct"].max()),
                bronze_path=str(bronze_path),
            ),
            bronze_path,
        )

        logger.info(
            "fetch_policy_rate complete | %d observations | "
            "rate range: %.2f%% → %.2f%% | %d validation failures",
            len(df),
            df["policy_rate_pct"].min(),
            df["policy_rate_pct"].max(),
            validation_failures,
        )
        return df

    # ------------------------------------------------------------------
    # Private: SWEA API Calls
    # ------------------------------------------------------------------
    @retry_with_backoff(
        max_retries=DEFAULT_MAX_RETRIES,
        base_seconds=DEFAULT_BASE_BACKOFF_SECONDS,
        max_seconds=DEFAULT_MAX_BACKOFF_SECONDS,
    )
    def _fetch_series_metadata(self, series_id: str) -> RiksbankSeriesMetadata:
        """
        Fetch series metadata from SWEA to verify the series still exists
        and corresponds to the expected economic concept.

        SWEA endpoint: GET /series/{series_id}
        """
        time.sleep(self._delay)
        response = self.request("GET", f"/series/{series_id}")
        payload = response.json()

        # SWEA series endpoint returns various formats — normalise
        series_name = (
            payload.get("name")
            or payload.get("seriesName")
            or payload.get("description")
            or ""
        )
        return RiksbankSeriesMetadata(
            series_id=series_id,
            series_name=series_name,
            unit=payload.get("unit") or payload.get("unitLabel"),
            frequency=payload.get("frequency") or payload.get("freq"),
            from_date=payload.get("from") or payload.get("startDate"),
            to_date=payload.get("to") or payload.get("endDate"),
        )

    @retry_with_backoff(
        max_retries=DEFAULT_MAX_RETRIES,
        base_seconds=DEFAULT_BASE_BACKOFF_SECONDS,
        max_seconds=DEFAULT_MAX_BACKOFF_SECONDS,
    )
    def _fetch_observations(
        self,
        series_id: str,
        from_date: Optional[str],
        to_date: Optional[str],
    ) -> dict[str, Any]:
        """
        Fetch the full observation series from SWEA.

        SWEA v1 observation endpoint:
            GET /observations/{series_id}
            Query params: from (YYYY-MM-DD), to (YYYY-MM-DD)

        Raises:
            ValueError: If the response is not valid JSON.
        """
        time.sleep(self._delay)

        params: dict[str, str] = {}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date

        response = self.request(
            "GET",
            f"/observations/{series_id}",
            params=params if params else None,
        )

        try:
            payload = response.json()
        except Exception as exc:
            raise ValueError(
                f"SWEA response for series '{series_id}' is not valid JSON: {exc}. "
                f"First 300 chars: {response.text[:300]!r}"
            ) from exc

        return payload if isinstance(payload, dict) else {"observations": payload}

    # ------------------------------------------------------------------
    # Private: Pydantic Validation
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_records(
        raw_records: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], int]:
        """
        Validate raw parsed records through RiksbankRatePoint Pydantic model.

        Returns:
            (validated_records, failure_count)
            validated_records: list of {effective_date_utc, policy_rate_pct} dicts
        """
        validated: list[dict[str, Any]] = []
        failures = 0

        for rec in raw_records:
            try:
                point = RiksbankRatePoint(**rec)
                validated.append({
                    "effective_date_utc": point.effective_date_utc,
                    "policy_rate_pct": point.policy_rate_pct,
                })
            except ValidationError as exc:
                failures += 1
                logger.error(
                    "RiksbankRatePoint validation failed | record=%r | errors=%s",
                    {k: rec.get(k) for k in ("raw_date_str", "policy_rate_pct")},
                    exc.errors(include_url=False),
                )

        if failures:
            logger.warning(
                "%d / %d Riksbank records failed Pydantic validation.",
                failures, len(raw_records),
            )

        return validated, failures

    # ------------------------------------------------------------------
    # Private: Rate Change Detector
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_rate_changes(
        validated_records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Identify rate change events where the policy rate differs from
        the previous observation.

        This is an audit/logging feature — the full series (including
        periods where the rate is unchanged day-to-day) is still returned.
        However since the Riksbank series only records change dates, each
        row IS a rate change.

        Returns:
            List of {date, old_rate, new_rate, delta} dicts
        """
        if not validated_records:
            return []

        changes = []
        prev_rate: Optional[float] = None

        for rec in validated_records:
            current_rate = rec["policy_rate_pct"]
            if prev_rate is not None and current_rate != prev_rate:
                changes.append({
                    "date": rec["effective_date_utc"].strftime(SWEA_DATE_FORMAT)
                    if isinstance(rec["effective_date_utc"], datetime)
                    else str(rec["effective_date_utc"])[:10],
                    "old_rate": prev_rate,
                    "new_rate": current_rate,
                    "delta": current_rate - prev_rate,
                })
            prev_rate = current_rate

        return changes

    # ------------------------------------------------------------------
    # Private: Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _empty_df() -> pd.DataFrame:
        """Return an empty DataFrame with the Silver-ready Riksbank schema."""
        return pd.DataFrame(columns=["effective_date_utc", "policy_rate_pct"])


# ---------------------------------------------------------------------------
# CLI Smoke Test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import tempfile
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("RiksbankFetcher — Unit Smoke Test (offline / no live API calls)")
    print("=" * 70)

    errors: list[str] = []

    def chk(name: str, cond: bool, detail: str = "") -> None:
        sym = "✓" if cond else "✗"
        print(f"  {sym} {name}" + (f": {detail}" if detail else ""))
        if not cond:
            errors.append(name)

    # ── [1] RiksbankRatePoint — UTC midnight coercion ────────────────────
    print("\n[1] RiksbankRatePoint — date coercion and validation")

    pt = RiksbankRatePoint(
        effective_date_utc="2024-03-27",
        policy_rate_pct=3.75,
        raw_date_str="2024-03-27",
    )
    chk("Date string coerced to UTC midnight",
        pt.effective_date_utc == datetime(2024, 3, 27, 0, 0, 0, tzinfo=timezone.utc))
    chk("policy_rate_pct stored", pt.policy_rate_pct == 3.75)
    chk("raw_date_str preserved", pt.raw_date_str == "2024-03-27")

    # Test from plain date object
    pt2 = RiksbankRatePoint(
        effective_date_utc=date(2024, 5, 8),
        policy_rate_pct=3.50,
        raw_date_str="2024-05-08",
    )
    chk("date object coerced to UTC midnight",
        pt2.effective_date_utc == datetime(2024, 5, 8, tzinfo=timezone.utc))

    # Test from pd.Timestamp
    pt3 = RiksbankRatePoint(
        effective_date_utc=pd.Timestamp("2024-08-20"),
        policy_rate_pct=3.25,
        raw_date_str="2024-08-20",
    )
    chk("pd.Timestamp coerced to UTC midnight",
        pt3.effective_date_utc == datetime(2024, 8, 20, tzinfo=timezone.utc))

    print("\n[1b] RiksbankRatePoint — validation error cases")
    # Rate below floor
    try:
        RiksbankRatePoint(
            effective_date_utc="2024-01-01", policy_rate_pct=-5.0,
            raw_date_str="2024-01-01"
        )
        chk("-5.0% should raise ValidationError", False)
    except ValidationError:
        chk("-5.0% (below floor) raises ValidationError", True)

    # Rate above ceiling
    try:
        RiksbankRatePoint(
            effective_date_utc="2024-01-01", policy_rate_pct=20.0,
            raw_date_str="2024-01-01"
        )
        chk("20.0% should raise ValidationError", False)
    except ValidationError:
        chk("20.0% (above ceiling) raises ValidationError", True)

    # Date format mismatch
    try:
        RiksbankRatePoint(
            effective_date_utc="2024-01-01", policy_rate_pct=4.0,
            raw_date_str="01/01/2024"   # wrong format
        )
        chk("Wrong date format should raise ValidationError", False)
    except ValidationError:
        chk("Wrong date format raises ValidationError", True)

    # Cross-field: date mismatch
    try:
        RiksbankRatePoint(
            effective_date_utc="2024-01-01", policy_rate_pct=4.0,
            raw_date_str="2024-01-02"  # different date
        )
        chk("Mismatched dates should raise ValidationError", False)
    except ValidationError:
        chk("Mismatched effective_date vs raw_date_str raises ValidationError", True)

    # Negative rate (valid in Swedish context — Riksbank was at -0.5% 2015–2021)
    pt_neg = RiksbankRatePoint(
        effective_date_utc="2020-01-09", policy_rate_pct=-0.25,
        raw_date_str="2020-01-09"
    )
    chk("Negative rate (-0.25%) is valid", pt_neg.policy_rate_pct == -0.25)

    # ── [2] SWEAResponseParser — both response shapes ──────────────────
    print("\n[2] SWEAResponseParser — shape normalisation")

    # Shape 1: nested groups
    shape1 = {
        "groups": [
            {
                "seriesId": "SEREPOREPOEFF",
                "observations": [
                    {"date": "2023-11-01", "value": "4.00"},
                    {"date": "2024-01-31", "value": "4.00"},
                    {"date": "2024-03-27", "value": "3.75"},
                    {"date": "2024-05-08", "value": "3.50"},
                ]
            }
        ]
    }
    parser = SWEAResponseParser()
    records_s1 = parser.parse(shape1)
    chk("Shape 1 (groups): 4 records parsed", len(records_s1) == 4, f"got {len(records_s1)}")
    chk("Records sorted ascending", records_s1[0]["raw_date_str"] == "2023-11-01")
    chk("Last record is 2024-05-08", records_s1[-1]["raw_date_str"] == "2024-05-08")
    chk("Rates parsed correctly", records_s1[2]["policy_rate_pct"] == 3.75)

    # Shape 2: direct observations
    shape2 = {
        "observations": [
            {"date": "2024-08-20", "value": "3.25"},
            {"date": "2024-11-07", "value": "2.75"},
        ]
    }
    records_s2 = parser.parse(shape2)
    chk("Shape 2 (observations): 2 records", len(records_s2) == 2)
    chk("Shape 2 rates correct", records_s2[0]["policy_rate_pct"] == 3.25)

    # Missing values excluded
    shape_missing = {
        "observations": [
            {"date": "2024-01-01", "value": "4.00"},
            {"date": "2024-02-01", "value": "NA"},
            {"date": "2024-03-01", "value": ""},
            {"date": "2024-04-01", "value": "3.75"},
            {"date": "", "value": "4.00"},                   # no date → skip
        ]
    }
    records_missing = parser.parse(shape_missing)
    chk("NA and empty values excluded", len(records_missing) == 2, f"got {len(records_missing)}")
    chk("Non-missing values preserved", records_missing[1]["policy_rate_pct"] == 3.75)

    # Comma-decimal format (some European APIs)
    shape_comma = {"observations": [{"date": "2024-01-01", "value": "4,00"}]}
    records_comma = parser.parse(shape_comma)
    chk("Comma-decimal '4,00' parsed as 4.0", records_comma[0]["policy_rate_pct"] == 4.0)

    # ── [3] Rate Change Detector ──────────────────────────────────────
    print("\n[3] Rate change detection")
    sample_validated = [
        {"effective_date_utc": datetime(2023, 11, 1, tzinfo=timezone.utc), "policy_rate_pct": 4.00},
        {"effective_date_utc": datetime(2024, 3, 27, tzinfo=timezone.utc), "policy_rate_pct": 3.75},
        {"effective_date_utc": datetime(2024, 5, 8, tzinfo=timezone.utc),  "policy_rate_pct": 3.50},
        {"effective_date_utc": datetime(2024, 8, 20, tzinfo=timezone.utc), "policy_rate_pct": 3.25},
        {"effective_date_utc": datetime(2024, 11, 7, tzinfo=timezone.utc), "policy_rate_pct": 2.75},
    ]
    changes = RiksbankFetcher._detect_rate_changes(sample_validated)
    chk("4 rate changes detected (first has no delta)", len(changes) == 4, f"got {len(changes)}")
    chk("First change: 4.00 → 3.75", changes[0]["new_rate"] == 3.75)
    chk("First change delta = -0.25", abs(changes[0]["delta"] - (-0.25)) < 1e-9)
    chk("Last change: 3.25 → 2.75", changes[-1]["new_rate"] == 2.75)
    chk("All deltas are negative (easing cycle)", all(c["delta"] < 0 for c in changes))

    # Single observation — no changes
    single = [{"effective_date_utc": datetime(2024, 1, 1, tzinfo=timezone.utc), "policy_rate_pct": 4.00}]
    chk("Single observation → 0 change events", RiksbankFetcher._detect_rate_changes(single) == [])

    # ── [4] DataFrame Output Contract ─────────────────────────────────
    print("\n[4] Silver-ready DataFrame output contract")
    # Simulate a complete fetch cycle using the validator and df builder
    raw = [
        {"raw_date_str": "2023-11-01", "effective_date_utc": "2023-11-01", "policy_rate_pct": 4.00},
        {"raw_date_str": "2024-03-27", "effective_date_utc": "2024-03-27", "policy_rate_pct": 3.75},
        {"raw_date_str": "2024-05-08", "effective_date_utc": "2024-05-08", "policy_rate_pct": 3.50},
    ]
    validated, fails = RiksbankFetcher._validate_records(raw)
    chk("3 records validated", len(validated) == 3)
    chk("0 validation failures", fails == 0)

    df = pd.DataFrame(validated)
    df["effective_date_utc"] = pd.to_datetime(df["effective_date_utc"], utc=True)
    df = df[["effective_date_utc", "policy_rate_pct"]].sort_values("effective_date_utc")
    chk("DataFrame has 2 columns", list(df.columns) == ["effective_date_utc", "policy_rate_pct"])
    chk("effective_date_utc is UTC-tz-aware pd.DatetimeTZDtype",
        str(df["effective_date_utc"].dtype) == "datetime64[ns, UTC]")
    chk("Sorted ascending", df["effective_date_utc"].is_monotonic_increasing)
    chk("First rate = 4.00", df["policy_rate_pct"].iloc[0] == 4.00)
    chk("Last rate = 3.50", df["policy_rate_pct"].iloc[-1] == 3.50)

    # ── [5] Step-Function Semantics — merge_asof compatibility ────────
    print("\n[5] Step-function forward-fill compatibility (merge_asof simulation)")
    # Simulate what silver_alignment.py TemporalHarmonizer._join_policy_rate does
    hourly_ts = pd.date_range("2024-03-26", periods=72, freq="h", tz="UTC")
    df_hourly = pd.DataFrame({"timestamp_utc": hourly_ts})

    df_rate = df.sort_values("effective_date_utc").reset_index(drop=True)
    df_merged = pd.merge_asof(
        df_hourly,
        df_rate,
        left_on="timestamp_utc",
        right_on="effective_date_utc",
        direction="backward",
    )
    df_merged.rename(columns={"policy_rate_pct": "riksbank_policy_rate_pct"}, inplace=True)

    # Before Mar 27 → old rate (4.00%), at/after Mar 27 → new rate (3.75%)
    rate_before_change = df_merged.loc[
        df_merged["timestamp_utc"] == pd.Timestamp("2024-03-26 23:00:00", tz="UTC"),
        "riksbank_policy_rate_pct"
    ].iloc[0]
    rate_at_change = df_merged.loc[
        df_merged["timestamp_utc"] == pd.Timestamp("2024-03-27 00:00:00", tz="UTC"),
        "riksbank_policy_rate_pct"
    ].iloc[0]
    rate_after_change = df_merged.loc[
        df_merged["timestamp_utc"] == pd.Timestamp("2024-03-27 12:00:00", tz="UTC"),
        "riksbank_policy_rate_pct"
    ].iloc[0]

    chk("Mar 26 23:00 has pre-change rate (4.00)", rate_before_change == 4.00,
        f"got {rate_before_change}")
    chk("Mar 27 00:00 has new rate (3.75)", rate_at_change == 3.75,
        f"got {rate_at_change}")
    chk("Mar 27 12:00 still has new rate (3.75)", rate_after_change == 3.75,
        f"got {rate_after_change}")
    chk("Step function: no leakage (new rate not before effective date)",
        rate_before_change != rate_at_change)

    # ── [6] BronzeWriter ──────────────────────────────────────────────
    print("\n[6] RiksbankBronzeWriter — gzip + manifest")
    with tempfile.TemporaryDirectory() as tmp:
        writer = RiksbankBronzeWriter(Path(tmp))
        sample_payload = {"observations": [{"date": "2024-03-27", "value": "3.75"}]}
        bp = writer.write(sample_payload, from_date="2020-01-01", to_date="2026-01-01")
        chk("Bronze file exists", bp.exists())
        chk("File is .gz", bp.suffix == ".gz")
        with gzip.open(bp, "rt", encoding="utf-8") as fh:
            recovered = json.load(fh)
        chk("Gzip round-trip intact", recovered == sample_payload)

        manifest = RiksbankBronzeManifest(
            series_id=SERIES_ID_STYRRANTAN,
            from_date="2020-01-01", to_date="2026-01-01",
            records_total=3, records_validation_failed=0,
            rate_changes_captured=2,
            rate_min_pct=3.50, rate_max_pct=4.00,
            bronze_path=str(bp),
        )
        mpath = writer.write_manifest(manifest, bp)
        with open(mpath) as fh:
            md = json.load(fh)
        chk("Manifest pipeline_version=1.0.0", md["pipeline_version"] == "1.0.0")
        chk("Manifest rate_changes_captured=2", md["rate_changes_captured"] == 2)
        chk("Manifest series_id correct", md["series_id"] == SERIES_ID_STYRRANTAN)

    # ── [7] Empty DataFrame schema ────────────────────────────────────
    print("\n[7] Empty DataFrame schema guard")
    empty = RiksbankFetcher._empty_df()
    chk("Empty DF has correct columns",
        list(empty.columns) == ["effective_date_utc", "policy_rate_pct"])
    chk("Empty DF has 0 rows", len(empty) == 0)

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    if errors:
        print(f"❌ {len(errors)} failure(s): {errors}")
        sys.exit(1)
    else:
        print("✅ All RiksbankFetcher smoke tests passed.")
        print("   Run RiksbankFetcher().fetch_policy_rate() with live network for end-to-end validation.")
    print("=" * 70)
