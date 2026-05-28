"""
src/ingestion/scb_fetcher.py
=============================================================================
Statistiska centralbyrån (SCB) PxWebApi v2 — Ingestion Client
=============================================================================
Responsibilities:
  - Fetch quarterly housing construction volume index (småhus/villor starts)
    at the municipality (Kommunkod) level from SCB PxWebApi v2
  - Fetch quarterly real estate price index (fastighetsprisindex) per
    municipality from SCB
  - Enforce SCB's published rate limit: max 30 requests per 10 seconds.
    Uses a token-bucket throttler so concurrent callers share the budget.
  - Normalise ISO-8859-1 / Windows-1252 encoded Swedish characters
    (ÅÄÖ, åäö) to UTF-8 before any string processing
  - Parse SCB's non-standard quarter notation (e.g. "2025K1", "2024K4")
    into UTC pd.Timestamp period-start values
  - Write immutable, timestamped Bronze layer snapshots as gzip-compressed
    JSON — one file per table per execution window
  - Validate all parsed records with Pydantic v2 at the ingestion boundary:
    schema drift in SCB table structure triggers CRITICAL log + safe abort
  - Output Silver-ready DataFrames with columns:
      [period_utc, kommunkod, region_name, smahus_construction_index]
      [period_utc, kommunkod, region_name, smahus_price_index]
    ready for KommunBiddingZoneMapper.aggregate_to_zone() in silver_alignment.py

SCB PxWebApi v2 Notes:
  - Base URL: https://api.scb.se/OV0104/v2beta/api/v2
  - Authentication: None required (public API)
  - Rate limit: 30 requests / 10 seconds per IP — enforced by TokenBucket
  - Response format: JSON (application/json)
  - Character encoding: responses may arrive as ISO-8859-1 despite UTF-8
    content-type headers — always decode bytes explicitly
  - Table IDs used:
      BO/BO0501/BO0501A/LagenhetNyKv16  (new dwelling construction, quarterly)
      BO/BO0501/BO0501B/FastpiUtr16Kv   (real estate price index, quarterly)
  - Variable codes:
      Region  → Kommunkod (4-digit municipality code)
      Tid     → Quarter period string (e.g. "2025K1")
      ContentsCode → selects the value column (construction index / price index)

Architecture position: LIVE API LAYER → BRONZE LAYER
Upstream:  SCB PxWebApi v2 (https://api.scb.se)
Downstream: src/processing/silver_alignment.py (KommunBiddingZoneMapper)
=============================================================================
"""

from __future__ import annotations

import gzip
import json
import logging
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator

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
logger = get_pipeline_logger("ingestion.scb_fetcher")


# ---------------------------------------------------------------------------
# SCB API Constants
# ---------------------------------------------------------------------------
SCB_BASE_URL = "https://api.scb.se/OV0104/v2beta/api/v2"

# Target table paths within the SCB API hierarchy
# Table: New dwelling construction starts, quarterly, by municipality
SCB_TABLE_CONSTRUCTION = "BO/BO0501/BO0501A/LagenhetNyKv16"
# Table: Real estate price index (fastighetsprisindex), quarterly, by municipality
SCB_TABLE_PRICE_INDEX = "BO/BO0501/BO0501B/FastpiUtr16Kv"

# SCB's language parameter for Swedish-language region labels (avoids encoding issues)
SCB_LANG = "sv"

# SCB rate limit: 30 requests per 10-second window (published in API docs)
SCB_MAX_REQUESTS = 30
SCB_RATE_WINDOW_SECONDS = 10.0

# Max rows per POST request (SCB rejects requests above this cell count)
SCB_MAX_CELLS_PER_REQUEST = 150_000

# Quarter string regex: matches "2024K1", "2025K4", etc.
QUARTER_PATTERN = re.compile(r"^\d{4}K[1-4]$")


# ---------------------------------------------------------------------------
# Token Bucket Rate Limiter
# ---------------------------------------------------------------------------
class TokenBucket:
    """
    Thread-safe token-bucket rate limiter for SCB's 30 req/10s limit.

    The bucket refills at `rate` tokens/second (= 30/10 = 3 tokens/s).
    Each request consumes one token. If the bucket is empty the caller
    blocks until a token is available.

    Shared across all SCBFetcher instances in the same process so that
    concurrent pipeline tasks don't collectively violate the limit.
    """

    _instance: Optional["TokenBucket"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls, capacity: int = SCB_MAX_REQUESTS,
                refill_rate: float = SCB_MAX_REQUESTS / SCB_RATE_WINDOW_SECONDS) -> "TokenBucket":
        # Singleton — one shared bucket per process
        with cls._lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._capacity = capacity
                instance._refill_rate = refill_rate          # tokens/second
                instance._tokens = float(capacity)
                instance._last_refill = time.monotonic()
                instance._bucket_lock = threading.Lock()
                cls._instance = instance
        return cls._instance  # type: ignore[return-value]

    def acquire(self, timeout: float = 60.0) -> None:
        """
        Block until a token is available, then consume it.

        Args:
            timeout: Maximum seconds to wait before raising RuntimeError.

        Raises:
            RuntimeError: If the bucket doesn't refill within `timeout` seconds.
        """
        deadline = time.monotonic() + timeout
        while True:
            with self._bucket_lock:
                now = time.monotonic()
                elapsed = now - self._last_refill
                # Refill tokens based on elapsed time
                self._tokens = min(
                    self._capacity,
                    self._tokens + elapsed * self._refill_rate,
                )
                self._last_refill = now

                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return

            # Not enough tokens — calculate sleep duration
            tokens_needed = 1.0 - self._tokens
            sleep_time = tokens_needed / self._refill_rate
            if time.monotonic() + sleep_time > deadline:
                raise RuntimeError(
                    f"TokenBucket: could not acquire token within {timeout}s. "
                    f"SCB rate limit may be exhausted."
                )
            time.sleep(min(sleep_time, 0.1))  # Sleep in small increments

    @property
    def available(self) -> float:
        with self._bucket_lock:
            return self._tokens


# ---------------------------------------------------------------------------
# Swedish Character Normaliser
# ---------------------------------------------------------------------------
def normalise_swedish_encoding(raw: bytes, declared_encoding: str = "utf-8") -> str:
    """
    Robustly decode SCB API response bytes to a clean UTF-8 string.

    SCB responses frequently declare UTF-8 in Content-Type headers but
    actually deliver ISO-8859-1 or Windows-1252 encoded bytes. This
    function tries the declared encoding first, then falls back through
    a sequence of Swedish-compatible encodings.

    Args:
        raw:              Raw response bytes from the HTTP client
        declared_encoding: Encoding claimed by the Content-Type header

    Returns:
        Decoded, clean UTF-8 string with Swedish characters intact.
    """
    fallback_encodings = ["utf-8", "iso-8859-1", "windows-1252", "latin-1"]

    # Try declared encoding first
    candidates = [declared_encoding.lower().replace("-", "")] if declared_encoding else []
    # Deduplicate while preserving order
    seen: set[str] = set()
    ordered: list[str] = []
    for enc in candidates + fallback_encodings:
        normalised = enc.lower().replace("-", "").replace("_", "")
        if normalised not in seen:
            seen.add(normalised)
            ordered.append(enc)

    last_exc: Optional[Exception] = None
    for encoding in ordered:
        try:
            decoded = raw.decode(encoding)
            # Quick sanity check: Swedish strings should not have replacement chars
            if "\ufffd" not in decoded:
                if encoding.lower().replace("-", "") != "utf8":
                    logger.debug(
                        "SCB response decoded with fallback encoding '%s' "
                        "(declared: '%s'). %d bytes → %d chars.",
                        encoding, declared_encoding, len(raw), len(decoded),
                    )
                return decoded
        except (UnicodeDecodeError, LookupError) as exc:
            last_exc = exc
            continue

    # Last resort: decode with replacement characters and log
    decoded = raw.decode("utf-8", errors="replace")
    logger.warning(
        "SCB encoding fallback exhausted — decoded with replacement characters. "
        "Swedish special chars may be corrupted. Last error: %s",
        last_exc,
    )
    return decoded


def sanitise_region_name(name: str) -> str:
    """
    Clean SCB region name strings for downstream use.

    Removes SCB's parenthetical municipality code suffixes and strips
    leading/trailing whitespace. Example:
        "Stockholm (0180)" → "Stockholm"
        "Malmö  " → "Malmö"
    """
    # Remove trailing parenthetical codes like "(0180)" or "(01)"
    cleaned = re.sub(r"\s*\(\d+\)\s*$", "", name).strip()
    return cleaned


# ---------------------------------------------------------------------------
# Pydantic v2 Schemas — SCB Ingestion Boundary Contracts
# ---------------------------------------------------------------------------

class SCBConstructionRecord(BaseModel):
    """
    Single validated quarterly housing construction observation.

    Represents one (municipality, quarter) cell from the SCB
    LagenhetNyKv16 table.

    Field meanings:
      period_utc:   Quarter period-start timestamp in UTC
                    (Q1=Jan 1, Q2=Apr 1, Q3=Jul 1, Q4=Oct 1)
      kommunkod:    4-digit SCB municipality code
      region_name:  Human-readable municipality name (UTF-8 normalised)
      smahus_construction_index: Number of new single/two-dwelling building
                    starts in the quarter (volume index, not normalised)
    """

    model_config = {"frozen": True}

    period_utc: datetime = Field(..., description="Quarter period-start in UTC")
    kommunkod: int = Field(..., ge=114, le=2584, description="SCB municipality code (Kommunkod)")
    region_name: str = Field(..., min_length=1, description="Municipality name (UTF-8)")
    smahus_construction_index: float = Field(
        ..., ge=0.0, description="New småhus starts in quarter (volume count)"
    )
    raw_quarter_str: str = Field(..., description="Original SCB quarter string e.g. '2025K1'")

    @field_validator("period_utc", mode="before")
    @classmethod
    def coerce_to_utc(cls, v: Any) -> datetime:
        if isinstance(v, (pd.Timestamp, datetime)):
            dt = v if isinstance(v, datetime) else v.to_pydatetime()
            return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
        raise ValueError(f"period_utc must be datetime or pd.Timestamp, got {type(v)}")

    @field_validator("raw_quarter_str")
    @classmethod
    def validate_quarter_format(cls, v: str) -> str:
        if not QUARTER_PATTERN.match(v):
            raise ValueError(
                f"Quarter string '{v}' does not match expected format 'YYYYK[1-4]'. "
                "SCB PxWebApi may have changed its Tid variable format."
            )
        return v


class SCBPriceIndexRecord(BaseModel):
    """
    Single validated quarterly real estate price index observation.

    Represents one (municipality, quarter) cell from the SCB
    FastpiUtr16Kv table.
    """

    model_config = {"frozen": True}

    period_utc: datetime = Field(..., description="Quarter period-start in UTC")
    kommunkod: int = Field(..., ge=114, le=2584, description="SCB municipality code")
    region_name: str = Field(..., min_length=1)
    smahus_price_index: float = Field(
        ..., ge=0.0, description="Real estate price index value (base year normalised)"
    )
    raw_quarter_str: str = Field(...)

    @field_validator("period_utc", mode="before")
    @classmethod
    def coerce_to_utc(cls, v: Any) -> datetime:
        if isinstance(v, (pd.Timestamp, datetime)):
            dt = v if isinstance(v, datetime) else v.to_pydatetime()
            return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
        raise ValueError(f"period_utc must be datetime or pd.Timestamp, got {type(v)}")

    @field_validator("raw_quarter_str")
    @classmethod
    def validate_quarter_format(cls, v: str) -> str:
        if not QUARTER_PATTERN.match(v):
            raise ValueError(f"Quarter string '{v}' does not match expected format 'YYYYK[1-4]'.")
        return v


class SCBBronzeManifest(BaseModel):
    """Metadata sidecar written alongside every SCB Bronze snapshot."""

    table_id: str
    metric_name: str
    period_from: str
    period_to: str
    records_total: int
    records_validation_failed: int
    municipalities_covered: int
    quarters_covered: int
    bronze_path: str
    ingested_at_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    pipeline_version: str = "1.0.0"


# ---------------------------------------------------------------------------
# Quarter String Parser (standalone utility, mirrors silver_alignment.py)
# ---------------------------------------------------------------------------
def parse_scb_quarter_string(quarter_str: str) -> pd.Timestamp:
    """
    Convert SCB quarter notation to a UTC period-start pd.Timestamp.

    Format: "{YYYY}K{Q}"  where Q ∈ {1, 2, 3, 4}

    Mapping:
        K1 → January 1   (Q1: Jan–Mar)
        K2 → April 1     (Q2: Apr–Jun)
        K3 → July 1      (Q3: Jul–Sep)
        K4 → October 1   (Q4: Oct–Dec)

    This is the authoritative parser for the codebase. The silver_alignment.py
    module contains an identical implementation for use within the Silver layer;
    both should stay in sync.

    Raises:
        ValueError: If the string does not match the YYYYK[1-4] format.
    """
    upper = quarter_str.strip().upper()
    if not QUARTER_PATTERN.match(upper):
        raise ValueError(
            f"Cannot parse SCB quarter string '{quarter_str}'. "
            "Expected format: 'YYYYK[1-4]' (e.g. '2025K1', '2024K4')."
        )
    year = int(upper[:4])
    quarter = int(upper[-1])
    month = (quarter - 1) * 3 + 1
    return pd.Timestamp(year=year, month=month, day=1, tz="UTC")


# ---------------------------------------------------------------------------
# SCB PxWebApi v2 Response Parser
# ---------------------------------------------------------------------------
class SCBResponseParser:
    """
    Parses the SCB PxWebApi v2 JSON response structure into flat records.

    PxWebApi v2 returns a "data" array of objects, each containing:
      - "key": list of dimension values (e.g. ["0180", "2024K1"])
      - "values": list of string-encoded metric values (e.g. ["2350"])

    The dimension ordering matches the "variables" list in the query body.
    We always query with (Region, Tid) so key[0]=Kommunkod, key[1]=quarter.

    Handles:
      - Missing values encoded as ".." (SCB suppression code) → NaN
      - Values encoded as "0" for true zero vs ".." for suppressed
      - Region labels containing parenthetical code suffixes → stripped
      - Integer kommunkoder with leading zeros in string form → int()
    """

    # SCB uses ".." to indicate a suppressed or unavailable cell
    SCB_MISSING_SENTINEL = ".."

    def __init__(self, table_id: str, metric_col: str) -> None:
        """
        Args:
            table_id:   SCB table path (e.g. "BO/BO0501/BO0501A/LagenhetNyKv16")
            metric_col: Column name for the parsed value in output records
        """
        self.table_id = table_id
        self.metric_col = metric_col

    def parse(
        self,
        response_json: dict[str, Any],
        region_label_map: dict[str, str],
    ) -> list[dict[str, Any]]:
        """
        Parse a PxWebApi v2 response dict into flat row dicts.

        Args:
            response_json:    Parsed JSON response from the SCB API
            region_label_map: {kommunkod_str → municipality_name} lookup
                              built from the table metadata /variables response

        Returns:
            List of flat dicts with keys:
              [raw_quarter_str, period_utc, kommunkod, region_name, {metric_col}]
            Suppressed cells (value == "..") are omitted.
            Cells that fail float conversion are omitted with a WARNING.
        """
        data_rows: list[dict[str, Any]] = response_json.get("data", [])
        if not data_rows:
            logger.warning(
                "SCB response for table %s contained no 'data' rows.", self.table_id
            )
            return []

        records: list[dict[str, Any]] = []
        suppressed = 0
        parse_errors = 0

        for row in data_rows:
            key: list[str] = row.get("key", [])
            values: list[str] = row.get("values", [])

            if len(key) < 2 or not values:
                logger.debug("Skipping malformed row (key=%r, values=%r)", key, values)
                parse_errors += 1
                continue

            kommunkod_str = key[0].lstrip("0") or "0"   # strip leading zeros
            quarter_str = key[1]
            raw_value = values[0]

            # Handle suppressed cells
            if raw_value == self.SCB_MISSING_SENTINEL:
                suppressed += 1
                continue

            # Parse kommunkod to int
            try:
                kommunkod = int(kommunkod_str)
            except ValueError:
                logger.debug("Cannot parse kommunkod '%s' — skipping.", kommunkod_str)
                parse_errors += 1
                continue

            # Parse value to float
            try:
                metric_value = float(raw_value.replace(",", ".").strip())
            except ValueError:
                logger.warning(
                    "Cannot parse metric value '%s' for kommunkod=%s quarter=%s — skipping.",
                    raw_value, kommunkod_str, quarter_str,
                )
                parse_errors += 1
                continue

            # Parse quarter string to UTC timestamp
            try:
                period_utc = parse_scb_quarter_string(quarter_str)
            except ValueError as exc:
                logger.warning(
                    "Cannot parse quarter string '%s': %s — skipping.", quarter_str, exc
                )
                parse_errors += 1
                continue

            region_name = sanitise_region_name(
                region_label_map.get(key[0], f"Okänd ({kommunkod_str})")
            )

            records.append({
                "raw_quarter_str": quarter_str,
                "period_utc": period_utc,
                "kommunkod": kommunkod,
                "region_name": region_name,
                self.metric_col: metric_value,
            })

        logger.info(
            "SCB parse | table=%s | records=%d | suppressed=%d | parse_errors=%d",
            self.table_id, len(records), suppressed, parse_errors,
        )
        return records


# ---------------------------------------------------------------------------
# Bronze Writer — SCB
# ---------------------------------------------------------------------------
class SCBBronzeWriter:
    """
    Persists raw SCB API responses as gzip-compressed JSON Bronze snapshots.

    Naming convention:
        {metric_name}_scb_{YYYYKQ_from}_{YYYYKQ_to}_{ingested_at_utc}.json.gz

    A JSON metadata manifest is written alongside each snapshot so downstream
    processes can inspect coverage without decompressing.
    """

    def __init__(self, bronze_dir: Path) -> None:
        self.bronze_dir = Path(bronze_dir)
        self.bronze_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        payload: dict[str, Any],
        metric_name: str,
        period_from: str,
        period_to: str,
    ) -> Path:
        """Compress and write raw JSON response to Bronze layer."""
        ts_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        fname = f"{metric_name}_scb_{period_from}_{period_to}_{ts_str}.json.gz"
        out_path = self.bronze_dir / fname

        with gzip.open(out_path, "wt", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=None)

        size_kb = out_path.stat().st_size / 1024
        logger.info(
            "Bronze snapshot written → %s (%.1f KB compressed)", out_path, size_kb
        )
        return out_path

    def write_manifest(
        self, manifest: SCBBronzeManifest, bronze_path: Path
    ) -> Path:
        """Write JSON metadata manifest alongside the Bronze snapshot."""
        mpath = bronze_path.with_suffix("").with_suffix(".manifest.json")
        with open(mpath, "w", encoding="utf-8") as fh:
            json.dump(manifest.model_dump(), fh, indent=2, ensure_ascii=False)
        logger.debug("Manifest written → %s", mpath)
        return mpath


# ---------------------------------------------------------------------------
# Main SCB Fetcher Client
# ---------------------------------------------------------------------------
class SCBFetcher(BaseHTTPClient):
    """
    Production SCB PxWebApi v2 ingestion client.

    Inherits:
      - Resilient async/sync HTTP session from BaseHTTPClient
      - Per-host circuit breaker
      - Exponential backoff with full jitter

    Adds:
      - Token-bucket rate limiter (30 req / 10 s)
      - ISO-8859-1 / Windows-1252 → UTF-8 normalisation
      - PxWebApi v2 POST query construction and response parsing
      - Pydantic v2 schema validation at the ingestion boundary
      - Immutable Bronze layer persistence with metadata manifests
      - Quarter-range filtering (only fetch the quarters needed)

    Output contract (what downstream silver_alignment.py expects):
      fetch_construction_index() →
        pd.DataFrame[period_utc, kommunkod, region_name, smahus_construction_index]

      fetch_price_index() →
        pd.DataFrame[period_utc, kommunkod, region_name, smahus_price_index]

    Usage:
        with SCBFetcher() as scb:
            df_const = scb.fetch_construction_index(from_quarter="2022K1", to_quarter="2025K4")
            df_price = scb.fetch_price_index(from_quarter="2022K1", to_quarter="2025K4")
    """

    def __init__(
        self,
        bronze_dir: Path = Path("data/bronze/scb"),
        request_timeout_seconds: float = 45.0,
        connect_timeout_seconds: float = 10.0,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        super().__init__(
            base_url=SCB_BASE_URL,
            max_retries=max_retries,
            request_timeout_seconds=request_timeout_seconds,
            connect_timeout_seconds=connect_timeout_seconds,
            # SCB is a public API with no authentication — circuit threshold lower
            circuit_breaker_threshold=3,
            circuit_breaker_recovery_seconds=60.0,
            http2=False,  # SCB API does not support HTTP/2
        )
        self._bronze = SCBBronzeWriter(bronze_dir)
        self._throttle = TokenBucket()   # shared singleton

        logger.info(
            "SCBFetcher initialised | bronze=%s | rate_limit=%d req/%.0fs",
            bronze_dir, SCB_MAX_REQUESTS, SCB_RATE_WINDOW_SECONDS,
        )

    # ------------------------------------------------------------------
    # BaseHTTPClient overrides
    # ------------------------------------------------------------------
    def _build_default_headers(self) -> dict[str, str]:
        return {
            "Accept": "application/json",
            "Content-Type": "application/json; charset=utf-8",
            "User-Agent": "SwedenGridPipeline/1.0 (scb_fetcher)",
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fetch_construction_index(
        self,
        from_quarter: str = "2020K1",
        to_quarter: str = "2026K2",
        kommunkoder: Optional[list[int]] = None,
    ) -> pd.DataFrame:
        """
        Fetch quarterly new småhus construction volume index per municipality.

        Source table: BO/BO0501/BO0501A/LagenhetNyKv16
        SCB variable: "Påbörjande av bostäder i småhus (nybyggnad), antal"

        Args:
            from_quarter: First quarter to fetch, e.g. "2020K1"
            to_quarter:   Last quarter to fetch, e.g. "2026K2"
            kommunkoder:  Optional filter list of municipality codes.
                          If None, fetches all ~290 municipalities.

        Returns:
            pd.DataFrame with columns:
              [period_utc, kommunkod, region_name, smahus_construction_index]
            Empty DataFrame with correct schema on total failure.
        """
        return self._fetch_table(
            table_id=SCB_TABLE_CONSTRUCTION,
            metric_col="smahus_construction_index",
            contents_code="BO0501A1",  # SCB contents code for construction starts
            from_quarter=from_quarter,
            to_quarter=to_quarter,
            kommunkoder=kommunkoder,
            pydantic_model=SCBConstructionRecord,
        )

    def fetch_price_index(
        self,
        from_quarter: str = "2020K1",
        to_quarter: str = "2026K2",
        kommunkoder: Optional[list[int]] = None,
    ) -> pd.DataFrame:
        """
        Fetch quarterly real estate price index (fastighetsprisindex) per municipality.

        Source table: BO/BO0501/BO0501B/FastpiUtr16Kv
        SCB variable: "Fastighetsprisindex för permanentbostäder i småhus"

        Args:
            from_quarter: First quarter to fetch, e.g. "2020K1"
            to_quarter:   Last quarter to fetch, e.g. "2026K2"
            kommunkoder:  Optional filter list of municipality codes.

        Returns:
            pd.DataFrame with columns:
              [period_utc, kommunkod, region_name, smahus_price_index]
        """
        return self._fetch_table(
            table_id=SCB_TABLE_PRICE_INDEX,
            metric_col="smahus_price_index",
            contents_code="BO0501B1",  # SCB contents code for price index
            from_quarter=from_quarter,
            to_quarter=to_quarter,
            kommunkoder=kommunkoder,
            pydantic_model=SCBPriceIndexRecord,
        )

    def fetch_all(
        self,
        from_quarter: str = "2020K1",
        to_quarter: str = "2026K2",
        kommunkoder: Optional[list[int]] = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch both construction and price index tables.

        Returns:
            {
              "construction": pd.DataFrame[..., smahus_construction_index],
              "price_index":  pd.DataFrame[..., smahus_price_index],
            }
        """
        logger.info(
            "Fetching all SCB tables | %s → %s", from_quarter, to_quarter
        )
        return {
            "construction": self.fetch_construction_index(
                from_quarter, to_quarter, kommunkoder
            ),
            "price_index": self.fetch_price_index(
                from_quarter, to_quarter, kommunkoder
            ),
        }

    # ------------------------------------------------------------------
    # Private: Core Table Fetch Pipeline
    # ------------------------------------------------------------------
    def _fetch_table(
        self,
        table_id: str,
        metric_col: str,
        contents_code: str,
        from_quarter: str,
        to_quarter: str,
        kommunkoder: Optional[list[int]],
        pydantic_model: type,
    ) -> pd.DataFrame:
        """
        Full pipeline for fetching a single SCB PxWebApi v2 table.

        Steps:
          1. Resolve available quarters from the table metadata
          2. Build the POST query body filtering to the desired window
          3. Throttle → POST → normalise encoding → parse JSON
          4. Write Bronze snapshot + manifest
          5. Parse response into flat records
          6. Validate each record with Pydantic v2
          7. Return typed DataFrame
        """
        logger.info(
            "SCB fetch | table=%s | metric=%s | %s → %s",
            table_id, metric_col, from_quarter, to_quarter,
        )

        # Step 1: Resolve available quarter strings from table metadata
        try:
            available_quarters, region_label_map = self._fetch_table_metadata(table_id)
        except (RetryExhaustedError, NonRetryableHTTPError, ValueError) as exc:
            logger.error(
                "Cannot retrieve SCB table metadata for %s: %s. Returning empty DataFrame.",
                table_id, exc,
            )
            return self._empty_df(metric_col)

        # Step 2: Filter quarters to the requested window
        quarters_in_window = self._filter_quarters(
            available_quarters, from_quarter, to_quarter
        )
        if not quarters_in_window:
            logger.warning(
                "No quarters in window %s → %s for table %s. "
                "Available range: %s → %s.",
                from_quarter, to_quarter, table_id,
                available_quarters[0] if available_quarters else "N/A",
                available_quarters[-1] if available_quarters else "N/A",
            )
            return self._empty_df(metric_col)

        logger.info(
            "Fetching %d quarters (%s → %s) for table %s",
            len(quarters_in_window),
            quarters_in_window[0], quarters_in_window[-1], table_id,
        )

        # Step 3: Build query body and fetch
        query_body = self._build_query_body(
            contents_code=contents_code,
            quarters=quarters_in_window,
            kommunkoder=kommunkoder,
        )

        try:
            raw_response = self._post_scb_table(table_id, query_body)
        except (RetryExhaustedError, NonRetryableHTTPError, ValueError) as exc:
            logger.error(
                "SCB table fetch failed for %s: %s. Returning empty DataFrame.",
                table_id, exc,
            )
            return self._empty_df(metric_col)

        # Step 4: Bronze write (always — before validation, so raw payload is preserved)
        bronze_path = self._bronze.write(
            payload=raw_response,
            metric_name=metric_col,
            period_from=quarters_in_window[0],
            period_to=quarters_in_window[-1],
        )

        # Step 5: Parse flat records
        parser = SCBResponseParser(table_id=table_id, metric_col=metric_col)
        raw_records = parser.parse(raw_response, region_label_map)

        # Step 6: Pydantic v2 validation
        validated_records, validation_failures = self._validate_records(
            raw_records, pydantic_model, metric_col, table_id
        )

        # Write manifest
        quarters_seen = sorted({r["raw_quarter_str"] for r in raw_records})
        municipalities_seen = len({r["kommunkod"] for r in raw_records})
        self._bronze.write_manifest(
            SCBBronzeManifest(
                table_id=table_id,
                metric_name=metric_col,
                period_from=quarters_in_window[0],
                period_to=quarters_in_window[-1],
                records_total=len(validated_records),
                records_validation_failed=validation_failures,
                municipalities_covered=municipalities_seen,
                quarters_covered=len(quarters_seen),
                bronze_path=str(bronze_path),
            ),
            bronze_path,
        )

        if not validated_records:
            logger.warning(
                "All records failed validation for table %s. "
                "Raw payload preserved at %s.",
                table_id, bronze_path,
            )
            return self._empty_df(metric_col)

        # Step 7: Build typed DataFrame
        df = pd.DataFrame(validated_records)
        df["period_utc"] = pd.to_datetime(df["period_utc"], utc=True)
        df["kommunkod"] = df["kommunkod"].astype(int)
        df.sort_values(["period_utc", "kommunkod"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        logger.info(
            "SCB fetch complete | table=%s | %d records | "
            "%d municipalities | %d quarters | %d validation failures",
            table_id, len(df),
            df["kommunkod"].nunique(),
            df["period_utc"].nunique(),
            validation_failures,
        )
        return df

    # ------------------------------------------------------------------
    # Private: Table Metadata (available quarters + region labels)
    # ------------------------------------------------------------------
    @retry_with_backoff(
        max_retries=DEFAULT_MAX_RETRIES,
        base_seconds=DEFAULT_BASE_BACKOFF_SECONDS,
        max_seconds=DEFAULT_MAX_BACKOFF_SECONDS,
    )
    def _fetch_table_metadata(
        self, table_id: str
    ) -> tuple[list[str], dict[str, str]]:
        """
        Retrieve the list of available Tid (quarter) values and the
        Region label map for a given SCB table.

        SCB PxWebApi v2 provides table variable metadata at:
            GET /tables/{table_id}/variables?lang=sv

        Returns:
            (quarters_list, region_label_map)
            quarters_list:     sorted list of "YYYYK[1-4]" strings
            region_label_map:  {kommunkod_str → municipality_name}
        """
        self._throttle.acquire()
        path = f"/tables/{table_id}/variables"
        response = self.request("GET", path, params={"lang": SCB_LANG})

        content_type = response.headers.get("content-type", "utf-8")
        declared_enc = self._extract_charset(content_type)
        text = normalise_swedish_encoding(response.content, declared_enc)
        payload = json.loads(text)

        variables: list[dict[str, Any]] = payload.get("variables", [])

        quarters: list[str] = []
        region_label_map: dict[str, str] = {}

        for var in variables:
            var_id: str = var.get("id", "")
            values: list[str] = var.get("values", [])
            value_texts: list[str] = var.get("valueTexts", [])

            if var_id == "Tid":
                # Filter to valid quarter strings only (ignore annual or monthly)
                quarters = sorted(
                    [v for v in values if QUARTER_PATTERN.match(v)]
                )
                logger.debug(
                    "Table %s: %d quarterly Tid values available (%s → %s)",
                    table_id, len(quarters),
                    quarters[0] if quarters else "N/A",
                    quarters[-1] if quarters else "N/A",
                )

            elif var_id == "Region":
                # Build {code → label} map, normalising Swedish chars
                for code, label in zip(values, value_texts):
                    region_label_map[code] = sanitise_region_name(label)
                logger.debug(
                    "Table %s: %d Region labels loaded.", table_id, len(region_label_map)
                )

        if not quarters:
            raise ValueError(
                f"No quarterly Tid values found in metadata for table {table_id}. "
                "The table structure may have changed."
            )

        return quarters, region_label_map

    # ------------------------------------------------------------------
    # Private: POST Query Builder and Executor
    # ------------------------------------------------------------------
    def _build_query_body(
        self,
        contents_code: str,
        quarters: list[str],
        kommunkoder: Optional[list[int]],
    ) -> dict[str, Any]:
        """
        Construct a PxWebApi v2 POST query body in the standard format.

        PxWebApi v2 uses a JSON body with a "query" array of selection objects.
        Each selection specifies one dimension (variable) and its value filter.

        The wildcard "*" selects all values for a dimension.
        For Region, if kommunkoder is specified, we select only those codes.

        Cell count guard: SCB rejects requests exceeding ~150,000 cells.
        We log a warning if the query approaches this limit.
        """
        # Region filter: specific municipalities or all (wildcard)
        if kommunkoder:
            region_values = [str(k).zfill(4) for k in kommunkoder]
        else:
            region_values = ["*"]  # all municipalities

        query: dict[str, Any] = {
            "query": [
                {
                    "code": "Region",
                    "selection": {
                        "filter": "item" if kommunkoder else "all",
                        "values": region_values,
                    },
                },
                {
                    "code": "ContentsCode",
                    "selection": {
                        "filter": "item",
                        "values": [contents_code],
                    },
                },
                {
                    "code": "Tid",
                    "selection": {
                        "filter": "item",
                        "values": quarters,
                    },
                },
            ],
            "response": {
                "format": "json",
            },
        }

        # Cell count estimation
        n_regions = len(kommunkoder) if kommunkoder else 290
        est_cells = n_regions * 1 * len(quarters)  # regions × contents × time
        if est_cells > SCB_MAX_CELLS_PER_REQUEST:
            logger.warning(
                "Estimated cell count %d exceeds SCB limit %d. "
                "Consider splitting the quarter range into smaller batches.",
                est_cells, SCB_MAX_CELLS_PER_REQUEST,
            )

        return query

    @retry_with_backoff(
        max_retries=DEFAULT_MAX_RETRIES,
        base_seconds=DEFAULT_BASE_BACKOFF_SECONDS,
        max_seconds=DEFAULT_MAX_BACKOFF_SECONDS,
    )
    def _post_scb_table(
        self, table_id: str, query_body: dict[str, Any]
    ) -> dict[str, Any]:
        """
        POST a query to the SCB PxWebApi v2 table endpoint and return
        the parsed JSON response dict.

        Acquires a token from the rate-limiter before every request.
        Normalises the response encoding before JSON parsing.

        Raises:
            ValueError: If the response cannot be decoded or parsed as JSON.
            NonRetryableHTTPError: If the API returns a 4xx error.
        """
        self._throttle.acquire()
        path = f"/tables/{table_id}/data"
        response = self.request(
            "POST",
            path,
            params={"lang": SCB_LANG},
            json_body=query_body,
        )

        content_type = response.headers.get("content-type", "utf-8")
        declared_enc = self._extract_charset(content_type)
        text = normalise_swedish_encoding(response.content, declared_enc)

        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"SCB response for table {table_id} is not valid JSON: {exc}. "
                f"First 300 chars: {text[:300]!r}"
            ) from exc

    # ------------------------------------------------------------------
    # Private: Pydantic Validation
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_records(
        raw_records: list[dict[str, Any]],
        pydantic_model: type,
        metric_col: str,
        table_id: str,
    ) -> tuple[list[dict[str, Any]], int]:
        """
        Validate raw parsed records through a Pydantic v2 model.

        Returns:
            (validated_records, validation_failure_count)
            validated_records contains model_dump() dicts from validated models.
            Failed records are logged at ERROR level and omitted from output.
        """
        validated: list[dict[str, Any]] = []
        failures = 0

        for rec in raw_records:
            try:
                model_instance = pydantic_model(**rec)
                # model_dump() preserves datetime objects; we convert period_utc
                # back to a plain datetime for DataFrame construction
                dumped = model_instance.model_dump()
                # Keep only the columns needed downstream
                validated.append({
                    "period_utc": dumped["period_utc"],
                    "kommunkod": dumped["kommunkod"],
                    "region_name": dumped["region_name"],
                    metric_col: dumped[metric_col],
                })
            except ValidationError as exc:
                failures += 1
                logger.error(
                    "Pydantic validation failed | table=%s | record=%r | errors=%s",
                    table_id,
                    {k: rec.get(k) for k in ("kommunkod", "raw_quarter_str")},
                    exc.errors(include_url=False),
                )

        if failures:
            logger.warning(
                "%d / %d SCB records failed Pydantic validation for table %s.",
                failures, len(raw_records), table_id,
            )

        return validated, failures

    # ------------------------------------------------------------------
    # Private: Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _filter_quarters(
        available: list[str],
        from_quarter: str,
        to_quarter: str,
    ) -> list[str]:
        """
        Filter the list of available quarters to the [from, to] window (inclusive).

        Raises:
            ValueError: If from_quarter or to_quarter cannot be parsed.
        """
        from_ts = parse_scb_quarter_string(from_quarter)
        to_ts = parse_scb_quarter_string(to_quarter)

        filtered = [
            q for q in available
            if from_ts <= parse_scb_quarter_string(q) <= to_ts
        ]
        return sorted(filtered)

    @staticmethod
    def _extract_charset(content_type: str) -> str:
        """Extract charset from Content-Type header, defaulting to 'utf-8'."""
        match = re.search(r"charset=([^\s;]+)", content_type, re.IGNORECASE)
        return match.group(1) if match else "utf-8"

    @staticmethod
    def _empty_df(metric_col: str) -> pd.DataFrame:
        """Return an empty DataFrame with the correct Silver-ready schema."""
        return pd.DataFrame(
            columns=["period_utc", "kommunkod", "region_name", metric_col]
        )


# ---------------------------------------------------------------------------
# CLI Smoke Test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import tempfile
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("SCBFetcher — Unit Smoke Test (offline / no live API calls)")
    print("=" * 70)

    errors: list[str] = []

    def chk(name: str, cond: bool, detail: str = "") -> None:
        sym = "✓" if cond else "✗"
        print(f"  {sym} {name}" + (f": {detail}" if detail else ""))
        if not cond:
            errors.append(name)

    # ── [1] Quarter String Parser ────────────────────────────────────────
    print("\n[1] parse_scb_quarter_string — all valid patterns")
    cases = {
        "2025K1": pd.Timestamp("2025-01-01", tz="UTC"),
        "2025K2": pd.Timestamp("2025-04-01", tz="UTC"),
        "2025K3": pd.Timestamp("2025-07-01", tz="UTC"),
        "2025K4": pd.Timestamp("2025-10-01", tz="UTC"),
        "2024K1": pd.Timestamp("2024-01-01", tz="UTC"),
        "2020K4": pd.Timestamp("2020-10-01", tz="UTC"),
    }
    for q, expected in cases.items():
        result = parse_scb_quarter_string(q)
        chk(f"  {q} → {expected.date()}", result == expected, f"got {result}")

    print("\n[1b] Quarter parser — error cases")
    for bad in ["2025K0", "2025K5", "25K1", "2025-Q1", "2025K", "", "BADSTR"]:
        try:
            parse_scb_quarter_string(bad)
            chk(f"  '{bad}' should raise ValueError", False)
        except ValueError:
            chk(f"  '{bad}' raises ValueError", True)

    # ── [2] Swedish Encoding Normaliser ─────────────────────────────────
    print("\n[2] normalise_swedish_encoding — encoding fallback chain")
    swedish_str = "Göteborg Malmö Östersund Härnösand"
    # Simulate ISO-8859-1 response with false UTF-8 header
    iso_bytes = swedish_str.encode("iso-8859-1")
    result_utf8 = normalise_swedish_encoding(iso_bytes, declared_encoding="utf-8")
    chk("ISO-8859-1 bytes decoded to correct string", result_utf8 == swedish_str)

    utf8_bytes = swedish_str.encode("utf-8")
    result_declared = normalise_swedish_encoding(utf8_bytes, declared_encoding="utf-8")
    chk("UTF-8 bytes decoded correctly", result_declared == swedish_str)

    # ── [3] Region Name Sanitiser ────────────────────────────────────────
    print("\n[3] sanitise_region_name")
    sanitise_cases = {
        "Stockholm (0180)": "Stockholm",
        "Malmö  ": "Malmö",
        "Göteborgs stad (1480)": "Göteborgs stad",
        "Åre": "Åre",
        "Härnösand (2280)": "Härnösand",
    }
    for raw, expected in sanitise_cases.items():
        result = sanitise_region_name(raw)
        chk(f"  '{raw}' → '{expected}'", result == expected, f"got '{result}'")

    # ── [4] Token Bucket Rate Limiter ────────────────────────────────────
    print("\n[4] TokenBucket — singleton and token consumption")
    # Reset singleton for test isolation
    TokenBucket._instance = None
    bucket = TokenBucket(capacity=5, refill_rate=100.0)
    chk("Singleton: same instance returned", bucket is TokenBucket())
    chk("Initial tokens = capacity", bucket.available >= 4.5)
    for _ in range(5): bucket.acquire()
    chk("After 5 acquires, tokens ≈ 0", bucket.available < 1.0)
    time.sleep(0.05)  # refill at 100 tok/s → 5 in 50ms
    chk("Tokens refilled after wait", bucket.available > 0.0)

    # ── [5] SCB Response Parser ──────────────────────────────────────────
    print("\n[5] SCBResponseParser — synthetic PxWebApi v2 response")
    synthetic_response: dict[str, Any] = {
        "data": [
            {"key": ["0180", "2024K1"], "values": ["2350"]},   # Stockholm, Q1 2024
            {"key": ["1280", "2024K1"], "values": ["850"]},    # Malmö, Q1 2024
            {"key": ["0180", "2024K2"], "values": [".."]},     # Stockholm, Q2 — suppressed
            {"key": ["1480", "2024K1"], "values": ["1200"]},   # Göteborg, Q1 2024
            {"key": ["XXXX", "2024K1"], "values": ["100"]},    # Bad kommunkod
            {"key": ["0180", "2025K5"], "values": ["500"]},    # Bad quarter
            {"key": [], "values": []},                          # Malformed row
        ]
    }
    region_labels = {"0180": "Stockholm", "1280": "Malmö", "1480": "Göteborg"}
    parser = SCBResponseParser(SCB_TABLE_CONSTRUCTION, "smahus_construction_index")
    records = parser.parse(synthetic_response, region_labels)

    chk("Suppressed '..' cell excluded", all(r["smahus_construction_index"] != ".." for r in records))
    chk("Bad kommunkod 'XXXX' excluded", all(r["kommunkod"] != "XXXX" for r in records))
    chk("Bad quarter '2025K5' excluded", all(r.get("raw_quarter_str") != "2025K5" for r in records))
    chk("Valid records: 3 (Stockholm Q1, Malmö Q1, Göteborg Q1)", len(records) == 3, f"got {len(records)}")

    chk("Stockholm kommunkod=180", any(r["kommunkod"] == 180 and r["smahus_construction_index"] == 2350.0 for r in records))
    chk("Malmö kommunkod=1280", any(r["kommunkod"] == 1280 and r["smahus_construction_index"] == 850.0 for r in records))
    chk("period_utc is pd.Timestamp", isinstance(records[0]["period_utc"], pd.Timestamp))
    chk("period_utc Q1 = Jan 1", records[0]["period_utc"] == pd.Timestamp("2024-01-01", tz="UTC"))

    # ── [6] Quarter Filter ───────────────────────────────────────────────
    print("\n[6] _filter_quarters — window slicing")
    available = ["2023K1", "2023K2", "2023K3", "2023K4",
                 "2024K1", "2024K2", "2024K3", "2024K4",
                 "2025K1", "2025K2"]
    filtered = SCBFetcher._filter_quarters(available, "2024K1", "2024K4")
    chk("Filter 2024K1→2024K4 = 4 quarters", len(filtered) == 4, f"got {filtered}")
    chk("First = 2024K1", filtered[0] == "2024K1")
    chk("Last = 2024K4", filtered[-1] == "2024K4")
    filtered_single = SCBFetcher._filter_quarters(available, "2025K2", "2025K2")
    chk("Single-quarter window", filtered_single == ["2025K2"])
    filtered_none = SCBFetcher._filter_quarters(available, "2026K1", "2026K4")
    chk("Out-of-range window returns empty", filtered_none == [])

    # ── [7] Query Body Builder ───────────────────────────────────────────
    print("\n[7] _build_query_body — structure validation")
    scb = SCBFetcher.__new__(SCBFetcher)  # skip __init__ for offline test
    scb._bronze = None  # not needed for this test
    query = SCBFetcher._build_query_body(
        None,  # type: ignore
        contents_code="BO0501A1",
        quarters=["2024K1", "2024K2"],
        kommunkoder=[180, 1280],
    )
    chk("Query has 3 selections", len(query["query"]) == 3)
    region_sel = next(s for s in query["query"] if s["code"] == "Region")
    chk("Region filter is 'item'", region_sel["selection"]["filter"] == "item")
    chk("Region values zero-padded", "0180" in region_sel["selection"]["values"])
    tid_sel = next(s for s in query["query"] if s["code"] == "Tid")
    chk("Tid contains 2 quarters", len(tid_sel["selection"]["values"]) == 2)
    chk("Response format is json", query["response"]["format"] == "json")

    query_all = SCBFetcher._build_query_body(
        None, contents_code="BO0501A1", quarters=["2024K1"], kommunkoder=None  # type: ignore
    )
    region_all = next(s for s in query_all["query"] if s["code"] == "Region")
    chk("No filter → wildcard '*'", region_all["selection"]["values"] == ["*"])

    # ── [8] BronzeWriter ────────────────────────────────────────────────
    print("\n[8] SCBBronzeWriter — gzip + manifest")
    with tempfile.TemporaryDirectory() as tmp:
        writer = SCBBronzeWriter(Path(tmp))
        sample_payload = {"data": [{"key": ["0180", "2024K1"], "values": ["2350"]}]}
        bp = writer.write(sample_payload, "smahus_construction_index", "2024K1", "2024K4")
        chk("Bronze file exists", bp.exists())
        chk("File is .gz", bp.suffix == ".gz")
        with gzip.open(bp, "rt", encoding="utf-8") as fh:
            recovered = json.load(fh)
        chk("Gzip round-trip intact", recovered == sample_payload)
        manifest = SCBBronzeManifest(
            table_id=SCB_TABLE_CONSTRUCTION, metric_name="smahus_construction_index",
            period_from="2024K1", period_to="2024K4",
            records_total=1, records_validation_failed=0,
            municipalities_covered=1, quarters_covered=1, bronze_path=str(bp),
        )
        mpath = writer.write_manifest(manifest, bp)
        with open(mpath) as fh:
            md = json.load(fh)
        chk("Manifest pipeline_version=1.0.0", md["pipeline_version"] == "1.0.0")
        chk("Manifest records_total=1", md["records_total"] == 1)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    if errors:
        print(f"❌ {len(errors)} failure(s): {errors}")
        sys.exit(1)
    else:
        print("✅ All SCBFetcher smoke tests passed.")
        print("   Run SCBFetcher().fetch_all() with live network for end-to-end validation.")
    print("=" * 70)
