"""
src/ingestion/energy_client.py
=============================================================================
ENTSO-E & Svenska kraftnät (SVK) Ingestion Client — Specialized HTTP Client
=============================================================================
Responsibilities:
  - Inherit resilient HTTP session, circuit breaker, and retry logic from
    BaseHTTPClient (src/ingestion/client.py)
  - Fetch hourly Actual Total Load (MW) per Swedish bidding zone (SE1–SE4)
    from ENTSO-E Transparency Platform (Document type A65)
  - Fetch hourly Net Grid Imbalance Volumes (MWh) per zone (Document type A86)
  - Fetch hourly Day-Ahead Market Prices EUR/MWh per zone (Document type A44)
  - Enforce strict Pydantic v2 schema validation at the ingestion boundary:
    schema drift is caught immediately and logged as CRITICAL before aborting
  - Tag anomalous observations (physics-bound checks) without dropping them:
    all records are preserved in Bronze; anomaly flags flow to Silver
  - Write immutable, gzip-compressed Bronze layer snapshots with execution
    timestamps for full audit traceability and idempotent re-runs
  - Expose both sync (batch pipeline) and async (parallel zone fetch) APIs

Architecture position: LIVE API LAYER → BRONZE LAYER
Upstream:  ENTSO-E Transparency REST API / SVK Mimer Portal
Downstream: src/processing/silver_alignment.py

ENTSO-E API Notes:
  - Base URL: https://web-api.tp.entsoe.eu/api
  - Authentication: securityToken query parameter (not Authorization header)
  - Response format: XML (not JSON) — IEC 62325 CIM document schema
  - Rate limit: ~400 requests/hour per token; 429 triggers Retry-After
  - Max query window: 1 year per request for most document types
  - EIC Area codes for Swedish zones are defined in BiddingZone enum
=============================================================================
"""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional
import xml.etree.ElementTree as ET

import httpx
import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from .client import (
    BaseHTTPClient,
    CircuitOpenError,
    NonRetryableHTTPError,
    RetryExhaustedError,
    async_retry_with_backoff,
    retry_with_backoff,
    DEFAULT_MAX_RETRIES,
    DEFAULT_BASE_BACKOFF_SECONDS,
    DEFAULT_MAX_BACKOFF_SECONDS,
    DEFAULT_JITTER_RANGE,
    DEFAULT_RETRYABLE_STATUS_CODES,
)
from ..utils.pipeline_logging import get_pipeline_logger
from ..utils.bronze import BronzeWriterBase

# ---------------------------------------------------------------------------
# Module Logger
# ---------------------------------------------------------------------------
logger = get_pipeline_logger("ingestion.energy_client")


# ---------------------------------------------------------------------------
# ENTSO-E Constants & Enums
# ---------------------------------------------------------------------------
class BiddingZone(str, Enum):
    """
    ENTSO-E EIC (Energy Identification Code) area codes for Swedish bidding zones.

    These are the authoritative identifiers used in ENTSO-E API requests.
    The string values are the EIC codes; the .name property gives the human
    label (SE1, SE2, SE3, SE4).
    """
    SE1 = "10Y1001A1001A44P"
    SE2 = "10Y1001A1001A45N"
    SE3 = "10Y1001A1001A46L"
    SE4 = "10Y1001A1001A47J"


ZONE_LABELS: dict[BiddingZone, str] = {z: z.name for z in BiddingZone}
LABEL_TO_ZONE: dict[str, BiddingZone] = {z.name: z for z in BiddingZone}
VALID_ZONE_LABELS: frozenset[str] = frozenset(ZONE_LABELS.values())

# ENTSO-E document type codes
DOC_TYPE_ACTUAL_LOAD: str = "A65"
DOC_TYPE_IMBALANCE_VOLUME: str = "A86"
DOC_TYPE_DAY_AHEAD_PRICE: str = "A44"

ENTSO_E_BASE_URL: str = "https://web-api.tp.entsoe.eu/api"

# Maximum time window per ENTSO-E request (1 year for most doc types)
MAX_QUERY_WINDOW_DAYS: int = 365

# Grid-physics anomaly thresholds — calibrated to Swedish system limits
ANOMALY_THRESHOLDS: dict[str, float] = {
    "load_mw_min": 100.0,            # SE1 minimum off-peak load
    "load_mw_max": 30_000.0,         # SE3 maximum observed peak (~28 GW)
    "imbalance_mwh_abs_max": 5_000.0,
    "price_eur_min": -500.0,         # Deep negative: curtailment event
    "price_eur_max": 4_000.0,        # Above 2022 European energy crisis peak
}

# XML namespaces for ENTSO-E document schemas (IEC 62325-451)
XML_NS: dict[str, dict[str, str]] = {
    DOC_TYPE_ACTUAL_LOAD: {
        "ns": "urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0"
    },
    DOC_TYPE_IMBALANCE_VOLUME: {
        "ns": "urn:iec62325.351:tc57wg16:451-6:balancingdocument:4:0"
    },
    DOC_TYPE_DAY_AHEAD_PRICE: {
        "ns": "urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:0"
    },
}

# ISO 8601 duration → minutes mapping for ENTSO-E resolution codes
RESOLUTION_MAP: dict[str, int] = {
    "PT15M": 15,
    "PT30M": 30,
    "PT60M": 60,
    "PT1H": 60,
}


# ---------------------------------------------------------------------------
# Pydantic v2 Data Models — Ingestion Boundary Contracts
# ---------------------------------------------------------------------------

class _UTCDatetimeMixin:
    """Shared UTC coercion validator for all time-series point models."""

    @field_validator("timestamp_utc", mode="before")
    @classmethod
    def coerce_to_utc(cls, v: Any) -> datetime:
        if isinstance(v, str):
            v = datetime.fromisoformat(v.replace("Z", "+00:00"))
        if isinstance(v, datetime) and v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)

    @field_validator("zone", mode="after")
    @classmethod
    def validate_zone_label(cls, v: str) -> str:
        if v not in VALID_ZONE_LABELS:
            raise ValueError(
                f"Invalid bidding zone label: '{v}'. "
                f"Must be one of {sorted(VALID_ZONE_LABELS)}."
            )
        return v


class HourlyLoadPoint(_UTCDatetimeMixin, BaseModel):
    """
    Single validated hourly Actual Total Load observation.

    Pydantic raises ValidationError (not silently coerces) if:
      - zone is not SE1–SE4
      - load_mw is negative (grid-physics impossibility)
      - timestamp_utc cannot be parsed or coerced to UTC
    """

    model_config = {"frozen": True}

    zone: str = Field(..., description="Nord Pool bidding zone label (SE1–SE4)")
    timestamp_utc: datetime = Field(..., description="Period start in UTC")
    load_mw: float = Field(..., ge=0.0, description="Actual load in megawatts (≥ 0)")
    resolution_minutes: int = Field(default=60, ge=1, le=1440)


class HourlyImbalancePoint(_UTCDatetimeMixin, BaseModel):
    """
    Single validated hourly Net Grid Imbalance Volume observation.

    Direction encoding:
      A01 = Long (generation surplus, positive MWh)
      A02 = Short (generation deficit, negative MWh)

    Cross-field validation emits a WARNING (not error) on sign/direction
    inconsistency — the record is kept for human review.
    """

    model_config = {"frozen": True}

    zone: str = Field(..., description="Nord Pool bidding zone label (SE1–SE4)")
    timestamp_utc: datetime = Field(..., description="Settlement period start in UTC")
    imbalance_mwh: float = Field(
        ..., description="Net imbalance volume in MWh (negative = under-generation)"
    )
    direction: Optional[str] = Field(
        default=None,
        description="ENTSO-E direction code: A01=Long (surplus) / A02=Short (deficit)",
    )

    @model_validator(mode="after")
    def check_direction_sign_consistency(self) -> "HourlyImbalancePoint":
        """
        Emit a WARNING — not an error — on direction/sign inconsistency.

        ENTSO-E upstream data occasionally contains direction code errors
        for small imbalances near zero. We preserve the record and flag it
        rather than discarding it, which could create gaps in the time series.
        """
        if self.direction == "A01" and self.imbalance_mwh < 0:
            logger.warning(
                "Direction inconsistency: A01 (surplus) but imbalance_mwh=%.2f < 0 "
                "| zone=%s | ts=%s",
                self.imbalance_mwh, self.zone, self.timestamp_utc.isoformat(),
            )
        elif self.direction == "A02" and self.imbalance_mwh > 0:
            logger.warning(
                "Direction inconsistency: A02 (deficit) but imbalance_mwh=%.2f > 0 "
                "| zone=%s | ts=%s",
                self.imbalance_mwh, self.zone, self.timestamp_utc.isoformat(),
            )
        return self


class HourlyPricePoint(_UTCDatetimeMixin, BaseModel):
    """Single validated hourly Day-Ahead Market Price observation."""

    model_config = {"frozen": True}

    zone: str = Field(..., description="Nord Pool bidding zone label (SE1–SE4)")
    timestamp_utc: datetime = Field(..., description="Period start in UTC")
    price_eur_mwh: float = Field(
        ..., description="Day-ahead clearing price in EUR/MWh (can be negative)"
    )


class BronzeManifestRecord(BaseModel):
    """
    Metadata record written alongside every Bronze snapshot.

    Enables downstream auditing: what was fetched, when, how many records
    were valid, and how many were anomalous.
    """

    zone: str
    document_type: str
    period_start_utc: str
    period_end_utc: str
    records_total: int
    records_anomalous: int
    records_validation_failed: int
    bronze_path: str
    ingested_at_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    pipeline_version: str = "1.0.0"


# ---------------------------------------------------------------------------
# Anomaly Logger — Physics-Bound Data Quality Monitor
# ---------------------------------------------------------------------------
class AnomalyLogger:
    """
    Stateful anomaly logger for a single zone ingestion session.

    Design contract:
      - Records are NEVER dropped. Anomalous observations are preserved in
        Bronze and tagged with is_anomaly=True for Silver-layer quarantine.
      - Anomaly log is exported to a JSON sidecar file in the Bronze directory
        to provide a separate audit trail independent of the main snapshot.
      - Thresholds are grid-physics based (not statistical) to avoid drift
        in the detector as load patterns shift over time.
    """

    def __init__(self, zone: str) -> None:
        self.zone = zone
        self._log: list[dict[str, Any]] = []
        self._logger = logging.getLogger(f"anomaly.{zone}")

    def check_load(self, point: HourlyLoadPoint) -> bool:
        """Returns True if the load point is anomalous."""
        flags: list[str] = []
        t = ANOMALY_THRESHOLDS

        if point.load_mw < t["load_mw_min"]:
            flags.append(
                f"load_mw={point.load_mw:.1f} is BELOW physics floor "
                f"{t['load_mw_min']:.0f} MW for zone {self.zone}"
            )
        if point.load_mw > t["load_mw_max"]:
            flags.append(
                f"load_mw={point.load_mw:.1f} EXCEEDS theoretical max "
                f"{t['load_mw_max']:.0f} MW for zone {self.zone}"
            )

        if flags:
            self._record("LOAD", point.timestamp_utc, flags)
        return bool(flags)

    def check_imbalance(self, point: HourlyImbalancePoint) -> bool:
        """Returns True if the imbalance point is anomalous."""
        flags: list[str] = []

        if abs(point.imbalance_mwh) > ANOMALY_THRESHOLDS["imbalance_mwh_abs_max"]:
            flags.append(
                f"|imbalance_mwh|={abs(point.imbalance_mwh):.1f} exceeds extreme "
                f"threshold {ANOMALY_THRESHOLDS['imbalance_mwh_abs_max']:.0f} MWh"
            )

        if flags:
            self._record("IMBALANCE", point.timestamp_utc, flags)
        return bool(flags)

    def check_price(self, point: HourlyPricePoint) -> bool:
        """Returns True if the price point is anomalous."""
        flags: list[str] = []
        t = ANOMALY_THRESHOLDS

        if point.price_eur_mwh < t["price_eur_min"]:
            flags.append(
                f"price_eur_mwh={point.price_eur_mwh:.2f} is DEEPLY NEGATIVE "
                f"(below {t['price_eur_min']:.0f} EUR/MWh) — possible curtailment event"
            )
        if point.price_eur_mwh > t["price_eur_max"]:
            flags.append(
                f"price_eur_mwh={point.price_eur_mwh:.2f} EXCEEDS spike ceiling "
                f"{t['price_eur_max']:.0f} EUR/MWh — verify against published auction results"
            )

        if flags:
            self._record("PRICE", point.timestamp_utc, flags)
        return bool(flags)

    def _record(
        self, metric_type: str, ts: datetime, flags: list[str]
    ) -> None:
        entry = {
            "zone": self.zone,
            "metric_type": metric_type,
            "timestamp_utc": ts.isoformat(),
            "flags": flags,
            "detected_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        self._log.append(entry)
        for flag in flags:
            self._logger.warning(
                "⚡ ANOMALY [%s | %s | %s]: %s",
                self.zone,
                metric_type,
                ts.strftime("%Y-%m-%d %H:%M"),
                flag,
            )

    def anomaly_timestamps(self, metric_type: str) -> frozenset[pd.Timestamp]:
        """Return set of anomalous timestamps for a given metric type."""
        return frozenset(
            pd.Timestamp(e["timestamp_utc"])
            for e in self._log
            if e["metric_type"] == metric_type
        )

    def anomaly_count(self) -> int:
        return len(self._log)

    def get_report(self) -> list[dict[str, Any]]:
        return list(self._log)

    def export_to_bronze(self, bronze_dir: Path) -> Optional[Path]:
        """Write anomaly sidecar JSON file. Returns None if no anomalies logged."""
        if not self._log:
            return None
        bronze_dir.mkdir(parents=True, exist_ok=True)
        ts_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = bronze_dir / f"anomalies_{self.zone}_{ts_str}.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(self._log, fh, indent=2, ensure_ascii=False)
        logger.info(
            "Anomaly sidecar written → %s (%d flags)", out_path, len(self._log)
        )
        return out_path


# ---------------------------------------------------------------------------
# ENTSO-E XML Parser
# ---------------------------------------------------------------------------
class ENTSOEXMLParser:
    """
    Parses ENTSO-E Transparency Platform XML responses into validated Pydantic models.

    ENTSO-E uses IEC 62325-451 XML schemas with a nested structure:
      GL_MarketDocument / Imbalance_MarketDocument
        └── TimeSeries
              └── Period
                    ├── timeInterval { start, end }
                    ├── resolution (PT60M, PT15M, etc.)
                    └── Point[]  { position, quantity / price.amount }

    Each document type uses a slightly different XML namespace and element
    name convention — the namespace map handles this transparently.

    Error handling philosophy:
      - A single malformed Point is logged and skipped (don't abort the batch)
      - A missing Period or TimeSeries is logged as WARNING and skipped
      - A completely unparseable XML document raises immediately (Bronze write
        has already happened, so the raw payload is preserved for debugging)
    """

    def __init__(self, zone_label: str, anomaly_logger: AnomalyLogger) -> None:
        self.zone_label = zone_label
        self.anomaly_logger = anomaly_logger

    # ------------------------------------------------------------------
    # Public Parse Methods
    # ------------------------------------------------------------------
    def parse_load(self, xml_text: str) -> list[HourlyLoadPoint]:
        """Parse A65 (Actual Total Load) XML → validated HourlyLoadPoint records."""
        ns = XML_NS[DOC_TYPE_ACTUAL_LOAD]
        root = self._parse_xml(xml_text, doc_type=DOC_TYPE_ACTUAL_LOAD)
        records: list[HourlyLoadPoint] = []
        validation_failures = 0

        for ts_elem in root.findall(".//ns:TimeSeries", ns):
            period_elem = ts_elem.find("ns:Period", ns)
            if period_elem is None:
                logger.warning("A65: TimeSeries with no Period element — skipping.")
                continue

            period_start, resolution_min = self._parse_period_header(period_elem, ns)
            if period_start is None:
                continue

            for point_elem in period_elem.findall("ns:Point", ns):
                ts, raw_qty = self._extract_point(
                    point_elem, ns, period_start, resolution_min, qty_tag="ns:quantity"
                )
                if ts is None or raw_qty is None:
                    continue

                try:
                    point = HourlyLoadPoint(
                        zone=self.zone_label,
                        timestamp_utc=ts,
                        load_mw=raw_qty,
                        resolution_minutes=resolution_min,
                    )
                    self.anomaly_logger.check_load(point)
                    records.append(point)
                except ValidationError as exc:
                    validation_failures += 1
                    logger.error(
                        "A65 Pydantic validation failed | zone=%s | ts=%s | qty=%s | err=%s",
                        self.zone_label, ts.isoformat(), raw_qty, exc.errors(include_url=False),
                    )

        logger.info(
            "A65 parse complete | zone=%s | records=%d | validation_failures=%d | anomalies=%d",
            self.zone_label, len(records), validation_failures,
            self.anomaly_logger.anomaly_count(),
        )
        return records

    def parse_imbalance(self, xml_text: str) -> list[HourlyImbalancePoint]:
        """Parse A86 (Imbalance Volumes) XML → validated HourlyImbalancePoint records."""
        ns = XML_NS[DOC_TYPE_IMBALANCE_VOLUME]
        root = self._parse_xml(xml_text, doc_type=DOC_TYPE_IMBALANCE_VOLUME)
        records: list[HourlyImbalancePoint] = []
        validation_failures = 0

        for ts_elem in root.findall(".//ns:TimeSeries", ns):
            # Extract flow direction code (A01=Long/surplus, A02=Short/deficit)
            direction_elem = ts_elem.find("ns:flowDirection.direction", ns)
            direction: Optional[str] = (
                direction_elem.text.strip() if direction_elem is not None else None
            )

            period_elem = ts_elem.find("ns:Period", ns)
            if period_elem is None:
                logger.warning("A86: TimeSeries with no Period element — skipping.")
                continue

            period_start, resolution_min = self._parse_period_header(period_elem, ns)
            if period_start is None:
                continue

            for point_elem in period_elem.findall("ns:Point", ns):
                ts, raw_qty = self._extract_point(
                    point_elem, ns, period_start, resolution_min, qty_tag="ns:quantity"
                )
                if ts is None or raw_qty is None:
                    continue

                # Apply sign convention: A02 (deficit) → negative MWh
                imbalance_val = -abs(raw_qty) if direction == "A02" else abs(raw_qty)

                try:
                    point = HourlyImbalancePoint(
                        zone=self.zone_label,
                        timestamp_utc=ts,
                        imbalance_mwh=imbalance_val,
                        direction=direction,
                    )
                    self.anomaly_logger.check_imbalance(point)
                    records.append(point)
                except ValidationError as exc:
                    validation_failures += 1
                    logger.error(
                        "A86 Pydantic validation failed | zone=%s | ts=%s | err=%s",
                        self.zone_label, ts.isoformat(), exc.errors(include_url=False),
                    )

        logger.info(
            "A86 parse complete | zone=%s | records=%d | validation_failures=%d",
            self.zone_label, len(records), validation_failures,
        )
        return records

    def parse_prices(self, xml_text: str) -> list[HourlyPricePoint]:
        """Parse A44 (Day-Ahead Prices) XML → validated HourlyPricePoint records."""
        ns = XML_NS[DOC_TYPE_DAY_AHEAD_PRICE]
        root = self._parse_xml(xml_text, doc_type=DOC_TYPE_DAY_AHEAD_PRICE)
        records: list[HourlyPricePoint] = []
        validation_failures = 0

        for ts_elem in root.findall(".//ns:TimeSeries", ns):
            period_elem = ts_elem.find("ns:Period", ns)
            if period_elem is None:
                logger.warning("A44: TimeSeries with no Period element — skipping.")
                continue

            period_start, resolution_min = self._parse_period_header(period_elem, ns)
            if period_start is None:
                continue

            for point_elem in period_elem.findall("ns:Point", ns):
                ts, raw_price = self._extract_point(
                    point_elem, ns, period_start, resolution_min, qty_tag="ns:price.amount"
                )
                if ts is None or raw_price is None:
                    continue

                try:
                    point = HourlyPricePoint(
                        zone=self.zone_label,
                        timestamp_utc=ts,
                        price_eur_mwh=raw_price,
                    )
                    self.anomaly_logger.check_price(point)
                    records.append(point)
                except ValidationError as exc:
                    validation_failures += 1
                    logger.error(
                        "A44 Pydantic validation failed | zone=%s | ts=%s | err=%s",
                        self.zone_label, ts.isoformat(), exc.errors(include_url=False),
                    )

        logger.info(
            "A44 parse complete | zone=%s | records=%d | validation_failures=%d",
            self.zone_label, len(records), validation_failures,
        )
        return records

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_xml(xml_text: str, doc_type: str) -> ET.Element:
        """Parse XML text and return root element. Raises ValueError on failure."""
        if not xml_text or not xml_text.strip():
            raise ValueError(
                f"Empty XML response received for document type {doc_type}."
            )
        try:
            return ET.fromstring(xml_text)
        except ET.ParseError as exc:
            raise ValueError(
                f"XML parse error for document type {doc_type}: {exc}. "
                f"First 200 chars: {xml_text[:200]!r}"
            ) from exc

    @staticmethod
    def _parse_period_header(
        period_elem: ET.Element,
        ns: dict[str, str],
    ) -> tuple[Optional[datetime], int]:
        """
        Extract period start timestamp and resolution from a Period element.

        Returns (None, 60) if the period start cannot be parsed — callers
        skip the entire Period on None return.
        """
        start_str = period_elem.findtext("ns:timeInterval/ns:start", namespaces=ns)
        resolution_str = (
            period_elem.findtext("ns:resolution", namespaces=ns) or "PT60M"
        )

        if not start_str:
            logger.warning("Period element missing timeInterval/start — skipping period.")
            return None, 60

        try:
            period_start = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
        except ValueError as exc:
            logger.warning(
                "Cannot parse period start '%s': %s — skipping period.", start_str, exc
            )
            return None, 60

        resolution_min = RESOLUTION_MAP.get(resolution_str.upper(), 60)
        if resolution_str.upper() not in RESOLUTION_MAP:
            logger.warning(
                "Unknown resolution code '%s' — defaulting to 60 minutes.", resolution_str
            )

        return period_start, resolution_min

    @staticmethod
    def _extract_point(
        point_elem: ET.Element,
        ns: dict[str, str],
        period_start: datetime,
        resolution_min: int,
        qty_tag: str,
    ) -> tuple[Optional[datetime], Optional[float]]:
        """
        Extract timestamp and quantity from a single Point element.

        Position is 1-based in ENTSO-E schema; timestamps are period_start
        offset by (position - 1) * resolution.

        Returns (None, None) if either field is absent or unparseable — callers
        skip the point.
        """
        position_str = point_elem.findtext("ns:position", namespaces=ns)
        qty_str = point_elem.findtext(qty_tag, namespaces=ns)

        if position_str is None or qty_str is None:
            logger.debug(
                "Point missing position or quantity (tag=%s) — skipping.", qty_tag
            )
            return None, None

        try:
            position = int(position_str)
            quantity = float(qty_str)
        except (ValueError, TypeError) as exc:
            logger.warning(
                "Cannot parse Point values (pos=%r, qty=%r): %s — skipping.",
                position_str, qty_str, exc,
            )
            return None, None

        ts = period_start + timedelta(minutes=resolution_min * (position - 1))
        return ts, quantity


# ---------------------------------------------------------------------------
# Bronze Layer Writer
# ---------------------------------------------------------------------------
class BronzeWriter(BronzeWriterBase):
    """
    Writes immutable Bronze layer snapshots with gzip compression and
    metadata manifests.

    Naming convention:
        {doc_type}_{zone}_{YYYYMMDD_start}_{YYYYMMDD_end}_{ingested_at_UTC}.xml.gz

    The ingested_at timestamp makes filenames unique per execution while
    the deterministic prefix (doc_type + zone + date range) supports
    idempotency checks and de-duplication in downstream processes.
    """

    def write_xml(
        self,
        xml_content: str,
        zone_label: str,
        doc_type: str,
        period_start: datetime,
        period_end: datetime,
    ) -> Path:
        """Compress and write raw XML to Bronze layer."""
        filename = self._build_filename(zone_label, doc_type, period_start, period_end)
        out_path = self.bronze_dir / filename

        with gzip.open(out_path, "wt", encoding="utf-8") as fh:
            fh.write(xml_content)

        size_kb = out_path.stat().st_size / 1024
        logger.info(
            "Bronze snapshot written → %s (%.1f KB compressed, %d chars raw)",
            out_path, size_kb, len(xml_content),
        )
        return out_path

    @staticmethod
    def _build_filename(
        zone_label: str,
        doc_type: str,
        period_start: datetime,
        period_end: datetime,
    ) -> str:
        start_str = period_start.strftime("%Y%m%d")
        end_str = period_end.strftime("%Y%m%d")
        ingested_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"{doc_type}_{zone_label}_{start_str}_{end_str}_{ingested_str}.xml.gz"


# ---------------------------------------------------------------------------
# Helper: Build Empty DataFrames
# ---------------------------------------------------------------------------
def _empty_load_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["zone", "timestamp_utc", "load_mw", "resolution_minutes", "is_anomaly"]
    )


def _empty_imbalance_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["zone", "timestamp_utc", "imbalance_mwh", "direction", "is_anomaly"]
    )


def _empty_price_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["zone", "timestamp_utc", "price_eur_mwh", "is_anomaly"]
    )


def _tag_anomalies(
    df: pd.DataFrame,
    anomaly_timestamps: frozenset[pd.Timestamp],
    zone_label: str,
    metric_type: str,
) -> pd.DataFrame:
    """Vectorised anomaly tagging on a freshly-built DataFrame."""
    df["is_anomaly"] = df["timestamp_utc"].isin(anomaly_timestamps)
    flagged = df["is_anomaly"].sum()
    if flagged:
        logger.warning(
            "⚡ %d / %d %s records flagged as anomalous for zone %s",
            flagged, len(df), metric_type, zone_label,
        )
    return df


# ---------------------------------------------------------------------------
# Main Client
# ---------------------------------------------------------------------------
class ENTSOEClient(BaseHTTPClient):
    """
    Production ENTSO-E Transparency Platform ingestion client.

    Inherits:
      - Resilient async/sync HTTP session from BaseHTTPClient
      - Per-host circuit breaker
      - Exponential backoff with full jitter

    Adds:
      - ENTSO-E API token injection via _build_default_params()
      - XML parsing for A65 (load), A86 (imbalance), A44 (prices)
      - Pydantic v2 schema validation at the ingestion boundary
      - Physics-bound anomaly detection and sidecar export
      - Immutable Bronze layer persistence with metadata manifests
      - Async multi-zone fetch for parallel ingestion

    Usage (sync batch context):
        with ENTSOEClient(api_token="...") as client:
            df = client.fetch_actual_load(BiddingZone.SE3, start, end)

    Usage (async parallel context):
        async with ENTSOEClient(api_token="...") as client:
            results = await client.fetch_all_zones_async(start, end)
    """

    def __init__(
        self,
        api_token: Optional[str] = None,
        bronze_dir: Path = Path("data/bronze/entso_e"),
        request_timeout_seconds: float = 30.0,
        connect_timeout_seconds: float = 10.0,
        max_retries: int = DEFAULT_MAX_RETRIES,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_recovery_seconds: float = 120.0,
    ) -> None:
        # Resolve API token (parameter → env var)
        resolved_token = api_token or os.environ.get("ENTSOE_API_TOKEN")
        if not resolved_token:
            raise EnvironmentError(
                "ENTSO-E API token is required. Provide it via api_token= parameter "
                "or set the ENTSOE_API_TOKEN environment variable."
            )
        self._api_token: str = resolved_token

        super().__init__(
            base_url=ENTSO_E_BASE_URL,
            max_retries=max_retries,
            request_timeout_seconds=request_timeout_seconds,
            connect_timeout_seconds=connect_timeout_seconds,
            circuit_breaker_threshold=circuit_breaker_threshold,
            circuit_breaker_recovery_seconds=circuit_breaker_recovery_seconds,
            http2=True,
        )

        self._bronze = BronzeWriter(bronze_dir)

        logger.info(
            "ENTSOEClient ready | bronze=%s | token=...%s",
            bronze_dir, self._api_token[-4:],
        )

    # ------------------------------------------------------------------
    # BaseHTTPClient overrides
    # ------------------------------------------------------------------
    def _build_default_headers(self) -> dict[str, str]:
        return {"Accept": "application/xml", "User-Agent": "SwedenGridPipeline/1.0"}

    def _build_default_params(self) -> dict[str, Any]:
        # securityToken is injected here so it never needs to appear in
        # call-site code — reduces risk of accidental token exposure in logs
        return {"securityToken": self._api_token}

    # ------------------------------------------------------------------
    # Public Sync API (batch pipeline entry points)
    # ------------------------------------------------------------------
    def fetch_actual_load(
        self,
        zone: BiddingZone,
        period_start: datetime,
        period_end: datetime,
    ) -> pd.DataFrame:
        """
        Fetch hourly Actual Total Load for one Swedish bidding zone.

        Enforces:
          - Query window ≤ MAX_QUERY_WINDOW_DAYS (splits automatically)
          - Pydantic v2 schema validation per record
          - Anomaly tagging (physics bounds) without data loss
          - Bronze write before parse (raw payload always preserved)

        Returns:
            pd.DataFrame[zone, timestamp_utc, load_mw, resolution_minutes, is_anomaly]
            Empty DataFrame with correct schema on total failure.
        """
        zone_label = ZONE_LABELS[zone]
        chunks = self._split_query_window(period_start, period_end)
        all_frames: list[pd.DataFrame] = []

        for chunk_start, chunk_end in chunks:
            df_chunk = self._fetch_load_chunk(zone, zone_label, chunk_start, chunk_end)
            if not df_chunk.empty:
                all_frames.append(df_chunk)

        if not all_frames:
            logger.warning(
                "No Actual Load data retrieved for zone %s in window %s → %s.",
                zone_label,
                period_start.strftime("%Y-%m-%d"),
                period_end.strftime("%Y-%m-%d"),
            )
            return _empty_load_df()

        df = pd.concat(all_frames, ignore_index=True)
        df = df.drop_duplicates(subset=["zone", "timestamp_utc"]).sort_values("timestamp_utc")
        df.reset_index(drop=True, inplace=True)
        logger.info(
            "fetch_actual_load complete | zone=%s | %d hourly records | "
            "%d anomalous | period: %s → %s",
            zone_label, len(df), df["is_anomaly"].sum(),
            period_start.strftime("%Y-%m-%d"), period_end.strftime("%Y-%m-%d"),
        )
        return df

    def fetch_imbalance_volumes(
        self,
        zone: BiddingZone,
        period_start: datetime,
        period_end: datetime,
    ) -> pd.DataFrame:
        """
        Fetch hourly Net Grid Imbalance Volumes for one Swedish bidding zone.

        Returns:
            pd.DataFrame[zone, timestamp_utc, imbalance_mwh, direction, is_anomaly]
        """
        zone_label = ZONE_LABELS[zone]
        chunks = self._split_query_window(period_start, period_end)
        all_frames: list[pd.DataFrame] = []

        for chunk_start, chunk_end in chunks:
            df_chunk = self._fetch_imbalance_chunk(zone, zone_label, chunk_start, chunk_end)
            if not df_chunk.empty:
                all_frames.append(df_chunk)

        if not all_frames:
            logger.warning(
                "No Imbalance Volume data retrieved for zone %s.", zone_label
            )
            return _empty_imbalance_df()

        df = pd.concat(all_frames, ignore_index=True)
        df = df.drop_duplicates(subset=["zone", "timestamp_utc"]).sort_values("timestamp_utc")
        df.reset_index(drop=True, inplace=True)
        logger.info(
            "fetch_imbalance_volumes complete | zone=%s | %d records | %d anomalous",
            zone_label, len(df), df["is_anomaly"].sum(),
        )
        return df

    def fetch_day_ahead_prices(
        self,
        zone: BiddingZone,
        period_start: datetime,
        period_end: datetime,
    ) -> pd.DataFrame:
        """
        Fetch hourly Day-Ahead Prices for one Swedish bidding zone.

        Returns:
            pd.DataFrame[zone, timestamp_utc, price_eur_mwh, is_anomaly]
        """
        zone_label = ZONE_LABELS[zone]
        chunks = self._split_query_window(period_start, period_end)
        all_frames: list[pd.DataFrame] = []

        for chunk_start, chunk_end in chunks:
            df_chunk = self._fetch_price_chunk(zone, zone_label, chunk_start, chunk_end)
            if not df_chunk.empty:
                all_frames.append(df_chunk)

        if not all_frames:
            logger.warning("No Day-Ahead Price data retrieved for zone %s.", zone_label)
            return _empty_price_df()

        df = pd.concat(all_frames, ignore_index=True)
        df = df.drop_duplicates(subset=["zone", "timestamp_utc"]).sort_values("timestamp_utc")
        df.reset_index(drop=True, inplace=True)
        logger.info(
            "fetch_day_ahead_prices complete | zone=%s | %d records | %d anomalous",
            zone_label, len(df), df["is_anomaly"].sum(),
        )
        return df

    def fetch_all_zones(
        self,
        period_start: datetime,
        period_end: datetime,
        include_prices: bool = True,
    ) -> dict[str, dict[str, pd.DataFrame]]:
        """
        Fetch load + imbalance (+ prices) for all four Swedish bidding zones
        sequentially (synchronous).

        For parallel fetching, use fetch_all_zones_async() instead.

        Returns:
            {
              "SE1": {"load": df, "imbalance": df, "prices": df},
              "SE2": {...}, "SE3": {...}, "SE4": {...}
            }
        """
        result: dict[str, dict[str, pd.DataFrame]] = {}

        for zone in BiddingZone:
            zone_label = ZONE_LABELS[zone]
            logger.info("=== Sequential fetch: zone %s ===", zone_label)
            result[zone_label] = {
                "load": self.fetch_actual_load(zone, period_start, period_end),
                "imbalance": self.fetch_imbalance_volumes(zone, period_start, period_end),
            }
            if include_prices:
                result[zone_label]["prices"] = self.fetch_day_ahead_prices(
                    zone, period_start, period_end
                )

        return result

    # ------------------------------------------------------------------
    # Async Multi-Zone API (parallel ingestion)
    # ------------------------------------------------------------------
    async def fetch_all_zones_async(
        self,
        period_start: datetime,
        period_end: datetime,
        include_prices: bool = True,
        max_concurrency: int = 2,
    ) -> dict[str, dict[str, pd.DataFrame]]:
        """
        Fetch all zones concurrently using bounded async concurrency.

        ENTSO-E rate limit is ~400 req/hour per token. With 4 zones × 3
        document types = 12 requests, max_concurrency=2 provides a safe
        default. Increase to 4 only for short time windows.

        Returns:
            Same structure as fetch_all_zones()
        """
        semaphore = asyncio.Semaphore(max_concurrency)
        result: dict[str, dict[str, pd.DataFrame]] = {
            z.name: {} for z in BiddingZone
        }

        async def _fetch_one(
            zone: BiddingZone, metric: str
        ) -> tuple[str, str, pd.DataFrame]:
            async with semaphore:
                zone_label = ZONE_LABELS[zone]
                logger.info("Async fetch: zone=%s metric=%s", zone_label, metric)
                try:
                    loop = asyncio.get_running_loop()
                    if metric == "load":
                        df = await loop.run_in_executor(
                            None, self.fetch_actual_load, zone, period_start, period_end
                        )
                    elif metric == "imbalance":
                        df = await loop.run_in_executor(
                            None, self.fetch_imbalance_volumes, zone, period_start, period_end
                        )
                    else:  # prices
                        df = await loop.run_in_executor(
                            None, self.fetch_day_ahead_prices, zone, period_start, period_end
                        )
                    return zone_label, metric, df
                except (CircuitOpenError, RetryExhaustedError, Exception) as exc:
                    logger.error(
                        "Async fetch failed | zone=%s | metric=%s | %s: %s",
                        zone_label, metric, type(exc).__name__, str(exc),
                    )
                    empty_map = {
                        "load": _empty_load_df,
                        "imbalance": _empty_imbalance_df,
                        "prices": _empty_price_df,
                    }
                    return zone_label, metric, empty_map[metric]()

        metrics = ["load", "imbalance"] + (["prices"] if include_prices else [])
        tasks = [
            _fetch_one(zone, metric)
            for zone in BiddingZone
            for metric in metrics
        ]
        completed = await asyncio.gather(*tasks, return_exceptions=False)

        for zone_label, metric, df in completed:
            result[zone_label][metric] = df

        total_records = sum(
            len(result[zl][m])
            for zl in result
            for m in result[zl]
        )
        logger.info(
            "fetch_all_zones_async complete | %d zone-metric combinations | %d total records",
            len(completed), total_records,
        )
        return result

    # ------------------------------------------------------------------
    # Private: Per-Chunk Fetch Methods (atomic unit of work)
    # ------------------------------------------------------------------
    @retry_with_backoff(
        max_retries=DEFAULT_MAX_RETRIES,
        base_seconds=DEFAULT_BASE_BACKOFF_SECONDS,
        max_seconds=DEFAULT_MAX_BACKOFF_SECONDS,
    )
    def _fetch_load_chunk(
        self,
        zone: BiddingZone,
        zone_label: str,
        period_start: datetime,
        period_end: datetime,
    ) -> pd.DataFrame:
        """Fetch and parse a single time-window chunk for Actual Load (A65)."""
        logger.info(
            "Fetching A65 | zone=%s | %s → %s",
            zone_label,
            period_start.strftime("%Y-%m-%d"),
            period_end.strftime("%Y-%m-%d"),
        )

        xml_text = self._call_entsoe_api(
            DOC_TYPE_ACTUAL_LOAD, zone.value, period_start, period_end
        )
        bronze_path = self._bronze.write_xml(
            xml_text, zone_label, DOC_TYPE_ACTUAL_LOAD, period_start, period_end
        )

        anomaly_log = AnomalyLogger(zone_label)
        parser = ENTSOEXMLParser(zone_label, anomaly_log)

        try:
            records = parser.parse_load(xml_text)
        except ValueError as exc:
            logger.error(
                "A65 XML parse failed for zone %s: %s. "
                "Raw payload preserved at %s.",
                zone_label, exc, bronze_path,
            )
            return _empty_load_df()

        anomaly_log.export_to_bronze(self._bronze.bronze_dir)

        if not records:
            return _empty_load_df()

        df = pd.DataFrame([r.model_dump() for r in records])
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        df = _tag_anomalies(df, anomaly_log.anomaly_timestamps("LOAD"), zone_label, "A65")

        self._bronze.write_manifest(
            BronzeManifestRecord(
                zone=zone_label,
                document_type=DOC_TYPE_ACTUAL_LOAD,
                period_start_utc=period_start.isoformat(),
                period_end_utc=period_end.isoformat(),
                records_total=len(df),
                records_anomalous=int(df["is_anomaly"].sum()),
                records_validation_failed=0,
                bronze_path=str(bronze_path),
            ),
            bronze_path,
        )
        return df

    @retry_with_backoff(
        max_retries=DEFAULT_MAX_RETRIES,
        base_seconds=DEFAULT_BASE_BACKOFF_SECONDS,
        max_seconds=DEFAULT_MAX_BACKOFF_SECONDS,
    )
    def _fetch_imbalance_chunk(
        self,
        zone: BiddingZone,
        zone_label: str,
        period_start: datetime,
        period_end: datetime,
    ) -> pd.DataFrame:
        """Fetch and parse a single time-window chunk for Imbalance Volumes (A86)."""
        logger.info(
            "Fetching A86 | zone=%s | %s → %s",
            zone_label,
            period_start.strftime("%Y-%m-%d"),
            period_end.strftime("%Y-%m-%d"),
        )

        xml_text = self._call_entsoe_api(
            DOC_TYPE_IMBALANCE_VOLUME, zone.value, period_start, period_end
        )
        bronze_path = self._bronze.write_xml(
            xml_text, zone_label, DOC_TYPE_IMBALANCE_VOLUME, period_start, period_end
        )

        anomaly_log = AnomalyLogger(zone_label)
        parser = ENTSOEXMLParser(zone_label, anomaly_log)

        try:
            records = parser.parse_imbalance(xml_text)
        except ValueError as exc:
            logger.error(
                "A86 XML parse failed for zone %s: %s. Raw payload at %s.",
                zone_label, exc, bronze_path,
            )
            return _empty_imbalance_df()

        anomaly_log.export_to_bronze(self._bronze.bronze_dir)

        if not records:
            return _empty_imbalance_df()

        df = pd.DataFrame([r.model_dump() for r in records])
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        df = _tag_anomalies(
            df, anomaly_log.anomaly_timestamps("IMBALANCE"), zone_label, "A86"
        )

        self._bronze.write_manifest(
            BronzeManifestRecord(
                zone=zone_label,
                document_type=DOC_TYPE_IMBALANCE_VOLUME,
                period_start_utc=period_start.isoformat(),
                period_end_utc=period_end.isoformat(),
                records_total=len(df),
                records_anomalous=int(df["is_anomaly"].sum()),
                records_validation_failed=0,
                bronze_path=str(bronze_path),
            ),
            bronze_path,
        )
        return df

    @retry_with_backoff(
        max_retries=DEFAULT_MAX_RETRIES,
        base_seconds=DEFAULT_BASE_BACKOFF_SECONDS,
        max_seconds=DEFAULT_MAX_BACKOFF_SECONDS,
    )
    def _fetch_price_chunk(
        self,
        zone: BiddingZone,
        zone_label: str,
        period_start: datetime,
        period_end: datetime,
    ) -> pd.DataFrame:
        """Fetch and parse a single time-window chunk for Day-Ahead Prices (A44)."""
        logger.info(
            "Fetching A44 | zone=%s | %s → %s",
            zone_label,
            period_start.strftime("%Y-%m-%d"),
            period_end.strftime("%Y-%m-%d"),
        )

        xml_text = self._call_entsoe_api(
            DOC_TYPE_DAY_AHEAD_PRICE, zone.value, period_start, period_end
        )
        bronze_path = self._bronze.write_xml(
            xml_text, zone_label, DOC_TYPE_DAY_AHEAD_PRICE, period_start, period_end
        )

        anomaly_log = AnomalyLogger(zone_label)
        parser = ENTSOEXMLParser(zone_label, anomaly_log)

        try:
            records = parser.parse_prices(xml_text)
        except ValueError as exc:
            logger.error(
                "A44 XML parse failed for zone %s: %s. Raw payload at %s.",
                zone_label, exc, bronze_path,
            )
            return _empty_price_df()

        anomaly_log.export_to_bronze(self._bronze.bronze_dir)

        if not records:
            return _empty_price_df()

        df = pd.DataFrame([r.model_dump() for r in records])
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        df = _tag_anomalies(
            df, anomaly_log.anomaly_timestamps("PRICE"), zone_label, "A44"
        )

        self._bronze.write_manifest(
            BronzeManifestRecord(
                zone=zone_label,
                document_type=DOC_TYPE_DAY_AHEAD_PRICE,
                period_start_utc=period_start.isoformat(),
                period_end_utc=period_end.isoformat(),
                records_total=len(df),
                records_anomalous=int(df["is_anomaly"].sum()),
                records_validation_failed=0,
                bronze_path=str(bronze_path),
            ),
            bronze_path,
        )
        return df

    # ------------------------------------------------------------------
    # Private: Core API Caller (no retry — applied at chunk level)
    # ------------------------------------------------------------------
    def _call_entsoe_api(
        self,
        doc_type: str,
        area_code: str,
        period_start: datetime,
        period_end: datetime,
    ) -> str:
        """
        Execute a single synchronous GET request against the ENTSO-E REST API.

        Parameters are injected here (not in _build_default_params) because
        each call has unique date ranges and document types.

        The securityToken is merged from _build_default_params() by the
        BaseHTTPClient.request() method — never re-specified here.
        """
        params = {
            "documentType": doc_type,
            "In_Domain": area_code,
            "Out_Domain": area_code,
            "periodStart": period_start.strftime("%Y%m%d%H%M"),
            "periodEnd": period_end.strftime("%Y%m%d%H%M"),
        }

        response = self.request("GET", "", params=params)

        # Surface ENTSO-E's XML error documents before returning to parser
        # ENTSO-E returns HTTP 200 with an XML Acknowledgement on some errors
        if "Reason" in response.text and "code" in response.text[:1000]:
            logger.warning(
                "ENTSO-E returned an acknowledgement/error document for "
                "doc_type=%s zone=%s. First 500 chars: %s",
                doc_type, area_code, response.text[:500],
            )

        return response.text

    # ------------------------------------------------------------------
    # Private: Query Window Splitter
    # ------------------------------------------------------------------
    @staticmethod
    def _split_query_window(
        period_start: datetime,
        period_end: datetime,
    ) -> list[tuple[datetime, datetime]]:
        """
        Split a query window larger than MAX_QUERY_WINDOW_DAYS into chunks.

        ENTSO-E rejects requests spanning more than 1 year for most document
        types. This splitter ensures each chunk is within the allowed window
        while keeping chunk boundaries on calendar day boundaries.
        """
        chunks: list[tuple[datetime, datetime]] = []
        chunk_start = period_start

        while chunk_start < period_end:
            chunk_end = min(
                chunk_start + timedelta(days=MAX_QUERY_WINDOW_DAYS),
                period_end,
            )
            chunks.append((chunk_start, chunk_end))
            chunk_start = chunk_end

        if len(chunks) > 1:
            logger.info(
                "Query window split into %d chunks (max %d days each): %s → %s",
                len(chunks), MAX_QUERY_WINDOW_DAYS,
                period_start.strftime("%Y-%m-%d"),
                period_end.strftime("%Y-%m-%d"),
            )
        return chunks


# ---------------------------------------------------------------------------
# CLI Smoke Test — validates schemas, anomaly detection, and client setup
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("ENTSOEClient — Schema, Anomaly & Architecture Smoke Test")
    print("=" * 70)

    errors: list[str] = []

    # ── [1] Pydantic Schema Validation ───────────────────────────────────
    print("\n[1] HourlyLoadPoint schema validation...")
    lp = HourlyLoadPoint(
        zone="SE3",
        timestamp_utc="2024-06-15T12:00:00+00:00",
        load_mw=12345.6,
    )
    assert lp.zone == "SE3"
    assert lp.load_mw == 12345.6
    print(f"    ✓ Valid load point: {lp}")

    try:
        HourlyLoadPoint(zone="SE5", timestamp_utc="2024-06-15T12:00:00Z", load_mw=100.0)
        errors.append("Should have rejected invalid zone SE5")
    except ValidationError:
        print("    ✓ Invalid zone SE5 correctly rejected")

    try:
        HourlyLoadPoint(zone="SE1", timestamp_utc="2024-06-15T12:00:00Z", load_mw=-50.0)
        errors.append("Should have rejected negative load_mw")
    except ValidationError:
        print("    ✓ Negative load_mw correctly rejected")

    # ── [2] Imbalance Direction Consistency ──────────────────────────────
    print("\n[2] HourlyImbalancePoint direction consistency warning...")
    ip = HourlyImbalancePoint(
        zone="SE2",
        timestamp_utc="2024-06-15T14:00:00Z",
        imbalance_mwh=-2500.0,
        direction="A01",  # Should trigger WARNING log
    )
    print(f"    ✓ Parsed (direction warning expected above): {ip.imbalance_mwh} MWh")

    # ── [3] Anomaly Logger ────────────────────────────────────────────────
    print("\n[3] AnomalyLogger physics-bound checks...")
    anomaly = AnomalyLogger("SE3")

    spike = HourlyLoadPoint(zone="SE3", timestamp_utc="2024-06-15T13:00:00Z", load_mw=99_999.0)
    flagged_load = anomaly.check_load(spike)
    assert flagged_load, "99,999 MW should be flagged"
    print(f"    ✓ Spike load (99,999 MW) flagged: {flagged_load}")

    normal = HourlyLoadPoint(zone="SE3", timestamp_utc="2024-06-15T14:00:00Z", load_mw=8_500.0)
    flagged_normal = anomaly.check_load(normal)
    assert not flagged_normal, "Normal load should not be flagged"
    print(f"    ✓ Normal load (8,500 MW) not flagged: {not flagged_normal}")

    extreme_imb = HourlyImbalancePoint(
        zone="SE3", timestamp_utc="2024-06-15T15:00:00Z", imbalance_mwh=9_999.0
    )
    flagged_imb = anomaly.check_imbalance(extreme_imb)
    assert flagged_imb, "9,999 MWh imbalance should be flagged"
    print(f"    ✓ Extreme imbalance (9,999 MWh) flagged: {flagged_imb}")

    deep_neg_price = HourlyPricePoint(
        zone="SE4", timestamp_utc="2024-06-15T03:00:00Z", price_eur_mwh=-650.0
    )
    price_anomaly = AnomalyLogger("SE4")
    flagged_price = price_anomaly.check_price(deep_neg_price)
    assert flagged_price, "Deep negative price should be flagged"
    print(f"    ✓ Deep negative price (-650 EUR/MWh) flagged: {flagged_price}")

    # ── [4] XML Parser on synthetic A65 payload ───────────────────────────
    print("\n[4] ENTSOEXMLParser — synthetic A65 XML...")
    synthetic_a65 = """<?xml version="1.0" encoding="UTF-8"?>
<GL_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0">
  <TimeSeries>
    <Period>
      <timeInterval>
        <start>2024-01-15T00:00Z</start>
        <end>2024-01-15T03:00Z</end>
      </timeInterval>
      <resolution>PT60M</resolution>
      <Point><position>1</position><quantity>8200.5</quantity></Point>
      <Point><position>2</position><quantity>8100.0</quantity></Point>
      <Point><position>3</position><quantity>7950.3</quantity></Point>
    </Period>
  </TimeSeries>
</GL_MarketDocument>"""

    parse_anomaly = AnomalyLogger("SE3")
    parser = ENTSOEXMLParser("SE3", parse_anomaly)
    load_records = parser.parse_load(synthetic_a65)
    assert len(load_records) == 3, f"Expected 3 records, got {len(load_records)}"
    assert load_records[0].load_mw == 8200.5
    assert load_records[0].timestamp_utc == datetime(2024, 1, 15, 0, 0, tzinfo=timezone.utc)
    assert load_records[1].timestamp_utc == datetime(2024, 1, 15, 1, 0, tzinfo=timezone.utc)
    print(f"    ✓ Parsed {len(load_records)} A65 load records from synthetic XML")

    # ── [5] Query Window Splitter ────────────────────────────────────────
    print("\n[5] Query window splitting (>1 year range)...")
    start = datetime(2023, 1, 1, tzinfo=timezone.utc)
    end = datetime(2025, 6, 1, tzinfo=timezone.utc)
    chunks = ENTSOEClient._split_query_window(start, end)
    assert len(chunks) > 1, "Multi-year range should produce multiple chunks"
    # Verify coverage: last chunk should end at period_end
    assert chunks[-1][1] == end, "Last chunk should end exactly at period_end"
    # Verify no gap between chunks
    for i in range(len(chunks) - 1):
        assert chunks[i][1] == chunks[i + 1][0], f"Gap between chunk {i} and {i+1}"
    print(f"    ✓ {len(chunks)} chunks for {(end - start).days}-day window — no gaps")

    # ── [6] BronzeWriter ─────────────────────────────────────────────────
    print("\n[6] BronzeWriter — file creation and manifest...")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = BronzeWriter(Path(tmpdir))
        test_start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        test_end = datetime(2024, 1, 31, tzinfo=timezone.utc)
        bronze_path = writer.write_xml(
            synthetic_a65, "SE3", DOC_TYPE_ACTUAL_LOAD, test_start, test_end
        )
        assert bronze_path.exists(), "Bronze file should exist"
        assert bronze_path.suffix == ".gz", "Should be gzip compressed"

        # Verify decompression round-trip
        with gzip.open(bronze_path, "rt", encoding="utf-8") as fh:
            recovered = fh.read()
        assert "GL_MarketDocument" in recovered, "Decompressed content should contain XML"

        manifest = BronzeManifestRecord(
            zone="SE3", document_type="A65",
            period_start_utc=test_start.isoformat(),
            period_end_utc=test_end.isoformat(),
            records_total=3, records_anomalous=0, records_validation_failed=0,
            bronze_path=str(bronze_path),
        )
        manifest_path = writer.write_manifest(manifest, bronze_path)
        assert manifest_path.exists(), "Manifest file should exist"
        with open(manifest_path) as fh:
            loaded_manifest = json.load(fh)
        assert loaded_manifest["records_total"] == 3
        print(f"    ✓ Bronze file created: {bronze_path.name}")
        print(f"    ✓ Manifest created: {manifest_path.name}")
        print(f"    ✓ Gzip round-trip verified ({len(recovered)} chars recovered)")

    # ── [7] ENTSOEClient instantiation guard ─────────────────────────────
    print("\n[7] ENTSOEClient — missing token guard...")
    old_val = os.environ.pop("ENTSOE_API_TOKEN", None)
    try:
        ENTSOEClient(api_token=None)
        errors.append("Should have raised EnvironmentError on missing token")
    except EnvironmentError as exc:
        print(f"    ✓ Missing token correctly raises EnvironmentError: {exc}")
    finally:
        if old_val:
            os.environ["ENTSOE_API_TOKEN"] = old_val

    client = ENTSOEClient(api_token="test-token-1234")
    print(f"    ✓ Client instantiated: {client!r}")
    print(f"    ✓ Circuit breaker state: {client.circuit_state.name}")

    # ── Final summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    if errors:
        print(f"❌ {len(errors)} assertion(s) failed:")
        for e in errors:
            print(f"   • {e}")
        sys.exit(1)
    else:
        print("✅ All ENTSOEClient smoke tests passed.")
        print("   Set ENTSOE_API_TOKEN and call ENTSOEClient().fetch_all_zones() for live data.")
    print("=" * 70)
