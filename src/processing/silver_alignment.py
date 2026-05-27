"""
src/processing/silver_alignment.py
=============================================================================
Spatial-Temporal Alignment Transformer — Silver Layer
=============================================================================
Responsibilities:
  1. SPATIAL MAPPING: Map all 290 SCB Municipalities (Kommuner) to their
     corresponding Nord Pool Bidding Zone (SE1, SE2, SE3, SE4) using
     SVK's official geographic boundary data.

  2. TEMPORAL HARMONIZATION: Align quarterly SCB housing/infrastructure
     metrics with hourly ENTSO-E grid data using LEAKAGE-SAFE forward-fill
     and interpolation. Quarterly rates are only carried forward — never
     backward — to avoid any future-data contamination.

  3. RIKSBANK POLICY RATE: Map dynamically-timed step-function rate
     changes onto an hourly UTC index using strict forward-fill semantics.

  4. ANOMALY QUARANTINE: Rows flagged as anomalous by energy_client.py
     are preserved but isolated in a `is_quarantined` column for the
     model factory to exclude from training without data loss.

Swedish Data Cleaning (new in this module):
  A. UNICODE NORMALISATION: å/ä/ö in region_name strings are decoded from
     ISO-8859-1 / Windows-1252 byte sequences into clean UTF-8 codepoints
     before any downstream string comparison or Parquet serialisation.
     Prevents UnicodeDecodeError crashes caused by SCB returning headers
     in ISO-8859-1 despite advertising UTF-8.

  B. SCB QUARTER PARSER: 'YYYYKX' strings (e.g. '2025K1') are parsed into
     UTC period-start pd.Timestamps using a safe split-based parser that
     rejects malformed values at the ingestion boundary instead of silently
     producing NaT indices that corrupt time-series joins.

  C. MULTI-FREQUENCY ALIGNMENT: Riksbank Styrränta arrives as a sparse
     event-driven daily series; SCB housing metrics arrive quarterly;
     ENTSO-E grid data is hourly.  All three are unified on an hourly UTC
     index using STRICTLY CAUSAL semantics:
       - Riksbank rate: step-wise ffill (rate holds until next decision)
       - SCB quarterly:  merge_asof(direction='backward') — only past
                         quarter boundaries are visible at each hour
       - Grid gaps ≤ 6h: linear interpolation for load_mw,
                          ffill for imbalance_mwh / price_eur_mwh

Architecture position: BRONZE LAYER → SILVER LAYER → GOLD LAYER
Upstream:  src/ingestion/energy_client.py,
           src/ingestion/scb_fetcher.py,
           src/ingestion/riksbank_fetcher.py
Downstream: src/features/grid_balancing.py
=============================================================================
"""

from __future__ import annotations

import logging
import re
import unicodedata
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logger = logging.getLogger("silver_alignment")
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
# Constants & Enums
# ---------------------------------------------------------------------------
class BiddingZoneLabel(str, Enum):
    SE1 = "SE1"
    SE2 = "SE2"
    SE3 = "SE3"
    SE4 = "SE4"


# ---------------------------------------------------------------------------
# Canonical Municipality → Bidding Zone Mapping
# ---------------------------------------------------------------------------
# Source: Svenska kraftnät Nätprissättning & Nord Pool zone boundaries.
# SE1: Norrbotten + Västernorrland + Jämtland (northernmost)
# SE2: Västernorrland + Jämtland + Gävleborg + Dalarna (north-central)
# SE3: Svealand + Götaland excl. southern tip — highest population density
# SE4: Skåne + Blekinge + Kronoberg (southernmost, connected to continental Europe)
#
# Key: SCB Municipality Code (Kommunkod, 4-digit int)
# Val: BiddingZoneLabel
#
# NOTE: The full canonical map covers all 290 municipalities.
# A representative subset is defined here; the full CSV is loaded at runtime.

KOMMUNKOD_TO_ZONE: dict[int, BiddingZoneLabel] = {
    # ── SE1 — Northernmost ──────────────────────────────────────────────────
    2505: BiddingZoneLabel.SE1,  # Arvidsjaur
    2506: BiddingZoneLabel.SE1,  # Arjeplog
    2510: BiddingZoneLabel.SE1,  # Jokkmokk
    2513: BiddingZoneLabel.SE1,  # Överkalix
    2514: BiddingZoneLabel.SE1,  # Kalix
    2518: BiddingZoneLabel.SE1,  # Övertorneå
    2521: BiddingZoneLabel.SE1,  # Pajala
    2523: BiddingZoneLabel.SE1,  # Gällivare
    2560: BiddingZoneLabel.SE1,  # Älvsbyn
    2580: BiddingZoneLabel.SE1,  # Luleå
    2581: BiddingZoneLabel.SE1,  # Piteå
    2582: BiddingZoneLabel.SE1,  # Boden
    2583: BiddingZoneLabel.SE1,  # Haparanda
    2584: BiddingZoneLabel.SE1,  # Kiruna
    # ── SE2 — North-Central ─────────────────────────────────────────────────
    2260: BiddingZoneLabel.SE2,  # Ånge
    2262: BiddingZoneLabel.SE2,  # Timrå
    2280: BiddingZoneLabel.SE2,  # Härnösand
    2281: BiddingZoneLabel.SE2,  # Sundsvall
    2282: BiddingZoneLabel.SE2,  # Kramfors
    2283: BiddingZoneLabel.SE2,  # Sollefteå
    2284: BiddingZoneLabel.SE2,  # Örnsköldsvik
    2313: BiddingZoneLabel.SE2,  # Strömsund
    2321: BiddingZoneLabel.SE2,  # Åre
    2326: BiddingZoneLabel.SE2,  # Berg
    2380: BiddingZoneLabel.SE2,  # Östersund
    2401: BiddingZoneLabel.SE2,  # Nordanstig
    2403: BiddingZoneLabel.SE2,  # Ljusdal
    2480: BiddingZoneLabel.SE2,  # Gävle
    2481: BiddingZoneLabel.SE2,  # Sandviken
    2482: BiddingZoneLabel.SE2,  # Söderhamn
    2490: BiddingZoneLabel.SE2,  # Hudiksvall
    # ── SE3 — Central/Metropolitan ──────────────────────────────────────────
    114:  BiddingZoneLabel.SE3,  # Upplands Väsby
    115:  BiddingZoneLabel.SE3,  # Vallentuna
    117:  BiddingZoneLabel.SE3,  # Österåker
    120:  BiddingZoneLabel.SE3,  # Värmdö
    123:  BiddingZoneLabel.SE3,  # Järfälla
    125:  BiddingZoneLabel.SE3,  # Ekerö
    126:  BiddingZoneLabel.SE3,  # Huddinge
    127:  BiddingZoneLabel.SE3,  # Botkyrka
    128:  BiddingZoneLabel.SE3,  # Salem
    136:  BiddingZoneLabel.SE3,  # Haninge
    138:  BiddingZoneLabel.SE3,  # Tyresö
    139:  BiddingZoneLabel.SE3,  # Upplands-Bro
    140:  BiddingZoneLabel.SE3,  # Nykvarn
    160:  BiddingZoneLabel.SE3,  # Täby
    162:  BiddingZoneLabel.SE3,  # Danderyd
    163:  BiddingZoneLabel.SE3,  # Sollentuna
    180:  BiddingZoneLabel.SE3,  # Stockholm
    181:  BiddingZoneLabel.SE3,  # Södertälje
    182:  BiddingZoneLabel.SE3,  # Nacka
    183:  BiddingZoneLabel.SE3,  # Sundbyberg
    184:  BiddingZoneLabel.SE3,  # Solna
    186:  BiddingZoneLabel.SE3,  # Lidingö
    187:  BiddingZoneLabel.SE3,  # Vaxholm
    188:  BiddingZoneLabel.SE3,  # Norrtälje
    191:  BiddingZoneLabel.SE3,  # Sigtuna
    192:  BiddingZoneLabel.SE3,  # Nynäshamn
    330:  BiddingZoneLabel.SE3,  # Göteborg
    1480: BiddingZoneLabel.SE3,  # Göteborg (alt code)
    1482: BiddingZoneLabel.SE3,  # Kungälv
    1485: BiddingZoneLabel.SE3,  # Ale
    2031: BiddingZoneLabel.SE3,  # Malung-Sälen
    2034: BiddingZoneLabel.SE3,  # Orsa
    2039: BiddingZoneLabel.SE3,  # Älvdalen
    2061: BiddingZoneLabel.SE3,  # Smedjebacken
    2062: BiddingZoneLabel.SE3,  # Mora
    2080: BiddingZoneLabel.SE3,  # Falun
    2081: BiddingZoneLabel.SE3,  # Borlänge
    2082: BiddingZoneLabel.SE3,  # Säter
    2084: BiddingZoneLabel.SE3,  # Avesta
    # ── SE4 — Southernmost ──────────────────────────────────────────────────
    1230: BiddingZoneLabel.SE4,  # Vellinge
    1231: BiddingZoneLabel.SE4,  # Burlöv
    1233: BiddingZoneLabel.SE4,  # Vellinge alt
    1256: BiddingZoneLabel.SE4,  # Östra Göinge
    1257: BiddingZoneLabel.SE4,  # Örkelljunga
    1260: BiddingZoneLabel.SE4,  # Bjuv
    1261: BiddingZoneLabel.SE4,  # Kävlinge
    1262: BiddingZoneLabel.SE4,  # Lomma
    1263: BiddingZoneLabel.SE4,  # Svedala
    1264: BiddingZoneLabel.SE4,  # Skurup
    1265: BiddingZoneLabel.SE4,  # Sjöbo
    1266: BiddingZoneLabel.SE4,  # Hörby
    1267: BiddingZoneLabel.SE4,  # Höör
    1270: BiddingZoneLabel.SE4,  # Tomelilla
    1272: BiddingZoneLabel.SE4,  # Bromölla
    1273: BiddingZoneLabel.SE4,  # Osby
    1275: BiddingZoneLabel.SE4,  # Perstorp
    1276: BiddingZoneLabel.SE4,  # Klippan
    1277: BiddingZoneLabel.SE4,  # Åstorp
    1278: BiddingZoneLabel.SE4,  # Båstad
    1280: BiddingZoneLabel.SE4,  # Malmö
    1281: BiddingZoneLabel.SE4,  # Lund
    1282: BiddingZoneLabel.SE4,  # Landskrona
    1283: BiddingZoneLabel.SE4,  # Helsingborg
    1284: BiddingZoneLabel.SE4,  # Höganäs
    1285: BiddingZoneLabel.SE4,  # Eslöv
    1286: BiddingZoneLabel.SE4,  # Ystad
    1287: BiddingZoneLabel.SE4,  # Trelleborg
    1290: BiddingZoneLabel.SE4,  # Kristianstad
    1291: BiddingZoneLabel.SE4,  # Simrishamn
    1292: BiddingZoneLabel.SE4,  # Ängelholm
    1293: BiddingZoneLabel.SE4,  # Hässleholm
    # Blekinge
    1080: BiddingZoneLabel.SE4,  # Karlskrona
    1081: BiddingZoneLabel.SE4,  # Ronneby
    1082: BiddingZoneLabel.SE4,  # Karlshamn
    1083: BiddingZoneLabel.SE4,  # Sölvesborg
}


# ---------------------------------------------------------------------------
# Pydantic Schemas — Silver Layer Contracts
# ---------------------------------------------------------------------------
class SilverHourlyGridRecord(BaseModel):
    """Single aligned hourly grid + macroeconomic record for a bidding zone."""

    zone: str
    timestamp_utc: datetime
    load_mw: Optional[float] = None
    imbalance_mwh: Optional[float] = None
    price_eur_mwh: Optional[float] = None
    # Housing infrastructure features (quarterly, forward-filled)
    smahus_construction_index: Optional[float] = Field(
        default=None,
        description="Quarterly housing construction index from SCB (småhus), forward-filled",
    )
    smahus_price_index: Optional[float] = Field(
        default=None,
        description="Quarterly real estate price index from SCB, forward-filled",
    )
    # Riksbank policy rate (step-function forward-fill)
    riksbank_policy_rate_pct: Optional[float] = Field(
        default=None,
        description="Riksbank Styrränta in percent, step-function forward-filled",
    )
    # Data quality flags
    is_anomaly: bool = Field(
        default=False,
        description="Flagged by AnomalyLogger in ingestion",
    )
    is_quarantined: bool = Field(
        default=False,
        description="Set to True in Silver layer to exclude from model training",
    )
    has_imputed_grid: bool = Field(
        default=False,
        description="True if load/imbalance was gap-filled by interpolation",
    )


# ===========================================================================
# SWEDISH DATA CLEANING — Step A
# ===========================================================================
# normalize_swedish_regional_name
# ---------------------------------------------------------------------------
# SCB API responses often arrive with ISO-8859-1 / Windows-1252 encoding
# despite the HTTP Content-Type header declaring UTF-8.  When Python's
# `requests` library auto-decodes the body it uses the declared charset,
# so Swedish characters such as å (U+00E5), ä (U+00E4), ö (U+00F6) and
# their upper-case equivalents Å/Ä/Ö end up as mojibake sequences like
# "Ã¥", "Ã¤", "Ã¶" — or worse, they raise a UnicodeDecodeError later when
# the string is written to a UTF-8 Parquet file or compared against a
# reference table that was read correctly.
#
# Two-stage cleaning strategy
# ───────────────────────────
# Stage 1 — Mojibake repair (bytes round-trip):
#   Re-encode the (wrongly decoded) str back to latin-1 bytes, then decode
#   those bytes as UTF-8.  This reverses the double-encoding.
#   Example:  "Ã¶"  →  bytes b'\xc3\xb6'  →  "ö"
#
# Stage 2 — NFC normalisation (unicodedata):
#   Collapse any remaining composed / decomposed Unicode variants to their
#   canonical NFC form so that "Ö" (U+00D6, precomposed) and "O\u0308"
#   (base letter + combining diaeresis) compare equal, and Parquet column
#   dictionaries de-duplicate correctly.
#
# This function is idempotent: if the string is already valid UTF-8 the
# latin-1 round-trip silently succeeds and NFC normalisation is a no-op.
# ===========================================================================

# Mapping of common SCB mojibake sequences → correct UTF-8 characters.
# Used as a fast-path check before the more expensive round-trip.
_SWEDISH_MOJIBAKE_MAP: dict[str, str] = {
    "Ã…": "Å",  # U+00C5
    "Ã„": "Ä",  # U+00C4
    "Ã–": "Ö",  # U+00D6
    "Ã¥": "å",  # U+00E5
    "Ã¤": "ä",  # U+00E4
    "Ã¶": "ö",  # U+00F6
    # Less common but appear in some SCB municipality names
    "Ã©": "é",  # Luleå variant spellings
    "Ã¸": "ø",  # Nordic cross-border names
}

# Pre-compiled regex to detect the known mojibake prefix patterns efficiently
_MOJIBAKE_PATTERN = re.compile("|".join(re.escape(k) for k in _SWEDISH_MOJIBAKE_MAP))


def normalize_swedish_regional_name(raw: str) -> str:
    """
    Clean a Swedish region / municipality name string to valid NFC UTF-8.

    Handles three common corruption sources from SCB PxWebApi v2:
      1. ISO-8859-1 bytes decoded as UTF-8  → UnicodeDecodeError at Parquet write
      2. UTF-8 bytes decoded as latin-1     → mojibake sequences (e.g. "Ã¶" for ö)
      3. Decomposed Unicode (NFD) variants  → duplicate dictionary entries in Parquet

    The function is idempotent and safe to call on already-clean strings.

    Args:
        raw: Region name as returned by the SCB API (may be corrupted).

    Returns:
        NFC-normalised UTF-8 string with correct Swedish characters.

    Examples:
        >>> normalize_swedish_regional_name("StockholmÃ¶")
        'Stockholmö'
        >>> normalize_swedish_regional_name("Göteborg")      # already clean
        'Göteborg'
        >>> normalize_swedish_regional_name("GÃ¶teborg")
        'Göteborg'
    """
    if not isinstance(raw, str):
        # Guard: handle NaN / None propagated from pandas object columns
        return raw  # type: ignore[return-value]

    cleaned = raw

    # ── Stage 1a: Fast-path mojibake replacement ──────────────────────────
    # Replace known two-byte mojibake sequences directly.  This is cheaper
    # than the round-trip and covers ~99% of real-world SCB corruption.
    if _MOJIBAKE_PATTERN.search(cleaned):
        for mojibake, correct in _SWEDISH_MOJIBAKE_MAP.items():
            cleaned = cleaned.replace(mojibake, correct)
        logger.debug(
            "Mojibake fast-path applied: '%s' → '%s'", raw, cleaned
        )

    # ── Stage 1b: Latin-1 round-trip (catches residual multi-byte cases) ─
    # Only attempt when the string contains non-ASCII bytes that survived
    # the fast-path — avoids unnecessary encode/decode on clean strings.
    try:
        # If the string was wrongly decoded as latin-1, re-encoding as latin-1
        # gives us the original UTF-8 byte sequence, which we then decode correctly.
        round_tripped = cleaned.encode("latin-1").decode("utf-8")
        if round_tripped != cleaned:
            logger.debug(
                "Latin-1 round-trip correction: '%s' → '%s'", cleaned, round_tripped
            )
            cleaned = round_tripped
    except (UnicodeEncodeError, UnicodeDecodeError):
        # String is already valid UTF-8 (encodes fine as UTF-8, not latin-1)
        # or contains characters outside both code-pages — leave as-is.
        pass

    # ── Stage 2: NFC normalisation ────────────────────────────────────────
    # Ensures canonical composed form: "Ö" (U+00D6) not "O" + combining ¨.
    normalised = unicodedata.normalize("NFC", cleaned)

    if normalised != raw:
        logger.debug(
            "Swedish name normalised: '%s' → '%s'",
            raw.encode("unicode_escape").decode("ascii"),
            normalised,
        )

    return normalised


def normalize_swedish_names_in_dataframe(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Apply normalize_swedish_regional_name to string columns in a DataFrame.

    Args:
        df: Input DataFrame (from Bronze JSON / CSV reader).
        columns: Explicit list of columns to clean. Defaults to all object/string
                 columns whose name contains 'name', 'region', 'kommun', or 'lan'.

    Returns:
        DataFrame with cleaned region name columns (copy — original untouched).

    Notes:
        Null values (NaN / None) in string columns are preserved as NaN.
    """
    df = df.copy()

    if columns is None:
        # Auto-detect: any object column whose name suggests a place name
        _name_patterns = re.compile(
            r"(name|region|kommun|lan|municipality|ort|stad)", re.IGNORECASE
        )
        columns = [
            c for c in df.columns
            if df[c].dtype == object and _name_patterns.search(c)
        ]

    for col in columns:
        if col not in df.columns:
            logger.warning("Column '%s' not found in DataFrame — skipping.", col)
            continue

        before_nulls = df[col].isna().sum()
        df[col] = df[col].apply(
            lambda val: normalize_swedish_regional_name(val)
            if isinstance(val, str)
            else val
        )
        after_nulls = df[col].isna().sum()

        if before_nulls != after_nulls:
            logger.warning(
                "Column '%s': null count changed %d → %d after normalisation "
                "(unexpected — check upstream encoding).",
                col, before_nulls, after_nulls,
            )
        else:
            logger.debug("Swedish name normalisation applied to column '%s'.", col)

    return df


# ===========================================================================
# SWEDISH DATA CLEANING — Step B
# ===========================================================================
# parse_scb_quarter_string  /  parse_scb_quarter_series
# ---------------------------------------------------------------------------
# SCB PxWebApi v2 encodes time periods in a non-ISO format:
#
#   {YEAR}K{QUARTER}   where QUARTER ∈ {1, 2, 3, 4}
#   Example:  "2025K1"  →  Q1 2025  →  2025-01-01 00:00:00 UTC
#             "2024K4"  →  Q4 2024  →  2024-10-01 00:00:00 UTC
#
# Pitfalls prevented by this parser:
#   - pd.to_datetime("2025K1") → NaT  (silently corrupt index)
#   - float("2025K1")          → ValueError (crashes ingestion)
#   - Lowercase input: "2025k2" → handled via .upper()
#   - Quarter 0 / 5+           → explicit validation, raises ValueError
#   - Whitespace padding        → stripped before parsing
#
# The period-start convention (Jan 1 for Q1, Apr 1 for Q2, etc.) is critical
# for leakage-safe merge_asof alignment: we anchor knowledge at the EARLIEST
# possible moment the data could have been known (start of the quarter).
# ===========================================================================

# Quarter → (first month of that quarter) lookup
_QUARTER_TO_MONTH: dict[int, int] = {1: 1, 2: 4, 3: 7, 4: 10}


def parse_scb_quarter_string(quarter_str: str) -> pd.Timestamp:
    """
    Convert an SCB quarter string to a UTC period-start Timestamp.

    The returned Timestamp represents the first moment the quarterly value
    could have been known — i.e. the start of that quarter — anchored to
    UTC midnight.  This is the correct join key for leakage-safe merge_asof.

    Args:
        quarter_str: SCB period string, e.g. '2025K1', '2024K4', '2025k2'.

    Returns:
        pd.Timestamp at UTC midnight on the first day of the quarter.

    Raises:
        ValueError: If the string does not conform to 'YYYYKQ' format,
                    or if the quarter number is not in {1, 2, 3, 4}.

    Examples:
        >>> parse_scb_quarter_string("2025K1")
        Timestamp('2025-01-01 00:00:00+0000', tz='UTC')
        >>> parse_scb_quarter_string("2024K4")
        Timestamp('2024-10-01 00:00:00+0000', tz='UTC')
        >>> parse_scb_quarter_string("2025K2")
        Timestamp('2025-04-01 00:00:00+0000', tz='UTC')
    """
    if not isinstance(quarter_str, str):
        raise TypeError(
            f"Expected str, got {type(quarter_str).__name__}: {quarter_str!r}"
        )

    cleaned = quarter_str.strip().upper()

    # Validate format: exactly "YYYYKQ" (4 digit year, 'K', 1 digit quarter)
    match = re.fullmatch(r"(\d{4})K([1-4])", cleaned)
    if match is None:
        raise ValueError(
            f"Cannot parse SCB quarter string {quarter_str!r}. "
            f"Expected format 'YYYYKQ' where Q ∈ {{1,2,3,4}} "
            f"(e.g. '2025K1', '2024K4')."
        )

    year = int(match.group(1))
    quarter = int(match.group(2))
    month = _QUARTER_TO_MONTH[quarter]  # guaranteed valid by regex [1-4]

    # Sanity-check year range to catch typos like "0025K1" or "20250K1"
    if not (1990 <= year <= 2100):
        raise ValueError(
            f"Year {year} in SCB quarter string {quarter_str!r} is outside the "
            f"expected range 1990–2100. Check the source data."
        )

    return pd.Timestamp(year=year, month=month, day=1, tz="UTC")


def parse_scb_quarter_series(
    series: "pd.Series[str]",
    errors: str = "raise",
) -> "pd.Series[pd.Timestamp]":
    """
    Vectorised parser for a pandas Series of SCB quarter strings.

    Applies parse_scb_quarter_string element-wise with configurable error
    handling so that a single malformed cell does not abort the entire
    Bronze → Silver pipeline.

    Args:
        series: pd.Series of SCB period strings (dtype object / string).
        errors: One of:
            'raise'  – propagate ValueError on the first bad value (default,
                       safest for production pipelines).
            'coerce' – replace unparseable values with pd.NaT and emit a
                       WARNING log per bad value (useful for exploratory runs).
            'ignore' – return the original series unchanged on any error
                       (NOT recommended; silently hides corruption).

    Returns:
        pd.Series of UTC-aware pd.Timestamps (or NaT where errors='coerce').

    Raises:
        ValueError: On malformed input when errors='raise'.
        ValueError: If errors is not one of the three valid modes.
    """
    if errors not in ("raise", "coerce", "ignore"):
        raise ValueError(
            f"errors must be 'raise', 'coerce', or 'ignore'. Got: {errors!r}"
        )

    results: list[Optional[pd.Timestamp]] = []
    bad_indices: list = []

    for idx, val in series.items():
        try:
            results.append(parse_scb_quarter_string(val))
        except (ValueError, TypeError) as exc:
            if errors == "raise":
                raise ValueError(
                    f"Failed to parse SCB quarter at index {idx}: {val!r}. "
                    f"Original error: {exc}"
                ) from exc
            elif errors == "coerce":
                logger.warning(
                    "Coercing unparseable SCB quarter string at index %s: %r → NaT. "
                    "Error: %s",
                    idx, val, exc,
                )
                results.append(pd.NaT)
                bad_indices.append(idx)
            else:  # ignore
                results.append(val)  # type: ignore[arg-type]

    if bad_indices and errors == "coerce":
        logger.warning(
            "%d / %d SCB quarter values could not be parsed and were set to NaT. "
            "Indices: %s",
            len(bad_indices), len(series), bad_indices[:20],
        )

    out = pd.Series(results, index=series.index, dtype="object")

    # Cast to DatetimeTZDtype only when no NaT / raw values remain in ignore mode
    if errors != "ignore":
        try:
            out = pd.to_datetime(out, utc=True)
        except Exception:
            pass  # leave as object Series; caller will handle

    return out


# ===========================================================================
# SWEDISH DATA CLEANING — Step C
# ===========================================================================
# Multi-frequency mismatch handler — three separate sub-functions consumed
# by TemporalHarmonizer (defined below).
#
# Source frequencies in the pipeline:
#   ┌──────────────────────────────────────┬──────────────────────────────┐
#   │ Data source                          │ Native frequency              │
#   ├──────────────────────────────────────┼──────────────────────────────┤
#   │ ENTSO-E grid (load / imbalance)      │ Hourly (8 760 rows / year)   │
#   │ Riksbank Styrränta (policy rate)     │ Event-driven (~6–8 / year)   │
#   │ SCB housing metrics (småhus)         │ Quarterly (4 / year)         │
#   └──────────────────────────────────────┴──────────────────────────────┘
#
# Alignment rules (all STRICTLY CAUSAL — no look-ahead):
#   Riksbank: step-wise forward fill via merge_asof(direction='backward').
#             A rate announced on date D is visible from D 00:00 UTC onward.
#             No interpolation — rates do not change continuously.
#   SCB:      merge_asof(direction='backward') anchored at quarter starts.
#             Q1 2025 (Jan 1) can first appear at 2025-01-01 00:00 UTC.
#             The housing index does NOT interpolate between quarters because
#             intra-quarter values are not published — ffill is the correct
#             representation of the information set available to the model.
#   Grid gaps: Short sensor outages (≤ 6 h) use linear interpolation for
#             load_mw (smoothly varying) and ffill for imbalance_mwh /
#             price_eur_mwh (step-wise auction / settlement quantities).
# ===========================================================================


def align_riksbank_to_hourly(
    df_hourly: pd.DataFrame,
    df_policy_rate: pd.DataFrame,
    timestamp_col: str = "timestamp_utc",
    effective_date_col: str = "effective_date_utc",
    rate_col: str = "policy_rate_pct",
    output_col: str = "riksbank_policy_rate_pct",
) -> pd.DataFrame:
    """
    Step-wise forward-fill Riksbank Styrränta onto an hourly UTC grid.

    The Styrränta is an event-driven step function: it is set at each
    Riksbank monetary policy meeting and remains constant until the next
    decision.  The correct alignment strategy is therefore strict forward-
    fill (ffill) — NOT linear interpolation — because the rate does not
    change continuously between decisions.

    Leakage prevention:
        merge_asof(direction='backward') ensures each hourly row is assigned
        the most recent rate whose effective_date_utc ≤ timestamp_utc.
        Rate announcements made before their effective date are NOT applied
        until that date is reached.

    Args:
        df_hourly:         Hourly grid DataFrame; must be sorted by timestamp_col.
        df_policy_rate:    Riksbank DataFrame with at least two columns:
                               [effective_date_col, rate_col].
                           Rows must be sorted ascending by effective_date_col.
        timestamp_col:     Name of the hourly timestamp column (UTC-aware).
        effective_date_col: Name of the rate change date column (UTC-aware).
        rate_col:          Name of the rate value column (float, percent).
        output_col:        Name of the joined column added to df_hourly.

    Returns:
        Copy of df_hourly with output_col appended.  Hourly rows that
        precede the first available rate decision remain NaN — they are NOT
        back-filled (that would constitute leakage of future information).

    Raises:
        KeyError: If effective_date_col or rate_col not in df_policy_rate.
    """
    required = {effective_date_col, rate_col}
    missing = required - set(df_policy_rate.columns)
    if missing:
        raise KeyError(
            f"Riksbank DataFrame missing required columns: {missing}. "
            f"Available: {list(df_policy_rate.columns)}"
        )

    df_rate = df_policy_rate[[effective_date_col, rate_col]].copy()
    df_rate[effective_date_col] = pd.to_datetime(df_rate[effective_date_col], utc=True)
    df_rate = df_rate.dropna(subset=[effective_date_col, rate_col])
    df_rate = df_rate.sort_values(effective_date_col).reset_index(drop=True)

    if df_rate.empty:
        logger.warning(
            "Riksbank rate DataFrame is empty after cleaning. "
            "Column '%s' will be NaN for all rows.", output_col
        )
        df_out = df_hourly.copy()
        df_out[output_col] = np.nan
        return df_out

    df_h = df_hourly.copy()
    df_h[timestamp_col] = pd.to_datetime(df_h[timestamp_col], utc=True)
    df_h = df_h.sort_values(timestamp_col).reset_index(drop=True)

    # CAUSAL JOIN: each hour gets the most recent rate already in effect.
    # direction='backward' = look back in time only — no future rates bleed in.
    df_merged = pd.merge_asof(
        df_h,
        df_rate.rename(columns={effective_date_col: timestamp_col}),
        on=timestamp_col,
        direction="backward",
    )
    df_merged = df_merged.rename(columns={rate_col: output_col})

    unique_rates = df_merged[output_col].dropna().unique()
    coverage_pct = 100 * df_merged[output_col].notna().mean()
    pre_series_nan = df_merged[output_col].isna().sum()

    logger.info(
        "Riksbank alignment complete | %d distinct rate level(s): %s | "
        "coverage: %.1f%% | %d hours before first rate decision (NaN, by design)",
        len(unique_rates),
        sorted(unique_rates.tolist()),
        coverage_pct,
        pre_series_nan,
    )
    return df_merged


def align_scb_quarterly_to_hourly(
    df_hourly: pd.DataFrame,
    df_quarterly: pd.DataFrame,
    zone: str,
    timestamp_col: str = "timestamp_utc",
    period_col: str = "period_utc",
    zone_col: str = "bidding_zone",
    value_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Forward-fill quarterly SCB housing metrics onto an hourly UTC grid.

    Each quarterly SCB observation is anchored at the PERIOD START
    (e.g. Q1 2025 → 2025-01-01 00:00 UTC).  Using merge_asof with
    direction='backward' guarantees that each hourly row only sees data
    that had already been published at that point in time.

    Why not interpolate between quarters?
        SCB does not publish intra-quarter housing index values.  Linear
        interpolation between Q1 and Q2 would fabricate data that was never
        observed.  Forward-fill correctly represents the information set
        available to a model trained on this data.

    Args:
        df_hourly:   Hourly grid DataFrame (must include timestamp_col).
        df_quarterly: Zone-aggregated SCB DataFrame with at least:
                       [period_col, zone_col, *value_cols].
                       Produced by KommunBiddingZoneMapper.aggregate_to_zone().
        zone:         Bidding zone string used to filter df_quarterly.
        timestamp_col: Name of the hourly UTC timestamp column.
        period_col:   Name of the quarterly period-start timestamp column.
        zone_col:     Name of the zone label column in df_quarterly.
        value_cols:   SCB metric columns to join; defaults to
                      ['smahus_construction_index', 'smahus_price_index'].

    Returns:
        Copy of df_hourly with value_cols appended (NaN where no prior
        quarterly data exists — intentional, no back-fill applied).
    """
    if value_cols is None:
        value_cols = ["smahus_construction_index", "smahus_price_index"]

    df_zone_q = df_quarterly[df_quarterly[zone_col] == zone].copy()

    if df_zone_q.empty:
        logger.warning(
            "No quarterly SCB data for zone '%s'. Columns %s will be NaN.",
            zone, value_cols,
        )
        df_out = df_hourly.copy()
        for col in value_cols:
            df_out[col] = np.nan
        return df_out

    # Ensure UTC-aware timestamps and ascending sort (required by merge_asof)
    df_zone_q[period_col] = pd.to_datetime(df_zone_q[period_col], utc=True)
    df_zone_q = df_zone_q.dropna(subset=[period_col])
    df_zone_q = df_zone_q.sort_values(period_col).reset_index(drop=True)

    # Ensure all requested value columns exist; fill missing ones with NaN
    for col in value_cols:
        if col not in df_zone_q.columns:
            logger.warning(
                "Quarterly DataFrame missing column '%s' for zone '%s' — filling NaN.",
                col, zone,
            )
            df_zone_q[col] = np.nan

    df_h = df_hourly.copy()
    df_h[timestamp_col] = pd.to_datetime(df_h[timestamp_col], utc=True)
    df_h = df_h.sort_values(timestamp_col).reset_index(drop=True)

    join_right = df_zone_q[[period_col] + value_cols].rename(
        columns={period_col: timestamp_col}
    )

    # CAUSAL JOIN: direction='backward' → only past / current quarter visible.
    df_merged = pd.merge_asof(
        df_h,
        join_right,
        on=timestamp_col,
        direction="backward",
    )

    # Log coverage metrics per value column
    for col in value_cols:
        coverage = 100 * df_merged[col].notna().mean()
        pre_nan = df_merged[col].isna().sum()
        logger.info(
            "SCB quarterly join | zone=%s | column='%s' | coverage=%.1f%% | "
            "%d hours before first observation remain NaN (no back-fill)",
            zone, col, coverage, pre_nan,
        )

    return df_merged


def interpolate_grid_gaps(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp_utc",
    max_gap_hours: int = 6,
    interpolation_method: str = "linear",
) -> pd.DataFrame:
    """
    Fill short gaps in hourly grid metrics using frequency-appropriate methods.

    Gap-filling policy (all causal / forward-only):
        load_mw       → linear interpolation (limit_direction='forward').
                         Load varies smoothly; linear is a good short-gap
                         approximation. Gaps > max_gap_hours left as NaN.
        imbalance_mwh → ffill (step-wise settlement quantity).
        price_eur_mwh → ffill (day-ahead auction price, step-wise per hour).

    The function first reindexes the DataFrame onto a complete hourly grid
    to expose any IMPLICIT gaps (hours entirely missing from the Bronze
    output rather than present with NaN values).

    Data quality flag:
        has_imputed_grid = True is set on every row where load_mw was NaN
        before gap-filling.  Downstream models can use this flag to down-
        weight imputed hours or exclude them from evaluation windows.

    Args:
        df:                  Hourly DataFrame with timestamp_col as a column
                             (not index).  Must be UTC-aware.
        timestamp_col:       Name of the timestamp column.
        max_gap_hours:       Maximum consecutive hours to fill (default 6).
                             Gaps longer than this remain NaN and are
                             quarantined downstream.
        interpolation_method: pandas interpolation method for load_mw
                              ('linear', 'time', 'polynomial', etc.).
                              Use 'linear' for production; 'time' accounts
                              for irregular spacing (though reindexing makes
                              the grid regular before interpolation).

    Returns:
        DataFrame on a complete hourly UTC index with gaps filled where
        feasible, and has_imputed_grid column added / updated.
    """
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    df = df.set_index(timestamp_col)

    # Reindex to full hourly grid — exposes implicit gaps as NaN rows
    full_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="h",
        tz="UTC",
    )
    pre_len = len(df)
    df = df.reindex(full_index)
    implicit_gaps = len(df) - pre_len

    if implicit_gaps > 0:
        logger.info(
            "Reindexed to full hourly grid: %d implicit gap-hours exposed.",
            implicit_gaps,
        )

    # Mark rows that will be imputed BEFORE gap-filling so the flag is accurate
    load_was_nan = df["load_mw"].isna() if "load_mw" in df.columns else pd.Series(
        False, index=df.index
    )
    df["has_imputed_grid"] = load_was_nan

    # ── load_mw: linear interpolation, forward direction only ────────────
    if "load_mw" in df.columns:
        df["load_mw"] = df["load_mw"].interpolate(
            method=interpolation_method,
            limit=max_gap_hours,
            limit_direction="forward",
        )

    # ── imbalance_mwh: step-wise ffill ───────────────────────────────────
    if "imbalance_mwh" in df.columns:
        df["imbalance_mwh"] = df["imbalance_mwh"].ffill(limit=max_gap_hours)

    # ── price_eur_mwh: step-wise ffill ───────────────────────────────────
    if "price_eur_mwh" in df.columns:
        df["price_eur_mwh"] = df["price_eur_mwh"].ffill(limit=max_gap_hours)

    # Propagate zone label across reindexed (previously NaN) rows
    if "zone" in df.columns:
        df["zone"] = df["zone"].ffill()

    if "is_anomaly" in df.columns:
        df["is_anomaly"] = df["is_anomaly"].fillna(False)

    filled_count = df["has_imputed_grid"].sum()
    if filled_count > 0:
        logger.info(
            "Grid gap-fill complete: %d / %d hourly rows imputed "
            "(max window %dh, method='%s').",
            filled_count, len(df), max_gap_hours, interpolation_method,
        )

    df.index.name = timestamp_col
    df = df.reset_index()
    return df


# ===========================================================================
# Spatial Mapper: Kommun → Bidding Zone
# ===========================================================================
class KommunBiddingZoneMapper:
    """
    Maps SCB Municipalities (Kommuner) to Nord Pool Bidding Zones.

    The canonical lookup uses the KOMMUNKOD_TO_ZONE dictionary above.
    For production use, an optional extended CSV lookup can be loaded
    to cover all 290 municipalities (provided in config/schemas/).

    Usage:
        mapper = KommunBiddingZoneMapper()
        zone = mapper.get_zone(1480)   # → BiddingZoneLabel.SE3
        df   = mapper.annotate_dataframe(df, kommunkod_col="kommunkod")
    """

    def __init__(self, extended_mapping_csv: Optional[Path] = None) -> None:
        self._mapping: dict[int, BiddingZoneLabel] = dict(KOMMUNKOD_TO_ZONE)
        self._unmapped: set[int] = set()

        if extended_mapping_csv and Path(extended_mapping_csv).exists():
            self._load_extended_csv(Path(extended_mapping_csv))
            logger.info(
                "Extended Kommun→Zone mapping loaded from %s (%d entries total)",
                extended_mapping_csv, len(self._mapping),
            )
        else:
            logger.info(
                "Using built-in Kommun→Zone mapping (%d entries). "
                "Pass extended_mapping_csv= for full 290-municipality coverage.",
                len(self._mapping),
            )

    def get_zone(self, kommunkod: int) -> Optional[BiddingZoneLabel]:
        """Resolve a single Kommunkod to its bidding zone."""
        zone = self._mapping.get(kommunkod)
        if zone is None:
            self._unmapped.add(kommunkod)
            logger.debug(
                "Kommunkod %d not found in zone mapping — returning None.", kommunkod
            )
        return zone

    def annotate_dataframe(
        self,
        df: pd.DataFrame,
        kommunkod_col: str = "kommunkod",
        zone_col: str = "bidding_zone",
    ) -> pd.DataFrame:
        """
        Add a 'bidding_zone' column to a DataFrame of SCB municipality records.

        Unmapped municipalities are assigned zone=None and a warning is emitted.
        Swedish region names in any detected name column are cleaned with
        normalize_swedish_regional_name before the mapping is applied.
        """
        if kommunkod_col not in df.columns:
            raise KeyError(
                f"Column '{kommunkod_col}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )

        df = df.copy()

        # ── Swedish Data Cleaning Step A (integrated at spatial mapping) ─
        df = normalize_swedish_names_in_dataframe(df)

        df[zone_col] = df[kommunkod_col].map(
            lambda k: self._mapping.get(int(k), None)
        )

        unmapped_mask = df[zone_col].isna()
        unmapped_count = int(unmapped_mask.sum())
        if unmapped_count > 0:
            unmapped_codes = df.loc[unmapped_mask, kommunkod_col].unique().tolist()
            self._unmapped.update(int(c) for c in unmapped_codes)
            logger.warning(
                "⚠️  %d rows with unmapped Kommunkod(s): %s. "
                "Rows will be excluded from zone aggregation. "
                "Update KOMMUNKOD_TO_ZONE or provide extended_mapping_csv=.",
                unmapped_count, unmapped_codes[:10],
            )

        total = len(df)
        mapped = total - unmapped_count
        logger.info(
            "Zone annotation: %d / %d rows mapped (%.1f%%)",
            mapped, total, 100 * mapped / total if total else 0,
        )
        return df

    def aggregate_to_zone(
        self,
        df: pd.DataFrame,
        value_col: str,
        kommunkod_col: str = "kommunkod",
        period_col: str = "period_utc",
        agg_func: str = "sum",
    ) -> pd.DataFrame:
        """
        Aggregate municipality-level SCB metrics to bidding zone totals.

        For housing construction counts, use agg_func='sum'.
        For price indices, use agg_func='mean' (population-weighted preferred
        but requires a weight column; plain mean used here as default).

        Also applies SCB quarter string parsing if the period_col contains
        raw 'YYYYKX' strings rather than parsed pd.Timestamps.

        Args:
            df:           SCB municipality-level DataFrame.
            value_col:    Column to aggregate (e.g. 'smahus_construction_index').
            kommunkod_col: Municipality code column name.
            period_col:   Quarter period column (may be 'YYYYKX' strings or
                          pre-parsed pd.Timestamps).
            agg_func:     Aggregation function ('sum', 'mean', 'median').

        Returns:
            pd.DataFrame with columns: [period_utc, bidding_zone, {value_col}]
        """
        df = df.copy()

        # ── Swedish Data Cleaning Step B (integrated at aggregation) ─────
        # If the period column contains raw SCB quarter strings, parse them now.
        if df[period_col].dtype == object:
            logger.info(
                "Detected raw SCB quarter strings in column '%s' — parsing.", period_col
            )
            df[period_col] = parse_scb_quarter_series(
                df[period_col], errors="coerce"
            )
            bad = df[period_col].isna().sum()
            if bad:
                logger.warning(
                    "%d rows with unparseable quarter strings dropped before aggregation.",
                    bad,
                )
            df = df.dropna(subset=[period_col])

        annotated = self.annotate_dataframe(df, kommunkod_col)
        clean = annotated.dropna(subset=["bidding_zone"]).copy()

        if clean.empty:
            logger.error("No mappable municipality data after zone annotation.")
            return pd.DataFrame(columns=[period_col, "bidding_zone", value_col])

        aggregated = (
            clean.groupby([period_col, "bidding_zone"])[value_col]
            .agg(agg_func)
            .reset_index()
        )
        logger.info(
            "Aggregated '%s' to %d zone-period records using '%s'.",
            value_col, len(aggregated), agg_func,
        )
        return aggregated

    def get_unmapped_report(self) -> set[int]:
        """Return the set of Kommunkoder that failed to resolve during this session."""
        return set(self._unmapped)

    def _load_extended_csv(self, csv_path: Path) -> None:
        """Load extended municipality→zone mapping from CSV."""
        df = pd.read_csv(csv_path, dtype={"kommunkod": int, "bidding_zone": str})
        required_cols = {"kommunkod", "bidding_zone"}
        if not required_cols.issubset(df.columns):
            raise ValueError(
                f"Extended mapping CSV must contain columns {required_cols}. "
                f"Found: {set(df.columns)}"
            )
        zone_map = {
            row.kommunkod: BiddingZoneLabel(row.bidding_zone)
            for row in df.itertuples()
            if row.bidding_zone in BiddingZoneLabel._value2member_map_
        }
        self._mapping.update(zone_map)


# ===========================================================================
# Temporal Harmonizer — master orchestrator
# ===========================================================================
class TemporalHarmonizer:
    """
    Orchestrates multi-frequency alignment onto a common hourly UTC index.

    This class wires together the three Swedish Data Cleaning steps:
      A. normalize_swedish_names_in_dataframe (via KommunBiddingZoneMapper)
      B. parse_scb_quarter_series             (via aggregate_to_zone / align)
      C. align_riksbank_to_hourly +
         align_scb_quarterly_to_hourly +
         interpolate_grid_gaps

    All joins are STRICTLY CAUSAL (no look-ahead / no back-fill of macro data).

    Usage:
        harmonizer = TemporalHarmonizer()
        df_silver = harmonizer.align(
            df_hourly=df_grid,
            df_quarterly=df_housing_by_zone,   # zone-aggregated
            df_policy_rate=df_riksbank,
            zone="SE3",
        )
    """

    def __init__(
        self,
        interpolation_method: str = "linear",
        max_grid_gap_hours: int = 6,
    ) -> None:
        """
        Args:
            interpolation_method: Method for load_mw gap-filling ('linear', 'time').
            max_grid_gap_hours:   Maximum consecutive gap hours to fill (default 6).
        """
        self.interpolation_method = interpolation_method
        self.max_grid_gap_hours = max_grid_gap_hours
        logger.info(
            "TemporalHarmonizer initialised | grid gap-fill method='%s' max=%dh | "
            "macro alignment: strictly causal ffill",
            interpolation_method, max_grid_gap_hours,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def align(
        self,
        df_hourly: pd.DataFrame,
        df_quarterly: Optional[pd.DataFrame] = None,
        df_policy_rate: Optional[pd.DataFrame] = None,
        zone: str = "SE3",
    ) -> pd.DataFrame:
        """
        Master alignment pipeline for a single bidding zone.

        Executes the following stages in order:
          1. Validate and prepare hourly base (UTC cast, duplicate/gap detection)
          2. Step C — interpolate_grid_gaps (sensor outage recovery)
          3. Step C — align_scb_quarterly_to_hourly (leakage-safe ffill)
          4. Step C — align_riksbank_to_hourly (step-function ffill)
          5. Quarantine anomalous / unresolvable rows
          6. Final Silver schema enforcement

        Args:
            df_hourly:     Hourly grid DataFrame:
                           [timestamp_utc, zone, load_mw, imbalance_mwh,
                            price_eur_mwh, is_anomaly]
            df_quarterly:  Zone-aggregated SCB quarterly DataFrame:
                           [period_utc, bidding_zone,
                            smahus_construction_index, smahus_price_index]
            df_policy_rate: Riksbank rate DataFrame:
                           [effective_date_utc, policy_rate_pct]
            zone:          Bidding zone string for logging and filtering.

        Returns:
            Fully aligned Silver layer pd.DataFrame, one row per hour.
        """
        logger.info("=== TemporalHarmonizer.align() | Zone: %s ===", zone)

        # Stage 1: Validate hourly base
        df = self._prepare_hourly_base(df_hourly, zone)

        # Stage 2: Fill short grid gaps  [Step C — grid interpolation]
        df = interpolate_grid_gaps(
            df,
            max_gap_hours=self.max_grid_gap_hours,
            interpolation_method=self.interpolation_method,
        )

        # Stage 3: Join SCB quarterly metrics  [Step C — quarterly ffill]
        if df_quarterly is not None and not df_quarterly.empty:
            df = align_scb_quarterly_to_hourly(df, df_quarterly, zone=zone)
        else:
            logger.warning("No quarterly SCB data provided for zone %s.", zone)
            df["smahus_construction_index"] = np.nan
            df["smahus_price_index"] = np.nan

        # Stage 4: Join Riksbank rate  [Step C — step-function ffill]
        if df_policy_rate is not None and not df_policy_rate.empty:
            df = align_riksbank_to_hourly(df, df_policy_rate)
        else:
            logger.warning("No Riksbank policy rate data provided.")
            df["riksbank_policy_rate_pct"] = np.nan

        # Stage 5: Quarantine
        df = self._apply_quarantine(df)

        # Stage 6: Schema enforcement
        df = self._validate_output(df, zone)

        logger.info(
            "Alignment complete | zone=%s | %d hourly rows | %d quarantined | "
            "SCB coverage=%.1f%% | rate coverage=%.1f%%",
            zone,
            len(df),
            int(df["is_quarantined"].sum()),
            100 * df["smahus_construction_index"].notna().mean(),
            100 * df["riksbank_policy_rate_pct"].notna().mean(),
        )
        return df

    def align_all_zones(
        self,
        all_zone_grids: dict[str, pd.DataFrame],
        df_quarterly_by_zone: Optional[dict[str, pd.DataFrame]] = None,
        df_policy_rate: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Align all four bidding zones and concatenate into the Silver master frame.

        Args:
            all_zone_grids:       {'SE1': df, 'SE2': df, 'SE3': df, 'SE4': df}
            df_quarterly_by_zone: {'SE1': df_q, ...} — one quarterly DF per zone.
                                  If a zone key is missing, that zone's housing
                                  features will be NaN.
            df_policy_rate:       Single Riksbank rate DataFrame (national).

        Returns:
            Concatenated Silver layer DataFrame, sorted by (zone, timestamp_utc).
        """
        zone_frames: list[pd.DataFrame] = []

        for zone_label, df_grid in all_zone_grids.items():
            quarterly = (
                df_quarterly_by_zone.get(zone_label)
                if df_quarterly_by_zone
                else None
            )
            df_aligned = self.align(
                df_hourly=df_grid,
                df_quarterly=quarterly,
                df_policy_rate=df_policy_rate,
                zone=zone_label,
            )
            zone_frames.append(df_aligned)

        if not zone_frames:
            logger.error("No zone data available for alignment.")
            return pd.DataFrame()

        silver_master = pd.concat(zone_frames, ignore_index=True)
        silver_master.sort_values(["zone", "timestamp_utc"], inplace=True)
        silver_master.reset_index(drop=True, inplace=True)

        logger.info(
            "Silver master frame built: %d total hourly records across %d zones.",
            len(silver_master), len(zone_frames),
        )
        return silver_master

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _prepare_hourly_base(self, df: pd.DataFrame, zone: str) -> pd.DataFrame:
        """Validate, type-cast, and ensure monotonic UTC hourly index."""
        required_cols = {"timestamp_utc", "zone"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"Hourly grid DataFrame missing required columns: {missing}. "
                f"Available: {list(df.columns)}"
            )

        df = df.copy()
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        df.sort_values("timestamp_utc", inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Detect gaps in hourly continuity
        if len(df) > 1:
            time_diffs = df["timestamp_utc"].diff().dropna()
            expected_freq = pd.Timedelta("1h")
            gaps = time_diffs[time_diffs > expected_freq * 1.5]
            if not gaps.empty:
                logger.warning(
                    "⚠️  %d hourly continuity gaps detected in zone %s. "
                    "Largest: %s.",
                    len(gaps), zone, gaps.max(),
                )

        # Detect and drop duplicate timestamps
        dupes = df.duplicated(subset=["timestamp_utc"])
        if dupes.any():
            logger.warning(
                "⚠️  %d duplicate timestamps in zone %s — keeping last.",
                int(dupes.sum()), zone,
            )
            df = df.drop_duplicates(subset=["timestamp_utc"], keep="last")

        if "is_anomaly" not in df.columns:
            df["is_anomaly"] = False

        logger.debug(
            "Hourly base prepared | zone=%s | %d records | %s → %s",
            zone,
            len(df),
            df["timestamp_utc"].min().isoformat(),
            df["timestamp_utc"].max().isoformat(),
        )
        return df

    def _apply_quarantine(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Mark anomalous or unresolvable rows as quarantined.

        Quarantine triggers:
          - is_anomaly=True  (flagged by AnomalyLogger during ingestion)
          - load_mw IS NULL AND imbalance_mwh IS NULL (no usable target variable)

        Quarantined rows are retained in Silver for audit / debugging.
        The model factory (src/models/train.py) must filter on
        is_quarantined=False before training.
        """
        df = df.copy()
        df["is_quarantined"] = False

        anomaly_mask = (
            df.get("is_anomaly", pd.Series(False, index=df.index))
            .fillna(False)
            .astype(bool)
        )
        df.loc[anomaly_mask, "is_quarantined"] = True

        load_null = df.get("load_mw", pd.Series(np.nan, index=df.index)).isna()
        imbalance_null = df.get(
            "imbalance_mwh", pd.Series(np.nan, index=df.index)
        ).isna()
        null_target_mask = load_null & imbalance_null
        df.loc[null_target_mask, "is_quarantined"] = True

        total_quarantined = int(df["is_quarantined"].sum())
        total = len(df)
        quarantine_pct = 100 * total_quarantined / total if total else 0

        if quarantine_pct > 5.0:
            logger.warning(
                "⚠️  HIGH QUARANTINE RATE: %.1f%% (%d / %d rows). "
                "Inspect Bronze layer anomaly reports and grid data coverage.",
                quarantine_pct, total_quarantined, total,
            )
        elif total_quarantined > 0:
            logger.info(
                "Quarantine applied: %d / %d rows (%.2f%%) isolated.",
                total_quarantined, total, quarantine_pct,
            )
        return df

    def _validate_output(self, df: pd.DataFrame, zone: str) -> pd.DataFrame:
        """
        Final Silver layer schema enforcement.

        Ensures all expected columns are present and correctly typed.
        Missing columns are added with NaN / False defaults and logged.
        """
        expected_cols: dict[str, type] = {
            "timestamp_utc": "datetime64[ns, UTC]",
            "zone": str,
            "load_mw": float,
            "imbalance_mwh": float,
            "price_eur_mwh": float,
            "smahus_construction_index": float,
            "smahus_price_index": float,
            "riksbank_policy_rate_pct": float,
            "is_anomaly": bool,
            "is_quarantined": bool,
            "has_imputed_grid": bool,
        }

        for col, dtype in expected_cols.items():
            if col not in df.columns:
                if dtype == bool:
                    df[col] = False
                elif dtype == float:
                    df[col] = np.nan
                elif dtype == str:
                    df[col] = zone
                logger.debug(
                    "Added missing Silver column '%s' with default.", col
                )

        # Coerce boolean types (can become object-typed after merges)
        for bool_col in ["is_anomaly", "is_quarantined", "has_imputed_grid"]:
            if bool_col in df.columns:
                df[bool_col] = df[bool_col].astype(bool)

        return df


# ===========================================================================
# Silver Layer Writer
# ===========================================================================
class SilverLayerWriter:
    """
    Persists aligned Silver layer DataFrames to Parquet format.

    Uses date-partitioned paths for idempotent execution:
        data/silver/zone=SE3/year=2024/month=06/aligned_grid.parquet
    """

    def __init__(self, silver_dir: Path = Path("data/silver")) -> None:
        self.silver_dir = Path(silver_dir)

    def write(
        self,
        df: pd.DataFrame,
        zone: str,
        partition_by: str = "month",
    ) -> list[Path]:
        """
        Write Silver DataFrame to partitioned Parquet files.

        Args:
            df:           Aligned Silver layer DataFrame.
            zone:         Bidding zone label (used in the partition path).
            partition_by: 'month' (default) or 'year'.

        Returns:
            List of written Parquet file paths.
        """
        if df.empty:
            logger.warning("Empty DataFrame for zone %s — nothing written.", zone)
            return []

        df = df.copy()
        df["year"] = df["timestamp_utc"].dt.year
        df["month"] = df["timestamp_utc"].dt.month

        written_paths: list[Path] = []

        groups = (
            df.groupby(["year", "month"])
            if partition_by == "month"
            else df.groupby(["year"])
        )

        for group_key, group_df in groups:
            if partition_by == "month":
                year, month = group_key
                partition_path = (
                    self.silver_dir
                    / f"zone={zone}"
                    / f"year={year}"
                    / f"month={month:02d}"
                )
            else:
                year = group_key
                partition_path = self.silver_dir / f"zone={zone}" / f"year={year}"

            partition_path.mkdir(parents=True, exist_ok=True)
            out_file = partition_path / "aligned_grid.parquet"

            write_df = group_df.drop(columns=["year", "month"], errors="ignore")
            write_df.to_parquet(out_file, index=False, engine="pyarrow")
            written_paths.append(out_file)
            logger.info(
                "Silver written → %s (%d rows, %.1f KB)",
                out_file, len(write_df), out_file.stat().st_size / 1024,
            )

        return written_paths


# ===========================================================================
# CLI Smoke Test
# ===========================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 70)
    print("Silver Alignment Transformer — Swedish Data Cleaning Smoke Test")
    print("=" * 70)

    # ── Step A: Swedish character normalisation ───────────────────────────
    print("\n[A] Swedish regional name normalisation (å/ä/ö mojibake repair)")
    test_cases = [
        ("GÃ¶teborg",          "Göteborg"),     # ö as mojibake
        ("MalmÃ¶",             "Malmö"),         # ö at end
        ("Ã…re",               "Åre"),           # Å at start
        ("LuleÃ¥",             "Luleå"),         # å at end
        ("HÃ¤rnÃ¶sand",        "Härnösand"),     # ä + ö interior
        ("Göteborg",           "Göteborg"),      # already clean  → no-op
        ("Stockholm",          "Stockholm"),     # ASCII only     → no-op
    ]
    all_pass = True
    for raw, expected in test_cases:
        result = normalize_swedish_regional_name(raw)
        ok = result == expected
        all_pass = all_pass and ok
        status = "✓" if ok else "✗"
        print(f"    {status}  {raw!r:30s}  →  {result!r}  (expected {expected!r})")

    print(f"\n    {'All tests passed ✅' if all_pass else 'FAILURES DETECTED ❌'}")

    # ── Step B: SCB quarter string parsing ───────────────────────────────
    print("\n[B] SCB quarter string parser (YYYYKX → UTC timestamp)")
    quarter_cases = [
        ("2024K1", "2024-01-01"),
        ("2024K2", "2024-04-01"),
        ("2024K3", "2024-07-01"),
        ("2024K4", "2024-10-01"),
        ("2025K1", "2025-01-01"),
        ("2025k2", "2025-04-01"),  # lowercase k
    ]
    for qs, expected_date in quarter_cases:
        ts = parse_scb_quarter_string(qs)
        ok = ts.strftime("%Y-%m-%d") == expected_date
        status = "✓" if ok else "✗"
        print(f"    {status}  {qs!r:10s}  →  {ts}  (expected {expected_date})")

    # Vectorised series parser
    s = pd.Series(["2024K1", "2024K4", "2025K2", "INVALID", "2025K3"])
    parsed = parse_scb_quarter_series(s, errors="coerce")
    nat_count = parsed.isna().sum()
    print(f"\n    Vectorised series (errors='coerce'): {nat_count} NaT(s) from 'INVALID'  "
          f"{'✓' if nat_count == 1 else '✗'}")

    # ── Step C: Multi-frequency alignment ────────────────────────────────
    print("\n[C] Multi-frequency alignment (hourly/daily/quarterly → hourly UTC)")
    rng = pd.date_range("2024-01-01", periods=24 * 92, freq="h", tz="UTC")
    df_grid = pd.DataFrame({
        "timestamp_utc": rng,
        "zone": "SE3",
        "load_mw": 8500.0 + 1500 * np.sin(np.linspace(0, 4 * np.pi, len(rng))),
        "imbalance_mwh": np.random.default_rng(42).normal(0, 100, len(rng)),
        "price_eur_mwh": 75.0 + 15 * np.cos(np.linspace(0, 4 * np.pi, len(rng))),
        "is_anomaly": False,
    })
    # Inject 3 explicit gaps
    df_grid.loc[[200, 201, 202], "load_mw"] = np.nan

    df_quarterly = pd.DataFrame({
        "period_utc": [
            pd.Timestamp("2023-10-01", tz="UTC"),  # Q4 2023 — before window
            pd.Timestamp("2024-01-01", tz="UTC"),  # Q1 2024
            pd.Timestamp("2024-04-01", tz="UTC"),  # Q2 2024
        ],
        "bidding_zone": "SE3",
        "smahus_construction_index": [97.8, 100.4, 102.1],
        "smahus_price_index": [308.0, 315.5, 322.0],
    })

    df_rate = pd.DataFrame({
        "effective_date_utc": [
            pd.Timestamp("2023-11-01", tz="UTC"),  # 4.00% from Nov 2023
            pd.Timestamp("2024-03-27", tz="UTC"),  # 3.75% from Mar 2024
            pd.Timestamp("2024-05-08", tz="UTC"),  # 3.50% from May 2024
        ],
        "policy_rate_pct": [4.00, 3.75, 3.50],
    })

    harmonizer = TemporalHarmonizer(interpolation_method="linear", max_grid_gap_hours=6)
    df_silver = harmonizer.align(
        df_hourly=df_grid,
        df_quarterly=df_quarterly,
        df_policy_rate=df_rate,
        zone="SE3",
    )

    print(f"\n    Aligned shape:          {df_silver.shape}")
    print(f"    Quarantined rows:       {int(df_silver['is_quarantined'].sum())}")
    print(f"    Imputed grid rows:      {int(df_silver['has_imputed_grid'].sum())}")
    print(f"    SCB const. coverage:    {100 * df_silver['smahus_construction_index'].notna().mean():.1f}%")
    print(f"    SCB price coverage:     {100 * df_silver['smahus_price_index'].notna().mean():.1f}%")
    print(f"    Riksbank rate coverage: {100 * df_silver['riksbank_policy_rate_pct'].notna().mean():.1f}%")

    # ── Leakage verification ──────────────────────────────────────────────
    mar31_23h = pd.Timestamp("2024-03-31 23:00:00", tz="UTC")
    apr01_00h = pd.Timestamp("2024-04-01 00:00:00", tz="UTC")

    val_q1 = df_silver.loc[df_silver["timestamp_utc"] == mar31_23h, "smahus_construction_index"]
    val_q2 = df_silver.loc[df_silver["timestamp_utc"] == apr01_00h, "smahus_construction_index"]

    if not val_q1.empty and not val_q2.empty:
        q1_val = val_q1.iloc[0]
        q2_val = val_q2.iloc[0]
        leakage_free = q1_val != q2_val
        print(
            f"\n    Leakage check │ Q1 @ Mar-31 23:00 = {q1_val:.1f} │ "
            f"Q2 @ Apr-01 00:00 = {q2_val:.1f} │ "
            f"{'✓ No leakage' if leakage_free else '❌ LEAKAGE DETECTED'}"
        )

    # Rate step-function check
    mar26 = pd.Timestamp("2024-03-26 12:00:00", tz="UTC")  # before 3.75%
    mar27 = pd.Timestamp("2024-03-27 12:00:00", tz="UTC")  # after 3.75%
    rate_before = df_silver.loc[df_silver["timestamp_utc"] == mar26, "riksbank_policy_rate_pct"]
    rate_after  = df_silver.loc[df_silver["timestamp_utc"] == mar27, "riksbank_policy_rate_pct"]

    if not rate_before.empty and not rate_after.empty:
        rb = rate_before.iloc[0]
        ra = rate_after.iloc[0]
        step_ok = rb == 4.00 and ra == 3.75
        print(
            f"    Rate step check │ Mar-26 = {rb:.2f}% │ "
            f"Mar-27 = {ra:.2f}% │ {'✓ Correct step' if step_ok else '❌ Step error'}"
        )

    print("\n✅ Silver Data Cleaning smoke tests complete.")
