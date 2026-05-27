"""
src/ingestion/client.py
=============================================================================
Resilient Base HTTP Client — Ingestion Foundation
=============================================================================
Responsibilities:
  - Provide a reusable async HTTP session wrapper built on httpx.AsyncClient
  - Expose a synchronous `request()` facade via asyncio.run() for use in
    batch pipeline contexts (scheduled Airflow/cron jobs) without requiring
    callers to manage event loops directly
  - Implement exponential backoff with full jitter on retryable conditions
    (HTTP 429/5xx, transport failures)
  - Implement a per-host Circuit Breaker to fast-fail after consecutive
    errors rather than hammering a degraded upstream endpoint
  - Emit structured, pipeline-traceable log records at every stage

Design decisions:
  - httpx over aiohttp: superior HTTP/2 support, sync+async parity, and
    tighter integration with Pydantic-validated response flows
  - Full jitter (random.uniform) over equal jitter: better thundering-herd
    prevention when multiple pipeline workers target the same endpoint
  - Circuit Breaker threshold configurable per-client instance: ENTSO-E
    may need a tighter threshold than SCB due to token rate caps

Architecture position: LIVE API LAYER (foundation for all ingestion clients)
Downstream: src/ingestion/energy_client.py, src/ingestion/scb_fetcher.py
=============================================================================
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from functools import wraps
from typing import Any, Callable, ClassVar, Optional

import httpx

# ---------------------------------------------------------------------------
# Module-Level Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger("ingestion.client")
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
# Retry Configuration Defaults
# ---------------------------------------------------------------------------
DEFAULT_MAX_RETRIES: int = 5
DEFAULT_BASE_BACKOFF_SECONDS: float = 2.0
DEFAULT_MAX_BACKOFF_SECONDS: float = 60.0
DEFAULT_JITTER_RANGE: float = 0.5          # ± 50% full jitter per step
DEFAULT_RETRYABLE_STATUS_CODES: tuple[int, ...] = (429, 500, 502, 503, 504)
DEFAULT_REQUEST_TIMEOUT_SECONDS: float = 30.0
DEFAULT_CONNECT_TIMEOUT_SECONDS: float = 10.0


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------
class CircuitState(Enum):
    CLOSED = auto()    # Normal operation — requests pass through
    OPEN = auto()      # Fast-fail — upstream is considered degraded
    HALF_OPEN = auto() # Probe state — one trial request allowed


@dataclass
class CircuitBreaker:
    """
    Per-host circuit breaker with configurable failure threshold and
    recovery timeout.

    States:
      CLOSED → OPEN:      failure_threshold consecutive failures
      OPEN → HALF_OPEN:   recovery_timeout_seconds elapsed
      HALF_OPEN → CLOSED: one successful request
      HALF_OPEN → OPEN:   one more failure (resets recovery timer)

    Thread / coroutine safety: the breaker is intentionally lock-free.
    In a single-process pipeline context this is sufficient; add asyncio.Lock
    if the client is shared across concurrent tasks.
    """

    failure_threshold: int = 5
    recovery_timeout_seconds: float = 120.0
    host: str = "unknown"

    _state: CircuitState = field(default=CircuitState.CLOSED, init=False, repr=False)
    _failure_count: int = field(default=0, init=False, repr=False)
    _last_failure_at: Optional[float] = field(default=None, init=False, repr=False)

    def record_success(self) -> None:
        if self._state == CircuitState.HALF_OPEN:
            logger.info(
                "CircuitBreaker[%s] probe succeeded — transitioning HALF_OPEN → CLOSED",
                self.host,
            )
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_at = None

    def record_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_at = time.monotonic()

        if self._failure_count >= self.failure_threshold:
            if self._state != CircuitState.OPEN:
                logger.error(
                    "CircuitBreaker[%s] threshold reached (%d failures) — "
                    "transitioning to OPEN. Fast-failing for %.0fs.",
                    self.host, self._failure_count, self.recovery_timeout_seconds,
                )
            self._state = CircuitState.OPEN

    def allow_request(self) -> bool:
        """Return True if a request should be allowed through."""
        if self._state == CircuitState.CLOSED:
            return True

        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - (self._last_failure_at or 0.0)
            if elapsed >= self.recovery_timeout_seconds:
                logger.info(
                    "CircuitBreaker[%s] recovery timeout elapsed (%.0fs) — "
                    "transitioning OPEN → HALF_OPEN.",
                    self.host, elapsed,
                )
                self._state = CircuitState.HALF_OPEN
                return True
            logger.warning(
                "CircuitBreaker[%s] is OPEN — fast-failing request "
                "(%.0fs remaining until probe attempt).",
                self.host,
                self.recovery_timeout_seconds - elapsed,
            )
            return False

        # HALF_OPEN: allow exactly one probe through
        return True

    @property
    def state(self) -> CircuitState:
        return self._state


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------
class CircuitOpenError(RuntimeError):
    """Raised when a request is rejected by an open circuit breaker."""


class RetryExhaustedError(RuntimeError):
    """Raised when all retry attempts have been consumed."""


class NonRetryableHTTPError(httpx.HTTPStatusError):
    """Raised immediately for 4xx errors that must not be retried."""


# ---------------------------------------------------------------------------
# Retry Backoff Helpers
# ---------------------------------------------------------------------------
def _compute_backoff(
    attempt: int,
    base_seconds: float,
    max_seconds: float,
    jitter_range: float,
) -> float:
    """
    Full-jitter exponential backoff.

    Formula:  wait = clip(base * 2^(attempt-1), 0, max) * uniform(1-j, 1+j)

    Full jitter prevents correlated retry storms when multiple pipeline
    workers share the same upstream rate limit.
    """
    raw = base_seconds * (2 ** (attempt - 1))
    capped = min(raw, max_seconds)
    jitter_factor = 1.0 + random.uniform(-jitter_range, jitter_range)
    return max(0.0, capped * jitter_factor)


def _extract_retry_after(response: httpx.Response) -> Optional[float]:
    """
    Parse Retry-After header from HTTP 429 responses.

    Supports both integer-seconds and HTTP-date formats.
    Returns None if the header is absent or unparsable.
    """
    retry_after = response.headers.get("Retry-After")
    if not retry_after:
        return None
    try:
        return float(retry_after)
    except ValueError:
        try:
            from email.utils import parsedate_to_datetime
            target = parsedate_to_datetime(retry_after)
            delta = (target - datetime.now(tz=timezone.utc)).total_seconds()
            return max(0.0, delta)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Retry Decorator (synchronous — wraps the async method via run_sync)
# ---------------------------------------------------------------------------
def retry_with_backoff(
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_seconds: float = DEFAULT_BASE_BACKOFF_SECONDS,
    max_seconds: float = DEFAULT_MAX_BACKOFF_SECONDS,
    jitter_range: float = DEFAULT_JITTER_RANGE,
    retryable_status_codes: tuple[int, ...] = DEFAULT_RETRYABLE_STATUS_CODES,
) -> Callable:
    """
    Decorator for synchronous methods that wraps retry logic around a
    function which may raise httpx.HTTPStatusError or httpx.TransportError.

    Non-retryable conditions (raise immediately):
      - HTTP 400 Bad Request  (fix the query parameters)
      - HTTP 401 Unauthorized (invalid API token)
      - HTTP 403 Forbidden    (token lacks permission)
      - HTTP 404 Not Found    (wrong endpoint / doc type)
      - pydantic.ValidationError (schema drift — alert and abort)

    Retryable conditions (exponential backoff + jitter):
      - HTTP 429 Too Many Requests  (honour Retry-After if present)
      - HTTP 500/502/503/504         (transient upstream errors)
      - httpx.TransportError         (connection reset, timeout)
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Optional[Exception] = None

            for attempt in range(1, max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    # Log recovery after prior failure
                    if attempt > 1:
                        logger.info(
                            "✓ Request succeeded on attempt %d/%d after prior failures.",
                            attempt, max_retries,
                        )
                    return result

                except httpx.HTTPStatusError as exc:
                    status = exc.response.status_code

                    if status not in retryable_status_codes:
                        logger.critical(
                            "Non-retryable HTTP %d — aborting immediately. "
                            "URL: %s | Response body (first 500 chars): %s",
                            status,
                            exc.request.url,
                            exc.response.text[:500],
                        )
                        raise NonRetryableHTTPError(
                            request=exc.request, response=exc.response
                        ) from exc

                    # Honour Retry-After on 429
                    wait = _compute_backoff(attempt, base_seconds, max_seconds, jitter_range)
                    if status == 429:
                        server_wait = _extract_retry_after(exc.response)
                        if server_wait is not None:
                            wait = max(wait, server_wait)
                            logger.warning(
                                "HTTP 429 — server Retry-After: %.1fs. "
                                "Using max(backoff, Retry-After) = %.1fs.",
                                server_wait, wait,
                            )

                    logger.warning(
                        "HTTP %d on attempt %d/%d — retrying in %.2fs | URL: %s",
                        status, attempt, max_retries, wait, exc.request.url,
                    )
                    last_exc = exc

                except httpx.TransportError as exc:
                    wait = _compute_backoff(attempt, base_seconds, max_seconds, jitter_range)
                    logger.warning(
                        "TransportError on attempt %d/%d — retrying in %.2fs | %s: %s",
                        attempt, max_retries, wait,
                        type(exc).__name__, str(exc),
                    )
                    last_exc = exc

                except CircuitOpenError:
                    # Do not retry — circuit is open, bubble up immediately
                    raise

                if attempt < max_retries:
                    time.sleep(wait)

            logger.error(
                "All %d retry attempts exhausted. Last error: %s",
                max_retries, repr(last_exc),
            )
            raise RetryExhaustedError(
                f"Request failed after {max_retries} attempts."
            ) from last_exc

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Async Retry Decorator (for async methods)
# ---------------------------------------------------------------------------
def async_retry_with_backoff(
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_seconds: float = DEFAULT_BASE_BACKOFF_SECONDS,
    max_seconds: float = DEFAULT_MAX_BACKOFF_SECONDS,
    jitter_range: float = DEFAULT_JITTER_RANGE,
    retryable_status_codes: tuple[int, ...] = DEFAULT_RETRYABLE_STATUS_CODES,
) -> Callable:
    """
    Async variant of retry_with_backoff for coroutine methods.
    Uses asyncio.sleep() to yield control back to the event loop during waits.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Optional[Exception] = None

            for attempt in range(1, max_retries + 1):
                try:
                    result = await func(*args, **kwargs)
                    if attempt > 1:
                        logger.info(
                            "✓ Async request succeeded on attempt %d/%d.",
                            attempt, max_retries,
                        )
                    return result

                except httpx.HTTPStatusError as exc:
                    status = exc.response.status_code

                    if status not in retryable_status_codes:
                        logger.critical(
                            "Non-retryable HTTP %d — aborting. URL: %s | Body: %s",
                            status, exc.request.url, exc.response.text[:500],
                        )
                        raise NonRetryableHTTPError(
                            request=exc.request, response=exc.response
                        ) from exc

                    wait = _compute_backoff(attempt, base_seconds, max_seconds, jitter_range)
                    if status == 429:
                        server_wait = _extract_retry_after(exc.response)
                        if server_wait is not None:
                            wait = max(wait, server_wait)

                    logger.warning(
                        "HTTP %d on async attempt %d/%d — retrying in %.2fs | URL: %s",
                        status, attempt, max_retries, wait, exc.request.url,
                    )
                    last_exc = exc

                except httpx.TransportError as exc:
                    wait = _compute_backoff(attempt, base_seconds, max_seconds, jitter_range)
                    logger.warning(
                        "Async TransportError attempt %d/%d — retrying in %.2fs | %s: %s",
                        attempt, max_retries, wait, type(exc).__name__, str(exc),
                    )
                    last_exc = exc

                except CircuitOpenError:
                    raise

                if attempt < max_retries:
                    await asyncio.sleep(wait)

            logger.error("All %d async retry attempts exhausted.", max_retries)
            raise RetryExhaustedError(
                f"Async request failed after {max_retries} attempts."
            ) from last_exc

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Request / Response Context — structured logging payload
# ---------------------------------------------------------------------------
@dataclass
class RequestContext:
    """
    Lightweight carrier for structured log fields on each request lifecycle.

    Designed to be passed through the call chain and serialised to the
    log record so downstream log aggregators (Datadog, Loki) can correlate
    entries by trace_id.
    """

    method: str
    url: str
    attempt: int = 1
    started_at: float = field(default_factory=time.monotonic)
    trace_id: str = field(
        default_factory=lambda: f"{datetime.now(tz=timezone.utc).strftime('%Y%m%dT%H%M%S')}_{random.randint(1000, 9999)}"
    )

    def elapsed_ms(self) -> float:
        return (time.monotonic() - self.started_at) * 1000.0

    def log_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "method": self.method,
            "url": self.url,
            "attempt": self.attempt,
            "elapsed_ms": round(self.elapsed_ms(), 2),
        }


# ---------------------------------------------------------------------------
# Base HTTP Client
# ---------------------------------------------------------------------------
class BaseHTTPClient:
    """
    Resilient async HTTP client base class.

    Provides:
      - httpx.AsyncClient session lifecycle (async context manager)
      - Synchronous `request()` facade via asyncio.run() for batch pipelines
      - Per-host Circuit Breaker integration
      - Structured request/response logging at DEBUG/INFO/WARNING levels
      - Configurable timeout (total, connect, read, write)

    Subclasses should override `_build_default_headers()` and
    `_build_default_params()` to inject API tokens and common parameters
    without duplicating them across every call site.

    Async usage (preferred for parallel zone fetches):
        async with MyClient() as client:
            response = await client.async_request("GET", "/endpoint", params={...})

    Sync usage (for batch pipeline compatibility):
        with MyClient() as client:
            response = client.request("GET", "/endpoint", params={...})
    """

    # Subclasses set these to enable automatic circuit-breaker per host
    _circuit_breakers: ClassVar[dict[str, CircuitBreaker]] = {}

    def __init__(
        self,
        base_url: str,
        *,
        max_retries: int = DEFAULT_MAX_RETRIES,
        base_backoff_seconds: float = DEFAULT_BASE_BACKOFF_SECONDS,
        max_backoff_seconds: float = DEFAULT_MAX_BACKOFF_SECONDS,
        jitter_range: float = DEFAULT_JITTER_RANGE,
        retryable_status_codes: tuple[int, ...] = DEFAULT_RETRYABLE_STATUS_CODES,
        request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        connect_timeout_seconds: float = DEFAULT_CONNECT_TIMEOUT_SECONDS,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_recovery_seconds: float = 120.0,
        follow_redirects: bool = True,
        http2: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.base_backoff_seconds = base_backoff_seconds
        self.max_backoff_seconds = max_backoff_seconds
        self.jitter_range = jitter_range
        self.retryable_status_codes = retryable_status_codes
        self.follow_redirects = follow_redirects
        self.http2 = http2

        self._timeout = httpx.Timeout(
            timeout=request_timeout_seconds,
            connect=connect_timeout_seconds,
        )

        # Per-host circuit breaker (shared across instances for same host)
        host = httpx.URL(base_url).host
        if host not in BaseHTTPClient._circuit_breakers:
            BaseHTTPClient._circuit_breakers[host] = CircuitBreaker(
                failure_threshold=circuit_breaker_threshold,
                recovery_timeout_seconds=circuit_breaker_recovery_seconds,
                host=host,
            )
        self._circuit_breaker = BaseHTTPClient._circuit_breakers[host]

        # Async client (lazily initialised on first async_request / __aenter__)
        self._async_client: Optional[httpx.AsyncClient] = None

        # Sync client (lazily initialised on first request / __enter__)
        self._sync_client: Optional[httpx.Client] = None

        logger.info(
            "BaseHTTPClient initialised | base_url=%s | max_retries=%d | "
            "timeout=%.1fs | circuit_breaker_threshold=%d | http2=%s",
            self.base_url, max_retries, request_timeout_seconds,
            circuit_breaker_threshold, http2,
        )

    # ------------------------------------------------------------------
    # Overridable hooks for subclass customisation
    # ------------------------------------------------------------------
    def _build_default_headers(self) -> dict[str, str]:
        """Return headers injected into every request. Override in subclasses."""
        return {"Accept": "application/xml", "User-Agent": "SwedenGridPipeline/1.0"}

    def _build_default_params(self) -> dict[str, Any]:
        """Return query parameters injected into every request. Override in subclasses."""
        return {}

    # ------------------------------------------------------------------
    # Sync Client Management
    # ------------------------------------------------------------------
    def _get_sync_client(self) -> httpx.Client:
        if self._sync_client is None or self._sync_client.is_closed:
            self._sync_client = httpx.Client(
                base_url=self.base_url,
                headers=self._build_default_headers(),
                timeout=self._timeout,
                follow_redirects=self.follow_redirects,
                http2=self.http2,
            )
            logger.debug("Sync httpx.Client session opened for %s", self.base_url)
        return self._sync_client

    def close_sync(self) -> None:
        if self._sync_client and not self._sync_client.is_closed:
            self._sync_client.close()
            logger.debug("Sync httpx.Client session closed for %s", self.base_url)

    # ------------------------------------------------------------------
    # Async Client Management
    # ------------------------------------------------------------------
    async def _get_async_client(self) -> httpx.AsyncClient:
        if self._async_client is None or self._async_client.is_closed:
            self._async_client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self._build_default_headers(),
                timeout=self._timeout,
                follow_redirects=self.follow_redirects,
                http2=self.http2,
            )
            logger.debug("Async httpx.AsyncClient session opened for %s", self.base_url)
        return self._async_client

    async def close_async(self) -> None:
        if self._async_client and not self._async_client.is_closed:
            await self._async_client.aclose()
            logger.debug("Async httpx.AsyncClient session closed for %s", self.base_url)

    # ------------------------------------------------------------------
    # Core Async Request (with circuit breaker + structured logging)
    # ------------------------------------------------------------------
    async def async_request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
        headers: Optional[dict[str, str]] = None,
        json_body: Optional[dict[str, Any]] = None,
        content: Optional[bytes] = None,
        extra_log: Optional[dict[str, Any]] = None,
    ) -> httpx.Response:
        """
        Execute a single async HTTP request with circuit breaker enforcement.

        Does NOT retry — use async_retry_with_backoff on the calling method.

        Args:
            method:     HTTP verb (GET, POST, etc.)
            path:       Path relative to base_url
            params:     Query parameters (merged with defaults)
            headers:    Additional headers (merged with defaults)
            json_body:  JSON request body
            content:    Raw bytes body
            extra_log:  Additional key-value pairs for the log record

        Returns:
            httpx.Response (raise_for_status NOT called — caller decides)

        Raises:
            CircuitOpenError: If the circuit breaker rejects the request
        """
        if not self._circuit_breaker.allow_request():
            raise CircuitOpenError(
                f"Circuit breaker OPEN for {self._circuit_breaker.host}. "
                f"Skipping request to {path}."
            )

        merged_params = {**self._build_default_params(), **(params or {})}
        merged_headers = {**self._build_default_headers(), **(headers or {})}

        ctx = RequestContext(method=method, url=f"{self.base_url}{path}")
        log_base = {**ctx.log_dict(), **(extra_log or {})}

        logger.debug(
            "→ %s %s%s | trace_id=%s | params_count=%d",
            method, self.base_url, path, ctx.trace_id, len(merged_params),
        )

        client = await self._get_async_client()

        try:
            response = await client.request(
                method=method,
                url=path,
                params=merged_params,
                headers=merged_headers,
                json=json_body,
                content=content,
            )
            elapsed_ms = ctx.elapsed_ms()

            if response.is_success:
                self._circuit_breaker.record_success()
                logger.info(
                    "← HTTP %d | %.0fms | %d bytes | trace_id=%s",
                    response.status_code,
                    elapsed_ms,
                    len(response.content),
                    ctx.trace_id,
                )
            else:
                # Record failure only for 5xx; 4xx are caller's fault
                if response.status_code >= 500:
                    self._circuit_breaker.record_failure()
                logger.warning(
                    "← HTTP %d | %.0fms | trace_id=%s | url=%s",
                    response.status_code, elapsed_ms, ctx.trace_id,
                    response.url,
                )
                response.raise_for_status()

            return response

        except httpx.TransportError as exc:
            self._circuit_breaker.record_failure()
            logger.warning(
                "TransportError | trace_id=%s | %s: %s",
                ctx.trace_id, type(exc).__name__, str(exc),
            )
            raise

    # ------------------------------------------------------------------
    # Synchronous Facade (runs the async method in a fresh event loop)
    # ------------------------------------------------------------------
    def request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
        headers: Optional[dict[str, str]] = None,
        json_body: Optional[dict[str, Any]] = None,
        content: Optional[bytes] = None,
        extra_log: Optional[dict[str, Any]] = None,
    ) -> httpx.Response:
        """
        Synchronous request wrapper for batch pipeline contexts.

        Internally runs async_request() in a dedicated event loop so that
        batch pipeline scripts (Airflow tasks, cron jobs) don't need to
        manage async contexts. The sync client is used directly for simpler
        request flows.

        For high-throughput parallel zone fetches, prefer the async API.
        """
        if not self._circuit_breaker.allow_request():
            raise CircuitOpenError(
                f"Circuit breaker OPEN for {self._circuit_breaker.host}."
            )

        merged_params = {**self._build_default_params(), **(params or {})}
        merged_headers = {**self._build_default_headers(), **(headers or {})}

        ctx = RequestContext(method=method, url=f"{self.base_url}{path}")
        logger.debug(
            "→ SYNC %s %s%s | trace_id=%s",
            method, self.base_url, path, ctx.trace_id,
        )

        client = self._get_sync_client()

        try:
            response = client.request(
                method=method,
                url=path,
                params=merged_params,
                headers=merged_headers,
                json=json_body,
                content=content,
            )
            elapsed_ms = ctx.elapsed_ms()

            if response.is_success:
                self._circuit_breaker.record_success()
                logger.info(
                    "← HTTP %d | %.0fms | %d bytes | trace_id=%s",
                    response.status_code,
                    elapsed_ms,
                    len(response.content),
                    ctx.trace_id,
                )
            else:
                if response.status_code >= 500:
                    self._circuit_breaker.record_failure()
                logger.warning(
                    "← HTTP %d | %.0fms | trace_id=%s",
                    response.status_code, elapsed_ms, ctx.trace_id,
                )
                response.raise_for_status()

            return response

        except httpx.TransportError as exc:
            self._circuit_breaker.record_failure()
            logger.warning(
                "Sync TransportError | trace_id=%s | %s: %s",
                ctx.trace_id, type(exc).__name__, str(exc),
            )
            raise

    # ------------------------------------------------------------------
    # Parallel Async Fetch (for multi-zone concurrent requests)
    # ------------------------------------------------------------------
    async def async_fetch_all(
        self,
        requests: list[dict[str, Any]],
        max_concurrency: int = 4,
    ) -> list[Optional[httpx.Response]]:
        """
        Execute multiple async requests with bounded concurrency.

        Args:
            requests:        List of dicts, each containing 'method', 'path',
                             and optionally 'params', 'headers', 'extra_log'
            max_concurrency: Max simultaneous in-flight requests (semaphore bound)

        Returns:
            List of responses (same order as input). Failed requests return None
            so callers can inspect which zones failed without aborting the batch.
        """
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _bounded_request(req: dict[str, Any]) -> Optional[httpx.Response]:
            async with semaphore:
                try:
                    return await self.async_request(
                        method=req["method"],
                        path=req.get("path", ""),
                        params=req.get("params"),
                        headers=req.get("headers"),
                        extra_log=req.get("extra_log"),
                    )
                except (CircuitOpenError, RetryExhaustedError, httpx.HTTPStatusError) as exc:
                    logger.error(
                        "Bounded request failed for path=%s | %s: %s",
                        req.get("path"), type(exc).__name__, str(exc),
                    )
                    return None

        return list(await asyncio.gather(*[_bounded_request(r) for r in requests]))

    # ------------------------------------------------------------------
    # Context Manager Support (sync)
    # ------------------------------------------------------------------
    def __enter__(self) -> "BaseHTTPClient":
        self._get_sync_client()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close_sync()

    # ------------------------------------------------------------------
    # Async Context Manager Support
    # ------------------------------------------------------------------
    async def __aenter__(self) -> "BaseHTTPClient":
        await self._get_async_client()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close_async()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    @property
    def circuit_state(self) -> CircuitState:
        return self._circuit_breaker.state

    def circuit_breaker_report(self) -> dict[str, Any]:
        cb = self._circuit_breaker
        return {
            "host": cb.host,
            "state": cb.state.name,
            "failure_count": cb._failure_count,
            "failure_threshold": cb.failure_threshold,
            "recovery_timeout_seconds": cb.recovery_timeout_seconds,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"base_url={self.base_url!r}, "
            f"circuit={self._circuit_breaker.state.name})"
        )


# ---------------------------------------------------------------------------
# CLI Smoke Test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG)

    print("=" * 70)
    print("BaseHTTPClient — Unit Smoke Test")
    print("=" * 70)

    # --- Test 1: Backoff computation ---
    print("\n[1] Backoff schedule (attempt 1-5, base=2s, max=60s, jitter=0.5):")
    random.seed(42)
    for i in range(1, 6):
        wait = _compute_backoff(i, 2.0, 60.0, 0.5)
        print(f"    Attempt {i}: {wait:.3f}s")

    # --- Test 2: Circuit Breaker state transitions ---
    print("\n[2] CircuitBreaker state transitions (threshold=3):")
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout_seconds=5.0, host="test.api")
    print(f"    Initial state: {cb.state.name}")
    assert cb.allow_request() is True

    for i in range(3):
        cb.record_failure()
        print(f"    After failure {i + 1}: {cb.state.name} | allow={cb.allow_request()}")

    assert cb.state == CircuitState.OPEN
    print(f"    ✓ Circuit opened after threshold reached")

    # Simulate recovery timeout
    cb._last_failure_at = time.monotonic() - 6.0  # force elapsed > timeout
    print(f"    After simulated timeout: allow={cb.allow_request()} | state={cb.state.name}")
    assert cb.state == CircuitState.HALF_OPEN

    cb.record_success()
    print(f"    After probe success: {cb.state.name}")
    assert cb.state == CircuitState.CLOSED
    print("    ✓ Circuit closed after successful probe")

    # --- Test 3: Retry-After header parsing ---
    print("\n[3] Retry-After header parsing:")
    class _FakeResponse:
        headers: dict[str, str]
        def __init__(self, h): self.headers = h

    r1 = _extract_retry_after(_FakeResponse({"Retry-After": "45"}))  # type: ignore[arg-type]
    print(f"    Integer '45' → {r1}s  ✓" if r1 == 45.0 else f"    FAIL: got {r1}")

    r2 = _extract_retry_after(_FakeResponse({}))  # type: ignore[arg-type]
    print(f"    Missing header → {r2}  ✓" if r2 is None else f"    FAIL: got {r2}")

    # --- Test 4: BaseHTTPClient instantiation ---
    print("\n[4] BaseHTTPClient instantiation:")
    client = BaseHTTPClient("https://web-api.tp.entsoe.eu/api")
    print(f"    ✓ {client!r}")
    print(f"    Circuit state: {client.circuit_state.name}")
    print(f"    Report: {client.circuit_breaker_report()}")

    print("\n✅ All BaseHTTPClient smoke tests passed.")
    sys.exit(0)
