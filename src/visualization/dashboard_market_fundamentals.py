"""
src/visualization/dashboard.py  — MARKET FUNDAMENTALS VIEWPORT ADDITIONS
=============================================================================
Additive patch for the existing Control Room Streamlit dashboard.
=============================================================================

Merge instructions:
  1. Add the two helper functions (_render_net_position_vs_imbalance_chart and
     _render_nuclear_alert_container) anywhere after the load_pipeline_data()
     cache function.
  2. Call render_market_fundamentals_section(df_gold, df_zone, selected_zone)
     as a new section at the bottom of the main dashboard body, after the
     existing "Deep Neural Feature Analysis" panel.
  3. The sidebar alert badge (render_nuclear_sidebar_alert) is called once
     inside the sidebar block after the existing selector widgets.
=============================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

try:
    import streamlit as st
    _STREAMLIT_AVAILABLE = True
except ImportError:
    _STREAMLIT_AVAILABLE = False

from src.ingestion.market_fundamentals import NUCLEAR_ALERT_THRESHOLD_MW

# ---------------------------------------------------------------------------
# Design constants (match existing dashboard palette)
# ---------------------------------------------------------------------------
_COLOR_NAVY      = "#0A2540"
_COLOR_TEAL      = "#639FAB"
_COLOR_RED       = "#FF6B6B"
_COLOR_ORANGE    = "#FFA500"
_COLOR_GREEN     = "#4CAF50"
_COLOR_YELLOW    = "#FFD700"
_COLOR_GREY      = "#B0BEC5"
_COLOR_BG        = "#F8F9FA"

# Alert thresholds
_ALERT_MW_CRITICAL: float = NUCLEAR_ALERT_THRESHOLD_MW         # 1,000 MW
_ALERT_MW_WARNING:  float = NUCLEAR_ALERT_THRESHOLD_MW * 0.5   # 500 MW
_HYDRO_CRUNCH_FILL: float = 0.25   # Below this = potential crunch


# ===========================================================================
# VIEWPORT 1 — Scheduled Net Positions vs Actual Imbalance
# ===========================================================================

def _render_net_position_vs_imbalance_chart(
    df_zone: pd.DataFrame,
    zone: str,
) -> None:
    """
    Multi-trace time-series comparing:
      - Scheduled Commercial Net Position (MW) — Day-Ahead gate-closed value
      - Actual Grid Imbalance Volume (MWh)     — Real-time ENTSO-E measurement

    Market intelligence narrative:
        When the scheduled net position is large and positive (Sweden exporting
        heavily), a subsequent large negative imbalance indicates that the actual
        generation mix under-delivered relative to the contracted export schedule.
        This divergence pattern is a leading indicator of system stress and is
        the signal that CLASS 4B features are designed to capture.

    The chart uses a dual Y-axis layout:
        Left axis:  scheduled_net_position_mw (MW, contractual)
        Right axis: imbalance_mwh (MWh, physical deviation)

    A divergence band is shaded when the two series are moving in opposite
    directions (signed product is negative), highlighting the hours where
    market contracts and physical reality are pulling apart.
    """
    if not _STREAMLIT_AVAILABLE:
        return

    has_net_pos   = "scheduled_net_position_mw" in df_zone.columns
    has_imbalance = "imbalance_mwh" in df_zone.columns

    if not has_imbalance:
        st.warning("imbalance_mwh column not found in the loaded dataset.")
        return

    ts = df_zone["timestamp_utc"]
    imb = df_zone["imbalance_mwh"]

    fig = go.Figure()

    # ── Trace 1: Scheduled Net Position ─────────────────────────────────
    if has_net_pos:
        net_pos = df_zone["scheduled_net_position_mw"]
        fig.add_trace(go.Scatter(
            x=ts,
            y=net_pos,
            name="Scheduled Net Position (MW)",
            line=dict(color=_COLOR_NAVY, width=1.8),
            yaxis="y1",
            hovertemplate="<b>Net Position</b>: %{y:.0f} MW<br>%{x}<extra></extra>",
        ))

        # ── Divergence band: shaded when net pos and imbalance oppose ────
        # Resample both to a common index; compute sign agreement
        sign_product = net_pos.values * imb.values
        diverging    = sign_product < 0  # True when they oppose

        # Fill band on diverging segments
        y_upper = np.where(diverging, np.maximum(net_pos.values,  0), 0)
        y_lower = np.where(diverging, np.minimum(imb.values, 0), 0)

        fig.add_trace(go.Scatter(
            x=pd.concat([ts, ts.iloc[::-1]]),
            y=np.concatenate([y_upper, y_lower[::-1]]),
            fill="toself",
            fillcolor="rgba(255, 107, 107, 0.10)",
            line=dict(width=0),
            showlegend=True,
            name="Contract–Physical Divergence",
            hoverinfo="skip",
        ))
    else:
        st.info(
            "scheduled_net_position_mw not yet available in this dataset. "
            "Run fetch_scheduled_net_positions() and re-align the Silver layer."
        )

    # ── Trace 2: Actual Imbalance ─────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=ts,
        y=imb,
        name="Actual Imbalance (MWh)",
        line=dict(color=_COLOR_RED, width=1.5, dash="dash"),
        yaxis="y2",
        hovertemplate="<b>Imbalance</b>: %{y:.1f} MWh<br>%{x}<extra></extra>",
    ))

    # ── Zero reference lines ──────────────────────────────────────────────
    fig.add_hline(
        y=0, line=dict(color=_COLOR_GREY, width=0.8, dash="dot"),
        annotation_text="Balanced", annotation_position="bottom right",
    )

    fig.update_layout(
        title=dict(
            text=f"📊 Scheduled Net Position vs Actual Imbalance — Zone {zone}",
            font=dict(color=_COLOR_NAVY, size=14),
        ),
        xaxis=dict(title="Time (UTC)", showgrid=False),
        yaxis=dict(
            title="Net Position (MW)",
            title_font=dict(color=_COLOR_NAVY),
            tickfont=dict(color=_COLOR_NAVY),
            showgrid=True,
            gridcolor="#ECEFF1",
        ),
        yaxis2=dict(
            title="Imbalance (MWh)",
            title_font=dict(color=_COLOR_RED),
            tickfont=dict(color=_COLOR_RED),
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right",  x=1,
        ),
        height=420,
        margin=dict(r=10, t=60, l=10, b=10),
        plot_bgcolor=_COLOR_BG,
        paper_bgcolor="white",
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── Divergence KPI below the chart ───────────────────────────────────
    if has_net_pos:
        net_pos = df_zone["scheduled_net_position_mw"]
        diverging_pct = 100 * (net_pos.values * imb.values < 0).mean()
        corr = net_pos.corr(imb)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Contract–Physical Divergence",
                f"{diverging_pct:.1f}% of hours",
                help="% of hours where scheduled position and actual imbalance "
                     "are in opposite directions.",
            )
        with col2:
            st.metric(
                "Net Position ↔ Imbalance Correlation",
                f"{corr:.3f}",
                help="Pearson r between scheduled net position and actual "
                     "imbalance. Near zero = contracts not predictive; "
                     "negative = systematic forecast bias.",
            )
        with col3:
            latest_net_pos = net_pos.dropna().iloc[-1] if not net_pos.dropna().empty else float("nan")
            latest_imb     = imb.dropna().iloc[-1] if not imb.dropna().empty else float("nan")
            st.metric(
                "Latest Net Position",
                f"{latest_net_pos:+.0f} MW",
                delta=f"Imbalance: {latest_imb:+.1f} MWh",
            )


# ===========================================================================
# VIEWPORT 2 — Nuclear Outage Alert Container
# ===========================================================================

def _render_nuclear_alert_container(
    df_zone:       pd.DataFrame,
    zone:          str,
    alert_threshold_mw: float = _ALERT_MW_CRITICAL,
) -> None:
    """
    Conditional alert panel for active nuclear / base-load outage events.

    Alert levels:
        🟢 Normal:   outage_mw < 500 MW  (< 50% of one Forsmark unit)
        🟡 Warning:  500 ≤ outage_mw < 1,000 MW (partial unit trip)
        🔴 Critical: outage_mw ≥ 1,000 MW (full unit or multiple units)

    The panel shows:
      - Current total offline MW and unit count
      - 24h and 7-day trailing maximum offline MW
      - A sparkline of outage_mw over the displayed horizon
      - Advisory text mapping the outage level to expected imbalance risk
    """
    if not _STREAMLIT_AVAILABLE:
        return

    if "outage_mw" not in df_zone.columns:
        st.info(
            "outage_mw not yet available in this dataset. "
            "Run fetch_remit_outages() and re-align the Silver layer."
        )
        return

    current_outage_mw = float(df_zone["outage_mw"].iloc[-1]) \
        if not df_zone.empty else 0.0
    peak_24h  = float(df_zone["outage_mw"].tail(24).max())
    peak_168h = float(df_zone["outage_mw"].tail(168).max())
    active_units = int(df_zone["active_unit_count"].iloc[-1]) \
        if "active_unit_count" in df_zone.columns else "N/A"

    # ── Determine alert level ─────────────────────────────────────────────
    if current_outage_mw >= _ALERT_MW_CRITICAL:
        alert_color   = "#FF6B6B"
        alert_bg      = "#FFF5F5"
        alert_icon    = "🔴"
        alert_level   = "CRITICAL"
        advisory_text = (
            f"≥ {_ALERT_MW_CRITICAL:.0f} MW of base-load capacity offline. "
            f"Zone {zone} is highly exposed to system-short imbalance events. "
            "Check REMIT UMM portal for unit status and expected return-to-service."
        )
    elif current_outage_mw >= _ALERT_MW_WARNING:
        alert_color   = _COLOR_ORANGE
        alert_bg      = "#FFFBF0"
        alert_icon    = "🟡"
        alert_level   = "WARNING"
        advisory_text = (
            f"Partial base-load reduction of {current_outage_mw:.0f} MW. "
            "Monitor imbalance volumes for upward pressure. "
            "Cross-border net positions may compensate if SE3/SE4 exports are available."
        )
    else:
        alert_color   = _COLOR_GREEN
        alert_bg      = "#F0FFF4"
        alert_icon    = "🟢"
        alert_level   = "NORMAL"
        advisory_text = (
            "No significant unplanned generation outages active. "
            "Grid operating from full dispatchable base-load stack."
        )

    # ── Alert banner ──────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="
            background-color:{alert_bg};
            border-left: 5px solid {alert_color};
            padding: 16px 20px;
            border-radius: 6px;
            margin-bottom: 12px;
        ">
            <h4 style="color:{alert_color}; margin:0 0 8px 0;">
                {alert_icon} Nuclear / Base-load Outage Status — Zone {zone}: {alert_level}
            </h4>
            <p style="margin:0; color:#37474F; font-size:14px;">{advisory_text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── KPI columns ───────────────────────────────────────────────────────
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric(
            "Current Offline Capacity",
            f"{current_outage_mw:,.0f} MW",
            delta=f"Threshold: {alert_threshold_mw:,.0f} MW",
            delta_color="inverse" if current_outage_mw >= alert_threshold_mw else "normal",
        )
    with kpi2:
        st.metric("Active Unit Trips", str(active_units))
    with kpi3:
        st.metric("Peak Offline — 24h", f"{peak_24h:,.0f} MW")
    with kpi4:
        st.metric("Peak Offline — 7d", f"{peak_168h:,.0f} MW")

    # ── Outage sparkline ─────────────────────────────────────────────────
    fig_spark = go.Figure()

    outage_series = df_zone["outage_mw"]
    ts_series     = df_zone["timestamp_utc"]

    # Colour the area based on alert level
    fig_spark.add_trace(go.Scatter(
        x=ts_series,
        y=outage_series,
        fill="tozeroy",
        fillcolor=f"rgba({_hex_to_rgb(alert_color)}, 0.20)",
        line=dict(color=alert_color, width=1.5),
        name="Offline Capacity (MW)",
        hovertemplate="%{y:.0f} MW offline<br>%{x}<extra></extra>",
    ))

    # Critical threshold reference line
    fig_spark.add_hline(
        y=_ALERT_MW_CRITICAL,
        line=dict(color=_COLOR_RED, width=1.0, dash="dash"),
        annotation_text=f"Alert: {_ALERT_MW_CRITICAL:.0f} MW",
        annotation_position="top right",
        annotation_font=dict(color=_COLOR_RED, size=11),
    )
    # Warning threshold
    fig_spark.add_hline(
        y=_ALERT_MW_WARNING,
        line=dict(color=_COLOR_ORANGE, width=0.8, dash="dot"),
        annotation_text=f"Warning: {_ALERT_MW_WARNING:.0f} MW",
        annotation_position="bottom right",
        annotation_font=dict(color=_COLOR_ORANGE, size=10),
    )

    fig_spark.update_layout(
        xaxis=dict(showgrid=False, title="Time (UTC)"),
        yaxis=dict(
            showgrid=True, gridcolor="#ECEFF1",
            title="Offline Capacity (MW)",
        ),
        height=250,
        margin=dict(r=10, t=20, l=10, b=10),
        plot_bgcolor=_COLOR_BG,
        paper_bgcolor="white",
        showlegend=False,
    )
    st.plotly_chart(fig_spark, use_container_width=True)


# ===========================================================================
# VIEWPORT 3 — Hydro Reservoir Status (supplementary)
# ===========================================================================

def _render_hydro_reservoir_panel(df_zone: pd.DataFrame) -> None:
    """
    Compact reservoir fill panel showing current fill ratio and depletion
    velocity.  Intended as a two-column companion to the outage alert panel.

    Displayed metrics:
      - Current reservoir_fill_ratio as a progress bar (0–100%)
      - hydro_depletion_velocity (weekly fill change)
      - is_hydro_crunch flag status
    """
    if not _STREAMLIT_AVAILABLE:
        return

    if "reservoir_fill_ratio" not in df_zone.columns:
        st.info("reservoir_fill_ratio not yet in dataset. Fetch hydro data first.")
        return

    current_fill = df_zone["reservoir_fill_ratio"].dropna().iloc[-1] \
        if not df_zone["reservoir_fill_ratio"].dropna().empty else float("nan")
    depletion_vel = df_zone["reservoir_fill_diff_1w"].dropna().iloc[-1] \
        if "reservoir_fill_diff_1w" in df_zone.columns \
            and not df_zone["reservoir_fill_diff_1w"].dropna().empty \
        else float("nan")
    crunch_active = bool(df_zone["is_hydro_crunch"].iloc[-1]) \
        if "is_hydro_crunch" in df_zone.columns else False

    fill_pct = current_fill * 100 if not np.isnan(current_fill) else 0.0

    # Colour by fill level
    if fill_pct >= 60:
        fill_color = _COLOR_GREEN
        fill_label = "Adequate"
    elif fill_pct >= 30:
        fill_color = _COLOR_ORANGE
        fill_label = "Moderate"
    else:
        fill_color = _COLOR_RED
        fill_label = "Low"

    st.markdown("#### 💧 Hydro Reservoir Buffer")
    st.markdown(
        f"""
        <div style="margin-bottom:8px;">
            <span style="font-size:13px; color:#546E7A;">
                Current Fill Level — SE1/SE2 Northern Basin
            </span>
        </div>
        <div style="background-color:#ECEFF1; border-radius:8px; height:20px; width:100%;">
            <div style="
                background-color:{fill_color};
                width:{fill_pct:.1f}%;
                height:100%;
                border-radius:8px;
                transition: width 0.5s ease;
            "></div>
        </div>
        <div style="display:flex; justify-content:space-between; margin-top:4px;">
            <span style="font-size:12px; color:#546E7A;">0%</span>
            <span style="font-weight:bold; color:{fill_color};">
                {fill_pct:.1f}% — {fill_label}
            </span>
            <span style="font-size:12px; color:#546E7A;">100%</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")
    col1, col2 = st.columns(2)
    with col1:
        vel_str = f"{depletion_vel:+.4f}/wk" if not np.isnan(depletion_vel) else "N/A"
        vel_delta_color = "inverse" if depletion_vel < 0 else "normal"
        st.metric(
            "Depletion Velocity (1w)",
            vel_str,
            help="Weekly change in fill ratio. Negative = reservoir draining.",
            delta_color=vel_delta_color,
        )
    with col2:
        crunch_text = "⚠️ ACTIVE" if crunch_active else "✅ None"
        crunch_color = _COLOR_RED if crunch_active else _COLOR_GREEN
        st.markdown(
            f"""
            <div style="
                padding: 8px 12px;
                background: {'#FFF5F5' if crunch_active else '#F0FFF4'};
                border-radius: 4px;
                border-left: 3px solid {crunch_color};
            ">
                <div style="font-size:11px; color:#546E7A;">Flexibility Crunch Flag</div>
                <div style="font-size:16px; font-weight:bold; color:{crunch_color};">
                    {crunch_text}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ===========================================================================
# Master Section Renderer
# ===========================================================================

def render_market_fundamentals_section(
    df_gold:       pd.DataFrame,
    df_zone:       pd.DataFrame,
    selected_zone: str,
) -> None:
    """
    Render the complete Market Intelligence section in the Streamlit dashboard.

    Call this function once in the main dashboard body after the existing
    structural analysis panel.

    Layout:
        ── Section header ──────────────────────────────────────────────────
        ── Full-width: Net Position vs Imbalance chart ──────────────────────
        ── Two-column row: Nuclear Alert | Hydro Reservoir ─────────────────

    Args:
        df_gold:       Full multi-zone Gold layer DataFrame (all zones).
        df_zone:       Pre-filtered, time-sliced DataFrame for selected_zone.
        selected_zone: Zone label string (e.g. 'SE3').
    """
    if not _STREAMLIT_AVAILABLE:
        return

    st.markdown("---")
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #0A2540 0%, #1a3a5c 100%);
            padding: 16px 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        ">
            <h3 style="color: white; margin: 0 0 4px 0;">
                🏭 Market Intelligence Panel
            </h3>
            <p style="color: #639FAB; margin: 0; font-size: 13px;">
                Real-time REMIT outage signals · DA scheduled net positions ·
                Northern hydro reservoir buffer state
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Full-width: Scheduled Net Position vs Actual Imbalance ───────────
    st.subheader("📈 Scheduled Contract Net Positions vs Actual Imbalance Volumes")
    _render_net_position_vs_imbalance_chart(df_zone, selected_zone)

    st.markdown("")

    # ── Two-column: Outage Alert + Hydro Reservoir ────────────────────────
    alert_col, hydro_col = st.columns([3, 2])

    with alert_col:
        st.subheader("⚡ Nuclear & Base-load Outage Alert")
        _render_nuclear_alert_container(
            df_zone,
            selected_zone,
            alert_threshold_mw=_ALERT_MW_CRITICAL,
        )

    with hydro_col:
        # Hydro features are meaningful for SE1/SE2; show context for others
        if selected_zone in {"SE1", "SE2"}:
            _render_hydro_reservoir_panel(df_zone)
        else:
            st.subheader("💧 Hydro Reservoir Buffer")
            st.info(
                f"Zone {selected_zone} is part of the thermal-dominant grid. "
                "The SE1/SE2 hydro reservoir signal is used as a cross-zone "
                "supply context feature. Select SE1 or SE2 for the primary "
                "reservoir status view."
            )
            # Still show reservoir fill if available (national aggregate)
            if "reservoir_fill_ratio" in df_zone.columns:
                fill = df_zone["reservoir_fill_ratio"].dropna()
                if not fill.empty:
                    national_fill = fill.iloc[-1] * 100
                    st.metric(
                        "SE1/SE2 Reservoir Fill (National)",
                        f"{national_fill:.1f}%",
                        help="Aggregate northern basin fill used as "
                             "supply-flexibility context for all zones.",
                    )


# ===========================================================================
# Sidebar Alert Badge (inject into existing sidebar block)
# ===========================================================================

def render_nuclear_sidebar_alert(
    df_zone: pd.DataFrame,
    selected_zone: str,
) -> None:
    """
    Compact nuclear outage badge for the existing sidebar controls block.

    Call immediately after the zone selector widget in the sidebar:

        selected_zone = st.sidebar.selectbox(...)
        render_nuclear_sidebar_alert(df_zone, selected_zone)

    Shows a colour-coded badge with current offline MW so operators can
    see the alert state at a glance without scrolling to the main panel.
    """
    if not _STREAMLIT_AVAILABLE:
        return

    if "outage_mw" not in df_zone.columns or df_zone.empty:
        return

    current_mw = float(df_zone["outage_mw"].iloc[-1])

    if current_mw >= _ALERT_MW_CRITICAL:
        bg, label = "#FFEBEE", f"🔴 {current_mw:,.0f} MW offline"
    elif current_mw >= _ALERT_MW_WARNING:
        bg, label = "#FFF8E1", f"🟡 {current_mw:,.0f} MW offline"
    else:
        bg, label = "#E8F5E9", "🟢 No outages"

    st.sidebar.markdown(
        f"""
        <div style="
            background:{bg};
            padding:8px 12px;
            border-radius:6px;
            font-size:13px;
            font-weight:bold;
            text-align:center;
            margin-bottom:8px;
        ">{label} — {selected_zone}</div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Internal utility
# ---------------------------------------------------------------------------

def _hex_to_rgb(hex_color: str) -> str:
    """Convert '#RRGGBB' hex to 'R, G, B' string for rgba() CSS calls."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r}, {g}, {b}"
