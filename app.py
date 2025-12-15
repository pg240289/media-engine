"""
FE Media Intelligence Platform - Main Streamlit Application

A unified analytics platform with AI-powered insights for marketing performance.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Set page config first
st.set_page_config(
    page_title="FE Media Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import modules
from config import (
    PLATFORM_NAME, PLATFORM_VERSION, PLATFORM_TAGLINE,
    COLORS, PLATFORMS, GEOS, DEMO_QUESTIONS,
    HIERARCHY_LEVELS, HIERARCHY_COLUMN_LABELS, OBJECTIVES, CREATIVE_TYPES
)
from utils.data_loader import load_data, filter_data, get_date_range, get_unique_values
from utils.formatters import format_inr, format_number, format_percentage, format_delta, format_ratio
from engines.analytics import AnalyticsEngine
from engines.diagnostics import DiagnosticsEngine
from engines.forecasting import ForecastingEngine
from engines.query_router import QueryRouter
from ui.styles import apply_custom_css
from ui.components import (
    metric_card, diagnostic_card, section_header, ai_response_card,
    kpi_row, empty_state
)
from ui.charts import (
    create_trend_chart, create_bar_chart, create_platform_comparison_chart,
    create_decomposition_chart, create_platform_attribution_chart,
    create_spend_efficiency_scatter, create_pie_chart, create_funnel_chart,
    create_stacked_area_chart, create_heatmap
)


def init_session_state():
    """Initialize session state variables"""
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    if 'current_query' not in st.session_state:
        st.session_state.current_query = ""
    if 'current_response' not in st.session_state:
        st.session_state.current_response = None


def load_app_data():
    """Load and cache data"""
    try:
        df = load_data()
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please run the data generator first.")
        st.code("python data/generate_mock_data.py", language="bash")
        st.stop()


def render_sidebar(client_df: pd.DataFrame):
    """Render sidebar with filters (client is selected in main header)"""
    st.sidebar.markdown(f"""
    <div style="padding: 1rem 0; border-bottom: 1px solid {COLORS['border']}; margin-bottom: 1rem;">
        <h2 style="margin: 0; font-size: 1.25rem; color: {COLORS['text_primary']};">
            📊 Filters
        </h2>
        <p style="margin: 0.25rem 0 0 0; font-size: 0.7rem; color: {COLORS['text_secondary']};">
            Refine your data view
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Date range - use client-specific data for date range
    min_date, max_date = get_date_range(client_df)

    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=max_date - timedelta(days=13),
            min_value=min_date,
            max_value=max_date,
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
        )

    # Platform filter - only show platforms available for this client
    available_platforms = get_unique_values(client_df, 'platform')
    platforms = st.sidebar.multiselect(
        "Platforms",
        options=available_platforms,
        default=available_platforms,
    )

    # Geo filter - only show geos available for this client
    available_geos = get_unique_values(client_df, 'geo')
    geos = st.sidebar.multiselect(
        "Geographies",
        options=available_geos,
        default=[],
        placeholder="All Geos"
    )

    st.sidebar.markdown("---")

    # Quick date presets
    st.sidebar.markdown(f"""
    <p style="font-size: 0.7rem; color: {COLORS['text_secondary']}; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">
        Quick Select
    </p>
    """, unsafe_allow_html=True)

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Last 7 Days", width="stretch"):
            st.session_state.start_date = max_date - timedelta(days=6)
            st.session_state.end_date = max_date
            st.rerun()
    with col2:
        if st.button("Last 14 Days", width="stretch"):
            st.session_state.start_date = max_date - timedelta(days=13)
            st.session_state.end_date = max_date
            st.rerun()

    return {
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d'),
        'platforms': platforms if platforms else None,
        'geos': geos if geos else None,
    }


def render_control_panel(df: pd.DataFrame):
    """
    Smart control panel with:
    - Industry filter (view all clients in an industry)
    - Client filter (specific client or All)
    - Date preset buttons (All, 7D, 14D, 30D, MTD) + custom range
    - Platform/Geo filters
    Returns filtered dataframe, filter settings, and whether to show deltas.
    """
    # Get unique industries and clients
    industries = sorted(df['industry'].unique().tolist())
    all_clients = sorted(df['client'].unique().tolist())

    # Get global date bounds
    global_max_date = df['date'].max().date()
    global_min_date = df['date'].min().date()

    # Initialize date preset in session state (default to "All")
    if 'date_preset' not in st.session_state:
        st.session_state.date_preset = "All"

    # === HEADER ===
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #1A1D29 0%, #242836 50%, #1A1D29 100%);
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 20px 28px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    ">
        <div>
            <div style="
                font-size: 1.75rem;
                font-weight: 700;
                color: #F9FAFB;
                letter-spacing: -0.02em;
                display: flex;
                align-items: center;
                gap: 12px;
            ">
                <span style="
                    background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                ">FE</span>
                <span>Media Intelligence Platform</span>
            </div>
            <div style="
                font-size: 0.875rem;
                color: #9CA3AF;
                margin-top: 4px;
            ">{PLATFORM_TAGLINE}</div>
        </div>
        <div style="
            display: flex;
            align-items: center;
            gap: 16px;
        ">
            <div style="
                background: rgba(99, 102, 241, 0.1);
                border: 1px solid rgba(99, 102, 241, 0.3);
                border-radius: 8px;
                padding: 8px 16px;
                text-align: center;
            ">
                <div style="font-size: 0.65rem; color: #9CA3AF; text-transform: uppercase; letter-spacing: 0.05em;">Version</div>
                <div style="font-size: 0.9rem; color: #6366F1; font-weight: 600;">{PLATFORM_VERSION}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # === ROW 1: Industry, Client, Platforms, Geos ===
    col1, col2, col3, col4 = st.columns([1.5, 1.5, 1.5, 1.5])

    with col1:
        selected_industry = st.selectbox(
            "Industry",
            options=["All Industries"] + industries,
            key="industry_select"
        )

    # Filter clients based on industry selection
    if selected_industry == "All Industries":
        available_clients = all_clients
        industry_df = df
    else:
        industry_df = df[df['industry'] == selected_industry]
        available_clients = sorted(industry_df['client'].unique().tolist())

    with col2:
        client_options = ["All Clients"] + available_clients
        selected_client = st.selectbox(
            "Client",
            options=client_options,
            key="client_select"
        )

    # Get filtered dataframe based on selections
    if selected_client == "All Clients":
        filtered_df = industry_df.copy()
        display_name = selected_industry if selected_industry != "All Industries" else "All Data"
    else:
        filtered_df = df[df['client'] == selected_client].copy()
        display_name = selected_client

    # Get date bounds for filtered data
    min_date, max_date = get_date_range(filtered_df)
    min_date = min_date.date() if hasattr(min_date, 'date') else min_date
    max_date = max_date.date() if hasattr(max_date, 'date') else max_date

    with col3:
        available_platforms = sorted(filtered_df['platform'].unique().tolist())
        selected_platforms = st.multiselect(
            "Platforms",
            options=available_platforms,
            default=[],
            placeholder="All Platforms",
            key="platforms"
        )

    with col4:
        available_geos = sorted(filtered_df['geo'].unique().tolist())
        selected_geos = st.multiselect(
            "Geographies",
            options=available_geos,
            default=[],
            placeholder="All Geos",
            key="geos"
        )

    # === ROW 2: Date Range (Presets + Custom) ===
    # Use radio for proper selection highlighting
    current_preset = st.radio(
        "Date Range",
        options=["All", "7D", "14D", "30D", "MTD", "Custom"],
        index=["All", "7D", "14D", "30D", "MTD", "Custom"].index(st.session_state.date_preset) if st.session_state.date_preset in ["All", "7D", "14D", "30D", "MTD", "Custom"] else 0,
        horizontal=True,
        key="date_preset_radio",
        label_visibility="visible"
    )

    # Update session state with current preset
    st.session_state.date_preset = current_preset

    # Calculate dates based on preset
    show_deltas = False  # Only true for specific presets

    if current_preset == "All":
        start_date = min_date
        end_date = max_date
        show_deltas = False
    elif current_preset == "7D":
        start_date = max_date - timedelta(days=6)
        end_date = max_date
        show_deltas = True
    elif current_preset == "14D":
        start_date = max_date - timedelta(days=13)
        end_date = max_date
        show_deltas = True
    elif current_preset == "30D":
        start_date = max_date - timedelta(days=29)
        end_date = max_date
        show_deltas = True
    elif current_preset == "MTD":
        start_date = max_date.replace(day=1)
        end_date = max_date
        show_deltas = True
    else:  # Custom - show date pickers
        show_deltas = False
        # Initialize custom dates if not set
        if 'custom_start' not in st.session_state:
            st.session_state.custom_start = max_date - timedelta(days=13)
        if 'custom_end' not in st.session_state:
            st.session_state.custom_end = max_date

        # Show date pickers only for Custom mode
        date_col1, date_col2 = st.columns(2)
        with date_col1:
            start_date = st.date_input(
                "From",
                value=st.session_state.custom_start,
                min_value=min_date,
                max_value=max_date,
                key="custom_start_input"
            )
            st.session_state.custom_start = start_date

        with date_col2:
            end_date = st.date_input(
                "To",
                value=st.session_state.custom_end,
                min_value=min_date,
                max_value=max_date,
                key="custom_end_input"
            )
            st.session_state.custom_end = end_date

    # Ensure dates are within valid range
    if start_date < min_date:
        start_date = min_date
    if end_date > max_date:
        end_date = max_date

    # Build filters dict
    filters = {
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d'),
        'platforms': selected_platforms if selected_platforms else None,
        'geos': selected_geos if selected_geos else None,
        'show_deltas': show_deltas,  # Only show % change for preset periods
    }

    # Return values
    industry_display = selected_industry if selected_industry != "All Industries" else "All Industries"

    return display_name, filtered_df, industry_display, filters


def render_data_context(display_name: str, industry: str, filters: dict):
    """
    Render the data context bar - shows what data is being displayed.
    This appears ABOVE the main content, BELOW the filters.
    """
    start = datetime.strptime(filters['start_date'], '%Y-%m-%d')
    end = datetime.strptime(filters['end_date'], '%Y-%m-%d')
    days = (end - start).days + 1

    date_str = f"{start.strftime('%b %d, %Y')} — {end.strftime('%b %d, %Y')} ({days} days)"

    # Build filter badges - show actual filter values
    filter_badges = []

    # Client/Industry display - determine what's actually selected
    # If display_name equals industry, user selected an industry (not a specific client)
    # If display_name differs from industry, user selected a specific client
    if display_name != "All Data":
        if display_name == industry:
            # Industry selected (viewing all clients in that industry)
            filter_badges.append(f"<span style='background: {COLORS['info']}22; color: {COLORS['info']}; padding: 0.25rem 0.6rem; border-radius: 4px; font-size: 0.8rem;'>Industry: {industry}</span>")
        else:
            # Specific client selected
            filter_badges.append(f"<span style='background: {COLORS['primary']}22; color: {COLORS['primary_light']}; padding: 0.25rem 0.6rem; border-radius: 4px; font-size: 0.8rem;'>Client: {display_name}</span>")
            # Also show industry if it's not "All Industries"
            if industry != "All Industries":
                filter_badges.append(f"<span style='background: {COLORS['info']}22; color: {COLORS['info']}; padding: 0.25rem 0.6rem; border-radius: 4px; font-size: 0.8rem;'>Industry: {industry}</span>")

    # Platform filters - show actual values
    if filters.get('platforms'):
        platforms_str = ", ".join(filters['platforms'])
        filter_badges.append(f"<span style='background: {COLORS['success']}22; color: {COLORS['success']}; padding: 0.25rem 0.6rem; border-radius: 4px; font-size: 0.8rem;'>Platform: {platforms_str}</span>")

    # Geo filters - show actual values
    if filters.get('geos'):
        geos_str = ", ".join(filters['geos'])
        filter_badges.append(f"<span style='background: {COLORS['warning']}22; color: {COLORS['warning']}; padding: 0.25rem 0.6rem; border-radius: 4px; font-size: 0.8rem;'>Geography: {geos_str}</span>")

    # If no filters, show "All Data"
    if not filter_badges:
        filter_badges.append(f"<span style='background: {COLORS['text_muted']}22; color: {COLORS['text_secondary']}; padding: 0.25rem 0.6rem; border-radius: 4px; font-size: 0.8rem;'>All Data</span>")

    filters_html = " ".join(filter_badges)

    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {COLORS['surface']} 0%, {COLORS['surface_light']} 100%);
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0 1rem 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.5rem;
    ">
        <div style="display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap;">
            {filters_html}
        </div>
        <div style="display: flex; align-items: center;">
            <span style="color: {COLORS['text_secondary']}; font-size: 0.85rem;">
                📅 {date_str}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def generate_executive_insights(analytics: AnalyticsEngine, diagnostics: DiagnosticsEngine, filters: dict) -> list:
    """Generate smart insights based on data analysis"""
    insights = []

    summary = analytics.get_summary_metrics(
        start_date=filters['start_date'],
        end_date=filters['end_date'],
        platforms=filters['platforms'],
        geos=filters['geos'],
    )

    current = summary['current']
    deltas = summary.get('deltas', {})

    # Insight 1: Overall performance trend
    cpa_delta = deltas.get('cpa', {}).get('percentage', 0)
    conv_delta = deltas.get('conversions', {}).get('percentage', 0)

    if cpa_delta < -5 and conv_delta > 5:
        insights.append({
            'type': 'success',
            'icon': '🚀',
            'title': 'Strong Performance',
            'text': f"CPA improved by {abs(cpa_delta):.1f}% while conversions grew {conv_delta:.1f}%. Campaign efficiency is trending positively."
        })
    elif cpa_delta > 10:
        insights.append({
            'type': 'warning',
            'icon': '⚠️',
            'title': 'CPA Alert',
            'text': f"CPA increased by {cpa_delta:.1f}% vs previous period. Recommend reviewing audience targeting and bid strategies."
        })
    elif conv_delta < -10:
        insights.append({
            'type': 'warning',
            'icon': '📉',
            'title': 'Volume Decline',
            'text': f"Conversions dropped {abs(conv_delta):.1f}%. Consider increasing budget allocation or expanding targeting."
        })

    # Insight 2: Platform performance
    platform_breakdown = analytics.get_breakdown_by_dimension(
        dimension='platform',
        start_date=filters['start_date'],
        end_date=filters['end_date'],
        platforms=filters['platforms'],
        geos=filters['geos'],
    )

    if len(platform_breakdown) > 1:
        platform_data = platform_breakdown[platform_breakdown['platform'] != 'Total']
        if len(platform_data) > 0:
            best_roas = platform_data.loc[platform_data['roas'].idxmax()]
            worst_cpa = platform_data.loc[platform_data['cpa'].idxmax()]

            if best_roas['roas'] > 5:
                insights.append({
                    'type': 'info',
                    'icon': '💎',
                    'title': f"{best_roas['platform']} Excelling",
                    'text': f"Delivering {best_roas['roas']:.1f}x ROAS - highest among all platforms. Consider increasing budget allocation here."
                })

            if worst_cpa['cpa'] > current['cpa'] * 1.5:
                insights.append({
                    'type': 'warning',
                    'icon': '🔍',
                    'title': f"{worst_cpa['platform']} Needs Attention",
                    'text': f"CPA of {format_inr(worst_cpa['cpa'])} is 50%+ above average. Review targeting and creative performance."
                })

    # Insight 3: Efficiency insight
    roas = current.get('roas', 0)
    if roas > 8:
        insights.append({
            'type': 'success',
            'icon': '✨',
            'title': 'High Efficiency',
            'text': f"ROAS of {roas:.1f}x indicates strong campaign ROI. Revenue is {roas:.1f}x the ad spend."
        })

    return insights[:4]  # Limit to 4 insights


def render_executive_summary(analytics: AnalyticsEngine, diagnostics: DiagnosticsEngine, filters: dict):
    """Render executive summary with key insights"""
    summary = analytics.get_summary_metrics(
        start_date=filters['start_date'],
        end_date=filters['end_date'],
        platforms=filters['platforms'],
        geos=filters['geos'],
    )

    current = summary['current']
    deltas = summary.get('deltas', {})

    # Only show deltas if the filter says so (preset periods only)
    show_deltas = filters.get('show_deltas', False)

    # Key Performance Indicators in styled cards
    st.markdown("### Key Performance Indicators")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        delta_val = deltas.get('spend', {}).get('percentage', 0) if show_deltas else None
        st.metric(
            label="💰 Total Spend",
            value=format_inr(current['spend']),
            delta=f"{delta_val:+.1f}% vs prev" if delta_val else None,
            delta_color="off"
        )

    with col2:
        delta_val = deltas.get('conversions', {}).get('percentage', 0) if show_deltas else None
        st.metric(
            label="🎯 Conversions",
            value=format_number(current['conversions']),
            delta=f"{delta_val:+.1f}% vs prev" if delta_val else None,
            delta_color="normal"
        )

    with col3:
        delta_val = deltas.get('cpa', {}).get('percentage', 0) if show_deltas else None
        st.metric(
            label="📊 CPA",
            value=format_inr(current['cpa']),
            delta=f"{delta_val:+.1f}% vs prev" if delta_val else None,
            delta_color="inverse"
        )

    with col4:
        delta_val = deltas.get('roas', {}).get('percentage', 0) if show_deltas else None
        st.metric(
            label="📈 ROAS",
            value=format_ratio(current['roas']),
            delta=f"{delta_val:+.1f}% vs prev" if delta_val else None,
            delta_color="normal"
        )

    # Secondary metrics row
    col5, col6, col7, col8 = st.columns(4)

    with col5:
        delta_val = deltas.get('impressions', {}).get('percentage', 0) if show_deltas else None
        st.metric(
            label="👁️ Impressions",
            value=format_number(current.get('impressions', 0)),
            delta=f"{delta_val:+.1f}%" if delta_val else None,
            delta_color="normal"
        )

    with col6:
        delta_val = deltas.get('clicks', {}).get('percentage', 0) if show_deltas else None
        st.metric(
            label="👆 Clicks",
            value=format_number(current.get('clicks', 0)),
            delta=f"{delta_val:+.1f}%" if delta_val else None,
            delta_color="normal"
        )

    with col7:
        delta_val = deltas.get('ctr', {}).get('percentage', 0) if show_deltas else None
        st.metric(
            label="🖱️ CTR",
            value=format_percentage(current['ctr']),
            delta=f"{delta_val:+.1f}%" if delta_val else None,
            delta_color="normal"
        )

    with col8:
        delta_val = deltas.get('cvr', {}).get('percentage', 0) if show_deltas else None
        st.metric(
            label="🔄 CVR",
            value=format_percentage(current['cvr']),
            delta=f"{delta_val:+.1f}%" if delta_val else None,
            delta_color="normal"
        )

    st.markdown("---")

    # AI-Generated Insights - only show if we have a comparison period
    if show_deltas:
        st.markdown("### 🧠 AI-Generated Insights")

        insights = generate_executive_insights(analytics, diagnostics, filters)

        if insights:
            cols = st.columns(len(insights))
            for i, insight in enumerate(insights):
                with cols[i]:
                    if insight['type'] == 'success':
                        st.success(f"**{insight['icon']} {insight['title']}**\n\n{insight['text']}")
                    elif insight['type'] == 'warning':
                        st.warning(f"**{insight['icon']} {insight['title']}**\n\n{insight['text']}")
                    else:
                        st.info(f"**{insight['icon']} {insight['title']}**\n\n{insight['text']}")
        else:
            st.info("No significant insights to report for this period.")


def render_kpi_summary(analytics: AnalyticsEngine, filters: dict):
    """Render KPI summary cards using native Streamlit metrics"""
    summary = analytics.get_summary_metrics(
        start_date=filters['start_date'],
        end_date=filters['end_date'],
        platforms=filters['platforms'],
        geos=filters['geos'],
    )

    current = summary['current']
    deltas = summary.get('deltas', {})

    # Use native Streamlit columns and metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        delta_val = deltas.get('spend', {}).get('percentage', 0)
        st.metric(
            label="Total Spend",
            value=format_inr(current['spend']),
            delta=f"{delta_val:+.1f}%" if delta_val else None,
            delta_color="off"
        )

    with col2:
        delta_val = deltas.get('conversions', {}).get('percentage', 0)
        st.metric(
            label="Conversions",
            value=format_number(current['conversions']),
            delta=f"{delta_val:+.1f}%" if delta_val else None,
            delta_color="normal"
        )

    with col3:
        delta_val = deltas.get('cpa', {}).get('percentage', 0)
        st.metric(
            label="CPA",
            value=format_inr(current['cpa']),
            delta=f"{delta_val:+.1f}%" if delta_val else None,
            delta_color="inverse"  # Lower CPA is better
        )

    with col4:
        delta_val = deltas.get('roas', {}).get('percentage', 0)
        st.metric(
            label="ROAS",
            value=format_ratio(current['roas']),
            delta=f"{delta_val:+.1f}%" if delta_val else None,
            delta_color="normal"
        )

    with col5:
        delta_val = deltas.get('ctr', {}).get('percentage', 0)
        st.metric(
            label="CTR",
            value=format_percentage(current['ctr']),
            delta=f"{delta_val:+.1f}%" if delta_val else None,
            delta_color="normal"
        )

    with col6:
        delta_val = deltas.get('cvr', {}).get('percentage', 0)
        st.metric(
            label="CVR",
            value=format_percentage(current['cvr']),
            delta=f"{delta_val:+.1f}%" if delta_val else None,
            delta_color="normal"
        )


def render_unified_view(analytics: AnalyticsEngine, filters: dict):
    """Render Unified Performance View section with rich visualizations"""
    st.subheader("📊 Platform & Channel Analysis")

    # Platform breakdown
    breakdown = analytics.get_breakdown_by_dimension(
        dimension='platform',
        start_date=filters['start_date'],
        end_date=filters['end_date'],
        platforms=filters['platforms'],
        geos=filters['geos'],
    )

    if len(breakdown) == 0:
        st.markdown(empty_state("No data available for selected filters"), unsafe_allow_html=True)
        return

    platform_data = breakdown[breakdown['platform'] != 'Total'].copy()

    # Row 1: Platform Performance Overview
    col1, col2 = st.columns([1.2, 1])

    with col1:
        # Spend Efficiency Bubble Chart
        st.markdown("**Spend vs ROAS by Platform** *(bubble size = conversions)*")
        fig = create_spend_efficiency_scatter(platform_data, title="")
        st.plotly_chart(fig, width="stretch")

    with col2:
        # Spend Distribution Donut
        st.markdown("**Spend Distribution**")
        fig = create_pie_chart(
            labels=platform_data['platform'].tolist(),
            values=platform_data['spend'].tolist(),
            title=""
        )
        st.plotly_chart(fig, width="stretch")

    # Row 2: Platform Metrics Comparison
    st.markdown("---")
    st.markdown("**Platform Performance Comparison**")

    col1, col2, col3 = st.columns(3)

    with col1:
        fig = create_platform_comparison_chart(platform_data, metric='spend', title='Spend by Platform')
        st.plotly_chart(fig, width="stretch")

    with col2:
        fig = create_platform_comparison_chart(platform_data, metric='cpa', title='CPA by Platform')
        st.plotly_chart(fig, width="stretch")

    with col3:
        fig = create_platform_comparison_chart(platform_data, metric='roas', title='ROAS by Platform')
        st.plotly_chart(fig, width="stretch")

    # Row 3: Detailed Tables in Tabs
    st.markdown("---")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📱 Platform Details", "📝 Ad Type Breakdown", "🎯 Campaign Breakdown", "📊 Objective Analysis", "🗺️ Geographic Analysis"])

    with tab1:
        display_df = platform_data.copy()
        display_df['Spend'] = display_df['spend'].apply(lambda x: format_inr(x, short=False))
        display_df['Revenue'] = display_df['revenue'].apply(lambda x: format_inr(x, short=False))
        display_df['Conversions'] = display_df['conversions'].apply(lambda x: f"{x:,.0f}")
        display_df['CPA'] = display_df['cpa'].apply(lambda x: format_inr(x))
        display_df['ROAS'] = display_df['roas'].apply(lambda x: f"{x:.2f}x")
        display_df['CTR'] = display_df['ctr'].apply(lambda x: f"{x:.2f}%")
        display_df['CVR'] = display_df['cvr'].apply(lambda x: f"{x:.2f}%")
        display_df['Share'] = display_df['spend_share'].apply(lambda x: f"{x:.1f}%")

        st.dataframe(
            display_df[['platform', 'Spend', 'Revenue', 'Conversions', 'CPA', 'ROAS', 'CTR', 'CVR', 'Share']].rename(
                columns={'platform': 'Platform'}
            ),
            width="stretch",
            hide_index=True,
        )

    with tab2:
        # Get raw filtered data to group by platform and ad_type
        filtered_df = analytics._filter_data(
            start_date=filters['start_date'],
            end_date=filters['end_date'],
            platforms=filters['platforms'],
            geos=filters['geos'],
        )

        if len(filtered_df) > 0:
            # Group by platform and ad_type
            adtype_data = filtered_df.groupby(['platform', 'ad_type']).agg({
                'spend': 'sum',
                'impressions': 'sum',
                'clicks': 'sum',
                'conversions': 'sum',
                'revenue': 'sum',
            }).reset_index()

            # Compute derived metrics
            adtype_data['cpa'] = adtype_data['spend'] / adtype_data['conversions'].replace(0, np.nan)
            adtype_data['roas'] = adtype_data['revenue'] / adtype_data['spend'].replace(0, np.nan)
            adtype_data['ctr'] = adtype_data['clicks'] / adtype_data['impressions'].replace(0, np.nan) * 100
            adtype_data['cvr'] = adtype_data['conversions'] / adtype_data['clicks'].replace(0, np.nan) * 100
            adtype_data = adtype_data.fillna(0)

            # Create combined label for display
            adtype_data['platform_adtype'] = adtype_data['platform'] + ' - ' + adtype_data['ad_type']

            # Sort by spend descending
            adtype_data = adtype_data.sort_values('spend', ascending=False)

            col1, col2 = st.columns([1, 1])

            with col1:
                # Ad type spend distribution with platform context
                fig = create_pie_chart(
                    labels=adtype_data['platform_adtype'].tolist()[:10],
                    values=adtype_data['spend'].tolist()[:10],
                    title="Spend by Platform - Ad Type"
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                display_df = adtype_data.copy()
                display_df['Spend'] = display_df['spend'].apply(lambda x: format_inr(x, short=False))
                display_df['Conversions'] = display_df['conversions'].apply(lambda x: f"{x:,.0f}")
                display_df['CPA'] = display_df['cpa'].apply(lambda x: format_inr(x))
                display_df['ROAS'] = display_df['roas'].apply(lambda x: f"{x:.2f}x")

                st.dataframe(
                    display_df[['platform', 'ad_type', 'Spend', 'Conversions', 'CPA', 'ROAS']].rename(
                        columns={'platform': 'Platform', 'ad_type': 'Ad Type'}
                    ).head(12),
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.info("No data available for the selected filters.")

    with tab3:
        # Campaign breakdown
        campaign_breakdown = analytics.get_breakdown_by_dimension(
            dimension='campaign_name',
            start_date=filters['start_date'],
            end_date=filters['end_date'],
            platforms=filters['platforms'],
            geos=filters['geos'],
        )

        campaign_data = campaign_breakdown[campaign_breakdown['campaign_name'] != 'Total'].copy()

        if len(campaign_data) > 0:
            col1, col2 = st.columns([1, 1])

            with col1:
                fig = create_pie_chart(
                    labels=campaign_data.head(10)['campaign_name'].tolist(),
                    values=campaign_data.head(10)['spend'].tolist(),
                    title="Spend by Campaign (Top 10)"
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                display_df = campaign_data.copy()
                display_df['Spend'] = display_df['spend'].apply(lambda x: format_inr(x, short=False))
                display_df['Conversions'] = display_df['conversions'].apply(lambda x: f"{x:,.0f}")
                display_df['CPA'] = display_df['cpa'].apply(lambda x: format_inr(x))
                display_df['ROAS'] = display_df['roas'].apply(lambda x: f"{x:.2f}x")
                display_df['Share'] = display_df['spend_share'].apply(lambda x: f"{x:.1f}%")

                st.dataframe(
                    display_df[['campaign_name', 'Spend', 'Conversions', 'CPA', 'ROAS', 'Share']].rename(
                        columns={'campaign_name': 'Campaign'}
                    ).head(15),
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.info("No campaign data available for selected filters.")

    with tab4:
        # Objective breakdown
        objective_breakdown = analytics.get_breakdown_by_dimension(
            dimension='objective',
            start_date=filters['start_date'],
            end_date=filters['end_date'],
            platforms=filters['platforms'],
            geos=filters['geos'],
        )

        objective_data = objective_breakdown[objective_breakdown['objective'] != 'Total'].copy()

        if len(objective_data) > 0:
            col1, col2 = st.columns([1, 1])

            with col1:
                fig = create_pie_chart(
                    labels=objective_data['objective'].tolist(),
                    values=objective_data['spend'].tolist(),
                    title="Spend by Objective"
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                display_df = objective_data.copy()
                display_df['Spend'] = display_df['spend'].apply(lambda x: format_inr(x, short=False))
                display_df['Conversions'] = display_df['conversions'].apply(lambda x: f"{x:,.0f}")
                display_df['CPA'] = display_df['cpa'].apply(lambda x: format_inr(x))
                display_df['ROAS'] = display_df['roas'].apply(lambda x: f"{x:.2f}x")
                display_df['CVR'] = display_df['cvr'].apply(lambda x: f"{x:.2f}%")
                display_df['Share'] = display_df['spend_share'].apply(lambda x: f"{x:.1f}%")

                st.dataframe(
                    display_df[['objective', 'Spend', 'Conversions', 'CPA', 'ROAS', 'CVR', 'Share']].rename(
                        columns={'objective': 'Objective'}
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.info("No objective data available for selected filters.")

    with tab5:
        geo_breakdown = analytics.get_breakdown_by_dimension(
            dimension='geo',
            start_date=filters['start_date'],
            end_date=filters['end_date'],
            platforms=filters['platforms'],
            geos=filters['geos'],
        )

        geo_data = geo_breakdown[geo_breakdown['geo'] != 'Total'].copy()

        col1, col2 = st.columns([1, 1])

        with col1:
            # Geo spend distribution
            fig = create_pie_chart(
                labels=geo_data['geo'].tolist(),
                values=geo_data['spend'].tolist(),
                title="Spend by Geography"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            display_df = geo_data.copy()
            display_df['Spend'] = display_df['spend'].apply(lambda x: format_inr(x, short=False))
            display_df['Conversions'] = display_df['conversions'].apply(lambda x: f"{x:,.0f}")
            display_df['CPA'] = display_df['cpa'].apply(lambda x: format_inr(x))
            display_df['Share'] = display_df['spend_share'].apply(lambda x: f"{x:.1f}%")

            st.dataframe(
                display_df[['geo', 'Spend', 'Conversions', 'CPA', 'Share']].rename(
                    columns={'geo': 'Geography'}
                ),
                use_container_width=True,
                hide_index=True,
            )


def render_conversion_funnel(analytics: AnalyticsEngine, filters: dict):
    """Render conversion metrics as horizontal cards"""
    summary = analytics.get_summary_metrics(
        start_date=filters['start_date'],
        end_date=filters['end_date'],
        platforms=filters['platforms'],
        geos=filters['geos'],
    )

    current = summary['current']

    impressions = current.get('impressions', 0)
    clicks = current.get('clicks', 0)
    conversions = current.get('conversions', 0)

    # Calculate rates
    ctr = (clicks / impressions * 100) if impressions > 0 else 0
    cvr = (conversions / clicks * 100) if clicks > 0 else 0

    # Horizontal layout with 3 columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Impressions", format_number(impressions))

    with col2:
        st.metric("Clicks", format_number(clicks), delta=f"{ctr:.2f}% CTR", delta_color="off")

    with col3:
        st.metric("Conversions", format_number(conversions), delta=f"{cvr:.2f}% CVR", delta_color="off")


def render_diagnostics(diagnostics: DiagnosticsEngine, filters: dict):
    """Render Performance Diagnostics section - Clean, clear design"""

    # Compute date ranges
    end_date = pd.to_datetime(filters['end_date'])
    start_date = pd.to_datetime(filters['start_date'])
    duration = (end_date - start_date).days + 1

    # Get the data's actual min date to check if "previous period" would exist
    data_min_date = diagnostics.df['date'].min()

    # Calculate ideal previous period
    previous_end = start_date - timedelta(days=1)
    previous_start = previous_end - timedelta(days=duration - 1)

    # Check if the previous period would have data
    use_split_comparison = previous_start < data_min_date

    if use_split_comparison:
        mid_point = start_date + timedelta(days=duration // 2)
        current_start_str = mid_point.strftime('%Y-%m-%d')
        current_end_str = end_date.strftime('%Y-%m-%d')
        previous_start_str = start_date.strftime('%Y-%m-%d')
        previous_end_str = (mid_point - timedelta(days=1)).strftime('%Y-%m-%d')
        comparison_label = "2nd half vs 1st half"
    else:
        current_start_str = filters['start_date']
        current_end_str = filters['end_date']
        previous_start_str = previous_start.strftime('%Y-%m-%d')
        previous_end_str = previous_end.strftime('%Y-%m-%d')
        comparison_label = "vs previous period"

    # Run analysis
    analysis = diagnostics.analyze_period_change(
        current_start=current_start_str,
        current_end=current_end_str,
        previous_start=previous_start_str,
        previous_end=previous_end_str,
        platforms=filters['platforms'],
        geos=filters['geos'],
    )

    if not analysis.get('cpa_decomposition') or analysis['cpa_decomposition'].get('primary_driver') == 'insufficient_data':
        st.info("Insufficient data for diagnostics. Try selecting a different date range or filters.")
        return

    decomp = analysis['cpa_decomposition']
    current_metrics = analysis.get('current_metrics', {})
    previous_metrics = analysis.get('previous_metrics', {})
    platform_attr = analysis.get('platform_attribution', [])

    cpa_change = decomp.get('total_change', 0)
    cpa_change_pct = decomp.get('total_change_pct', 0)
    is_improving = cpa_change < 0

    # ═══════════════════════════════════════════════════════════════
    # SECTION 1: THE HEADLINE - What happened?
    # ═══════════════════════════════════════════════════════════════

    st.markdown("### What Happened to Performance?")
    st.caption(f"Comparing {current_start_str} to {current_end_str} ({comparison_label})")

    # Big headline metric card
    if abs(cpa_change_pct) > 0.5:
        if is_improving:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(16, 185, 129, 0.05) 100%);
                        border-left: 4px solid #10B981; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                <div style="font-size: 14px; color: #10B981; font-weight: 600; margin-bottom: 8px;">PERFORMANCE IMPROVED</div>
                <div style="font-size: 32px; font-weight: 700; color: #F9FAFB;">CPA decreased by {abs(cpa_change_pct):.1f}%</div>
                <div style="font-size: 16px; color: #9CA3AF; margin-top: 8px;">{format_inr(previous_metrics.get('cpa', 0))} → {format_inr(current_metrics.get('cpa', 0))} (saving {format_inr(abs(cpa_change))} per conversion)</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(239, 68, 68, 0.05) 100%);
                        border-left: 4px solid #EF4444; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                <div style="font-size: 14px; color: #EF4444; font-weight: 600; margin-bottom: 8px;">PERFORMANCE DECLINED</div>
                <div style="font-size: 32px; font-weight: 700; color: #F9FAFB;">CPA increased by {abs(cpa_change_pct):.1f}%</div>
                <div style="font-size: 16px; color: #9CA3AF; margin-top: 8px;">{format_inr(previous_metrics.get('cpa', 0))} → {format_inr(current_metrics.get('cpa', 0))} (costing {format_inr(abs(cpa_change))} more per conversion)</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(59, 130, 246, 0.05) 100%);
                    border-left: 4px solid #3B82F6; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
            <div style="font-size: 14px; color: #3B82F6; font-weight: 600; margin-bottom: 8px;">PERFORMANCE STABLE</div>
            <div style="font-size: 32px; font-weight: 700; color: #F9FAFB;">CPA remained steady</div>
            <div style="font-size: 16px; color: #9CA3AF; margin-top: 8px;">Current CPA: {format_inr(current_metrics.get('cpa', 0))}</div>
        </div>
        """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════
    # SECTION 2: KEY METRICS COMPARISON - Before vs After
    # ═══════════════════════════════════════════════════════════════

    st.markdown("### Key Metrics: Before vs After")

    metric_cols = st.columns(4)

    metrics_to_show = [
        ('spend', 'Spend', 'currency', None),
        ('conversions', 'Conversions', 'number', True),
        ('cpa', 'CPA', 'currency', False),
        ('roas', 'ROAS', 'ratio', True),
    ]

    for i, (key, label, fmt, higher_is_better) in enumerate(metrics_to_show):
        curr_val = current_metrics.get(key, 0)
        prev_val = previous_metrics.get(key, 0)
        change_pct = ((curr_val - prev_val) / prev_val * 100) if prev_val != 0 else 0

        # Determine if change is good or bad
        if higher_is_better is None:
            delta_color = "off"
        elif higher_is_better:
            delta_color = "normal"  # Green if up, red if down
        else:
            delta_color = "inverse"  # Red if up, green if down

        # Format value
        if fmt == 'currency':
            display_val = format_inr(curr_val)
            delta_str = f"{change_pct:+.1f}%"
        elif fmt == 'ratio':
            display_val = f"{curr_val:.2f}x"
            delta_str = f"{change_pct:+.1f}%"
        else:
            display_val = f"{curr_val:,.0f}"
            delta_str = f"{change_pct:+.1f}%"

        with metric_cols[i]:
            st.metric(label=label, value=display_val, delta=delta_str, delta_color=delta_color)

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════
    # SECTION 3: ROOT CAUSE - Why did this happen?
    # ═══════════════════════════════════════════════════════════════

    st.markdown("### Why Did This Happen?")
    st.caption("CPA = Cost Per Click (CPC) ÷ Conversion Rate (CVR). Changes in either affect your CPA.")

    col_why1, col_why2 = st.columns(2)

    # CPC Analysis
    cpc_change_pct = decomp.get('cpc_change_pct', 0)
    cpc_contribution_pct = decomp.get('cpc_contribution_pct', 0)
    cpc_curr = decomp.get('current_cpc', 0)
    cpc_prev = decomp.get('previous_cpc', 0)

    # CVR Analysis
    cvr_change_pct = decomp.get('cvr_change_pct', 0)
    cvr_contribution_pct = decomp.get('cvr_contribution_pct', 0)
    cvr_curr = decomp.get('current_cvr', 0)
    cvr_prev = decomp.get('previous_cvr', 0)

    # Determine primary driver
    primary_driver = decomp.get('primary_driver', 'both')

    with col_why1:
        # CPC Card
        cpc_is_bad = (cpc_change_pct > 0 and cpa_change > 0) or (cpc_change_pct < 0 and cpa_change < 0)
        cpc_color = "#EF4444" if cpc_is_bad else "#10B981"
        cpc_label = "COST DRIVER" if primary_driver == 'cpc' else "COST FACTOR"

        st.markdown(f"""
        <div style="background: #1A1D29; border: 1px solid #374151; border-radius: 8px; padding: 16px; height: 180px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <span style="font-size: 12px; color: #9CA3AF; font-weight: 600;">{cpc_label}</span>
                <span style="font-size: 12px; color: {cpc_color}; background: rgba({('239, 68, 68' if cpc_is_bad else '16, 185, 129')}, 0.2);
                       padding: 2px 8px; border-radius: 4px;">{cpc_contribution_pct:.0f}% of change</span>
            </div>
            <div style="font-size: 18px; color: #F9FAFB; font-weight: 600; margin-bottom: 8px;">Cost Per Click (CPC)</div>
            <div style="font-size: 28px; color: {cpc_color}; font-weight: 700;">{cpc_change_pct:+.1f}%</div>
            <div style="font-size: 13px; color: #9CA3AF; margin-top: 8px;">{format_inr(cpc_prev)} → {format_inr(cpc_curr)}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_why2:
        # CVR Card
        # For CVR: improvement (positive change) is good, decline (negative) is bad
        cvr_is_bad = (cvr_change_pct < 0 and cpa_change > 0) or (cvr_change_pct > 0 and cpa_change < 0)
        cvr_color = "#EF4444" if cvr_is_bad else "#10B981"
        cvr_label = "CONVERSION DRIVER" if primary_driver == 'cvr' else "CONVERSION FACTOR"

        st.markdown(f"""
        <div style="background: #1A1D29; border: 1px solid #374151; border-radius: 8px; padding: 16px; height: 180px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <span style="font-size: 12px; color: #9CA3AF; font-weight: 600;">{cvr_label}</span>
                <span style="font-size: 12px; color: {cvr_color}; background: rgba({('239, 68, 68' if cvr_is_bad else '16, 185, 129')}, 0.2);
                       padding: 2px 8px; border-radius: 4px;">{cvr_contribution_pct:.0f}% of change</span>
            </div>
            <div style="font-size: 18px; color: #F9FAFB; font-weight: 600; margin-bottom: 8px;">Conversion Rate (CVR)</div>
            <div style="font-size: 28px; color: {cvr_color}; font-weight: 700;">{cvr_change_pct:+.1f}%</div>
            <div style="font-size: 13px; color: #9CA3AF; margin-top: 8px;">{cvr_prev:.2f}% → {cvr_curr:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    # Insight explanation
    st.markdown("")
    if primary_driver == 'cpc':
        if cpc_change_pct > 0:
            st.info(f"**Primary Driver: Rising Click Costs** — Your CPC increased by {abs(cpc_change_pct):.1f}%, accounting for {cpc_contribution_pct:.0f}% of the CPA increase. This could be due to increased competition, higher bids, or changes in auction dynamics.")
        else:
            st.success(f"**Primary Driver: Lower Click Costs** — Your CPC decreased by {abs(cpc_change_pct):.1f}%, accounting for {cpc_contribution_pct:.0f}% of the CPA improvement. Your bidding strategy or reduced competition is paying off.")
    elif primary_driver == 'cvr':
        if cvr_change_pct < 0:
            st.info(f"**Primary Driver: Declining Conversions** — Your CVR dropped by {abs(cvr_change_pct):.1f}%, accounting for {cvr_contribution_pct:.0f}% of the CPA increase. Check landing pages, ad relevance, or audience targeting.")
        else:
            st.success(f"**Primary Driver: Better Conversions** — Your CVR improved by {abs(cvr_change_pct):.1f}%, accounting for {cvr_contribution_pct:.0f}% of the CPA improvement. Your targeting or landing page optimizations are working.")
    else:
        st.info(f"**Both Factors Contributing** — CPC changed by {cpc_change_pct:+.1f}% and CVR changed by {cvr_change_pct:+.1f}%. Both are affecting your performance roughly equally.")

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════
    # SECTION 4: PLATFORM BREAKDOWN - Which platform is responsible?
    # ═══════════════════════════════════════════════════════════════

    if platform_attr and len(platform_attr) > 0:
        st.markdown("### Which Platform is Responsible?")
        st.caption("Platforms ranked by their contribution to the overall CPA change")

        # Platform cards
        platform_cols = st.columns(len(platform_attr[:4]))  # Show top 4

        platform_colors = {
            'Google Ads': '#4285F4',
            'Meta': '#1877F2',
            'DV360': '#34A853',
            'Amazon Ads': '#FF9900',
        }

        for i, plat in enumerate(platform_attr[:4]):
            with platform_cols[i]:
                plat_name = plat['platform']
                plat_color = platform_colors.get(plat_name, '#6366F1')
                plat_cpa_change = plat.get('cpa_change_pct', 0)
                plat_contrib = plat.get('contribution_pct', 0)
                plat_spend_share = plat.get('spend_share', 0)

                # Determine if this platform is hurting or helping
                is_hurting = (plat_cpa_change > 0 and cpa_change > 0) or (plat_cpa_change < 0 and cpa_change < 0)
                status_color = "#EF4444" if is_hurting and abs(plat_contrib) > 20 else "#10B981" if not is_hurting and abs(plat_contrib) > 20 else "#9CA3AF"

                st.markdown(f"""
                <div style="background: #1A1D29; border-radius: 8px; padding: 16px; border-left: 4px solid {plat_color};">
                    <div style="font-size: 14px; color: #F9FAFB; font-weight: 600; margin-bottom: 12px;">{plat_name}</div>
                    <div style="font-size: 24px; color: {status_color}; font-weight: 700;">{plat_cpa_change:+.1f}%</div>
                    <div style="font-size: 12px; color: #9CA3AF; margin-top: 4px;">CPA change</div>
                    <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #374151;">
                        <div style="font-size: 11px; color: #6B7280;">Contribution: {abs(plat_contrib):.0f}%</div>
                        <div style="font-size: 11px; color: #6B7280;">Spend Share: {plat_spend_share:.0f}%</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Find the biggest contributor and explain
        top_platform = platform_attr[0]
        if abs(top_platform.get('contribution_pct', 0)) > 25:
            st.markdown("")
            top_name = top_platform['platform']
            top_contrib = abs(top_platform['contribution_pct'])
            top_cpa_change = top_platform['cpa_change_pct']
            top_spend = top_platform['spend_share']

            if top_cpa_change > 0:
                st.warning(f"**{top_name}** is responsible for {top_contrib:.0f}% of your CPA increase. With {top_spend:.0f}% of spend, its CPA rose {abs(top_cpa_change):.1f}%. Consider reviewing campaign settings, audiences, or creative performance on this platform.")
            else:
                st.success(f"**{top_name}** drove {top_contrib:.0f}% of your CPA improvement. With {top_spend:.0f}% of spend, its CPA dropped {abs(top_cpa_change):.1f}%. This platform is performing well.")


def render_ask_platform(query_router: QueryRouter, filters: dict):
    """Render Ask AI section - Conversational chat interface with history"""

    # Initialize session state for conversation history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []  # List of {question, answer, intent, analysis_summary}
    if 'pending_question' not in st.session_state:
        st.session_state.pending_question = None

    # ═══════════════════════════════════════════════════════════════
    # HEADER
    # ═══════════════════════════════════════════════════════════════

    st.markdown("### 🤖 AI Analytics Assistant")
    st.caption("Have a conversation about your campaign performance. Ask follow-up questions to dive deeper.")

    # ═══════════════════════════════════════════════════════════════
    # QUICK QUESTIONS - Collapsed by default when there's history
    # ═══════════════════════════════════════════════════════════════

    question_categories = {
        "🔍 Diagnostics": {
            "color": "#EF4444",
            "bg": "rgba(239, 68, 68, 0.12)",
            "questions": [
                "Why did CPA change this week?",
                "What's driving performance changes?",
                "Why is conversion rate dropping?",
            ],
        },
        "📊 Comparison": {
            "color": "#3B82F6",
            "bg": "rgba(59, 130, 246, 0.12)",
            "questions": [
                "Which platform has the best ROAS?",
                "Compare Google Ads vs Meta",
                "Which geo is performing best?",
            ],
        },
        "📈 Forecasting": {
            "color": "#F59E0B",
            "bg": "rgba(245, 158, 11, 0.12)",
            "questions": [
                "What's the conversion trend?",
                "Predict next week's performance",
                "How will CPA trend next week?",
            ],
        },
        "💡 Recommendations": {
            "color": "#10B981",
            "bg": "rgba(16, 185, 129, 0.12)",
            "questions": [
                "Should I reallocate budget?",
                "Where should I increase spend?",
                "How can I improve ROAS?",
            ],
        },
    }

    # Show quick questions only when starting fresh or collapsed
    has_history = len(st.session_state.chat_history) > 0
    with st.expander("💬 Quick Start Questions", expanded=not has_history):
        q_tabs = st.tabs(list(question_categories.keys()))
        for tab_idx, (tab, (category, cat_data)) in enumerate(zip(q_tabs, question_categories.items())):
            with tab:
                cols = st.columns(3)
                for i, q in enumerate(cat_data["questions"]):
                    with cols[i % 3]:
                        if st.button(q, key=f"quick_{tab_idx}_{i}", use_container_width=True):
                            st.session_state.pending_question = q
                            st.rerun()

    # ═══════════════════════════════════════════════════════════════
    # CONVERSATION HISTORY DISPLAY
    # ═══════════════════════════════════════════════════════════════

    if has_history:
        # Clear conversation button
        col1, col2, col3 = st.columns([1, 3, 1])
        with col3:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.pending_question = None
                st.rerun()

        st.markdown("---")

        # Display conversation history
        for idx, turn in enumerate(st.session_state.chat_history):
            # User question
            st.markdown(f"""
            <div style="display: flex; justify-content: flex-end; margin-bottom: 10px;">
                <div style="background: #3B82F6; color: white; padding: 12px 16px; border-radius: 16px 16px 4px 16px; max-width: 80%; font-size: 14px;">
                    {turn['question']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # AI response with intent badge
            intent = turn.get('intent', 'general')
            intent_colors = {
                'diagnostic': '#EF4444', 'comparison': '#3B82F6', 'forecast': '#F59E0B',
                'scenario': '#8B5CF6', 'recommendation': '#10B981', 'lookup': '#6B7280', 'general': '#6366F1',
            }
            intent_labels = {
                'diagnostic': '🔍 Diagnostic', 'comparison': '📊 Comparison', 'forecast': '📈 Forecast',
                'scenario': '🎯 Scenario', 'recommendation': '💡 Recommendation', 'lookup': '📋 Lookup', 'general': '💬 Response',
            }

            # Format the response with markdown support
            response_html = turn['answer'].replace('\n', '<br>').replace('**', '<strong>').replace('*', '<em>')

            st.markdown(f"""
            <div style="display: flex; justify-content: flex-start; margin-bottom: 20px;">
                <div style="background: #1A1D29; border: 1px solid #374151; padding: 16px 20px; border-radius: 4px 16px 16px 16px; max-width: 95%;">
                    <span style="background: {intent_colors.get(intent, '#6366F1')}22; color: {intent_colors.get(intent, '#6366F1')};
                                 padding: 2px 8px; border-radius: 8px; font-size: 11px; font-weight: 600; margin-bottom: 8px; display: inline-block;">
                        {intent_labels.get(intent, 'Response')}
                    </span>
                    <div style="font-size: 14px; color: #E5E7EB; line-height: 1.8; margin-top: 8px; white-space: pre-wrap;">{turn['answer']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Show visualization for forecasts
            if intent == 'forecast' and turn.get('analysis', {}).get('projection'):
                proj = turn['analysis']
                if proj.get('historical', {}).get('dates'):
                    with st.container():
                        hist_df = pd.DataFrame({
                            'period': proj['historical']['dates'],
                            proj['metric']: proj['historical']['values']
                        })
                        projection_data = {
                            'dates': proj['projection']['dates'],
                            'values': proj['projection']['values'],
                            'lower_bound': proj['projection'].get('lower_bound', []),
                            'upper_bound': proj['projection'].get('upper_bound', []),
                        }
                        fig = create_trend_chart(
                            hist_df, x_col='period', y_col=proj['metric'],
                            title=f"{proj['metric'].upper()} Trend & Projection",
                            show_projection=True, projection_data=projection_data
                        )
                        st.plotly_chart(fig, use_container_width=True, key=f"chart_{idx}")

            # Show comparison table
            elif intent == 'comparison' and turn.get('analysis', {}).get('breakdown'):
                breakdown = turn['analysis']['breakdown']
                if len(breakdown) > 0:
                    with st.expander("📊 View Detailed Data", expanded=False):
                        df = pd.DataFrame(breakdown)
                        if 'spend' in df.columns:
                            df['Spend'] = df['spend'].apply(lambda x: format_inr(x))
                        if 'cpa' in df.columns:
                            df['CPA'] = df['cpa'].apply(lambda x: format_inr(x))
                        if 'roas' in df.columns:
                            df['ROAS'] = df['roas'].apply(lambda x: f"{x:.2f}x")
                        st.dataframe(df, use_container_width=True, hide_index=True)

    # ═══════════════════════════════════════════════════════════════
    # PROCESS PENDING QUESTION
    # ═══════════════════════════════════════════════════════════════

    if st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = None

        # Show the question being processed
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-end; margin-bottom: 10px;">
            <div style="background: #3B82F6; color: white; padding: 12px 16px; border-radius: 16px 16px 4px 16px; max-width: 80%; font-size: 14px;">
                {question}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Process with conversation history context
        with st.spinner("🔍 Analyzing your data and considering conversation context..."):
            try:
                # Build conversation history for context
                history_for_llm = [
                    {
                        'question': turn['question'],
                        'answer_summary': turn['answer'][:500] if len(turn['answer']) > 500 else turn['answer'],
                    }
                    for turn in st.session_state.chat_history[-5:]  # Last 5 turns
                ]

                result = query_router.process_query(
                    user_question=question,
                    filters=filters,
                    conversation_history=history_for_llm if history_for_llm else None
                )

                # Add to conversation history
                st.session_state.chat_history.append({
                    'question': question,
                    'answer': result['explanation'],
                    'intent': result['intent'],
                    'analysis': result.get('analysis', {}),
                })

                st.rerun()

            except Exception as e:
                st.error(f"Error processing question: {str(e)}")

    # ═══════════════════════════════════════════════════════════════
    # SUGGESTED FOLLOW-UPS (contextual based on last response)
    # ═══════════════════════════════════════════════════════════════

    if has_history:
        last_intent = st.session_state.chat_history[-1].get('intent', 'general')

        # Contextual follow-up suggestions based on last question type
        follow_up_suggestions = {
            'diagnostic': [
                "What specific actions should I take?",
                "Which platform should I focus on first?",
                "How does this compare to last month?",
            ],
            'comparison': [
                "Why is the winner performing better?",
                "Should I shift budget to the best performer?",
                "What's the trend for each platform?",
            ],
            'forecast': [
                "What could change this projection?",
                "How confident is this forecast?",
                "What should I do to improve the outlook?",
            ],
            'recommendation': [
                "What's the risk of this recommendation?",
                "How much budget should I shift?",
                "What results should I expect?",
            ],
            'lookup': [
                "How does this compare to last week?",
                "Break this down by platform",
                "What's driving this number?",
            ],
            'general': [
                "Tell me more about this",
                "What should I do next?",
                "What are the key takeaways?",
            ],
        }

        suggestions = follow_up_suggestions.get(last_intent, follow_up_suggestions['general'])

        st.markdown("---")
        st.markdown("**💡 Suggested follow-ups:**")
        cols = st.columns(3)
        for idx, suggestion in enumerate(suggestions):
            with cols[idx]:
                if st.button(suggestion, key=f"followup_{idx}", use_container_width=True):
                    st.session_state.pending_question = suggestion
                    st.rerun()

    # ═══════════════════════════════════════════════════════════════
    # CHAT INPUT
    # ═══════════════════════════════════════════════════════════════

    st.markdown("---")

    # Use chat_input for a more natural chat experience
    user_input = st.chat_input(
        placeholder="Ask a question about your campaign performance...",
        key="chat_input"
    )

    if user_input:
        st.session_state.pending_question = user_input
        st.rerun()



def render_trends(analytics: AnalyticsEngine, filters: dict):
    """Render Trends section with enhanced visualizations"""
    st.markdown("### 📈 Performance Trends")

    # Metric and view type selectors in columns
    col1, col2 = st.columns([1, 1])

    with col1:
        metric = st.selectbox(
            "Select Metric",
            options=['conversions', 'spend', 'cpa', 'roas', 'ctr', 'cvr'],
            format_func=lambda x: x.upper(),
            index=0,
            key="trend_metric"
        )

    with col2:
        view_type = st.selectbox(
            "View Type",
            options=['Overall Trend', 'By Platform', 'Stacked by Platform'],
            index=0,
            key="trend_view"
        )

    # Get time series based on view type
    if view_type == 'Overall Trend':
        ts = analytics.get_time_series(
            metric=metric,
            granularity='daily',
            start_date=filters['start_date'],
            end_date=filters['end_date'],
            platforms=filters['platforms'],
            geos=filters['geos'],
        )

        if len(ts) > 0:
            fig = create_trend_chart(
                ts,
                x_col='period',
                y_col=metric,
                title=f"{metric.upper()} Over Time"
            )
            st.plotly_chart(fig, width="stretch")

    elif view_type == 'By Platform':
        ts_by_platform = analytics.get_time_series(
            metric=metric,
            granularity='daily',
            start_date=filters['start_date'],
            end_date=filters['end_date'],
            platforms=filters['platforms'],
            geos=filters['geos'],
            breakdown='platform'
        )

        if len(ts_by_platform) > 0:
            fig = create_trend_chart(
                ts_by_platform,
                x_col='period',
                y_col=metric,
                title=f"{metric.upper()} by Platform",
                color_by='platform'
            )
            st.plotly_chart(fig, width="stretch")

    else:  # Stacked by Platform
        ts_by_platform = analytics.get_time_series(
            metric=metric,
            granularity='daily',
            start_date=filters['start_date'],
            end_date=filters['end_date'],
            platforms=filters['platforms'],
            geos=filters['geos'],
            breakdown='platform'
        )

        if len(ts_by_platform) > 0:
            fig = create_stacked_area_chart(
                ts_by_platform,
                x_col='period',
                y_col=metric,
                stack_col='platform',
                title=f"{metric.upper()} Stacked by Platform"
            )
            st.plotly_chart(fig, width="stretch")


def render_campaign_explorer(analytics: AnalyticsEngine, filters: dict):
    """Render Campaign Explorer tab with hierarchical drill-down"""
    st.subheader("📊 Campaign Explorer")
    st.caption("Explore campaign hierarchy: Platform → Campaign → Objective → Ad Type → Adset → Ad → Creative")

    # Initialize hierarchy navigation state
    if 'explorer_level' not in st.session_state:
        st.session_state.explorer_level = 'campaign_name'
    if 'explorer_filters' not in st.session_state:
        st.session_state.explorer_filters = {}

    # Get filtered data
    filtered_df = analytics._filter_data(
        start_date=filters['start_date'],
        end_date=filters['end_date'],
        platforms=filters['platforms'],
        geos=filters['geos'],
    )

    if len(filtered_df) == 0:
        st.info("No data available for selected filters.")
        return

    # Hierarchy level selector
    hierarchy_options = [
        ('campaign_name', 'Campaign'),
        ('objective', 'Objective'),
        ('ad_type', 'Ad Type'),
        ('adset_name', 'Adset (Targeting)'),
        ('ad_name', 'Ad'),
        ('creative_type', 'Creative Type'),
    ]

    col1, col2 = st.columns([1, 3])

    with col1:
        selected_level = st.selectbox(
            "View By",
            options=[h[0] for h in hierarchy_options],
            format_func=lambda x: dict(hierarchy_options).get(x, x),
            key="explorer_view_by"
        )

    # Additional filters based on hierarchy
    with col2:
        filter_cols = st.columns(4)

        with filter_cols[0]:
            campaigns = sorted(filtered_df['campaign_name'].unique().tolist())
            selected_campaign = st.selectbox(
                "Campaign",
                options=["All"] + campaigns,
                key="explorer_campaign"
            )

        with filter_cols[1]:
            objectives = sorted(filtered_df['objective'].unique().tolist())
            selected_objective = st.selectbox(
                "Objective",
                options=["All"] + objectives,
                key="explorer_objective"
            )

        with filter_cols[2]:
            ad_types = sorted(filtered_df['ad_type'].unique().tolist())
            selected_ad_type = st.selectbox(
                "Ad Type",
                options=["All"] + ad_types,
                key="explorer_ad_type"
            )

        with filter_cols[3]:
            creative_types = sorted(filtered_df['creative_type'].unique().tolist())
            selected_creative = st.selectbox(
                "Creative Type",
                options=["All"] + creative_types,
                key="explorer_creative"
            )

    # Apply additional filters
    explorer_filters = {}
    if selected_campaign != "All":
        explorer_filters['campaign_name'] = [selected_campaign]
    if selected_objective != "All":
        explorer_filters['objective'] = [selected_objective]
    if selected_ad_type != "All":
        explorer_filters['ad_type'] = [selected_ad_type]
    if selected_creative != "All":
        explorer_filters['creative_type'] = [selected_creative]

    # Get breakdown data
    breakdown = analytics.get_hierarchical_breakdown(
        levels=[selected_level],
        start_date=filters['start_date'],
        end_date=filters['end_date'],
        platforms=filters['platforms'],
        geos=filters['geos'],
        filters=explorer_filters,
        include_totals=True
    )

    if len(breakdown) == 0:
        st.info("No data available for selected filters.")
        return

    # Summary metrics
    st.markdown("---")
    total_row = breakdown[breakdown[selected_level] == 'Total']
    data_rows = breakdown[breakdown[selected_level] != 'Total']

    if len(total_row) > 0:
        total = total_row.iloc[0]
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Total Spend", format_inr(total['spend']))
        with col2:
            st.metric("Conversions", format_number(total['conversions']))
        with col3:
            st.metric("CPA", format_inr(total['cpa']))
        with col4:
            st.metric("ROAS", format_ratio(total['roas']))
        with col5:
            st.metric(f"Unique {dict(hierarchy_options).get(selected_level, selected_level)}s", len(data_rows))

    st.markdown("---")

    # Visualizations
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown(f"**{dict(hierarchy_options).get(selected_level, selected_level)} Performance**")
        if len(data_rows) > 0:
            fig = create_bar_chart(
                data_rows.head(15),
                x_col=selected_level,
                y_col='spend',
                title=f"Spend by {dict(hierarchy_options).get(selected_level, selected_level)}"
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Spend Distribution**")
        if len(data_rows) > 0:
            fig = create_pie_chart(
                labels=data_rows.head(10)[selected_level].tolist(),
                values=data_rows.head(10)['spend'].tolist(),
                title=""
            )
            st.plotly_chart(fig, use_container_width=True)

    # Detailed table
    st.markdown("---")
    st.markdown(f"**Detailed {dict(hierarchy_options).get(selected_level, selected_level)} Performance**")

    display_df = data_rows.copy()
    display_df['Spend'] = display_df['spend'].apply(lambda x: format_inr(x, short=False))
    display_df['Revenue'] = display_df['revenue'].apply(lambda x: format_inr(x, short=False))
    display_df['Conversions'] = display_df['conversions'].apply(lambda x: f"{x:,.0f}")
    display_df['CPA'] = display_df['cpa'].apply(lambda x: format_inr(x))
    display_df['ROAS'] = display_df['roas'].apply(lambda x: f"{x:.2f}x")
    display_df['CTR'] = display_df['ctr'].apply(lambda x: f"{x:.2f}%")
    display_df['CVR'] = display_df['cvr'].apply(lambda x: f"{x:.2f}%")
    display_df['Spend %'] = display_df['spend_share'].apply(lambda x: f"{x:.1f}%")

    column_name = dict(hierarchy_options).get(selected_level, selected_level)
    display_cols = [selected_level, 'Spend', 'Revenue', 'Conversions', 'CPA', 'ROAS', 'CTR', 'CVR', 'Spend %']

    st.dataframe(
        display_df[display_cols].rename(columns={selected_level: column_name}),
        use_container_width=True,
        hide_index=True,
    )


def render_creative_analysis(analytics: AnalyticsEngine, filters: dict):
    """Render Creative Analysis tab"""
    st.subheader("🎨 Creative Performance Analysis")
    st.caption("Analyze performance by creative type and individual creatives")

    # Get creative analysis data
    creative_data = analytics.get_creative_analysis(
        start_date=filters['start_date'],
        end_date=filters['end_date'],
        platforms=filters['platforms'],
        geos=filters['geos'],
    )

    by_type = creative_data['by_type']
    by_creative = creative_data['by_creative']
    top_performers = creative_data['top_performers']

    if len(by_type) == 0:
        st.info("No creative data available for selected filters.")
        return

    # Summary row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Creatives", len(by_creative))
    with col2:
        st.metric("Creative Types", len(by_type))
    with col3:
        best_type = by_type.loc[by_type['roas'].idxmax()] if len(by_type) > 0 else None
        st.metric("Best ROAS Type", f"{best_type['creative_type']}" if best_type is not None else "N/A")
    with col4:
        total_spend = by_type['spend'].sum()
        st.metric("Total Creative Spend", format_inr(total_spend))

    st.markdown("---")

    # Row 1: Creative Type Analysis
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("**Performance by Creative Type**")
        if len(by_type) > 0:
            fig = create_bar_chart(
                by_type,
                x_col='creative_type',
                y_col='roas',
                title="ROAS by Creative Type"
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Spend Distribution by Type**")
        if len(by_type) > 0:
            fig = create_pie_chart(
                labels=by_type['creative_type'].tolist(),
                values=by_type['spend'].tolist(),
                title=""
            )
            st.plotly_chart(fig, use_container_width=True)

    # Creative Type Table
    st.markdown("---")
    st.markdown("**Creative Type Breakdown**")

    display_type = by_type.copy()
    display_type['Spend'] = display_type['spend'].apply(lambda x: format_inr(x, short=False))
    display_type['Conversions'] = display_type['conversions'].apply(lambda x: f"{x:,.0f}")
    display_type['CPA'] = display_type['cpa'].apply(lambda x: format_inr(x))
    display_type['ROAS'] = display_type['roas'].apply(lambda x: f"{x:.2f}x")
    display_type['CTR'] = display_type['ctr'].apply(lambda x: f"{x:.2f}%")
    display_type['CVR'] = display_type['cvr'].apply(lambda x: f"{x:.2f}%")
    display_type['Spend %'] = display_type['spend_share'].apply(lambda x: f"{x:.1f}%")

    st.dataframe(
        display_type[['creative_type', 'Spend', 'Conversions', 'CPA', 'ROAS', 'CTR', 'CVR', 'Spend %']].rename(
            columns={'creative_type': 'Creative Type'}
        ),
        use_container_width=True,
        hide_index=True,
    )

    # Top Performers
    st.markdown("---")
    st.markdown("**🏆 Top Performing Creatives** *(by ROAS, minimum spend threshold)*")

    if len(top_performers) > 0:
        display_top = top_performers.copy()
        display_top['Spend'] = display_top['spend'].apply(lambda x: format_inr(x, short=False))
        display_top['Conversions'] = display_top['conversions'].apply(lambda x: f"{x:,.0f}")
        display_top['CPA'] = display_top['cpa'].apply(lambda x: format_inr(x))
        display_top['ROAS'] = display_top['roas'].apply(lambda x: f"{x:.2f}x")
        display_top['CTR'] = display_top['ctr'].apply(lambda x: f"{x:.2f}%")

        st.dataframe(
            display_top[['creative_name', 'creative_type', 'platform', 'Spend', 'Conversions', 'CPA', 'ROAS', 'CTR']].rename(
                columns={
                    'creative_name': 'Creative',
                    'creative_type': 'Type',
                    'platform': 'Platform'
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No top performers data available.")

    # All Creatives (expandable)
    with st.expander("View All Creatives"):
        if len(by_creative) > 0:
            display_all = by_creative.head(50).copy()
            display_all['Spend'] = display_all['spend'].apply(lambda x: format_inr(x, short=False))
            display_all['Conversions'] = display_all['conversions'].apply(lambda x: f"{x:,.0f}")
            display_all['CPA'] = display_all['cpa'].apply(lambda x: format_inr(x))
            display_all['ROAS'] = display_all['roas'].apply(lambda x: f"{x:.2f}x")

            st.dataframe(
                display_all[['creative_id', 'creative_name', 'creative_type', 'platform', 'Spend', 'Conversions', 'CPA', 'ROAS']].rename(
                    columns={
                        'creative_id': 'ID',
                        'creative_name': 'Creative',
                        'creative_type': 'Type',
                        'platform': 'Platform'
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )


def main():
    """Main application entry point"""
    # Initialize
    init_session_state()

    # Apply custom CSS
    st.markdown(apply_custom_css(), unsafe_allow_html=True)

    # Load data
    df = load_app_data()

    # Render control panel (Industry + Client + Date + Platform/Geo filters)
    display_name, filtered_df, industry, filters = render_control_panel(df)

    # Render data context bar (shows what data is being displayed - appears above content)
    render_data_context(display_name, industry, filters)

    # Initialize engines with FILTERED data
    analytics = AnalyticsEngine(filtered_df)
    diagnostics = DiagnosticsEngine(filtered_df)
    forecasting = ForecastingEngine(filtered_df)

    # Check for API key - try multiple sources
    api_key = os.getenv('ANTHROPIC_API_KEY')

    # Also try Streamlit secrets if env var not found
    if not api_key:
        try:
            api_key = st.secrets.get('ANTHROPIC_API_KEY')
        except (FileNotFoundError, KeyError, AttributeError):
            pass

    use_mock = not api_key or api_key == 'your_api_key_here'

    # Only show the toast once per session to avoid repeated notifications
    if use_mock and 'api_warning_shown' not in st.session_state:
        st.session_state.api_warning_shown = True
        st.toast("Running in demo mode. Set ANTHROPIC_API_KEY for AI features.", icon="ℹ️")

    query_router = QueryRouter(
        df=filtered_df,
        llm_api_key=api_key,
        use_mock_llm=use_mock
    )

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Executive Summary",
        "📊 Platform Analysis",
        "🎯 Campaign Explorer",
        "🎨 Creative Analysis",
        "🔍 Diagnostics",
        "💬 Ask AI"
    ])

    with tab1:
        render_executive_summary(analytics, diagnostics, filters)
        st.markdown("---")

        # Trends - full width
        render_trends(analytics, filters)

        st.markdown("---")

        # Conversion Metrics - full width, horizontal layout
        st.markdown("### 🎯 Conversion Metrics")
        render_conversion_funnel(analytics, filters)

    with tab2:
        render_unified_view(analytics, filters)

    with tab3:
        render_campaign_explorer(analytics, filters)

    with tab4:
        render_creative_analysis(analytics, filters)

    with tab5:
        render_diagnostics(diagnostics, filters)

    with tab6:
        render_ask_platform(query_router, filters)


if __name__ == "__main__":
    main()
