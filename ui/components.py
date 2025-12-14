"""
Reusable UI components for FE Media Intelligence Platform
"""

from typing import Optional, Dict, Any
from config import COLORS, METRICS


def metric_card(
    label: str,
    value: str,
    delta: Optional[Dict[str, Any]] = None,
    metric_key: Optional[str] = None,
) -> str:
    """
    Generate HTML for a metric card

    Args:
        label: Metric label
        value: Formatted metric value
        delta: Delta information dict with 'formatted', 'direction', etc.
        metric_key: Key to determine if higher is better

    Returns:
        HTML string for the metric card
    """
    # Determine delta styling
    delta_html = ""
    if delta and delta.get('formatted') and delta.get('formatted') != '—':
        direction = delta.get('direction', 'neutral')

        # Adjust color based on whether higher is better
        if metric_key and metric_key in METRICS:
            higher_is_better = METRICS[metric_key].get('higher_is_better')
            if higher_is_better is not None:
                if higher_is_better:
                    delta_class = 'positive' if direction == 'up' else 'negative'
                else:
                    delta_class = 'negative' if direction == 'up' else 'positive'
            else:
                delta_class = 'neutral'
        else:
            delta_class = 'positive' if direction == 'up' else 'negative' if direction == 'down' else 'neutral'

        delta_html = f"""
        <div class="metric-delta {delta_class}">
            {delta.get('formatted', '')} vs prev period
        </div>
        """

    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """


def diagnostic_card(
    alert_level: str,
    title: str,
    description: str,
    drivers: list = None,
) -> str:
    """
    Generate HTML for a diagnostic alert card

    Args:
        alert_level: 'warning', 'success', 'danger', 'info'
        title: Alert title
        description: Alert description
        drivers: List of driver dicts with 'type', 'title', 'detail'
    """
    # Alert icons
    icons = {
        'warning': '⚠️',
        'success': '✅',
        'danger': '🚨',
        'info': 'ℹ️',
    }

    icon = icons.get(alert_level, '📊')

    # Build drivers HTML
    drivers_html = ""
    if drivers:
        for i, driver in enumerate(drivers):
            driver_type = 'primary' if i == 0 else 'secondary'
            label = 'Primary Driver' if i == 0 else 'Secondary Factor'

            drivers_html += f"""
            <div class="driver-card {driver_type}">
                <div class="driver-label">{label}</div>
                <div class="driver-title">{driver.get('title', '')}</div>
                <div class="driver-detail">{driver.get('detail', '')}</div>
            </div>
            """

    return f"""
    <div class="diagnostic-panel">
        <div class="diagnostic-header">
            <span style="font-size: 1.5rem;">{icon}</span>
            <div class="diagnostic-alert {alert_level}">
                {title}
            </div>
        </div>
        <p style="color: {COLORS['text_secondary']}; font-size: 0.9rem; margin-bottom: 1rem;">
            {description}
        </p>
        {drivers_html}
    </div>
    """


def platform_badge(platform: str) -> str:
    """Generate HTML for a platform badge"""
    platform_classes = {
        'Google Ads': 'platform-google',
        'Meta': 'platform-meta',
        'DV360': 'platform-dv360',
        'Amazon Ads': 'platform-amazon',
    }

    platform_icons = {
        'Google Ads': '🔍',
        'Meta': '📘',
        'DV360': '📺',
        'Amazon Ads': '🛒',
    }

    css_class = platform_classes.get(platform, '')
    icon = platform_icons.get(platform, '📊')

    return f"""
    <span class="platform-badge {css_class}">
        {icon} {platform}
    </span>
    """


def delta_indicator(
    value: float,
    format_type: str = 'percentage',
    higher_is_better: bool = True,
    show_arrow: bool = True,
) -> str:
    """
    Generate HTML for a delta indicator

    Args:
        value: Delta value (positive or negative)
        format_type: 'percentage', 'currency', 'number'
        higher_is_better: If True, positive is green; if False, positive is red
        show_arrow: Whether to show direction arrow
    """
    # Determine direction and color
    if abs(value) < 0.1:
        direction = 'neutral'
        color = COLORS['text_secondary']
        arrow = '→' if show_arrow else ''
    elif value > 0:
        direction = 'up'
        color = COLORS['success'] if higher_is_better else COLORS['danger']
        arrow = '▲' if show_arrow else ''
    else:
        direction = 'down'
        color = COLORS['danger'] if higher_is_better else COLORS['success']
        arrow = '▼' if show_arrow else ''

    # Format value
    if format_type == 'percentage':
        formatted = f"{abs(value):.1f}%"
    elif format_type == 'currency':
        formatted = f"₹{abs(value):,.0f}"
    else:
        formatted = f"{abs(value):,.0f}"

    return f"""
    <span style="color: {color}; font-weight: 500;">
        {arrow} {formatted}
    </span>
    """


def section_header(icon: str, title: str) -> str:
    """Generate HTML for a section header"""
    return f"""
    <div class="section-header">
        <span class="section-icon">{icon}</span>
        <span class="section-title">{title}</span>
    </div>
    """


def ai_response_card(response: str) -> str:
    """Generate HTML for an AI response card"""
    # Process markdown-like formatting
    response = response.replace('**', '<strong>').replace('**', '</strong>')

    return f"""
    <div class="ai-panel">
        <div class="section-header">
            <span class="section-icon">🤖</span>
            <span class="section-title">AI Analysis</span>
        </div>
        <div class="ai-response">
            {response}
        </div>
    </div>
    """


def suggested_questions(questions: list) -> str:
    """Generate HTML for suggested questions"""
    buttons = ""
    for q in questions:
        buttons += f"""
        <span class="suggested-question">{q}</span>
        """

    return f"""
    <div style="display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.5rem;">
        {buttons}
    </div>
    """


def loading_indicator(text: str = "Analyzing...") -> str:
    """Generate HTML for a loading indicator"""
    return f"""
    <div class="loading-pulse" style="
        color: {COLORS['text_secondary']};
        font-size: 0.9rem;
        padding: 1rem;
        text-align: center;
    ">
        ⏳ {text}
    </div>
    """


def empty_state(message: str, icon: str = "📭") -> str:
    """Generate HTML for an empty state"""
    return f"""
    <div style="
        text-align: center;
        padding: 3rem;
        color: {COLORS['text_secondary']};
    ">
        <div style="font-size: 3rem; margin-bottom: 1rem;">{icon}</div>
        <div style="font-size: 0.9rem;">{message}</div>
    </div>
    """


def format_table_row(row: dict, columns: list) -> str:
    """Format a table row with proper styling"""
    cells = ""
    for col in columns:
        value = row.get(col, '')
        cells += f"<td>{value}</td>"

    return f"<tr>{cells}</tr>"


def progress_bar(value: float, max_value: float = 100, color: str = None) -> str:
    """Generate HTML for a progress bar"""
    percentage = min(100, (value / max_value) * 100) if max_value > 0 else 0
    bar_color = color or COLORS['primary']

    return f"""
    <div style="
        background: {COLORS['surface_light']};
        border-radius: 4px;
        height: 8px;
        overflow: hidden;
    ">
        <div style="
            background: {bar_color};
            width: {percentage}%;
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s ease;
        "></div>
    </div>
    """


def kpi_row(kpis: list) -> str:
    """
    Generate HTML for a row of KPIs

    Args:
        kpis: List of dicts with 'label', 'value', 'delta' keys
    """
    cols = len(kpis)
    col_width = 100 // cols

    kpi_html = ""
    for kpi in kpis:
        delta_html = ""
        if kpi.get('delta'):
            delta = kpi['delta']
            direction = delta.get('direction', 'neutral')
            if direction == 'up':
                color = COLORS['success'] if kpi.get('higher_is_better', True) else COLORS['danger']
                arrow = '▲'
            elif direction == 'down':
                color = COLORS['danger'] if kpi.get('higher_is_better', True) else COLORS['success']
                arrow = '▼'
            else:
                color = COLORS['text_secondary']
                arrow = '→'

            delta_html = f"""
            <div style="font-size: 0.75rem; color: {color}; margin-top: 0.25rem;">
                {arrow} {delta.get('formatted', '')}
            </div>
            """

        kpi_html += f"""
        <div style="flex: 0 0 {col_width}%; text-align: center; padding: 0.5rem;">
            <div style="color: {COLORS['text_secondary']}; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em;">
                {kpi.get('label', '')}
            </div>
            <div style="color: {COLORS['text_primary']}; font-size: 1.5rem; font-weight: 600; margin-top: 0.25rem;">
                {kpi.get('value', '')}
            </div>
            {delta_html}
        </div>
        """

    return f"""
    <div style="display: flex; justify-content: space-between; background: {COLORS['surface']}; border: 1px solid {COLORS['border']}; border-radius: 12px; padding: 1rem;">
        {kpi_html}
    </div>
    """
