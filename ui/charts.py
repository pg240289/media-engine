"""
Chart components using Plotly for FE Media Intelligence Platform
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import List, Optional, Dict, Any
from config import COLORS, CHART_COLORS


def get_chart_layout(title: str = "", height: int = 400) -> dict:
    """Get consistent chart layout"""
    return dict(
        title=dict(
            text=title,
            font=dict(size=14, color=COLORS['text_primary']),
            x=0,
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(
            family="Inter, sans-serif",
            color=COLORS['text_secondary'],
            size=11,
        ),
        height=height,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(0,0,0,0)',
            font=dict(size=10),
        ),
        xaxis=dict(
            gridcolor=COLORS['border'],
            linecolor=COLORS['border'],
            tickfont=dict(size=10),
            showgrid=True,
            gridwidth=1,
        ),
        yaxis=dict(
            gridcolor=COLORS['border'],
            linecolor=COLORS['border'],
            tickfont=dict(size=10),
            showgrid=True,
            gridwidth=1,
        ),
        hoverlabel=dict(
            bgcolor=COLORS['surface'],
            bordercolor=COLORS['border'],
            font=dict(size=12, color=COLORS['text_primary']),
        ),
    )


def create_trend_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = "",
    color_by: Optional[str] = None,
    show_projection: bool = False,
    projection_data: Optional[Dict] = None,
) -> go.Figure:
    """
    Create a trend line chart

    Args:
        df: DataFrame with data
        x_col: Column for x-axis (usually date)
        y_col: Column for y-axis (metric)
        title: Chart title
        color_by: Optional column to color lines by
        show_projection: Whether to show projection
        projection_data: Dict with 'dates', 'values', 'lower_bound', 'upper_bound'
    """
    fig = go.Figure()

    if color_by:
        colors = CHART_COLORS
        for i, category in enumerate(df[color_by].unique()):
            mask = df[color_by] == category
            fig.add_trace(go.Scatter(
                x=df.loc[mask, x_col],
                y=df.loc[mask, y_col],
                mode='lines+markers',
                name=category,
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=4),
            ))
    else:
        fig.add_trace(go.Scatter(
            x=df[x_col],
            y=df[y_col],
            mode='lines+markers',
            name=y_col.upper(),
            line=dict(color=COLORS['primary'], width=2),
            marker=dict(size=4),
            fill='tozeroy',
            fillcolor=f"rgba(99, 102, 241, 0.1)",
        ))

    # Add projection if provided
    if show_projection and projection_data:
        # Projection line
        fig.add_trace(go.Scatter(
            x=projection_data['dates'],
            y=projection_data['values'],
            mode='lines',
            name='Projection',
            line=dict(color=COLORS['warning'], width=2, dash='dash'),
        ))

        # Confidence band
        if projection_data.get('lower_bound') and projection_data.get('upper_bound'):
            fig.add_trace(go.Scatter(
                x=projection_data['dates'] + projection_data['dates'][::-1],
                y=projection_data['upper_bound'] + projection_data['lower_bound'][::-1],
                fill='toself',
                fillcolor='rgba(245, 158, 11, 0.1)',
                line=dict(color='rgba(0,0,0,0)'),
                name='Confidence Band',
                showlegend=False,
            ))

    layout = get_chart_layout(title)
    fig.update_layout(**layout)

    return fig


def create_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = "",
    color_col: Optional[str] = None,
    orientation: str = 'v',
    show_values: bool = True,
) -> go.Figure:
    """
    Create a bar chart

    Args:
        df: DataFrame with data
        x_col: Column for x-axis (or y-axis if horizontal)
        y_col: Column for y-axis (or x-axis if horizontal)
        title: Chart title
        color_col: Column for bar colors
        orientation: 'v' for vertical, 'h' for horizontal
        show_values: Whether to show values on bars
    """
    if color_col:
        colors = [CHART_COLORS[i % len(CHART_COLORS)] for i in range(len(df))]
    else:
        colors = COLORS['primary']

    if orientation == 'h':
        fig = go.Figure(go.Bar(
            y=df[x_col],
            x=df[y_col],
            orientation='h',
            marker_color=colors,
            text=df[y_col].apply(lambda x: f"₹{x:,.0f}" if y_col in ['spend', 'cpa', 'revenue'] else f"{x:,.0f}"),
            textposition='outside' if show_values else 'none',
        ))
    else:
        fig = go.Figure(go.Bar(
            x=df[x_col],
            y=df[y_col],
            marker_color=colors,
            text=df[y_col].apply(lambda x: f"₹{x:,.0f}" if y_col in ['spend', 'cpa', 'revenue'] else f"{x:,.0f}"),
            textposition='outside' if show_values else 'none',
        ))

    layout = get_chart_layout(title)
    fig.update_layout(**layout)

    return fig


def create_platform_comparison_chart(
    df: pd.DataFrame,
    metric: str,
    title: str = "",
) -> go.Figure:
    """
    Create a platform comparison bar chart with proper formatting
    """
    # Define platform colors
    platform_colors = {
        'Google Ads': '#4285F4',
        'Meta': '#1877F2',
        'DV360': '#34A853',
        'Amazon Ads': '#FF9900',
    }

    # Sort by metric value
    df_sorted = df.sort_values(metric, ascending=False)

    colors = [platform_colors.get(p, COLORS['primary']) for p in df_sorted['platform']]

    # Format based on metric type
    if metric in ['spend', 'cpa', 'revenue', 'cpc']:
        text = df_sorted[metric].apply(lambda x: f"₹{x:,.0f}")
        prefix = '₹'
    elif metric in ['roas']:
        text = df_sorted[metric].apply(lambda x: f"{x:.2f}x")
        prefix = ''
    elif metric in ['ctr', 'cvr']:
        text = df_sorted[metric].apply(lambda x: f"{x:.2f}%")
        prefix = ''
    else:
        text = df_sorted[metric].apply(lambda x: f"{x:,.0f}")
        prefix = ''

    fig = go.Figure(go.Bar(
        x=df_sorted['platform'],
        y=df_sorted[metric],
        marker_color=colors,
        text=text,
        textposition='outside',
        textfont=dict(size=11, color=COLORS['text_primary']),
    ))

    layout = get_chart_layout(title)
    layout['yaxis']['tickprefix'] = prefix
    fig.update_layout(**layout)

    return fig


def create_decomposition_chart(
    cpc_contribution: float,
    cvr_contribution: float,
    title: str = "CPA Change Decomposition",
) -> go.Figure:
    """
    Create a waterfall-style chart showing CPA decomposition
    """
    fig = go.Figure(go.Waterfall(
        name="CPA Decomposition",
        orientation="v",
        measure=["relative", "relative", "total"],
        x=["CPC Impact", "CVR Impact", "Total Change"],
        y=[cpc_contribution, cvr_contribution, cpc_contribution + cvr_contribution],
        connector={"line": {"color": COLORS['border']}},
        increasing={"marker": {"color": COLORS['danger']}},
        decreasing={"marker": {"color": COLORS['success']}},
        totals={"marker": {"color": COLORS['primary']}},
        text=[f"₹{cpc_contribution:+.0f}", f"₹{cvr_contribution:+.0f}", f"₹{cpc_contribution + cvr_contribution:+.0f}"],
        textposition="outside",
    ))

    layout = get_chart_layout(title, height=300)
    fig.update_layout(**layout)

    return fig


def create_platform_attribution_chart(
    attributions: List[Dict],
    title: str = "Platform Contribution to CPA Change",
) -> go.Figure:
    """
    Create a horizontal bar chart showing platform attribution to CPA change
    """
    if not attributions:
        return go.Figure()

    platforms = [a['platform'] for a in attributions]
    contributions = [a['weighted_contribution'] for a in attributions]

    # Colors based on positive/negative contribution
    colors = [COLORS['danger'] if c > 0 else COLORS['success'] for c in contributions]

    fig = go.Figure(go.Bar(
        y=platforms,
        x=contributions,
        orientation='h',
        marker_color=colors,
        text=[f"₹{c:+.0f}" for c in contributions],
        textposition='outside',
    ))

    layout = get_chart_layout(title, height=250)
    layout['xaxis']['zeroline'] = True
    layout['xaxis']['zerolinecolor'] = COLORS['text_secondary']
    layout['xaxis']['zerolinewidth'] = 1
    fig.update_layout(**layout)

    return fig


def create_sparkline(
    values: List[float],
    color: str = None,
    height: int = 50,
    width: int = 120,
) -> go.Figure:
    """
    Create a small sparkline chart
    """
    fig = go.Figure(go.Scatter(
        y=values,
        mode='lines',
        line=dict(color=color or COLORS['primary'], width=1.5),
        fill='tozeroy',
        fillcolor=f"rgba(99, 102, 241, 0.1)",
    ))

    fig.update_layout(
        height=height,
        width=width,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
    )

    return fig


def create_scenario_comparison_chart(
    current: Dict,
    projected: Dict,
    title: str = "Scenario Impact",
) -> go.Figure:
    """
    Create a grouped bar chart comparing current vs projected metrics
    """
    metrics = ['Conversions', 'CPA', 'ROAS']
    current_values = [
        current.get('conversions', 0),
        current.get('cpa', 0),
        current.get('roas', 0),
    ]
    projected_values = [
        projected.get('conversions', 0),
        projected.get('cpa', 0),
        projected.get('roas', 0),
    ]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Current',
        x=metrics,
        y=[1, 1, 1],  # Normalized to 1
        marker_color=COLORS['text_secondary'],
        opacity=0.5,
    ))

    # Normalize projected to current
    normalized_projected = [
        projected_values[i] / current_values[i] if current_values[i] > 0 else 1
        for i in range(len(metrics))
    ]

    fig.add_trace(go.Bar(
        name='Projected',
        x=metrics,
        y=normalized_projected,
        marker_color=[
            COLORS['success'] if metrics[i] != 'CPA' and n > 1 else
            COLORS['success'] if metrics[i] == 'CPA' and n < 1 else
            COLORS['danger']
            for i, n in enumerate(normalized_projected)
        ],
    ))

    layout = get_chart_layout(title, height=300)
    layout['barmode'] = 'group'
    layout['yaxis']['tickformat'] = '.0%'
    fig.update_layout(**layout)

    return fig


def create_pie_chart(
    labels: List[str],
    values: List[float],
    title: str = "",
) -> go.Figure:
    """
    Create a donut chart
    """
    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.6,
        marker_colors=CHART_COLORS[:len(labels)],
        textinfo='label+percent',
        textposition='outside',
        textfont=dict(size=10, color=COLORS['text_secondary']),
    ))

    layout = get_chart_layout(title, height=300)
    layout['showlegend'] = False
    fig.update_layout(**layout)

    return fig


def create_spend_efficiency_scatter(
    df: pd.DataFrame,
    title: str = "Spend vs Efficiency by Platform",
) -> go.Figure:
    """
    Create a bubble chart showing spend vs ROAS with conversions as bubble size
    """
    platform_colors = {
        'Google Ads': '#4285F4',
        'Meta': '#1877F2',
        'DV360': '#34A853',
        'Amazon Ads': '#FF9900',
    }

    fig = go.Figure()

    # Normalize bubble sizes based on actual data range
    min_conv = df['conversions'].min()
    max_conv = df['conversions'].max()
    conv_range = max_conv - min_conv if max_conv > min_conv else 1

    for _, row in df.iterrows():
        platform = row.get('platform', 'Unknown')
        # Scale conversions to bubble size between 30 and 80
        normalized = (row['conversions'] - min_conv) / conv_range
        bubble_size = 30 + (normalized * 50)  # Range: 30-80

        fig.add_trace(go.Scatter(
            x=[row['spend']],
            y=[row['roas']],
            mode='markers+text',
            name=platform,
            marker=dict(
                size=bubble_size,
                color=platform_colors.get(platform, COLORS['primary']),
                opacity=0.7,
                line=dict(width=2, color='white')
            ),
            text=[platform],
            textposition='top center',
            textfont=dict(size=10, color=COLORS['text_primary']),
            hovertemplate=(
                f"<b>{platform}</b><br>"
                f"Spend: ₹{row['spend']:,.0f}<br>"
                f"ROAS: {row['roas']:.2f}x<br>"
                f"Conversions: {row['conversions']:,.0f}<br>"
                f"CPA: ₹{row['cpa']:,.0f}<extra></extra>"
            )
        ))

    layout = get_chart_layout(title, height=400)
    layout['xaxis']['title'] = 'Spend (₹)'
    layout['yaxis']['title'] = 'ROAS'
    layout['showlegend'] = False
    fig.update_layout(**layout)

    return fig


def create_period_comparison_bars(
    current: Dict,
    previous: Dict,
    metrics: List[str],
    title: str = "Period-over-Period Comparison",
) -> go.Figure:
    """
    Create side-by-side bar comparison for current vs previous period
    """
    fig = go.Figure()

    # Previous period
    fig.add_trace(go.Bar(
        name='Previous Period',
        x=metrics,
        y=[previous.get(m, 0) for m in metrics],
        marker_color=COLORS['text_muted'],
        opacity=0.6,
    ))

    # Current period
    fig.add_trace(go.Bar(
        name='Current Period',
        x=metrics,
        y=[current.get(m, 0) for m in metrics],
        marker_color=COLORS['primary'],
    ))

    layout = get_chart_layout(title, height=300)
    layout['barmode'] = 'group'
    layout['legend'] = dict(
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='right',
        x=1
    )
    fig.update_layout(**layout)

    return fig


def create_metric_gauge(
    value: float,
    title: str,
    target: float = None,
    min_val: float = 0,
    max_val: float = 100,
    is_inverse: bool = False,
) -> go.Figure:
    """
    Create a gauge chart for a single metric
    """
    # Determine color based on value vs target
    if target:
        if is_inverse:
            color = COLORS['success'] if value <= target else COLORS['danger']
        else:
            color = COLORS['success'] if value >= target else COLORS['danger']
    else:
        color = COLORS['primary']

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 14, 'color': COLORS['text_primary']}},
        number={'font': {'size': 28, 'color': COLORS['text_primary']}},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickcolor': COLORS['text_muted']},
            'bar': {'color': color},
            'bgcolor': COLORS['surface'],
            'borderwidth': 0,
            'steps': [
                {'range': [min_val, max_val * 0.33], 'color': 'rgba(239, 68, 68, 0.2)'},
                {'range': [max_val * 0.33, max_val * 0.66], 'color': 'rgba(245, 158, 11, 0.2)'},
                {'range': [max_val * 0.66, max_val], 'color': 'rgba(34, 197, 94, 0.2)'},
            ] if not is_inverse else [
                {'range': [min_val, max_val * 0.33], 'color': 'rgba(34, 197, 94, 0.2)'},
                {'range': [max_val * 0.33, max_val * 0.66], 'color': 'rgba(245, 158, 11, 0.2)'},
                {'range': [max_val * 0.66, max_val], 'color': 'rgba(239, 68, 68, 0.2)'},
            ],
            'threshold': {
                'line': {'color': COLORS['warning'], 'width': 2},
                'thickness': 0.75,
                'value': target
            } if target else None
        }
    ))

    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLORS['text_secondary'])
    )

    return fig


def create_heatmap(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    value_col: str,
    title: str = "",
    color_scale: str = 'RdYlGn',
) -> go.Figure:
    """
    Create a heatmap for dimensional analysis
    """
    pivot = df.pivot_table(values=value_col, index=y_col, columns=x_col, aggfunc='mean')

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=color_scale,
        hoverongaps=False,
        hovertemplate='%{y} - %{x}<br>Value: %{z:.2f}<extra></extra>'
    ))

    layout = get_chart_layout(title, height=350)
    fig.update_layout(**layout)

    return fig


def create_funnel_chart(
    stages: List[str],
    values: List[float],
    title: str = "Conversion Funnel",
) -> go.Figure:
    """
    Create a funnel chart showing conversion stages
    """
    fig = go.Figure(go.Funnel(
        y=stages,
        x=values,
        textposition="inside",
        textinfo="value+percent initial",
        marker=dict(
            color=[COLORS['primary'], COLORS['info'], COLORS['success'], COLORS['warning']][:len(stages)]
        ),
        connector={"line": {"color": COLORS['border'], "width": 1}}
    ))

    layout = get_chart_layout(title, height=300)
    fig.update_layout(**layout)

    return fig


def create_stacked_area_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    stack_col: str,
    title: str = "",
) -> go.Figure:
    """
    Create a stacked area chart for time series by category
    """
    fig = go.Figure()

    categories = df[stack_col].unique()
    colors = CHART_COLORS

    for i, cat in enumerate(categories):
        cat_data = df[df[stack_col] == cat].sort_values(x_col)
        fig.add_trace(go.Scatter(
            x=cat_data[x_col],
            y=cat_data[y_col],
            mode='lines',
            name=cat,
            stackgroup='one',
            line=dict(width=0.5, color=colors[i % len(colors)]),
            fillcolor=colors[i % len(colors)],
        ))

    layout = get_chart_layout(title, height=350)
    fig.update_layout(**layout)

    return fig


def create_bullet_chart(
    actual: float,
    target: float,
    title: str,
    ranges: List[float] = None,
) -> go.Figure:
    """
    Create a bullet chart for goal tracking
    """
    if ranges is None:
        ranges = [target * 0.5, target * 0.75, target * 1.25]

    fig = go.Figure(go.Indicator(
        mode="number+gauge+delta",
        value=actual,
        delta={'reference': target, 'relative': True, 'valueformat': '.1%'},
        title={'text': title},
        gauge={
            'shape': 'bullet',
            'axis': {'range': [0, max(ranges)]},
            'threshold': {
                'line': {'color': 'red', 'width': 2},
                'thickness': 0.75,
                'value': target
            },
            'steps': [
                {'range': [0, ranges[0]], 'color': 'rgba(239, 68, 68, 0.3)'},
                {'range': [ranges[0], ranges[1]], 'color': 'rgba(245, 158, 11, 0.3)'},
                {'range': [ranges[1], ranges[2]], 'color': 'rgba(34, 197, 94, 0.3)'},
            ],
            'bar': {'color': COLORS['primary']}
        }
    ))

    fig.update_layout(
        height=120,
        margin=dict(l=120, r=20, t=30, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
    )

    return fig


def create_treemap(
    labels: List[str],
    parents: List[str],
    values: List[float],
    title: str = "Spend Distribution",
) -> go.Figure:
    """
    Create a treemap for hierarchical data visualization
    """
    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        values=values,
        marker=dict(
            colors=values,
            colorscale='Blues',
            line=dict(width=2, color=COLORS['background'])
        ),
        textinfo='label+value+percent parent',
        hovertemplate='<b>%{label}</b><br>Spend: ₹%{value:,.0f}<br>Share: %{percentParent:.1%}<extra></extra>'
    ))

    layout = get_chart_layout(title, height=400)
    fig.update_layout(**layout)

    return fig
