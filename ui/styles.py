"""
Custom CSS styles for FE Media Intelligence Platform

Professional dark theme with enterprise-grade aesthetics.
"""

from config import COLORS


def apply_custom_css():
    """Return custom CSS for Streamlit app"""
    return f"""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Styles */
    .stApp {{
        background-color: {COLORS['background']};
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    /* Main container */
    .main .block-container {{
        padding: 2rem 3rem;
        max-width: 1400px;
    }}

    /* Headers */
    h1, h2, h3 {{
        font-weight: 600;
        color: {COLORS['text_primary']};
    }}

    h1 {{
        font-size: 1.75rem;
        letter-spacing: -0.02em;
    }}

    h2 {{
        font-size: 1.25rem;
        letter-spacing: -0.01em;
    }}

    h3 {{
        font-size: 1rem;
        font-weight: 500;
    }}

    /* Sidebar */
    .css-1d391kg, [data-testid="stSidebar"] {{
        background-color: {COLORS['surface']};
        border-right: 1px solid {COLORS['border']};
    }}

    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label,
    [data-testid="stSidebar"] .stDateInput label {{
        color: {COLORS['text_secondary']};
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    /* Metric Cards */
    .metric-card {{
        background: linear-gradient(135deg, {COLORS['surface']} 0%, {COLORS['surface_light']} 100%);
        border: 1px solid {COLORS['border']};
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 0.75rem;
        transition: all 0.2s ease;
    }}

    .metric-card:hover {{
        border-color: {COLORS['primary']};
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.15);
    }}

    .metric-label {{
        color: {COLORS['text_secondary']};
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }}

    .metric-value {{
        color: {COLORS['text_primary']};
        font-size: 1.75rem;
        font-weight: 600;
        font-variant-numeric: tabular-nums;
        line-height: 1.2;
    }}

    .metric-delta {{
        font-size: 0.875rem;
        font-weight: 500;
        margin-top: 0.5rem;
    }}

    .metric-delta.positive {{
        color: {COLORS['success']};
    }}

    .metric-delta.negative {{
        color: {COLORS['danger']};
    }}

    .metric-delta.neutral {{
        color: {COLORS['text_secondary']};
    }}

    /* Diagnostic Panel */
    .diagnostic-panel {{
        background: linear-gradient(135deg, {COLORS['surface']} 0%, {COLORS['surface_light']} 100%);
        border: 1px solid {COLORS['border']};
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }}

    .diagnostic-header {{
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1rem;
    }}

    .diagnostic-alert {{
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-size: 0.875rem;
        font-weight: 500;
    }}

    .diagnostic-alert.warning {{
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid rgba(245, 158, 11, 0.3);
        color: {COLORS['warning']};
    }}

    .diagnostic-alert.success {{
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: {COLORS['success']};
    }}

    .diagnostic-alert.danger {{
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.3);
        color: {COLORS['danger']};
    }}

    .diagnostic-alert.info {{
        background: rgba(59, 130, 246, 0.15);
        border: 1px solid rgba(59, 130, 246, 0.3);
        color: {COLORS['info']};
    }}

    /* Driver Card */
    .driver-card {{
        background: {COLORS['surface_light']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 1rem;
        margin: 0.75rem 0;
    }}

    .driver-card.primary {{
        border-left: 3px solid {COLORS['primary']};
    }}

    .driver-card.secondary {{
        border-left: 3px solid {COLORS['text_secondary']};
    }}

    .driver-label {{
        color: {COLORS['text_secondary']};
        font-size: 0.7rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
    }}

    .driver-title {{
        color: {COLORS['text_primary']};
        font-size: 0.9rem;
        font-weight: 500;
    }}

    .driver-detail {{
        color: {COLORS['text_secondary']};
        font-size: 0.8rem;
        margin-top: 0.25rem;
    }}

    /* Chat/AI Panel */
    .ai-panel {{
        background: linear-gradient(135deg, {COLORS['surface']} 0%, {COLORS['surface_light']} 100%);
        border: 1px solid {COLORS['border']};
        border-radius: 12px;
        padding: 1.5rem;
    }}

    .ai-response {{
        background: {COLORS['surface_light']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
        color: {COLORS['text_primary']};
        font-size: 0.9rem;
        line-height: 1.6;
    }}

    .ai-response strong {{
        color: {COLORS['primary_light']};
    }}

    /* Tables */
    .stDataFrame {{
        border-radius: 8px;
        overflow: hidden;
    }}

    .stDataFrame table {{
        border-collapse: collapse;
    }}

    .stDataFrame th {{
        background-color: {COLORS['surface_light']} !important;
        color: {COLORS['text_secondary']} !important;
        font-size: 0.7rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 0.75rem 1rem !important;
    }}

    .stDataFrame td {{
        background-color: {COLORS['surface']} !important;
        color: {COLORS['text_primary']} !important;
        font-size: 0.875rem;
        padding: 0.75rem 1rem !important;
        border-bottom: 1px solid {COLORS['border']} !important;
    }}

    /* Platform Badges */
    .platform-badge {{
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
    }}

    .platform-google {{
        background: rgba(66, 133, 244, 0.15);
        color: #4285F4;
    }}

    .platform-meta {{
        background: rgba(24, 119, 242, 0.15);
        color: #1877F2;
    }}

    .platform-dv360 {{
        background: rgba(52, 168, 83, 0.15);
        color: #34A853;
    }}

    .platform-amazon {{
        background: rgba(255, 153, 0, 0.15);
        color: #FF9900;
    }}

    /* Buttons */
    .stButton > button {{
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }}

    .stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
    }}

    /* Text Input */
    .stTextInput > div > div > input {{
        background-color: {COLORS['surface_light']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        color: {COLORS['text_primary']};
        padding: 0.75rem 1rem;
    }}

    .stTextInput > div > div > input:focus {{
        border-color: {COLORS['primary']};
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2);
    }}

    /* Select boxes */
    .stSelectbox > div > div {{
        background-color: {COLORS['surface_light']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
    }}

    .stMultiSelect > div > div {{
        background-color: {COLORS['surface_light']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.5rem;
        background-color: transparent;
    }}

    .stTabs [data-baseweb="tab"] {{
        background-color: {COLORS['surface']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        color: {COLORS['text_secondary']};
        padding: 0.5rem 1rem;
        font-weight: 500;
    }}

    .stTabs [aria-selected="true"] {{
        background-color: {COLORS['primary']};
        border-color: {COLORS['primary']};
        color: white;
    }}

    /* Expander */
    .streamlit-expanderHeader {{
        background-color: {COLORS['surface']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        color: {COLORS['text_primary']};
    }}

    /* Section Headers */
    .section-header {{
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid {COLORS['border']};
    }}

    .section-icon {{
        font-size: 1.25rem;
    }}

    .section-title {{
        font-size: 1rem;
        font-weight: 600;
        color: {COLORS['text_primary']};
    }}

    /* Scrollbar */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}

    ::-webkit-scrollbar-track {{
        background: {COLORS['surface']};
    }}

    ::-webkit-scrollbar-thumb {{
        background: {COLORS['border']};
        border-radius: 4px;
    }}

    ::-webkit-scrollbar-thumb:hover {{
        background: {COLORS['border_light']};
    }}

    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    /* Suggested Questions */
    .suggested-question {{
        background: {COLORS['surface_light']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        cursor: pointer;
        transition: all 0.2s ease;
        font-size: 0.8rem;
        color: {COLORS['text_secondary']};
    }}

    .suggested-question:hover {{
        border-color: {COLORS['primary']};
        color: {COLORS['text_primary']};
    }}

    /* Loading animation */
    .loading-pulse {{
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }}

    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.5; }}
    }}

    /* Date Filter Bar - Enhanced Radio Buttons */
    div[role="radiogroup"] {{
        gap: 0.25rem !important;
    }}

    div[role="radiogroup"] label {{
        background: {COLORS['surface']} !important;
        border: 1px solid {COLORS['border']} !important;
        border-radius: 6px !important;
        padding: 0.4rem 0.8rem !important;
        font-size: 0.75rem !important;
        font-weight: 500 !important;
        color: {COLORS['text_secondary']} !important;
        transition: all 0.15s ease !important;
        cursor: pointer !important;
    }}

    div[role="radiogroup"] label:hover {{
        border-color: {COLORS['primary']} !important;
        color: {COLORS['text_primary']} !important;
        background: {COLORS['surface_light']} !important;
    }}

    div[role="radiogroup"] label[data-checked="true"],
    div[role="radiogroup"] label[data-baseweb="radio"]:has(input:checked) {{
        background: {COLORS['primary']} !important;
        border-color: {COLORS['primary']} !important;
        color: white !important;
    }}

    /* Date input styling */
    div[data-testid="stDateInput"] {{
        min-width: 100px;
    }}

    div[data-testid="stDateInput"] input {{
        background: {COLORS['surface']} !important;
        border: 1px solid {COLORS['border']} !important;
        border-radius: 6px !important;
        padding: 0.4rem 0.6rem !important;
        font-size: 0.8rem !important;
        color: {COLORS['text_primary']} !important;
    }}

    div[data-testid="stDateInput"] input:focus {{
        border-color: {COLORS['primary']} !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.15) !important;
    }}

    /* Client selector dropdown styling */
    div[data-testid="stSelectbox"] > div > div {{
        background: {COLORS['surface']} !important;
        border-radius: 8px !important;
    }}

    /* AI Quick Questions - Category-specific button colors */
    /* Diagnostics - Red */
    button[kind="secondary"]:has([data-testid]) {{
        transition: all 0.2s ease !important;
    }}

    /* Override default button gradient for AI questions */
    .stExpander button[kind="secondary"] {{
        background: {COLORS['surface_light']} !important;
        border: 1px solid {COLORS['border']} !important;
        color: {COLORS['text_primary']} !important;
    }}

    .stExpander button[kind="secondary"]:hover {{
        border-color: {COLORS['primary']} !important;
        background: {COLORS['surface']} !important;
    }}
    </style>
    """


def get_metric_card_style(metric_type: str = 'default') -> str:
    """Get inline style for metric card based on type"""
    base_style = f"""
        background: linear-gradient(135deg, {COLORS['surface']} 0%, {COLORS['surface_light']} 100%);
        border: 1px solid {COLORS['border']};
        border-radius: 12px;
        padding: 1.25rem;
    """
    return base_style
