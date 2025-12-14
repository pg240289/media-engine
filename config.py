"""
FE Media Intelligence Platform - Configuration
"""

# Platform Branding
PLATFORM_NAME = "FE Media Intelligence Platform"
PLATFORM_VERSION = "2.4.1"
PLATFORM_TAGLINE = "Unified Performance Analytics & AI-Powered Insights"

# Color Palette - Professional Dark Theme
COLORS = {
    'background': '#0E1117',
    'surface': '#1A1D29',
    'surface_light': '#242836',
    'primary': '#6366F1',        # Indigo
    'primary_light': '#818CF8',
    'secondary': '#8B5CF6',      # Purple
    'success': '#10B981',        # Emerald
    'success_light': '#34D399',
    'danger': '#EF4444',         # Red
    'danger_light': '#F87171',
    'warning': '#F59E0B',        # Amber
    'warning_light': '#FBBF24',
    'info': '#3B82F6',           # Blue
    'text_primary': '#F9FAFB',
    'text_secondary': '#9CA3AF',
    'text_muted': '#6B7280',
    'border': '#374151',
    'border_light': '#4B5563',
}

# Chart Colors
CHART_COLORS = [
    '#6366F1',  # Indigo
    '#8B5CF6',  # Purple
    '#EC4899',  # Pink
    '#F59E0B',  # Amber
    '#10B981',  # Emerald
    '#3B82F6',  # Blue
    '#EF4444',  # Red
    '#14B8A6',  # Teal
]

# Platform Configuration
PLATFORMS = {
    'Google Ads': {
        'color': '#4285F4',
        'icon': '🔍',
        'channels': ['Search', 'Display', 'YouTube', 'Performance Max'],
    },
    'Meta': {
        'color': '#1877F2',
        'icon': '📘',
        'channels': ['Feed', 'Stories', 'Reels', 'Audience Network'],
    },
    'DV360': {
        'color': '#34A853',
        'icon': '📺',
        'channels': ['Programmatic Display', 'Connected TV', 'Audio'],
    },
    'Amazon Ads': {
        'color': '#FF9900',
        'icon': '🛒',
        'channels': ['Sponsored Products', 'Sponsored Brands', 'DSP'],
    },
}

# Indian Geos
GEOS = [
    'Pan-India',
    'Maharashtra',
    'Karnataka',
    'Tamil Nadu',
    'Delhi NCR',
    'Gujarat',
    'West Bengal',
    'Telangana',
    'Kerala',
    'Rajasthan',
    'Uttar Pradesh',
]

# Metric Definitions
METRICS = {
    'spend': {'label': 'Spend', 'format': 'currency', 'higher_is_better': None},
    'impressions': {'label': 'Impressions', 'format': 'number', 'higher_is_better': True},
    'clicks': {'label': 'Clicks', 'format': 'number', 'higher_is_better': True},
    'conversions': {'label': 'Conversions', 'format': 'number', 'higher_is_better': True},
    'revenue': {'label': 'Revenue', 'format': 'currency', 'higher_is_better': True},
    'cpa': {'label': 'CPA', 'format': 'currency', 'higher_is_better': False},
    'roas': {'label': 'ROAS', 'format': 'ratio', 'higher_is_better': True},
    'ctr': {'label': 'CTR', 'format': 'percentage', 'higher_is_better': True},
    'cvr': {'label': 'CVR', 'format': 'percentage', 'higher_is_better': True},
    'cpc': {'label': 'CPC', 'format': 'currency', 'higher_is_better': False},
    'cpm': {'label': 'CPM', 'format': 'currency', 'higher_is_better': False},
}

# Date Configuration
DATE_FORMAT = "%Y-%m-%d"
DISPLAY_DATE_FORMAT = "%d %b %Y"

# LLM Configuration
LLM_MODEL = "claude-sonnet-4-20250514"  # Claude 4 Sonnet - latest fast model
LLM_MAX_TOKENS = 1024
LLM_TEMPERATURE = 0.3  # Lower for more consistent, factual responses

# Demo Questions (pre-configured for natural flow)
DEMO_QUESTIONS = [
    "Why did CPA increase this week?",
    "Which platform is underperforming?",
    "What's driving the performance change in Maharashtra?",
    "Should I reallocate budget from Meta to Google?",
    "What's the trend for conversions over the last 4 weeks?",
    "Which ad type has the best ROAS?",
    "Why is Meta's conversion rate dropping?",
    "What will happen if I increase Google Ads spend by 20%?",
]

# Campaign Hierarchy Definition
# Platform → Campaign → Objective → Ad Type → Targeting (Adset) → Ads → Creatives
HIERARCHY_LEVELS = [
    {'column': 'platform', 'label': 'Platform', 'icon': '🌐'},
    {'column': 'campaign_name', 'label': 'Campaign', 'icon': '📊'},
    {'column': 'objective', 'label': 'Objective', 'icon': '🎯'},
    {'column': 'ad_type', 'label': 'Ad Type', 'icon': '📝'},
    {'column': 'adset_name', 'label': 'Adset (Targeting)', 'icon': '🎯'},
    {'column': 'ad_name', 'label': 'Ad', 'icon': '📢'},
    {'column': 'creative_type', 'label': 'Creative Type', 'icon': '🎨'},
]

# Campaign Objectives
OBJECTIVES = [
    'Conversions',
    'Traffic',
    'Brand Awareness',
    'Lead Generation',
    'App Installs',
]

# Ad Types by Platform
AD_TYPES_BY_PLATFORM = {
    'Google Ads': ['Search', 'Display', 'Video', 'Shopping', 'Performance Max'],
    'Meta': ['Image', 'Video', 'Carousel', 'Collection', 'Stories'],
    'DV360': ['Display', 'Video', 'Native', 'Audio'],
    'Amazon Ads': ['Sponsored Products', 'Sponsored Brands', 'Sponsored Display', 'Video'],
}

# Targeting Types by Platform
TARGETING_TYPES = {
    'Google Ads': ['Keywords', 'In-Market Audiences', 'Custom Intent', 'Remarketing', 'Demographics'],
    'Meta': ['Interests', 'Lookalike', 'Custom Audience', 'Retargeting', 'Broad'],
    'DV360': ['Contextual', 'Audience Segments', 'First-Party Data', 'Programmatic'],
    'Amazon Ads': ['Product Targeting', 'Category Targeting', 'Brand Defense', 'Competitor Conquest'],
}

# Creative Types
CREATIVE_TYPES = [
    'Static Image',
    'Video 15s',
    'Video 30s',
    'Carousel',
    'Text Ad',
    'Responsive',
    'HTML5',
    'Native',
]

# Hierarchy Column Mapping (for display names)
HIERARCHY_COLUMN_LABELS = {
    'platform': 'Platform',
    'campaign_id': 'Campaign ID',
    'campaign_name': 'Campaign',
    'objective': 'Objective',
    'ad_type': 'Ad Type',
    'adset_id': 'Adset ID',
    'adset_name': 'Adset',
    'targeting_type': 'Targeting',
    'ad_id': 'Ad ID',
    'ad_name': 'Ad',
    'creative_type': 'Creative Type',
    'creative_id': 'Creative ID',
    'creative_name': 'Creative',
    'channel': 'Channel',
    'geo': 'Geo',
}
