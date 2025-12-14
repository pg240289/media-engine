"""
Mock Data Generator for FE Media Intelligence Platform

Generates realistic Indian digital marketing data with FULL CAMPAIGN HIERARCHY:
Platform → Campaign → Objective → Ad Type → Targeting (Adset) → Ads → Creatives

CONSISTENCY RULES:
1. Each campaign belongs to ONE platform only (realistic)
2. Each campaign has ONE objective
3. Each adset belongs to ONE campaign
4. Each ad belongs to ONE adset
5. Ad types are platform-specific
6. Targeting types are platform-specific
7. Creative types are compatible with platform and ad type

Features:
- Multiple clients with different industries
- Realistic Indian market data (CPCs, CPAs, geos)
- Authentic campaign/adset/ad naming conventions
- Intentional issues for diagnostics demo
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import hashlib

# Seed for reproducibility
np.random.seed(42)

# Configuration
START_DATE = datetime(2024, 10, 1)
END_DATE = datetime(2024, 11, 30)
DIWALI_DATES = [datetime(2024, 11, 1), datetime(2024, 11, 2), datetime(2024, 11, 3)]

# =============================================================================
# CLIENT CONFIGURATION
# =============================================================================
CLIENTS = {
    'Sunrise Electronics': {
        'industry': 'Consumer Electronics',
        'budget_multiplier': 1.2,
        'aov_range': (8000, 25000),
        'cpa_multiplier': 1.3,
        'cvr_multiplier': 0.8,
        'primary_geos': ['Maharashtra', 'Karnataka', 'Delhi NCR', 'Tamil Nadu'],
        'diwali_boost': 2.8,
        'platforms': ['Google Ads', 'Meta', 'Amazon Ads'],
        'issue': 'meta_cvr_decline',
        'products': ['TV', 'Laptop', 'Mobile', 'Appliances', 'Audio'],
    },
    'FreshBasket': {
        'industry': 'Grocery & FMCG',
        'budget_multiplier': 0.8,
        'aov_range': (800, 2500),
        'cpa_multiplier': 0.6,
        'cvr_multiplier': 1.3,
        'primary_geos': ['Maharashtra', 'Delhi NCR', 'Karnataka', 'West Bengal', 'Gujarat'],
        'diwali_boost': 1.5,
        'platforms': ['Google Ads', 'Meta', 'DV360'],
        'issue': None,
        'products': ['Groceries', 'Fresh Produce', 'Dairy', 'Snacks', 'Beverages'],
    },
    'StyleKart Fashion': {
        'industry': 'Fashion & Apparel',
        'budget_multiplier': 1.0,
        'aov_range': (1500, 5000),
        'cpa_multiplier': 0.9,
        'cvr_multiplier': 1.0,
        'primary_geos': ['Maharashtra', 'Delhi NCR', 'Karnataka', 'Tamil Nadu', 'Telangana'],
        'diwali_boost': 2.2,
        'platforms': ['Google Ads', 'Meta', 'DV360', 'Amazon Ads'],
        'issue': 'cpc_increase',
        'products': ['Ethnic Wear', 'Western', 'Footwear', 'Accessories', 'Kidswear'],
    },
    'AutoZone India': {
        'industry': 'Automotive',
        'budget_multiplier': 1.5,
        'aov_range': (35000, 150000),  # Higher AOV for automotive (cars/bikes)
        'cpa_multiplier': 1.4,         # Slightly lower CPA multiplier for better profitability
        'cvr_multiplier': 0.7,         # Slightly higher CVR
        'primary_geos': ['Maharashtra', 'Delhi NCR', 'Gujarat', 'Tamil Nadu'],
        'diwali_boost': 1.3,
        'platforms': ['Google Ads', 'Meta', 'DV360'],
        'issue': None,
        'products': ['Cars', 'Bikes', 'Accessories', 'Service', 'Insurance'],
    },
    'WellnessFirst': {
        'industry': 'Health & Wellness',
        'budget_multiplier': 0.7,
        'aov_range': (1200, 4000),
        'cpa_multiplier': 0.8,
        'cvr_multiplier': 1.1,
        'primary_geos': ['Maharashtra', 'Karnataka', 'Delhi NCR', 'Kerala', 'Tamil Nadu'],
        'diwali_boost': 1.2,
        'platforms': ['Google Ads', 'Meta', 'Amazon Ads'],
        'issue': 'improving',
        'products': ['Supplements', 'Fitness', 'Ayurveda', 'Personal Care', 'Organic'],
    },
}

# =============================================================================
# HIERARCHY CONFIGURATION WITH REALISTIC MAPPINGS
# =============================================================================

# Campaign objectives with performance characteristics
OBJECTIVES = {
    'Conversions': {'weight': 0.35, 'cvr_mult': 1.2, 'cpc_mult': 1.1},
    'Traffic': {'weight': 0.20, 'cvr_mult': 0.6, 'cpc_mult': 0.7},
    'Brand Awareness': {'weight': 0.15, 'cvr_mult': 0.3, 'cpc_mult': 0.5},
    'Lead Generation': {'weight': 0.15, 'cvr_mult': 1.0, 'cpc_mult': 0.9},
    'App Installs': {'weight': 0.15, 'cvr_mult': 0.8, 'cpc_mult': 0.85},
}

# Ad types available per platform with realistic performance multipliers
AD_TYPES_BY_PLATFORM = {
    'Google Ads': {
        'Search': {'ctr_mult': 1.3, 'cvr_mult': 1.2, 'valid_objectives': ['Conversions', 'Traffic', 'Lead Generation']},
        'Display': {'ctr_mult': 0.5, 'cvr_mult': 0.6, 'valid_objectives': ['Brand Awareness', 'Traffic', 'Conversions']},
        'Video': {'ctr_mult': 0.8, 'cvr_mult': 0.7, 'valid_objectives': ['Brand Awareness', 'Traffic', 'App Installs']},
        'Shopping': {'ctr_mult': 1.1, 'cvr_mult': 1.3, 'valid_objectives': ['Conversions']},
        'Performance Max': {'ctr_mult': 0.9, 'cvr_mult': 1.0, 'valid_objectives': ['Conversions', 'Lead Generation']},
    },
    'Meta': {
        'Image': {'ctr_mult': 0.9, 'cvr_mult': 0.9, 'valid_objectives': ['Conversions', 'Traffic', 'Brand Awareness', 'Lead Generation']},
        'Video': {'ctr_mult': 1.1, 'cvr_mult': 1.0, 'valid_objectives': ['Brand Awareness', 'Traffic', 'Conversions', 'App Installs']},
        'Carousel': {'ctr_mult': 1.0, 'cvr_mult': 1.1, 'valid_objectives': ['Conversions', 'Traffic']},
        'Collection': {'ctr_mult': 0.85, 'cvr_mult': 1.2, 'valid_objectives': ['Conversions']},
        'Stories': {'ctr_mult': 1.2, 'cvr_mult': 0.8, 'valid_objectives': ['Brand Awareness', 'Traffic', 'App Installs']},
    },
    'DV360': {
        'Display': {'ctr_mult': 0.4, 'cvr_mult': 0.5, 'valid_objectives': ['Brand Awareness', 'Traffic']},
        'Video': {'ctr_mult': 0.6, 'cvr_mult': 0.6, 'valid_objectives': ['Brand Awareness', 'Traffic']},
        'Native': {'ctr_mult': 0.7, 'cvr_mult': 0.7, 'valid_objectives': ['Traffic', 'Conversions']},
        'Audio': {'ctr_mult': 0.3, 'cvr_mult': 0.4, 'valid_objectives': ['Brand Awareness']},
    },
    'Amazon Ads': {
        'Sponsored Products': {'ctr_mult': 1.4, 'cvr_mult': 1.5, 'valid_objectives': ['Conversions']},
        'Sponsored Brands': {'ctr_mult': 1.1, 'cvr_mult': 1.2, 'valid_objectives': ['Conversions', 'Brand Awareness']},
        'Sponsored Display': {'ctr_mult': 0.6, 'cvr_mult': 0.7, 'valid_objectives': ['Conversions', 'Traffic']},
        'Video': {'ctr_mult': 0.8, 'cvr_mult': 0.9, 'valid_objectives': ['Brand Awareness', 'Conversions']},
    },
}

# Targeting types per platform - realistic naming conventions
TARGETING_TYPES_BY_PLATFORM = {
    'Google Ads': {
        'Exact Match Keywords': {'cvr_mult': 1.3, 'adset_prefix': 'KW_Exact'},
        'Phrase Match Keywords': {'cvr_mult': 1.1, 'adset_prefix': 'KW_Phrase'},
        'Broad Match Keywords': {'cvr_mult': 0.8, 'adset_prefix': 'KW_Broad'},
        'In-Market Audiences': {'cvr_mult': 1.0, 'adset_prefix': 'AUD_InMarket'},
        'Custom Intent': {'cvr_mult': 1.1, 'adset_prefix': 'AUD_CustomIntent'},
        'Remarketing': {'cvr_mult': 1.5, 'adset_prefix': 'RMKT'},
        'Demographics': {'cvr_mult': 0.7, 'adset_prefix': 'DEMO'},
    },
    'Meta': {
        'Interest Based': {'cvr_mult': 0.8, 'adset_prefix': 'INT'},
        'Lookalike 1%': {'cvr_mult': 1.3, 'adset_prefix': 'LAL_1'},
        'Lookalike 3%': {'cvr_mult': 1.1, 'adset_prefix': 'LAL_3'},
        'Lookalike 5%': {'cvr_mult': 0.9, 'adset_prefix': 'LAL_5'},
        'Custom Audience': {'cvr_mult': 1.4, 'adset_prefix': 'CA'},
        'Website Retargeting': {'cvr_mult': 1.6, 'adset_prefix': 'WCA'},
        'Engagement Retargeting': {'cvr_mult': 1.4, 'adset_prefix': 'ECA'},
        'Broad Targeting': {'cvr_mult': 0.6, 'adset_prefix': 'BROAD'},
    },
    'DV360': {
        'Contextual Targeting': {'cvr_mult': 0.7, 'adset_prefix': 'CTX'},
        'Affinity Audiences': {'cvr_mult': 0.8, 'adset_prefix': 'AFF'},
        'Custom Affinity': {'cvr_mult': 0.9, 'adset_prefix': 'CAFF'},
        'First-Party Data': {'cvr_mult': 1.3, 'adset_prefix': '1PD'},
        'Similar Audiences': {'cvr_mult': 1.1, 'adset_prefix': 'SIM'},
    },
    'Amazon Ads': {
        'Product Targeting': {'cvr_mult': 1.3, 'adset_prefix': 'PROD'},
        'Category Targeting': {'cvr_mult': 1.0, 'adset_prefix': 'CAT'},
        'Brand Defense': {'cvr_mult': 1.5, 'adset_prefix': 'BRAND_DEF'},
        'Competitor Conquest': {'cvr_mult': 0.9, 'adset_prefix': 'COMP'},
        'Auto Targeting': {'cvr_mult': 1.1, 'adset_prefix': 'AUTO'},
    },
}

# Creative types with platform and ad type compatibility
CREATIVE_TYPES = {
    'Static Image': {
        'ctr_mult': 0.9,
        'compatible': {
            'Google Ads': ['Display', 'Performance Max'],
            'Meta': ['Image', 'Carousel'],
            'DV360': ['Display', 'Native'],
            'Amazon Ads': ['Sponsored Brands', 'Sponsored Display'],
        }
    },
    'Video 15s': {
        'ctr_mult': 1.15,
        'compatible': {
            'Google Ads': ['Video', 'Performance Max'],
            'Meta': ['Video', 'Stories'],
            'DV360': ['Video'],
            'Amazon Ads': ['Video'],
        }
    },
    'Video 30s': {
        'ctr_mult': 1.0,
        'compatible': {
            'Google Ads': ['Video'],
            'Meta': ['Video'],
            'DV360': ['Video'],
            'Amazon Ads': ['Video'],
        }
    },
    'Carousel': {
        'ctr_mult': 1.05,
        'compatible': {
            'Meta': ['Carousel'],
        }
    },
    'Text Ad': {
        'ctr_mult': 1.2,
        'compatible': {
            'Google Ads': ['Search'],
        }
    },
    'Responsive': {
        'ctr_mult': 0.95,
        'compatible': {
            'Google Ads': ['Display', 'Search'],
            'DV360': ['Display'],
        }
    },
    'Collection': {
        'ctr_mult': 1.1,
        'compatible': {
            'Meta': ['Collection'],
        }
    },
    'Product Image': {
        'ctr_mult': 1.1,
        'compatible': {
            'Google Ads': ['Shopping'],
            'Amazon Ads': ['Sponsored Products', 'Sponsored Brands'],
        }
    },
    'HTML5 Rich Media': {
        'ctr_mult': 0.85,
        'compatible': {
            'DV360': ['Display'],
        }
    },
    'Native Ad': {
        'ctr_mult': 0.9,
        'compatible': {
            'DV360': ['Native'],
        }
    },
    'Audio Spot': {
        'ctr_mult': 0.7,
        'compatible': {
            'DV360': ['Audio'],
        }
    },
}

# Platform configurations with Indian market characteristics
# IMPORTANT: These values are calibrated to produce realistic ROAS and CPA
PLATFORM_CONFIG = {
    'Google Ads': {
        'base_daily_spend': 45000,
        'cpc_range': (12, 28),        # Lower CPC for better CPA
        'cvr_range': (0.028, 0.055),  # Higher CVR for Search
        'aov_mult': 1.0,              # Standard AOV
        'weekend_spend_mult': 0.75,
        'weekend_cvr_mult': 0.85,
    },
    'Meta': {
        'base_daily_spend': 35000,
        'cpc_range': (6, 18),         # Lower CPC
        'cvr_range': (0.022, 0.045),  # Good CVR for social
        'aov_mult': 0.9,              # Slightly lower AOV
        'weekend_spend_mult': 1.15,
        'weekend_cvr_mult': 1.05,
    },
    'DV360': {
        'base_daily_spend': 20000,    # Lower spend (awareness focused)
        'cpc_range': (3, 10),         # Low CPM-based CPC
        'cvr_range': (0.015, 0.035),  # Better CVR (was too low)
        'aov_mult': 1.1,              # Higher AOV (premium placements)
        'weekend_spend_mult': 0.85,
        'weekend_cvr_mult': 0.90,
    },
    'Amazon Ads': {
        'base_daily_spend': 30000,
        'cpc_range': (10, 25),        # Amazon competitive CPC
        'cvr_range': (0.045, 0.095),  # High CVR (high intent)
        'aov_mult': 1.0,              # Standard AOV
        'weekend_spend_mult': 1.10,
        'weekend_cvr_mult': 1.08,
    },
}

# Geo distribution (based on Indian e-commerce patterns)
GEO_CONFIG = {
    'Maharashtra': {'weight': 0.22, 'cpa_mult': 0.95},
    'Karnataka': {'weight': 0.15, 'cpa_mult': 0.90},
    'Delhi NCR': {'weight': 0.18, 'cpa_mult': 1.05},
    'Tamil Nadu': {'weight': 0.12, 'cpa_mult': 0.92},
    'Gujarat': {'weight': 0.10, 'cpa_mult': 1.00},
    'West Bengal': {'weight': 0.08, 'cpa_mult': 1.08},
    'Telangana': {'weight': 0.07, 'cpa_mult': 0.88},
    'Kerala': {'weight': 0.05, 'cpa_mult': 1.02},
    'Rajasthan': {'weight': 0.03, 'cpa_mult': 1.15},
}

# =============================================================================
# REALISTIC CAMPAIGN NAMING PATTERNS
# =============================================================================

CAMPAIGN_NAME_PATTERNS = {
    'Google Ads': {
        'Conversions': [
            '{client}_Search_Conversions_{product}',
            '{client}_Shopping_{product}_Conv',
            '{client}_PMax_{product}',
            '{client}_DSA_Conversions',
        ],
        'Traffic': [
            '{client}_Display_Traffic_{product}',
            '{client}_Discovery_Traffic',
            '{client}_Video_Traffic_{product}',
        ],
        'Brand Awareness': [
            '{client}_YouTube_Brand',
            '{client}_Display_Awareness',
            '{client}_Video_Reach',
        ],
        'Lead Generation': [
            '{client}_Search_Leads',
            '{client}_PMax_LeadGen',
        ],
        'App Installs': [
            '{client}_UAC_Installs',
            '{client}_Video_AppInstall',
        ],
    },
    'Meta': {
        'Conversions': [
            '{client}_FB_Conversions_{product}',
            '{client}_IG_Sales_{product}',
            '{client}_META_CBO_{product}',
            '{client}_Advantage_Shopping',
        ],
        'Traffic': [
            '{client}_FB_Traffic_{product}',
            '{client}_IG_LandingPage',
        ],
        'Brand Awareness': [
            '{client}_FB_Reach',
            '{client}_IG_Brand_Awareness',
            '{client}_META_Video_Views',
        ],
        'Lead Generation': [
            '{client}_FB_LeadAds',
            '{client}_IG_LeadGen',
        ],
        'App Installs': [
            '{client}_FB_AppInstall',
            '{client}_IG_AppPromo',
        ],
    },
    'DV360': {
        'Conversions': [
            '{client}_DV360_Conversions',
            '{client}_Programmatic_Conv',
        ],
        'Traffic': [
            '{client}_DV360_Traffic',
            '{client}_Native_Traffic',
        ],
        'Brand Awareness': [
            '{client}_DV360_Awareness',
            '{client}_CTV_Brand',
            '{client}_Audio_Brand',
            '{client}_Video_Reach_DV360',
        ],
        'Lead Generation': [],
        'App Installs': [],
    },
    'Amazon Ads': {
        'Conversions': [
            '{client}_SP_{product}',
            '{client}_SB_{product}',
            '{client}_SD_Retargeting',
        ],
        'Traffic': [
            '{client}_SD_Traffic',
        ],
        'Brand Awareness': [
            '{client}_SBV_Brand',
            '{client}_SD_Awareness',
        ],
        'Lead Generation': [],
        'App Installs': [],
    },
}


def generate_hash_id(prefix: str, *args) -> str:
    """Generate a consistent hash-based ID"""
    content = "_".join(str(a) for a in args)
    hash_val = hashlib.md5(content.encode()).hexdigest()[:8].upper()
    return f"{prefix}_{hash_val}"


def get_client_short(client_name: str) -> str:
    """Get abbreviated client name for campaign naming"""
    words = client_name.split()
    if len(words) >= 2:
        return words[0]
    return client_name.split()[0]


def get_compatible_creative_types(platform: str, ad_type: str) -> list:
    """Get creative types compatible with platform and ad type"""
    compatible = []
    for creative_type, config in CREATIVE_TYPES.items():
        if platform in config['compatible']:
            if ad_type in config['compatible'][platform]:
                compatible.append(creative_type)
    return compatible if compatible else ['Static Image']  # Fallback


def get_valid_ad_types(platform: str, objective: str) -> list:
    """Get ad types valid for this platform and objective"""
    valid = []
    for ad_type, config in AD_TYPES_BY_PLATFORM[platform].items():
        if objective in config.get('valid_objectives', []):
            valid.append(ad_type)
    return valid if valid else list(AD_TYPES_BY_PLATFORM[platform].keys())[:2]


def get_diwali_multiplier(date: datetime, client_config: dict) -> dict:
    """Get multipliers for Diwali period"""
    base_boost = client_config.get('diwali_boost', 1.5)

    if date in DIWALI_DATES:
        return {'spend': base_boost, 'cvr': 1 + (base_boost - 1) * 0.5, 'cpc': 1 + (base_boost - 1) * 0.3}

    # Pre-Diwali buildup (Oct 25 - Oct 31)
    if datetime(2024, 10, 25) <= date <= datetime(2024, 10, 31):
        days_to_diwali = (datetime(2024, 11, 1) - date).days
        intensity = 1 + ((base_boost - 1) * 0.5 * (7 - days_to_diwali) / 7)
        return {'spend': intensity, 'cvr': 1 + 0.1 * intensity, 'cpc': 1 + 0.05 * intensity}

    # Post-Diwali (Nov 4 - Nov 10)
    if datetime(2024, 11, 4) <= date <= datetime(2024, 11, 10):
        days_after = (date - datetime(2024, 11, 3)).days
        decay = max(0.7, 1 - 0.05 * days_after)
        return {'spend': decay, 'cvr': decay, 'cpc': 0.95}

    return {'spend': 1.0, 'cvr': 1.0, 'cpc': 1.0}


def get_client_issue_multiplier(client_name: str, client_config: dict, platform: str, date: datetime) -> dict:
    """Get multipliers based on client-specific issues for diagnostics"""
    days_elapsed = (date - START_DATE).days
    total_days = (END_DATE - START_DATE).days
    progress = days_elapsed / total_days

    issue = client_config.get('issue')

    if issue == 'meta_cvr_decline' and platform == 'Meta':
        if days_elapsed > 40:
            decline = 0.18 * ((days_elapsed - 40) / 20)
            return {'cvr': 1 - decline, 'cpc': 1.0}

    elif issue == 'cpc_increase' and platform in ['Google Ads', 'Meta']:
        if days_elapsed > 30:
            increase = 0.12 * ((days_elapsed - 30) / 30)
            return {'cvr': 1.0, 'cpc': 1 + increase}

    elif issue == 'improving':
        improvement = 0.15 * progress
        return {'cvr': 1 + improvement, 'cpc': 1 - improvement * 0.3}

    return {'cvr': 1.0, 'cpc': 1.0}


def generate_campaign_structure(client_name: str, client_config: dict) -> list:
    """
    Generate complete campaign hierarchy structure for a client.

    CONSISTENCY RULES:
    - Each campaign is tied to ONE platform
    - Each campaign has ONE objective
    - Ad types are valid for the platform AND objective
    - Creative types are compatible with platform AND ad type
    """
    campaigns = []
    client_short = get_client_short(client_name)
    client_platforms = client_config.get('platforms', list(PLATFORM_CONFIG.keys()))
    client_products = client_config.get('products', ['Product'])

    # Generate campaigns PER PLATFORM (realistic - campaigns exist within platforms)
    for platform in client_platforms:
        # Get objectives valid for this platform
        platform_objectives = []
        for obj in OBJECTIVES.keys():
            valid_ad_types = get_valid_ad_types(platform, obj)
            if valid_ad_types:
                platform_objectives.append(obj)

        # Generate 3-5 campaigns per platform
        num_campaigns = np.random.randint(3, 6)

        for camp_idx in range(num_campaigns):
            # Select objective (weighted)
            objectives_list = [o for o in platform_objectives]
            if not objectives_list:
                continue

            objective_weights = [OBJECTIVES[o]['weight'] for o in objectives_list]
            objective_weights = [w / sum(objective_weights) for w in objective_weights]
            objective = np.random.choice(objectives_list, p=objective_weights)

            # Generate campaign name using platform-specific patterns
            patterns = CAMPAIGN_NAME_PATTERNS.get(platform, {}).get(objective, [])
            if patterns:
                pattern = np.random.choice(patterns)
                product = np.random.choice(client_products)
                campaign_name = pattern.format(client=client_short, product=product)
            else:
                campaign_name = f"{client_short}_{platform.replace(' ', '')}_{objective}_{camp_idx + 1}"

            # Make campaign name unique
            campaign_name = f"{campaign_name}_{camp_idx + 1}"
            campaign_id = generate_hash_id("CAMP", client_name, platform, campaign_name)

            # Get valid ad types for this platform and objective
            valid_ad_types = get_valid_ad_types(platform, objective)
            num_ad_types = min(len(valid_ad_types), np.random.randint(1, 4))
            selected_ad_types = np.random.choice(valid_ad_types, num_ad_types, replace=False)

            for ad_type in selected_ad_types:
                # Get targeting types for this platform
                targeting_types = list(TARGETING_TYPES_BY_PLATFORM[platform].keys())
                num_adsets = min(len(targeting_types), np.random.randint(2, 5))
                selected_targeting = np.random.choice(targeting_types, num_adsets, replace=False)

                for targeting in selected_targeting:
                    targeting_cfg = TARGETING_TYPES_BY_PLATFORM[platform][targeting]
                    adset_prefix = targeting_cfg.get('adset_prefix', targeting[:4].upper())

                    # Realistic adset naming
                    adset_name = f"{adset_prefix}_{ad_type[:4]}_{np.random.randint(1, 100):02d}"
                    adset_id = generate_hash_id("ADSET", campaign_id, ad_type, targeting)

                    # Generate 2-4 ads per adset
                    num_ads = np.random.randint(2, 5)

                    for ad_idx in range(num_ads):
                        # Select creative type valid for this platform AND ad type
                        valid_creatives = get_compatible_creative_types(platform, ad_type)
                        creative_type = np.random.choice(valid_creatives)

                        # Realistic ad naming
                        ad_variant = chr(65 + ad_idx)  # A, B, C, D
                        ad_name = f"{adset_prefix}_{creative_type[:5]}_{ad_variant}"
                        ad_id = generate_hash_id("AD", adset_id, ad_idx)

                        creative_id = generate_hash_id("CRE", ad_id)
                        creative_name = f"{creative_type}_{ad_variant}_v{np.random.randint(1, 4)}"

                        campaigns.append({
                            'client': client_name,
                            'industry': client_config['industry'],
                            'campaign_id': campaign_id,
                            'campaign_name': campaign_name,
                            'objective': objective,
                            'platform': platform,
                            'ad_type': ad_type,
                            'adset_id': adset_id,
                            'adset_name': adset_name,
                            'targeting_type': targeting,
                            'ad_id': ad_id,
                            'ad_name': ad_name,
                            'creative_type': creative_type,
                            'creative_id': creative_id,
                            'creative_name': creative_name,
                        })

    return campaigns


def generate_daily_metrics(
    date: datetime,
    campaign_structure: dict,
    client_config: dict,
    geo: str,
) -> dict:
    """Generate daily metrics for a single campaign/ad/geo combination"""

    platform = campaign_structure['platform']
    objective = campaign_structure['objective']
    ad_type = campaign_structure['ad_type']
    targeting_type = campaign_structure['targeting_type']
    creative_type = campaign_structure['creative_type']

    is_weekend = date.weekday() >= 5
    platform_cfg = PLATFORM_CONFIG[platform]
    geo_cfg = GEO_CONFIG[geo]

    # Get all multipliers
    diwali_mult = get_diwali_multiplier(date, client_config)
    issue_mult = get_client_issue_multiplier(
        campaign_structure['client'], client_config, platform, date
    )
    objective_cfg = OBJECTIVES[objective]
    ad_type_cfg = AD_TYPES_BY_PLATFORM[platform][ad_type]
    targeting_cfg = TARGETING_TYPES_BY_PLATFORM[platform][targeting_type]
    creative_cfg = CREATIVE_TYPES[creative_type]

    # Base spend calculation
    base_spend = (
        platform_cfg['base_daily_spend']
        * geo_cfg['weight']
        * client_config['budget_multiplier']
        * 0.05  # Distribute among many ads
    )

    # Apply modifiers
    spend_mult = 1.0
    if is_weekend:
        spend_mult *= platform_cfg['weekend_spend_mult']
    spend_mult *= diwali_mult['spend']
    spend_mult *= np.random.uniform(0.7, 1.3)

    spend = base_spend * spend_mult

    # CPC calculation
    base_cpc = np.random.uniform(*platform_cfg['cpc_range'])
    cpc = base_cpc * client_config['cpa_multiplier']
    cpc *= diwali_mult['cpc'] * issue_mult['cpc']
    cpc *= geo_cfg['cpa_mult']
    cpc *= objective_cfg['cpc_mult']
    cpc *= np.random.uniform(0.85, 1.15)

    # Clicks from spend and CPC
    clicks = int(spend / cpc) if cpc > 0 else 0

    # CTR calculation for impressions
    base_ctr = np.random.uniform(0.008, 0.025)
    ctr = base_ctr * ad_type_cfg['ctr_mult'] * creative_cfg['ctr_mult']

    impressions = int(clicks / ctr) if ctr > 0 else 0

    # CVR calculation
    base_cvr = np.random.uniform(*platform_cfg['cvr_range'])
    cvr = base_cvr * client_config['cvr_multiplier']
    if is_weekend:
        cvr *= platform_cfg['weekend_cvr_mult']
    cvr *= diwali_mult['cvr'] * issue_mult['cvr']
    cvr *= objective_cfg['cvr_mult']
    cvr *= ad_type_cfg['cvr_mult']
    cvr *= targeting_cfg['cvr_mult']
    cvr *= np.random.uniform(0.8, 1.2)

    # Conversions
    conversions = int(clicks * cvr)

    # Revenue calculation with platform AOV multiplier
    # AOV is affected by platform (e.g., Amazon/DV360 premium placements get higher AOV)
    base_aov = np.random.uniform(*client_config['aov_range'])
    platform_aov_mult = platform_cfg.get('aov_mult', 1.0)

    # Ensure profitability: AOV should be meaningfully higher than CPA (spend/conversions)
    # This is critical for realistic ROAS (ROAS = revenue/spend = AOV*conv / spend)
    cpa_estimate = spend / max(conversions, 1)

    # Apply platform multiplier
    aov = base_aov * platform_aov_mult

    # For awareness/traffic objectives, lower CVR but still need viable AOV
    if objective in ['Brand Awareness', 'Traffic']:
        # These campaigns have lower CVR, so ensure AOV covers CPA for overall profitability
        aov = max(aov, cpa_estimate * 1.5)  # At least 1.5x CPA for minimum viable ROAS
    else:
        # Conversion-focused campaigns should have better ROAS
        aov = max(aov, cpa_estimate * 2.0)  # At least 2x CPA for healthy ROAS

    revenue = conversions * aov

    return {
        'spend': round(spend, 2),
        'impressions': impressions,
        'clicks': clicks,
        'conversions': conversions,
        'revenue': round(revenue, 2),
    }


def generate_all_data() -> pd.DataFrame:
    """Generate complete dataset with full campaign hierarchy"""
    print("Generating campaign structures...")

    # First, generate all campaign structures for all clients
    all_campaign_structures = {}
    for client_name, client_config in CLIENTS.items():
        all_campaign_structures[client_name] = generate_campaign_structure(client_name, client_config)
        print(f"  {client_name}: {len(all_campaign_structures[client_name])} ad combinations")

    print("\nGenerating daily performance data...")
    all_records = []

    current_date = START_DATE
    day_count = 0
    total_days = (END_DATE - START_DATE).days + 1

    while current_date <= END_DATE:
        day_count += 1
        if day_count % 10 == 0:
            print(f"  Processing day {day_count}/{total_days}...")

        for client_name, client_config in CLIENTS.items():
            campaign_structures = all_campaign_structures[client_name]
            primary_geos = client_config.get('primary_geos', list(GEO_CONFIG.keys()))

            for structure in campaign_structures:
                # Each ad runs in 2-4 geos
                num_geos = min(len(primary_geos), np.random.randint(2, 5))
                selected_geos = np.random.choice(primary_geos, num_geos, replace=False)

                for geo in selected_geos:
                    metrics = generate_daily_metrics(
                        current_date, structure, client_config, geo
                    )

                    # Only include if meaningful activity
                    if metrics['spend'] >= 50 and metrics['clicks'] >= 1:
                        record = {
                            'date': current_date.strftime('%Y-%m-%d'),
                            **{k: v for k, v in structure.items()},
                            'geo': geo,
                            **metrics,
                        }
                        all_records.append(record)

        current_date += timedelta(days=1)

    print(f"\nTotal records generated: {len(all_records):,}")

    df = pd.DataFrame(all_records)

    # Ensure data types
    df['date'] = pd.to_datetime(df['date'])
    df['spend'] = df['spend'].astype(float)
    df['impressions'] = df['impressions'].astype(int)
    df['clicks'] = df['clicks'].astype(int)
    df['conversions'] = df['conversions'].astype(int)
    df['revenue'] = df['revenue'].astype(float)

    return df


def validate_data(df: pd.DataFrame):
    """Validate generated data for consistency"""
    print("\n" + "=" * 60)
    print("DATA VALIDATION REPORT")
    print("=" * 60)

    # Basic stats
    print(f"\n[INFO] Total Records: {len(df):,}")
    print(f"[INFO] Date Range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"[INFO] Total Days: {df['date'].nunique()}")
    print(f"[INFO] Total Clients: {df['client'].nunique()}")

    # Hierarchy counts
    print(f"\n[HIERARCHY COUNTS]")
    print(f"  Campaigns: {df['campaign_id'].nunique()}")
    print(f"  Adsets: {df['adset_id'].nunique()}")
    print(f"  Ads: {df['ad_id'].nunique()}")
    print(f"  Creatives: {df['creative_id'].nunique()}")

    # Client breakdown
    print("\n[CLIENT SUMMARY]")
    client_summary = df.groupby('client').agg({
        'spend': 'sum',
        'conversions': 'sum',
        'revenue': 'sum',
        'campaign_id': 'nunique',
    }).round(0)
    client_summary['CPA'] = (client_summary['spend'] / client_summary['conversions']).round(0)
    client_summary['ROAS'] = (client_summary['revenue'] / client_summary['spend']).round(2)
    client_summary.columns = ['Spend', 'Conv', 'Revenue', 'Campaigns', 'CPA', 'ROAS']
    print(client_summary.to_string())

    # Platform breakdown
    print("\n[PLATFORM SUMMARY]")
    platform_summary = df.groupby('platform').agg({
        'spend': 'sum',
        'conversions': 'sum',
        'revenue': 'sum'
    }).round(0)
    platform_summary['CPA'] = (platform_summary['spend'] / platform_summary['conversions']).round(0)
    platform_summary['ROAS'] = (platform_summary['revenue'] / platform_summary['spend']).round(2)
    print(platform_summary.to_string())

    # Validate hierarchy consistency
    print("\n[HIERARCHY CONSISTENCY VALIDATION]")
    errors = 0

    # 1. Check campaign-platform consistency (each campaign should be on ONE platform)
    camp_platform = df.groupby('campaign_id')['platform'].nunique()
    inconsistent = (camp_platform > 1).sum()
    if inconsistent > 0:
        print(f"  [X] Campaigns spanning multiple platforms: {inconsistent}")
        errors += 1
    else:
        print(f"  [OK] All campaigns tied to single platform")

    # 2. Check campaign-objective consistency
    camp_obj = df.groupby('campaign_id')['objective'].nunique()
    inconsistent = (camp_obj > 1).sum()
    if inconsistent > 0:
        print(f"  [X] Campaigns with multiple objectives: {inconsistent}")
        errors += 1
    else:
        print(f"  [OK] All campaigns have single objective")

    # 3. Check adset-campaign consistency
    adset_camp = df.groupby('adset_id')['campaign_id'].nunique()
    inconsistent = (adset_camp > 1).sum()
    if inconsistent > 0:
        print(f"  [X] Adsets spanning multiple campaigns: {inconsistent}")
        errors += 1
    else:
        print(f"  [OK] All adsets belong to single campaign")

    # 4. Check ad-adset consistency
    ad_adset = df.groupby('ad_id')['adset_id'].nunique()
    inconsistent = (ad_adset > 1).sum()
    if inconsistent > 0:
        print(f"  [X] Ads spanning multiple adsets: {inconsistent}")
        errors += 1
    else:
        print(f"  [OK] All ads belong to single adset")

    # 5. Check creative-ad consistency
    creative_ad = df.groupby('creative_id')['ad_id'].nunique()
    inconsistent = (creative_ad > 1).sum()
    if inconsistent > 0:
        print(f"  [X] Creatives spanning multiple ads: {inconsistent}")
        errors += 1
    else:
        print(f"  [OK] All creatives belong to single ad")

    # Objective breakdown
    print("\n[OBJECTIVE SUMMARY]")
    obj_summary = df.groupby('objective').agg({
        'spend': 'sum',
        'conversions': 'sum',
    }).round(0)
    obj_summary['CPA'] = (obj_summary['spend'] / obj_summary['conversions']).round(0)
    print(obj_summary.to_string())

    # Ad Type breakdown
    print("\n[AD TYPE SUMMARY]")
    adtype_summary = df.groupby(['platform', 'ad_type']).agg({
        'spend': 'sum',
    }).round(0)
    print(adtype_summary.head(15).to_string())

    # Creative type breakdown
    print("\n[CREATIVE TYPE SUMMARY]")
    creative_summary = df.groupby('creative_type').agg({
        'spend': 'sum',
        'clicks': 'sum',
        'impressions': 'sum',
    }).round(0)
    creative_summary['CTR'] = (creative_summary['clicks'] / creative_summary['impressions'] * 100).round(2)
    print(creative_summary[['spend', 'CTR']].to_string())

    print("\n" + "=" * 60)
    if errors == 0:
        print("[OK] ALL CONSISTENCY CHECKS PASSED!")
    else:
        print(f"[!] {errors} consistency issues found")
    print("=" * 60)


def main():
    """Main entry point"""
    print("=" * 60)
    print("FE Media Intelligence Platform - Mock Data Generator")
    print("Generating CONSISTENT data with FULL CAMPAIGN HIERARCHY")
    print("=" * 60)

    # Generate data
    df = generate_all_data()

    # Validate
    validate_data(df)

    # Save
    output_path = Path(__file__).parent / "campaign_performance.csv"

    # Define column order
    column_order = [
        'date', 'client', 'industry', 'platform',
        'campaign_id', 'campaign_name', 'objective',
        'ad_type', 'adset_id', 'adset_name', 'targeting_type',
        'ad_id', 'ad_name', 'creative_type', 'creative_id', 'creative_name',
        'geo', 'spend', 'impressions', 'clicks', 'conversions', 'revenue'
    ]

    df = df[column_order]
    df.to_csv(output_path, index=False)

    print(f"\n[SAVED] Data saved to: {output_path}")
    print(f"[SAVED] File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    return df


if __name__ == "__main__":
    main()
