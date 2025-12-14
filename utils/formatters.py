"""
Indian number and currency formatting utilities
"""

def format_inr(amount: float, short: bool = True) -> str:
    """
    Format amount in Indian Rupees with proper lakhs/crores notation

    Indian number system:
    - 1,00,000 = 1 Lakh (1L)
    - 1,00,00,000 = 1 Crore (1Cr)

    Args:
        amount: The amount to format
        short: If True, use L/Cr notation; if False, use full Indian comma format
    """
    if amount is None or (isinstance(amount, float) and (amount != amount)):  # NaN check
        return "₹0"

    amount = float(amount)
    is_negative = amount < 0
    amount = abs(amount)

    if short:
        if amount >= 1_00_00_000:  # 1 Crore
            formatted = f"₹{amount/1_00_00_000:.2f}Cr"
        elif amount >= 1_00_000:  # 1 Lakh
            formatted = f"₹{amount/1_00_000:.2f}L"
        elif amount >= 1_000:
            formatted = f"₹{amount/1_000:.1f}K"
        else:
            formatted = f"₹{amount:.0f}"
    else:
        # Full Indian comma format: 12,34,567
        formatted = "₹" + _indian_comma_format(amount)

    if is_negative:
        formatted = "-" + formatted

    return formatted


def _indian_comma_format(num: float) -> str:
    """
    Convert number to Indian comma notation
    123456789 → 12,34,56,789
    """
    num = int(round(num))
    s = str(num)

    if len(s) <= 3:
        return s

    # Last 3 digits
    result = s[-3:]
    s = s[:-3]

    # Then groups of 2
    while s:
        result = s[-2:] + "," + result
        s = s[:-2]

    # Remove leading comma if present
    return result.lstrip(",")


def format_number(num: float, decimals: int = 0) -> str:
    """Format number with Indian comma notation"""
    if num is None or (isinstance(num, float) and (num != num)):
        return "0"

    num = float(num)

    if num >= 1_00_00_000:
        return f"{num/1_00_00_000:.2f}Cr"
    elif num >= 1_00_000:
        return f"{num/1_00_000:.2f}L"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    else:
        return f"{num:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format as percentage"""
    if value is None or (isinstance(value, float) and (value != value)):
        return "0%"
    return f"{value:.{decimals}f}%"


def format_delta(current: float, previous: float, format_type: str = 'percentage') -> dict:
    """
    Calculate and format delta between two values

    Returns:
        dict with 'value', 'formatted', 'direction', 'color'
    """
    if previous is None or previous == 0 or (isinstance(previous, float) and (previous != previous)):
        return {
            'value': 0,
            'formatted': '—',
            'direction': 'neutral',
            'color': 'text_secondary'
        }

    if current is None or (isinstance(current, float) and (current != current)):
        current = 0

    delta = current - previous
    delta_pct = (delta / abs(previous)) * 100

    if abs(delta_pct) < 0.1:
        direction = 'neutral'
        color = 'text_secondary'
        symbol = ''
    elif delta_pct > 0:
        direction = 'up'
        color = 'success'  # Will be interpreted based on metric context
        symbol = '▲'
    else:
        direction = 'down'
        color = 'danger'
        symbol = '▼'

    if format_type == 'percentage':
        formatted = f"{symbol} {abs(delta_pct):.1f}%"
    elif format_type == 'currency':
        formatted = f"{symbol} {format_inr(abs(delta))}"
    else:
        formatted = f"{symbol} {format_number(abs(delta))}"

    return {
        'value': delta,
        'percentage': delta_pct,
        'formatted': formatted,
        'direction': direction,
        'color': color
    }


def format_ratio(value: float, decimals: int = 2) -> str:
    """Format as ratio (e.g., ROAS)"""
    if value is None or (isinstance(value, float) and (value != value)):
        return "0.00x"
    return f"{value:.{decimals}f}x"


def get_delta_color(delta_direction: str, higher_is_better: bool = True) -> str:
    """
    Get appropriate color based on delta direction and whether higher is better

    Args:
        delta_direction: 'up', 'down', or 'neutral'
        higher_is_better: If True, up=green; if False, up=red
    """
    if delta_direction == 'neutral':
        return 'text_secondary'

    if higher_is_better:
        return 'success' if delta_direction == 'up' else 'danger'
    else:
        return 'danger' if delta_direction == 'up' else 'success'
