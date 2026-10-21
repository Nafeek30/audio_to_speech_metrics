"""Display percentage errors without altering analytical values."""

def percentage_error(value):
    """Use a 0.10% display floor for positive errors; retain exact zero."""
    if value < 0:
        raise ValueError('Absolute percentage errors cannot be negative.')
    displayed = max(value, 0.1) if value > 0 else 0.0
    return f'{displayed:.2f}%'
