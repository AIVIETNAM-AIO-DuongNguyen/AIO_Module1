"""Display helpers for metrics and experiment axes."""


def format_regime(regime: float) -> str:
    """Format a regime fraction as a percentage label."""
    return f"{int(round(float(regime) * 100))}%"


def format_metric(value: float | None, *, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "—"
    return f"{float(value):.{digits}f}"
