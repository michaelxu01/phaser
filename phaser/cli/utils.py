import re


def natural_key(val: object) -> list[int | str]:
    """Sort key which orders embedded numbers numerically ('acquisition_2' before 'acquisition_10')."""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', str(val))]
