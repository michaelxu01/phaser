import re

from rich.theme import Theme


CONSOLE_THEME = Theme({
    'warning': 'bold yellow',
    'error': 'bold red',
    'info': 'blue',
    'prompt.invalid': 'bold red',
})
"""Shared theme for CLI consoles, defining the 'warning', 'error', and 'info' markup styles"""


def natural_key(val: object) -> list[int | str]:
    """Sort key which orders embedded numbers numerically ('acquisition_2' before 'acquisition_10')."""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', str(val))]
