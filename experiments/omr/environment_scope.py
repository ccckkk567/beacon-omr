"""User-selected benchmark scope for site discovery and runtime checks."""
from pathlib import Path


def excluded_environment(path):
    """AppWorld is explicitly excluded, including aliases to its installation."""
    path = Path(path)
    return 'appworld' in str(path).casefold() or 'appworld' in str(path.resolve()).casefold()
