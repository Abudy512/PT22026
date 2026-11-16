"""Shared pytest configuration.

The project is imported from its root directory (``alm042``).  Adding that
folder to ``sys.path`` here means the tests work no matter which directory
pytest is started from.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
