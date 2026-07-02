"""Put the repository root on ``sys.path`` for Streamlit entry points.

``streamlit run src/ui/app.py`` executes scripts with ``src/ui`` on the path, not
the repo root, so absolute ``src.*`` imports need this bootstrap first.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def ensure_repo_on_path() -> Path:
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return REPO_ROOT
