"""Launch the Streamlit experiment dashboard.

Usage (from repo root, with the virtual environment active):

    python scripts/run_ui.py
"""

import os
import subprocess
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_APP_PATH = os.path.join(_REPO_ROOT, "src", "ui", "app.py")


def main() -> None:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = _REPO_ROOT if not existing else f"{_REPO_ROOT}{os.pathsep}{existing}"
    raise SystemExit(
        subprocess.call(
            [sys.executable, "-m", "streamlit", "run", _APP_PATH],
            cwd=_REPO_ROOT,
            env=env,
        )
    )


if __name__ == "__main__":
    main()
