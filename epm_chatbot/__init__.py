"""
EPM Chat — a standalone, independently deployable chatbot for EPM.

It reuses EPM's dataset registry, algorithm catalogue and leakage-free
preprocessing pipelines (``src/``) through explicit adapters in
``epm_chatbot.ml_service`` and never imports or modifies the existing website
(``web/``, ``api/``, ``run.py``).
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:  # make the EPM ``src`` package importable when run from anywhere
    sys.path.insert(0, str(_ROOT))

__version__ = "1.0.0"


def require_runtime() -> None:
    """Exit with clear instructions when the Python running us lacks EPM's packages
    (the usual cause: the project's virtual environment is not being used)."""
    import importlib.util

    missing = [m for m in ("numpy", "pandas", "sklearn", "joblib") if importlib.util.find_spec(m) is None]
    if not missing:
        return
    venv_py = _ROOT / "venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    lines = [
        f"EPM Chat can't start: this Python ({sys.executable}) is missing: {', '.join(missing)}.",
        "",
    ]
    if venv_py.exists():
        lines += ["Your project has a virtual environment with these packages. Run EPM Chat with it:",
                  f'    "{venv_py}" -m epm_chatbot.bootstrap',
                  f'    "{venv_py}" -m epm_chatbot',
                  "or activate it first (PowerShell: .\\venv\\Scripts\\Activate.ps1) and run the usual commands."]
    else:
        lines += ["Install the project's packages into this Python first:",
                  f'    "{sys.executable}" -m pip install -r requirements.txt']
    print("\n".join(lines), file=sys.stderr)
    raise SystemExit(1)
