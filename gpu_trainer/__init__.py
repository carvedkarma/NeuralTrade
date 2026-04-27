from __future__ import annotations

import sys
from pathlib import Path

# Compatibility bootstrap:
# Much of the legacy trainer stack imports sibling packages as top-level modules
# (for example: ``from config import config`` or ``from data.pipeline import ...``).
# When the CLI is launched as ``python -m gpu_trainer.research_cli`` or via the
# installed console script, those imports fail unless the gpu_trainer directory
# itself is on sys.path. Adding the package directory here keeps legacy modules
# importable without requiring invasive rewrites across the stack.
_PACKAGE_DIR = Path(__file__).resolve().parent
_PACKAGE_DIR_STR = str(_PACKAGE_DIR)
if _PACKAGE_DIR_STR not in sys.path:
    sys.path.insert(0, _PACKAGE_DIR_STR)
