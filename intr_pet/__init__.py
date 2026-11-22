"""Our project package: data pipeline and model wrapper around the official INTR code.

Importing this package puts third_party/INTR on sys.path, so the authors'
modules (`models`, `util`, `datasets`) can be imported unmodified.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTR_DIR = PROJECT_ROOT / "third_party" / "INTR"
if str(INTR_DIR) not in sys.path:
    sys.path.insert(0, str(INTR_DIR))
