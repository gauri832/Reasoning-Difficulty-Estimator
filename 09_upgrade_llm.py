"""Compatibility wrapper.

Use:
  python scripts/09_upgrade_llm.py

This file is kept so `python 09_upgrade_llm.py` still works.
"""

from pathlib import Path
import runpy


if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "scripts" / "09_upgrade_llm.py"
    runpy.run_path(str(target), run_name="__main__")
