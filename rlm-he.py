#!/usr/bin/env python3
"""
rlm-he.py — RLM Code עם ממשק עברי
בסיס: https://github.com/SuperagenticAI/rlm-code

הרצה:
    python rlm-he.py

דרישות:
    uv tool install "rlm-code[tui,llm-all]"
"""

import sys
from pathlib import Path

HERE = Path(__file__).parent

# --- טעינת site-packages של rlm-code ---
for candidate in [
    Path.home() / ".local/share/uv/tools/rlm-code/lib/python3.11/site-packages",
    Path.home() / ".local/share/uv/tools/rlm-code/lib/python3.12/site-packages",
]:
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
        break

# --- הוסף תיקיית הפרויקט לנתיב ---
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# --- טען .env אוטומטית אם קיים ---
_env = HERE / ".env"
if _env.exists():
    import os
    for line in _env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

# --- החלף מודולי UI במודולים העבריים ---
import importlib
sys.modules["rlm_code.ui.welcome"] = importlib.import_module("hebrew.welcome")
sys.modules["rlm_code.ui.prompts"] = importlib.import_module("hebrew.prompts")

# --- הפעל ---
from rlm_code.main import main
main()
