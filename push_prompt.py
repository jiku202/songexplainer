"""Push the local system prompt to Langfuse as a new version labeled 'production'.

Usage:  LANGFUSE_PUBLIC_KEY=... LANGFUSE_SECRET_KEY=... python scripts/push_prompt.py
Each run creates a new prompt version, giving you version history for your report.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langfuse import get_client  # noqa: E402

import prompts  # noqa: E402

lf = get_client()
lf.create_prompt(name=prompts.PROMPT_NAME, type="text", prompt=prompts.SYSTEM_PROMPT, labels=["production"])
print(f"Pushed '{prompts.PROMPT_NAME}' to Langfuse.")
