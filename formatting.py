"""Helpers for the bot's answer layout (kept out of app.py so they can be unit-tested)."""
import re

# Both the song explanation and the comparison layout start with a Theme label followed by a Meaning label.
_EXPLANATION = re.compile(r"\*\*Theme\s*:?\s*\*\*\s*:?.*\*\*Meaning\s*:?\s*\*\*\s*:?", re.S)


def is_explanation(text: str) -> bool:
    """True for a full song explanation or comparison."""
    return bool(_EXPLANATION.search(text or ""))


def with_line_breaks(text: str) -> str:
    """Markdown ignores single newlines; add hard breaks so each label sits above its text as written."""
    return (text or "").replace("\n", "  \n")
