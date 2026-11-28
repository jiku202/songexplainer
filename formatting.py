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


# A thin line between topics, with blank space around it. Change this one constant to restyle the dividers
# (use "\n\n" for spacing only, or "\n\n---\n\n" for a visible line).
SEPARATOR = "\n\n---\n\n"
# A label line such as "**Theme:**" or "**Overall Similarities**:" (the colon is what tells it apart from the title).
_LABEL = re.compile(r"^\*\*[^*\n]+?(?::\*\*|\*\*:)\s*$")


def format_reply(text: str) -> str:
    """Prepare a bot reply for display. Full explanations and comparisons are split into their topics
    (title block, Theme, Meaning, ...) with a divider between each; other replies only keep their line breaks."""
    text = (text or "").strip()
    if not is_explanation(text):
        return with_line_breaks(text)
    blocks: list[list[str]] = [[]]
    for line in text.split("\n"):
        if _LABEL.match(line.strip()) and any(l.strip() for l in blocks[-1]):
            blocks.append([])
        blocks[-1].append(line.rstrip())
    shown = []
    for lines in blocks:
        block = "\n".join(lines).strip()
        if not block:
            continue
        block = re.sub(r"\n(1\. )", r"\n\n\1", block, count=1)   # a numbered list needs a blank line before it
        shown.append(with_line_breaks(block))
    return SEPARATOR.join(shown)
