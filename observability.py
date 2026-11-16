"""Optional Langfuse integration. Every helper degrades to a no-op if Langfuse is not
installed or not configured, so the app runs the same with or without it."""
from __future__ import annotations

import functools
import os

import prompts

try:  # pragma: no cover - depends on environment
    from langfuse import get_client
    from langfuse import observe as _lf_observe
    _IMPORTED = True
except Exception:  # noqa: BLE001
    _IMPORTED = False
    _lf_observe = None

    def get_client():  # type: ignore[no-redef]
        return None


def observe(**kwargs):
    """Lazy Langfuse @observe: only wraps the function when Langfuse keys are present at call time,
    so the app never touches Langfuse (or prints auth errors) when tracing is not configured."""
    def deco(fn):
        real = None

        @functools.wraps(fn)
        def wrapper(*a, **k):
            nonlocal real
            if enabled():
                if real is None:
                    real = _lf_observe(**kwargs)(fn)
                return real(*a, **k)
            return fn(*a, **k)
        return wrapper
    return deco


def enabled() -> bool:
    return _IMPORTED and bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def load_system_prompt(canary: str) -> tuple[str, str]:
    """Return (compiled_prompt, source). Fetches the 'production' version from Langfuse
    when available, otherwise falls back to the local copy in prompts.py."""
    if enabled():
        try:
            lf = get_client()
            p = lf.get_prompt(prompts.PROMPT_NAME, label="production")
            return p.compile(canary=canary), f"langfuse:v{p.version}"
        except Exception:  # noqa: BLE001
            pass
    return prompts.compile_local(canary), "local"


def update_trace(**kwargs) -> None:
    if not enabled():
        return
    try:
        get_client().update_current_trace(**kwargs)
    except Exception:  # noqa: BLE001
        pass


def record_generation(**kwargs) -> None:
    if not enabled():
        return
    try:
        get_client().update_current_generation(**kwargs)
    except Exception:  # noqa: BLE001
        pass


def flush() -> None:
    if enabled():
        try:
            get_client().flush()
        except Exception:  # noqa: BLE001
            pass
