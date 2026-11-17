"""Check which Groq / Gemini models actually work with YOUR keys.

Run from the project folder (with your virtual environment active):
    pip install openai google-genai
    python scripts/check_providers.py

Keys are read from environment variables or from .streamlit/secrets.toml.
It makes a handful of tiny requests, so it uses very little of your free quota.
"""
import os
import sys
from pathlib import Path

NON_CHAT = ("whisper", "tts", "guard", "orpheus", "embed", "compound", "safeguard")
TOOL = [{"type": "function", "function": {
    "name": "wikipedia_lookup", "description": "Look up a song on Wikipedia.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}]


def load_secrets() -> dict:
    path = Path(__file__).resolve().parent.parent / ".streamlit" / "secrets.toml"
    if not path.exists():
        return {}
    try:
        import tomllib
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"(could not read secrets.toml: {e})")
        return {}


def short(e: Exception, n: int = 160) -> str:
    return f"{type(e).__name__}: {' '.join(str(e).split())[:n]}"


def check_groq(key: str) -> None:
    print("\n=== GROQ ===")
    from openai import OpenAI
    client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1", timeout=30)
    try:
        ids = [m.id for m in client.models.list().data]
    except Exception as e:  # noqa: BLE001
        print("Could not list models (bad key?):", short(e))
        return
    chat = [i for i in ids if not any(w in i.lower() for w in NON_CHAT)]
    print("Chat models your key can see:", ", ".join(chat) or "none")
    good = []
    for m in chat[:8]:
        extra = {"reasoning_effort": "low"} if m.startswith("openai/gpt-oss") else {}
        try:
            r = client.chat.completions.create(model=m, max_tokens=300, extra_body=extra,
                                               messages=[{"role": "user", "content": "Reply with the word OK."}])
            plain = "OK (replied: %r)" % (r.choices[0].message.content or "")[:30]
            plain_ok = True
        except Exception as e:  # noqa: BLE001
            plain, plain_ok = "FAIL " + short(e), False
        tool = "-"
        if plain_ok:
            try:
                r = client.chat.completions.create(
                    model=m, max_tokens=400, extra_body=extra, tools=TOOL, tool_choice="auto",
                    messages=[{"role": "user", "content": "Use the tool to look up the song 'Circles' by Post Malone."}])
                calls = r.choices[0].message.tool_calls or []
                tool = "OK (called %s)" % calls[0].function.name if calls else "no tool call made"
                if calls and calls[0].function.name == "wikipedia_lookup":
                    good.append(m)
            except Exception as e:  # noqa: BLE001
                tool = "FAIL " + short(e)
        print(f"- {m}\n    plain chat: {plain}\n    tool call:  {tool}")
    if good:
        print(f"\n>>> Best Groq choice: add  GROQ_MODEL = \"{good[0]}\"  to your secrets")
    else:
        print("\n>>> No Groq model passed the tool-call test. The app can still work in its no-tools mode.")


def check_gemini(key: str, configured: str) -> None:
    print("\n=== GEMINI ===")
    from google import genai
    client = genai.Client(api_key=key)
    for m in dict.fromkeys([configured, "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-3.1-flash-lite"]):
        try:
            r = client.models.generate_content(model=m, contents="Reply with the word OK.")
            print(f"- {m}: OK (replied: {(r.text or '')[:30]!r})")
        except Exception as e:  # noqa: BLE001
            print(f"- {m}: FAIL {short(e, 220)}")


def main() -> None:
    secrets = load_secrets()
    get = lambda k, d="": os.getenv(k) or str(secrets.get(k, d))
    groq, gemini = get("GROQ_API_KEY"), get("GEMINI_API_KEY")
    if not groq and not gemini:
        print("No GROQ_API_KEY or GEMINI_API_KEY found in the environment or .streamlit/secrets.toml.")
        sys.exit(1)
    if groq:
        check_groq(groq)
    else:
        print("\n(no GROQ_API_KEY found, skipping Groq)")
    if gemini:
        check_gemini(gemini, get("GEMINI_MODEL") or get("MODEL", "gemini-3.5-flash"))
    else:
        print("\n(no GEMINI_API_KEY found, skipping Gemini)")


if __name__ == "__main__":
    main()
