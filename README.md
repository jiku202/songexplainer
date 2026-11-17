# SongExplainer

*Every song has more than one story.* A Streamlit chatbot that explains song meaning, context, and
multiple valid interpretations in plain language.

## Architecture

| Requirement | Implementation |
|---|---|
| **Agentic AI** | `agent.py` - tool-use loop (max 4 tool steps) on Gemini or Groq with automatic fallback. The model decides when to call `search_knowledge_base`, `wikipedia_lookup`, or `pin_song` (fills the sidebar card), reads results, and answers. |
| **RAG** | `rag.py` + `knowledge_base/*.md` - document loader, paragraph chunking, TF-IDF vector index, cosine-similarity retrieval. Exposed to the agent as a tool (agentic RAG). |
| **Prompt defenses** | `security.py` - input screening, `<user_input>` delimiting + sandwich reminder, tool-output scrubbing (indirect injection), canary token + output guard, copyright quote limit, piracy-link stripping, rate limiting. |
| **Langfuse (optional)** | `observability.py` - traces, generations, and prompt versions. `scripts/push_prompt.py` pushes the prompt as a new version; the app fetches the `production` label at runtime and falls back to `prompts.py`. |

## Run locally

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # add GEMINI_API_KEY (free: aistudio.google.com/apikey)
streamlit run app.py
python -m pytest -q                                          # 67 tests, no API key needed
```

## Deploy (Streamlit Community Cloud)

1. Push this folder to a **public** GitHub repo (`.streamlit/secrets.toml` is git-ignored).
2. On share.streamlit.io, create an app pointing at `app.py`.
3. Paste your secrets into App settings -> Secrets.

## Langfuse setup (for the higher-grade option)

1. Create a free project at cloud.langfuse.com and copy the keys into your secrets.
2. Run `python scripts/push_prompt.py` (re-run after each prompt edit to create a new version).
3. Chat with the app, then open Langfuse to show traces, tool calls, token usage, and prompt versions.

## Extending the knowledge base

Add a markdown file to `knowledge_base/` with a `---` front-matter header (title, artist, year, tags).
Keep notes paraphrased - never paste lyrics. The sample notes were written as brief background
summaries; verify facts you plan to present before your demo.

## Models, cost, and fallback

Two free providers are supported through `agent.py`:

- **Gemini** (`google-genai`, default `gemini-3.5-flash`) - key from aistudio.google.com/apikey
- **Groq** (OpenAI-compatible API, default `openai/gpt-oss-120b`; if the model is missing it asks Groq which models the account can use and switches automatically) - key from console.groq.com

Add one or both keys as secrets. With both, the primary provider (`PROVIDER`, default `gemini`) answers first
and the app automatically retries the turn on the other provider if the first one errors (rate limit, quota,
outage, wrong model name). A small note under the reply tells you when the fallback answered.
If Groq's tool calling fails twice (malformed or unparseable tool call), the Groq backend switches to a
no-tools mode: it runs the knowledge-base search and Wikipedia lookup itself and answers from those notes. If a model's output still can't be parsed, it switches to a different
Groq model family; error messages list which chat models your key can see.
Secrets: `GEMINI_API_KEY`, `GROQ_API_KEY`, `PROVIDER`, `GEMINI_MODEL`, `GROQ_MODEL`.
Free-tier limits change without notice; check each provider's console. Free-tier requests may be used by
the provider to improve its products, so don't paste private information.

## Troubleshooting providers

`python scripts/check_providers.py` tests your Groq and Gemini keys directly and prints which models work
(plain chat and tool calling), plus a recommended `GROQ_MODEL`.

## Wireframe mapping

| Wireframe element | Streamlit implementation |
|---|---|
| Title + tagline + divider | `st.title`, `st.caption`, `st.divider` |
| Chat window, bot greeting first | `st.chat_message` bubbles; greeting is always the first bubble; user bubbles right-aligned via small CSS |
| Starter chips | two `st.button` chips (Explain a song, Compare two songs) shown while the chat is empty; each opens a small `st.form` (song title + artist, or two songs). **Explain like I'm 12** is not a chip: it appears under the latest explanation and rewrites it in simpler words |
| Input bar "Ask about any song..." | `st.chat_input` |
| Sidebar: Song of the Session | extended into a chat history: each chat is named after its song (the `pin_song` tool sets the name, otherwise the first message), New Chat saves the current chat, click a chat to reopen it, trash icon deletes it. Lives in `st.session_state` (kept while the tab stays open, lost on page refresh) |
| Sidebar: Level (Casual / Advanced) | `st.radio` |
| Sidebar: [New Chat] | button at the top of the sidebar (standard chat-app placement); starts a fresh chat without deleting the old one |
