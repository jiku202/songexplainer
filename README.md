# SongExplainer

*Every song has more than one story.* A Streamlit chatbot that explains song meaning, context, and
multiple valid interpretations in plain language.

## Architecture

| Requirement | Implementation |
|---|---|
| **Agentic AI** | `agent.py` - tool-use loop (max 4 tool steps). Gemini decides when to call `search_knowledge_base`, `wikipedia_lookup`, or `pin_song` (fills the sidebar card), reads results, and answers. |
| **RAG** | `rag.py` + `knowledge_base/*.md` - document loader, paragraph chunking, TF-IDF vector index, cosine-similarity retrieval. Exposed to the agent as a tool (agentic RAG). |
| **Prompt defenses** | `security.py` - input screening, `<user_input>` delimiting + sandwich reminder, tool-output scrubbing (indirect injection), canary token + output guard, copyright quote limit, piracy-link stripping, rate limiting. |
| **Langfuse (optional)** | `observability.py` - traces, generations, and prompt versions. `scripts/push_prompt.py` pushes the prompt as a new version; the app fetches the `production` label at runtime and falls back to `prompts.py`. |

## Run locally

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # add GEMINI_API_KEY (free: aistudio.google.com/apikey)
streamlit run app.py
python -m pytest -q                                          # 35 tests, no API key needed
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

## Model and cost

The app uses Google's Gemini API through the `google-genai` SDK. The default model is `gemini-3.5-flash`
(override with a `MODEL` secret). Gemini API keys from Google AI Studio work on a free tier with rate limits;
check the limits shown in your AI Studio console. Free-tier requests may be used by Google to improve its products,
so don't paste private information. All model calls live in `agent.py::call_model`.

## Wireframe mapping

| Wireframe element | Streamlit implementation |
|---|---|
| Title + tagline + divider | `st.title`, `st.caption`, `st.divider` |
| Chat window, bot greeting first | `st.chat_message` bubbles; greeting is always the first bubble; user bubbles right-aligned via small CSS |
| Starter chips (Explain a song / Compare two songs / Explain like I'm 12) | three `st.button`s below the chat, shown only while the chat is empty |
| Input bar "Ask about any song..." | `st.chat_input` |
| Sidebar: Song of the Session card + [Clear pin] | bordered container showing title/artist; `pin_song` tool fills it; Clear pin button |
| Sidebar: Level (Casual / Advanced) | `st.radio` |
| Sidebar: [New Chat] | button at the bottom of the sidebar; resets chat, pin, and canary |
