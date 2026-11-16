"""Tools the agent can call: knowledge-base search (RAG) and Wikipedia lookup."""
from __future__ import annotations

import requests

import security
from rag import SongKnowledgeBase

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/"
HEADERS = {"User-Agent": "SongExplainer/1.0 (class project; educational use)"}

TOOL_SCHEMAS = [
    {
        "name": "search_knowledge_base",
        "description": (
            "Search SongExplainer's curated notes on songs, artists, protest/OPM music history, "
            "and lyric-writing techniques. Use this FIRST for any song, artist, or songwriting question."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Song title, artist, or topic, e.g. 'Fix You Coldplay'"}},
            "required": ["query"],
        },
    },
    {
        "name": "wikipedia_lookup",
        "description": (
            "Look up a song, album, artist, or music-history topic on Wikipedia to verify facts such as "
            "year, album, artist, and documented background. Use when the knowledge base has no answer."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search phrase, e.g. 'Hotel California song Eagles'"}},
            "required": ["query"],
        },
    },
    {
        "name": "pin_song",
        "description": (
            "Pin the song the user is asking about so it shows in the sidebar 'Song of the Session' card and "
            "follow-up questions stay on it. Call it when the user's question centers on ONE specific song "
            "(not for comparisons or general questions). Use the song's real title and artist."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Song title, e.g. 'Fix You'"},
                "artist": {"type": "string", "description": "Artist or band, e.g. 'Coldplay'"},
            },
            "required": ["title", "artist"],
        },
    },
]


def clean_pin_field(value: str) -> str:
    ok, _, clean = security.check_input(str(value or ""), max_chars=80)
    return clean.replace("<", "").replace(">", "") if ok else ""


def wikipedia_lookup(query: str, timeout: int = 8) -> str:
    try:
        r = requests.get(
            WIKI_API,
            params={"action": "query", "list": "search", "srsearch": query, "srlimit": 1, "format": "json"},
            headers=HEADERS, timeout=timeout,
        )
        r.raise_for_status()
        hits = r.json().get("query", {}).get("search", [])
        if not hits:
            return "No Wikipedia article found for that query."
        title = hits[0]["title"]
        s = requests.get(WIKI_SUMMARY + requests.utils.quote(title.replace(" ", "_")), headers=HEADERS, timeout=timeout)
        s.raise_for_status()
        data = s.json()
        return f"Wikipedia: {data.get('title', title)}\n{data.get('extract', 'No summary available.')}"
    except requests.RequestException:
        return "Wikipedia lookup is unavailable right now. Say you could not verify the detail."


class ToolRunner:
    def __init__(self, kb: SongKnowledgeBase, on_pin=None):
        self.kb = kb
        self.on_pin = on_pin  # callable(title, artist) -> None, updates the sidebar card

    def run(self, name: str, args: dict) -> str:
        args = args or {}
        if name == "pin_song":
            title, artist = clean_pin_field(args.get("title")), clean_pin_field(args.get("artist"))
            if not title:
                return "Could not pin: invalid song title."
            if self.on_pin:
                self.on_pin(title, artist)
            return f"Pinned '{title}' by {artist or 'unknown artist'} for this session."
        query = str(args.get("query", ""))[:200]
        if name == "search_knowledge_base":
            raw = self.kb.format_results(query)
        elif name == "wikipedia_lookup":
            raw = wikipedia_lookup(query)
        else:
            raw = f"Unknown tool: {name}"
        return security.scrub_tool_output(raw)
