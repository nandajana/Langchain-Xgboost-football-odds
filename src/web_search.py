"""
Live search layer: finds relevant links via a search engine (DuckDuckGo, no
API key needed), fetches each page, and extracts readable text so it can be
used as RAG context.

Run: python src/web_search.py "arsenal man city injury news"
"""
import sys
import time

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS

HEADERS = {"User-Agent": "Mozilla/5.0 (research project; contact: you@example.com)"}


def search_links(query, max_results=5):
    """Search the web and return a list of {title, href, body} dicts."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    return results


def fetch_page_text(url, max_chars=3000, timeout=8):
    """Fetch a URL and extract readable text (strips nav/script/style)."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [skip] {url} -> {e}")
        return None

    soup = BeautifulSoup(resp.text, "lxml")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()

    text = " ".join(soup.get_text(separator=" ").split())
    return text[:max_chars] if text else None


def search_and_build_documents(query, max_results=5, delay_between_fetches=0.5):
    """
    Search for `query`, fetch each result page, and return a list of dicts
    {"text": ..., "source": url, "title": ...} ready to feed into a
    retriever later. Pages that fail to fetch are skipped rather than
    crashing the whole batch.
    """
    docs = []
    links = search_links(query, max_results=max_results)
    print(f"Found {len(links)} links for: {query!r}")

    for item in links:
        url = item.get("href") or item.get("link")
        title = item.get("title", "")
        if not url:
            continue

        text = fetch_page_text(url)
        if text:
            docs.append({"text": text, "source": url, "title": title})
            print(f"  [ok]   {title[:70]}  -> {url}")
        time.sleep(delay_between_fetches)  # be polite, avoid rate limits

    return docs


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "arsenal man city injury news"
    documents = search_and_build_documents(q, max_results=5)
    print(f"\nBuilt {len(documents)} documents.")
    for d in documents:
        print(f"\n--- {d['source']} ---")
        print(d["text"][:300], "...")
