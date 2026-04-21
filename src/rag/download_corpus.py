"""
download_corpus.py — Fetch real SEC filings and save them as plain text.

SEC EDGAR requires a User-Agent header on every request; requests without
one return HTTP 403. We also strip HTML tags with BeautifulSoup so the rest
of the pipeline works with clean text.
"""

import os
import re
import requests
from bs4 import BeautifulSoup

# The directory where plain-text files are saved.
CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "corpus")

# Each entry: (url, output_filename)
FILINGS = [
    (
        "https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm",
        "apple_10k_2023.txt",
    ),
    (
        "https://www.sec.gov/Archives/edgar/data/789019/000095017023035122/msft-20230630.htm",
        "microsoft_10k_2023.txt",
    ),
    (
        "https://www.sec.gov/Archives/edgar/data/1318605/000162828024002390/tsla-20231231.htm",
        "tesla_10k_2023.txt",
    ),
]

# SEC requires a descriptive User-Agent. Requests without one are blocked.
HEADERS = {"User-Agent": "Learning RAG Course contact@example.com"}


def strip_html(html: str) -> str:
    """Parse HTML and return clean plain text with collapsed whitespace."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove script and style blocks — they add noise but no meaning.
    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(separator=" ")

    # Collapse runs of whitespace (spaces, tabs, newlines) to a single space,
    # then strip leading/trailing whitespace.
    text = re.sub(r"\s+", " ", text).strip()
    return text


def download_filing(url: str, output_path: str) -> None:
    """Download one SEC filing URL, strip HTML, and write plain text to disk."""
    filename = os.path.basename(output_path)

    # Skip if already downloaded so re-running the script is safe (idempotent).
    if os.path.exists(output_path):
        size = os.path.getsize(output_path)
        print(f"  [skip] {filename} already exists ({size:,} bytes)")
        return

    print(f"  Downloading {filename} …")
    response = requests.get(url, headers=HEADERS, timeout=60)
    response.raise_for_status()  # Raise an exception for 4xx/5xx responses.

    text = strip_html(response.text)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"  [done]  {filename} — {len(text):,} characters")


def main() -> None:
    print("Downloading SEC corpus …\n")
    for url, filename in FILINGS:
        output_path = os.path.normpath(os.path.join(CORPUS_DIR, filename))
        download_filing(url, output_path)

    # Verify and print final sizes.
    print("\nCorpus summary:")
    for _, filename in FILINGS:
        path = os.path.normpath(os.path.join(CORPUS_DIR, filename))
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"  {filename}: {size:,} bytes")
        else:
            print(f"  {filename}: MISSING")


if __name__ == "__main__":
    main()
