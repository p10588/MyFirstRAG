import os

import hashlib
import requests
from bs4 import BeautifulSoup


DATA_PATH = os.getenv("DATA_PATH", "data")
WEB_CRAWL_PATH = os.path.join(DATA_PATH, "web_crawl")
WEBSITE_CRAWLER_URL = os.getenv("WEBSITE_CRAWLER_URL")

os.makedirs(WEB_CRAWL_PATH, exist_ok=True)  # Create directory if not exists

def get_clean_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove unwanted tags like script, style, header, footer, nav, aside
        for script in soup(["script", "style", "header", "footer", "nav", "aside"]):
            script.decompose()

        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines()]
        clean_text = "\n".join(line for line in lines if line)  # Join non-empty lines
        return clean_text
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def file_hash(content):
    # Calculate SHA256 hash of the content string (UTF-8 encoded)
    h = hashlib.sha256()
    h.update(content.encode("utf-8"))
    return h.hexdigest()

def save_text_file(content, filename):
    # Save content to file inside WEB_CRAWL_PATH with UTF-8 encoding
    filepath = os.path.join(WEB_CRAWL_PATH, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return filepath

def crawl_and_save(url, max_files=10):
    clean_text = get_clean_text_from_url(url)
    if not clean_text:
        print("No content extracted.")
        return None

    content_hash = file_hash(clean_text)

    # Check for duplicate content to avoid saving same file multiple times
    existing_hashes = set()
    hash_file = os.path.join(WEB_CRAWL_PATH, "hashes.txt")
    if os.path.exists(hash_file):
        with open(hash_file, "r") as f:
            existing_hashes = set(line.strip() for line in f.readlines())

    if content_hash in existing_hashes:
        print("Duplicate content found, skipping save.")
        return None

    # Use first 12 chars of hash as filename to ensure uniqueness
    filename = f"{content_hash[:12]}.txt"
    filepath = save_text_file(clean_text, filename)
    print(f"Saved cleaned text to {filepath}")

    # Append new hash to record file for future duplicate checks
    with open(hash_file, "a") as f:
        f.write(content_hash + "\n")

    return filepath


def main():
    crawl_and_save(WEBSITE_CRAWLER_URL)

main()