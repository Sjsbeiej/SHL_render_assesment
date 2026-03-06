"""
SHL Product Catalogue Scraper
Scrapes Individual Test Solutions from SHL's product catalogue.
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup

import config

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

# Test type code to full name mapping
TEST_TYPE_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behaviour",
    "S": "Simulations",
}


def _parse_table_rows(table) -> list[dict]:
    """Parse assessment rows from a catalogue table."""
    assessments = []
    rows = table.find_all("tr")[1:]  # skip header row

    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 4:
            continue

        link = cells[0].find("a")
        if not link:
            continue
        name = link.get_text(strip=True)
        href = link.get("href", "")
        full_url = f"https://www.shl.com{href}" if href.startswith("/") else href

        remote = bool(cells[1].find("span", class_="-yes"))
        adaptive = bool(cells[2].find("span", class_="-yes"))

        type_spans = cells[3].find_all("span", class_="product-catalogue__key")
        type_codes = [s.get_text(strip=True) for s in type_spans]
        test_types = [TEST_TYPE_MAP.get(c, c) for c in type_codes]

        assessments.append({
            "name": name,
            "url": full_url,
            "remote_testing": remote,
            "adaptive_irt": adaptive,
            "test_type_codes": type_codes,
            "test_types": test_types,
        })

    return assessments


def _fetch_page(url: str) -> BeautifulSoup | None:
    """Fetch and parse a catalogue page."""
    for attempt in range(config.MAX_RETRIES):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "html.parser")
        except requests.RequestException as e:
            if attempt == config.MAX_RETRIES - 1:
                print(f"  Failed to fetch {url}: {e}")
                return None
            time.sleep(2 ** attempt)


def get_catalogue_page(start: int, catalogue_type: int = 1) -> list[dict]:
    """Scrape one page of the catalogue.

    catalogue_type=1 paginates Individual Test Solutions (up to start=372).
    catalogue_type=2 paginates Pre-packaged Job Solutions (up to start=132).
    At start=0, both tables appear; at start>=12, only the paginated table shows.
    """
    url = f"{config.SHL_CATALOG_BASE}?start={start}&type={catalogue_type}"
    soup = _fetch_page(url)
    if not soup:
        return []

    tables = soup.find_all("table")
    if not tables:
        return []

    # Determine which table to use
    if start == 0 and len(tables) >= 2:
        # At start=0, table[0]=Pre-packaged, table[1]=Individual
        target_table = tables[1] if catalogue_type == 1 else tables[0]
    else:
        # At start>=12, only one table shows
        target_table = tables[0]

    return _parse_table_rows(target_table)


def scrape_detail_page(url: str) -> dict:
    """Scrape additional details from an individual assessment page."""
    for attempt in range(config.MAX_RETRIES):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt == config.MAX_RETRIES - 1:
                return {}
            time.sleep(2 ** attempt)

    soup = BeautifulSoup(resp.text, "html.parser")
    details = {}

    paragraphs = soup.find_all("p")
    all_text = []

    for p in paragraphs:
        text = p.get_text(strip=True)
        if not text:
            continue
        all_text.append(text)

        # Extract duration
        if "Completion Time" in text:
            match = re.search(r"(\d+)", text)
            if match:
                details["duration_minutes"] = int(match.group(1))

        # Extract job levels
        level_keywords = [
            "Entry-Level", "Mid-Professional", "Professional Individual Contributor",
            "Manager", "Director", "Executive", "Graduate", "Supervisor",
            "Front Line Manager", "Senior Manager"
        ]
        if any(kw.lower() in text.lower() for kw in level_keywords):
            details["job_levels"] = text.strip().rstrip(",")

        # Extract languages
        if "English" in text or "Spanish" in text or "French" in text:
            if len(text) < 200 and "Completion" not in text:
                details.setdefault("languages", text.strip().rstrip(","))

    # Try full paragraph text first (meta descriptions are often truncated)
    skip_phrases = ["recommend upgrading", "modern browser", "browser options",
                    "choose to continue", "Accelerate", "Speak to our team",
                    "Book a Demo", "SHL and its affiliates", "Product Fact Sheet"]
    desc_paragraphs = [t for t in all_text if len(t) > 20
                       and "Completion Time" not in t
                       and "Test Type:" not in t
                       and "Remote Testing:" not in t
                       and not any(skip in t for skip in skip_phrases)]
    if desc_paragraphs:
        details["description"] = desc_paragraphs[0]

    # Fallback to meta description if no paragraph found
    if "description" not in details:
        meta = soup.find("meta", attrs={"name": "description"})
        if meta and meta.get("content"):
            meta_desc = meta["content"].strip()
            if ":" in meta_desc:
                meta_desc = meta_desc.split(":", 1)[1].strip()
            if len(meta_desc) > 20 and "Browse through" not in meta_desc:
                details["description"] = meta_desc

    return details


def scrape_all_assessments() -> list[dict]:
    """Scrape all assessments from the SHL catalogue (Individual + Pre-packaged)."""
    all_assessments = []

    # Scrape Individual Test Solutions (type=1, up to start=384)
    print("Phase 1a: Scraping Individual Test Solutions...")
    start = 0
    while True:
        print(f"  Fetching Individual page start={start}...")
        page_assessments = get_catalogue_page(start, catalogue_type=1)
        if not page_assessments:
            break
        for a in page_assessments:
            a["category"] = "Individual Test Solutions"
        all_assessments.extend(page_assessments)
        start += 12
        time.sleep(0.3)
    individual_count = len(all_assessments)
    print(f"  Found {individual_count} Individual Test Solutions")

    # Scrape Pre-packaged Job Solutions (type=2, up to start=132)
    print("\nPhase 1b: Scraping Pre-packaged Job Solutions...")
    start = 0
    while True:
        print(f"  Fetching Pre-packaged page start={start}...")
        page_assessments = get_catalogue_page(start, catalogue_type=2)
        if not page_assessments:
            break
        for a in page_assessments:
            a["category"] = "Pre-packaged Job Solutions"
        all_assessments.extend(page_assessments)
        start += 12
        time.sleep(0.3)
    print(f"  Found {len(all_assessments) - individual_count} Pre-packaged Job Solutions")

    print(f"\nTotal found: {len(all_assessments)} assessments from listing pages.")

    # Deduplicate by URL
    seen_urls = set()
    unique = []
    for a in all_assessments:
        if a["url"] not in seen_urls:
            seen_urls.add(a["url"])
            unique.append(a)
    all_assessments = unique
    print(f"After deduplication: {len(all_assessments)} unique assessments.")

    print("\nPhase 2: Scraping detail pages (concurrent)...")
    done = 0

    def fetch_detail(assessment):
        details = scrape_detail_page(assessment["url"])
        assessment.update(details)
        return assessment["name"]

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_detail, a): a for a in all_assessments}
        for future in as_completed(futures):
            done += 1
            if done % 50 == 0 or done == len(all_assessments):
                print(f"  Detail pages: {done}/{len(all_assessments)}")

    return all_assessments


def save_assessments(assessments: list[dict]):
    """Save scraped assessments to JSON."""
    os.makedirs(config.DATA_DIR, exist_ok=True)
    with open(config.ASSESSMENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(assessments, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(assessments)} assessments to {config.ASSESSMENTS_FILE}")


if __name__ == "__main__":
    assessments = scrape_all_assessments()
    save_assessments(assessments)

    # Summary
    print(f"\n=== Scraping Summary ===")
    print(f"Total assessments: {len(assessments)}")
    with_desc = sum(1 for a in assessments if a.get("description"))
    with_duration = sum(1 for a in assessments if a.get("duration_minutes"))
    print(f"With description: {with_desc}")
    print(f"With duration: {with_duration}")

    # Test type distribution
    from collections import Counter
    type_counter = Counter()
    for a in assessments:
        for t in a.get("test_type_codes", []):
            type_counter[t] += 1
    print(f"Test type distribution: {dict(type_counter)}")
