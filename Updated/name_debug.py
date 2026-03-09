"""
NAME DEBUGGER — Facebook Profile
Run this to see exactly what the scraper sees on any profile page.
It shows ALL candidate names found so you can tune the detection.

Usage:
    python name_debug.py
"""

import time
import os
import warnings
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# ── Same blacklist as main scraper ─────────────────────────────
NAME_BLACKLIST = [
    "find friends", "log in", "sign up", "facebook", "messenger",
    "create account", "home", "watch", "marketplace", "groups",
    "gaming", "more", "create", "add friend", "message", "follow",
    "see more", "mutual friends", "pages", "people", "events",
    "timeline", "about", "friends", "photos", "videos", "check-ins",
    "sports", "music", "movies", "tv shows", "books", "liked pages",
    "reels", "notifications", "search", "profile", "settings",
]


def _is_valid_name(text: str) -> bool:
    if not text or len(text) < 3 or len(text) > 80:
        return False
    text_lower = text.lower().strip()
    if any(bad == text_lower or text_lower.startswith(bad) for bad in NAME_BLACKLIST):
        return False
    if not text[0].isalpha():
        return False
    if sum(c.isdigit() for c in text) / len(text) > 0.5:
        return False
    return True


def launch_browser():
    options = Options()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-notifications")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument("--disable-blink-features=AutomationControlled")
    service = Service(ChromeDriverManager().install())
    driver  = webdriver.Chrome(service=service, options=options)
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"}
    )
    return driver


def debug_name(driver, url: str):
    print(f"\n{'='*60}")
    print(f"URL: {url}")
    print(f"{'='*60}")

    driver.get(url)
    WebDriverWait(driver, 15).until(
        EC.presence_of_element_located((By.TAG_NAME, "body"))
    )
    time.sleep(3)

    # ── 1. Page Title ──────────────────────────────────────────
    title = driver.title.strip()
    print(f"\n[1] PAGE TITLE:  '{title}'")
    if "|" in title:
        candidate = title.split("|")[0].strip()
        valid = _is_valid_name(candidate)
        print(f"    Extracted:   '{candidate}'  -->  {'VALID NAME' if valid else 'REJECTED'}")

    # ── 2. All <h1> tags ───────────────────────────────────────
    print(f"\n[2] ALL <h1> TAGS:")
    h1s = driver.find_elements(By.TAG_NAME, "h1")
    if not h1s:
        print("    (none found)")
    for i, el in enumerate(h1s):
        text = el.text.strip()
        if text:
            valid = _is_valid_name(text)
            print(f"    h1[{i}]: '{text}'  -->  {'VALID NAME' if valid else 'rejected'}")

    # ── 3. All <h2> tags ───────────────────────────────────────
    print(f"\n[3] ALL <h2> TAGS:")
    h2s = driver.find_elements(By.TAG_NAME, "h2")
    shown = 0
    for el in h2s:
        text = el.text.strip()
        if text and shown < 5:
            valid = _is_valid_name(text)
            print(f"    h2: '{text}'  -->  {'VALID NAME' if valid else 'rejected'}")
            shown += 1

    # ── 4. span[dir='auto'] — Facebook name spans ─────────────
    print(f"\n[4] SPAN[dir='auto'] (first 10):")
    spans = driver.find_elements(By.CSS_SELECTOR, "span[dir='auto']")
    shown = 0
    for el in spans:
        text = el.text.strip()
        if text and shown < 10:
            valid = _is_valid_name(text)
            print(f"    span: '{text}'  -->  {'VALID NAME' if valid else 'rejected'}")
            shown += 1

    # ── 5. What the scraper would actually pick ────────────────
    print(f"\n[5] WHAT SCRAPER PICKS:")

    # Method 1: title
    picked = None
    if "|" in title:
        c = title.split("|")[0].strip()
        if _is_valid_name(c):
            picked = f"'{c}'  (from page title)"

    # Method 2: h1
    if not picked:
        for el in driver.find_elements(By.TAG_NAME, "h1"):
            text = el.text.strip()
            if _is_valid_name(text):
                picked = f"'{text}'  (from <h1>)"
                break

    # Method 3: span
    if not picked:
        for el in driver.find_elements(By.CSS_SELECTOR, "span[dir='auto']"):
            text = el.text.strip()
            if _is_valid_name(text):
                picked = f"'{text}'  (from span)"
                break

    print(f"    --> {picked if picked else 'NOTHING FOUND — would save as Unknown'}")

    # ── 6. First 300 chars of page text (for context) ─────────
    print(f"\n[6] FIRST 300 CHARS OF PAGE TEXT:")
    body_text = driver.find_element(By.TAG_NAME, "body").text
    print(f"    {repr(body_text[:300])}")

    print(f"\n{'='*60}\n")


def main():
    print("\nFACEBOOK NAME DEBUGGER")
    print("="*60)
    print("This tool shows exactly what the scraper sees on a profile.")
    print("Use it to diagnose why a name is being missed or wrong.\n")

    driver = launch_browser()
    driver.get("https://www.facebook.com")

    print("Log in to Facebook in the Chrome window.")
    input("Press ENTER when you are on your home feed...\n")

    while True:
        url = input("Enter a Facebook profile URL (or 'q' to quit): ").strip()
        if url.lower() == 'q':
            break
        if not url.startswith("http"):
            print("Please enter a full URL starting with https://")
            continue
        try:
            debug_name(driver, url)
        except Exception as e:
            print(f"Error: {e}")

    driver.quit()
    print("\nDone.")


if __name__ == "__main__":
    main()