import time
import os
import re
import pandas as pd
import numpy as np

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    WebDriverException
)
from webdriver_manager.chrome import ChromeDriverManager

# ══════════════════════════════════════════════════════════════
#  LOAD ML MODEL (optional — scraper works without it too)
# ══════════════════════════════════════════════════════════════

# Always look for pkl files in the SAME folder as this script
# (fixes the issue where running from a different directory causes FileNotFoundError)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(SCRIPT_DIR, 'fake_profile_model.pkl')
LE_PATH     = os.path.join(SCRIPT_DIR, 'label_encoder.pkl')

MODEL_AVAILABLE = False
try:
    import joblib
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # suppress sklearn version warnings
        model = joblib.load(MODEL_PATH)
        le    = joblib.load(LE_PATH)
    MODEL_AVAILABLE = True
    print("Model loaded - predictions ON.")
except FileNotFoundError:
    print(f"ERROR: Model files not found in: {SCRIPT_DIR}")
    print("Place fake_profile_model.pkl and label_encoder.pkl in the same folder as fb_scraper.py\n")
except Exception as e:
    print(f"ERROR loading model: {e}")
    print("Tip: run this to fix version mismatch:  pip install scikit-learn==1.6.1\n")


def build_features(df_input):
    """
    MUST match build_features() from the Kaggle notebook exactly.
    Same 11 features, same order — critical for correct predictions.
    """
    df = df_input.copy()
    df['name'] = df['name'].fillna('Unknown')

    binary_map = {'Yes': 1, 'No': 0}
    for col in ['has_profile_pic', 'has_profile_details']:
        if col in df.columns:
            df[col] = df[col].map(binary_map).fillna(0).astype(int)

    df['log_followers']  = np.log1p(df['followers'])
    df['log_following']  = np.log1p(df['following'])
    df['log_friends']    = np.log1p(df['friends'])
    df['social_ratio']   = df['followers'] / (df['following'] + 1)
    df['friend_ratio']   = df['friends']   / (df['followers'] + 1)
    df['name_len']       = df['name'].str.len()
    df['digit_count']    = df['name'].apply(lambda x: sum(c.isdigit() for c in str(x)))
    df['digit_ratio']    = df['digit_count'] / (df['name_len'] + 1)
    df['has_space']      = df['name'].str.contains(' ').astype(int)
    df['humanity_score'] = df['has_profile_pic'] + df['has_profile_details']

    # Exact same order as Kaggle notebook build_features()
    features = [
        'has_profile_pic',
        'has_profile_details',
        'log_followers',
        'log_following',
        'log_friends',
        'social_ratio',
        'friend_ratio',
        'name_len',
        'digit_ratio',
        'has_space',
        'humanity_score',
    ]
    return df[features]


def predict_profile(data: dict) -> str:
    """Returns 'Real', 'Fake', 'Suspicious', or 'N/A'."""
    if not MODEL_AVAILABLE:
        print("  [Prediction skipped: model not available]")
        return "N/A"
    try:
        row      = pd.DataFrame([data])
        features = build_features(row)
        print(f"  [DEBUG] Features shape: {features.shape}, columns: {list(features.columns)}")
        print(f"  [DEBUG] Model expects:  {model.n_features_in_} features")
        probs    = model.predict_proba(features)[0]
        label    = le.classes_[np.argmax(probs)]
        print(f"  [DEBUG] Probs: {dict(zip(le.classes_, [round(p,2) for p in probs]))}")
        return label
    except Exception as e:
        import traceback
        print(f"  [Prediction error: {e}]")
        traceback.print_exc()
        return "N/A"


# ══════════════════════════════════════════════════════════════
#  BROWSER SETUP
# ══════════════════════════════════════════════════════════════

def launch_browser():
    options = Options()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-popup-blocking")
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


# ══════════════════════════════════════════════════════════════
#  SCRAPING HELPERS
# ══════════════════════════════════════════════════════════════

# Phrases that mean the profile is locked / private
LOCKED_PHRASES = [
    "only her friends can see",
    "only his friends can see",
    "only their friends can see",
    "only friends can see",
    "add as friend to see",
]

# Facebook UI strings that are NOT profile names
NAME_BLACKLIST = [
    "find friends", "log in", "sign up", "facebook", "messenger",
    "create account", "home", "watch", "marketplace", "groups",
    "gaming", "more", "create", "add friend", "message", "follow",
    "see more", "mutual friends", "pages", "people", "events",
    "timeline", "about", "friends", "photos", "videos", "check-ins",
    "sports", "music", "movies", "tv shows", "books", "liked pages",
    "reels", "notifications", "search", "profile", "settings",
]


def get_page_text(driver) -> str:
    try:
        return driver.find_element(By.TAG_NAME, "body").text
    except Exception:
        return ""


def is_locked_profile(page_text: str) -> bool:
    text_lower = page_text.lower()
    return any(phrase in text_lower for phrase in LOCKED_PHRASES)


def parse_count(raw: str) -> int:
    """'1.2K' -> 1200, '4,500' -> 4500, '2M' -> 2000000"""
    if not raw:
        return 0
    raw = raw.replace(",", "").strip()
    try:
        if raw.upper().endswith("K"):
            return int(float(raw[:-1]) * 1_000)
        elif raw.upper().endswith("M"):
            return int(float(raw[:-1]) * 1_000_000)
        m = re.search(r"[\d.]+", raw)
        return int(float(m.group())) if m else 0
    except Exception:
        return 0


def _is_valid_name(text: str) -> bool:
    """Return True if the string looks like a real person/page name."""
    if not text or len(text) < 3 or len(text) > 80:
        return False
    text_lower = text.lower().strip()
    if any(bad == text_lower or text_lower.startswith(bad) for bad in NAME_BLACKLIST):
        return False
    if not text[0].isalpha():
        return False
    # Reject if more than half the characters are digits
    if sum(c.isdigit() for c in text) / len(text) > 0.5:
        return False
    return True


def scrape_profile_name(driver) -> str:
    """Get the profile owner's real name."""

    # Method 1: page <title> is the most reliable  e.g. "Isuru Lokuhettiarachchi | Facebook"
    try:
        title = driver.title.strip()
        if "|" in title:
            candidate = title.split("|")[0].strip()
            if _is_valid_name(candidate):
                return candidate
    except Exception:
        pass

    # Method 2: <h1> tag
    try:
        for el in driver.find_elements(By.TAG_NAME, "h1"):
            text = el.text.strip()
            if _is_valid_name(text):
                return text
    except Exception:
        pass

    # Method 3: span[dir='auto'] — Facebook name spans
    try:
        for el in driver.find_elements(By.CSS_SELECTOR, "span[dir='auto']"):
            text = el.text.strip()
            if _is_valid_name(text):
                return text
    except Exception:
        pass

    return "Unknown"


def scrape_counts(page_text: str) -> dict:
    """Extract follower / following / friend counts from page text."""
    counts = {"followers": 0, "following": 0, "friends": 0}
    patterns = {
        "followers": r"([\d,\.]+\s*[KkMm]?)\s*[Ff]ollower",
        "following": r"([\d,\.]+\s*[KkMm]?)\s*[Ff]ollowing",
        "friends":   r"([\d,\.]+\s*[KkMm]?)\s*[Ff]riend",
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, page_text)
        if m:
            counts[key] = parse_count(m.group(1).strip())
    return counts


def scrape_has_profile_pic(driver) -> str:
    """Check if profile has an actual (non-default) profile picture."""
    try:
        imgs = driver.find_elements(By.CSS_SELECTOR, "image[preserveAspectRatio]")
        for img in imgs:
            href = img.get_attribute("xlink:href") or img.get_attribute("href") or ""
            if href and "static" not in href.lower() and href.startswith("http"):
                return "Yes"
        avatars = driver.find_elements(
            By.XPATH,
            "//img[contains(@alt,'profile picture') or contains(@alt,'Profile picture')]"
        )
        if avatars:
            return "Yes"
    except Exception:
        pass
    return "No"


def scrape_has_details(page_text: str) -> str:
    """
    Check if the profile has account details by looking for the
    section headings visible in the left sidebar of a Facebook profile:
    Contact info, Personal details, Work, Lives in.
    """
    signals = [
        "contact info",
        "personal details",
        "work",
        "lives in",
        "lives ",
    ]
    text_lower = page_text.lower()
    return "Yes" if any(s in text_lower for s in signals) else "No"


# ══════════════════════════════════════════════════════════════
#  PROFILE SCRAPER  — returns exactly the 8 columns you want
# ══════════════════════════════════════════════════════════════

def scrape_profile(driver, url: str) -> dict:
    result = {
        "url":                  url,
        "name":                 "Unknown",
        "has_profile_pic":      "No",
        "followers":            0,
        "following":            0,
        "friends":              0,
        "has_profile_details":  "No",
        "Fake/Real/Suspicious": "N/A",
    }

    try:
        driver.get(url)
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        time.sleep(3)  # Let dynamic content finish loading

        page_text = get_page_text(driver)

        # ── Locked profile ────────────────────────────────────
        if is_locked_profile(page_text):
            result["Fake/Real/Suspicious"] = "Locked Profile"
            print("  >> LOCKED — Cannot analyze (friends only)")
            return result

        # ── Page not found ────────────────────────────────────
        if "page not found" in page_text.lower()[:300]:
            result["Fake/Real/Suspicious"] = "Profile Not Found"
            print("  >> Profile not found or deleted")
            return result

        # ── Scrape ────────────────────────────────────────────
        result["name"]                = scrape_profile_name(driver)
        result["has_profile_pic"]     = scrape_has_profile_pic(driver)
        counts                        = scrape_counts(page_text)
        result["followers"]           = counts["followers"]
        result["following"]           = counts["following"]
        result["friends"]             = counts["friends"]
        result["has_profile_details"] = scrape_has_details(page_text)

        # ── Predict ───────────────────────────────────────────
        result["Fake/Real/Suspicious"] = predict_profile({
            "name":                result["name"],
            "followers":           result["followers"],
            "following":           result["following"],
            "friends":             result["friends"],
            "has_profile_pic":     result["has_profile_pic"],
            "has_profile_details": result["has_profile_details"],
        })

        print(f"  Name:       {result['name']}")
        print(f"  Followers:  {result['followers']}  |  "
              f"Following: {result['following']}  |  "
              f"Friends: {result['friends']}")
        print(f"  Pic: {result['has_profile_pic']}  |  "
              f"Details: {result['has_profile_details']}  |  "
              f"Prediction: {result['Fake/Real/Suspicious']}")

    except TimeoutException:
        result["Fake/Real/Suspicious"] = "Error - Timeout"
        print("  >> Timeout — page too slow")
    except WebDriverException as e:
        result["Fake/Real/Suspicious"] = "Error - Browser"
        print(f"  >> Browser error: {str(e)[:80]}")
    except Exception as e:
        result["Fake/Real/Suspicious"] = "Error"
        print(f"  >> Error: {str(e)[:80]}")

    return result


# ══════════════════════════════════════════════════════════════
#  OUTPUT  — only 8 columns, UTF-8 with BOM (fixes garbled text)
# ══════════════════════════════════════════════════════════════

OUTPUT_COLUMNS = [
    "url",
    "name",
    "has_profile_pic",
    "followers",
    "following",
    "friends",
    "has_profile_details",
    "Fake/Real/Suspicious",
]


def save_results(results: list, filepath: str):
    """
    Save directly to the output CSV.
    If the file is locked by Excel, wait and keep retrying silently.
    Never creates temp files.
    """
    df = pd.DataFrame(results, columns=OUTPUT_COLUMNS)
    for attempt in range(10):
        try:
            df.to_csv(filepath, index=False, encoding="utf-8-sig")
            return  # saved OK
        except PermissionError:
            time.sleep(2)  # wait 2 seconds and retry silently
    # If still failing after 10 attempts, just skip — data is safe in memory
    # and will be saved after the next profile


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    # Always save in the same folder as the script — no hardcoded paths
    INPUT_CSV  = os.path.join(SCRIPT_DIR, "input_urls.csv")
    OUTPUT_CSV = os.path.join(SCRIPT_DIR, "scraped_results.csv")
    print(f"Input  file: {INPUT_CSV}")
    print(f"Output file: {OUTPUT_CSV}")

    if not os.path.exists(INPUT_CSV):
        print(f"ERROR: '{INPUT_CSV}' not found.")
        print("Create a CSV with one column named 'url', one URL per row.")
        return

    df_input = pd.read_csv(INPUT_CSV)
    if "url" not in df_input.columns:
        print("ERROR: input_urls.csv must have a column named 'url'.")
        return

    urls  = df_input["url"].dropna().str.strip().tolist()
    total = len(urls)
    print(f"\nLoaded {total} URLs from '{INPUT_CSV}'")

    print("\nOpening Chrome browser...")
    driver = launch_browser()
    driver.get("https://www.facebook.com")

    print("\n" + "="*52)
    print("  ACTION REQUIRED:")
    print("  1. Log in to your Facebook in the Chrome window")
    print("  2. Wait until you see your home feed")
    print("  3. Come back here and press ENTER")
    print("="*52)
    input("\n  Press ENTER when logged in and ready...\n")

    results = []
    for i, url in enumerate(urls, 1):
        print(f"\n[{i}/{total}]  {url}")
        result = scrape_profile(driver, url)
        results.append(result)
        save_results(results, OUTPUT_CSV)  # save after every profile

        if i < total:
            delay = 10 if i % 5 == 0 else 4
            print(f"  Waiting {delay}s...")
            time.sleep(delay)

    driver.quit()

    # Summary
    df_final = pd.DataFrame(results, columns=OUTPUT_COLUMNS)
    locked  = (df_final["Fake/Real/Suspicious"] == "Locked Profile").sum()
    errors  = df_final["Fake/Real/Suspicious"].str.startswith("Error").sum()
    scraped = total - locked - errors

    print("\n" + "="*52)
    print(f"  DONE! Saved to: {OUTPUT_CSV}")
    print(f"  Successfully scraped : {scraped}")
    print(f"  Locked profiles      : {locked}")
    print(f"  Errors               : {errors}")
    print("="*52)

    if MODEL_AVAILABLE:
        valid = df_final[
            ~df_final["Fake/Real/Suspicious"].isin(["Locked Profile", "Profile Not Found", "N/A"]) &
            ~df_final["Fake/Real/Suspicious"].str.startswith("Error")
        ]
        if not valid.empty:
            print("\n  Predictions:")
            for label, count in valid["Fake/Real/Suspicious"].value_counts().items():
                print(f"     {label:<15}: {count}")


if __name__ == "__main__":
    main()
