#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas>=2.0",
#   "pyarrow>=14.0",
#   "beautifulsoup4>=4.12",
#   "scikit-learn>=1.3",
#   "tqdm>=4.65",
# ]
# ///
"""
Parse downloaded email datasets into a unified format and create train/val/cal/test splits.

Usage:
    uv run scripts/parse_and_split.py [--raw-dir data/raw] [--out-dir data/splits]

Output (parquet files):
    train.parquet   — 60%, both classes (for BERT + scorer training)
    val.parquet     — 10%, both classes (for weight learning + threshold tuning)
    cal.parquet     — legitimate only, ~2000-4000 emails (for conformal calibration)
    test.parquet    — 20%, both classes (for evaluation)

Columns:
    email_id, label (0=legit, 1=phishing), source, subject, body_text,
    from_domain, return_path_domain, reply_to_domain,
    from_reply_to_mismatch, from_return_path_mismatch,
    received_hops, sender_ip, urls_json, url_count,
    has_ip_url, has_shortener, has_href_mismatch, body_length
"""

import argparse
import email
import email.policy
import hashlib
import json
import logging
import mailbox
import os
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── URL shorteners (common list) ──────────────────────────────────────────
SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd",
    "buff.ly", "rebrand.ly", "short.io", "cutt.ly", "tiny.cc",
}

# ── Domain extraction ─────────────────────────────────────────────────────
def extract_domain(addr: Optional[str]) -> str:
    """Extract domain from an email address or header value."""
    if not addr:
        return ""
    # Handle "Name <user@domain.com>" format
    match = re.search(r"@([\w.-]+)", addr)
    return match.group(1).lower() if match else ""


def extract_ip_from_received(received_headers: list[str]) -> str:
    """Extract the first IP address from Received headers."""
    for hdr in received_headers:
        match = re.search(r"\[(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\]", hdr)
        if match:
            return match.group(1)
    return ""


# ── URL feature extraction ────────────────────────────────────────────────
def extract_urls_from_html(html: str) -> list[dict]:
    """Extract URLs with display text from HTML body."""
    urls = []
    try:
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            urls.append({
                "href": a["href"],
                "display_text": a.get_text(strip=True),
            })
    except Exception:
        pass
    return urls


def extract_urls_from_text(text: str) -> list[dict]:
    """Extract URLs from plain text."""
    urls = []
    for match in re.finditer(r"https?://[^\s<>\"']+", text):
        urls.append({"href": match.group(), "display_text": ""})
    return urls


def url_features(urls: list[dict]) -> dict:
    """Compute URL-based features."""
    url_count = len(urls)
    has_ip = 0
    has_shortener = 0
    has_mismatch = 0

    for u in urls:
        href = u.get("href", "")
        display = u.get("display_text", "")
        # IP address in URL
        if re.search(r"https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", href):
            has_ip = 1
        # URL shortener
        try:
            domain = re.search(r"https?://([^/]+)", href)
            if domain and domain.group(1).lower() in SHORTENERS:
                has_shortener = 1
        except Exception:
            pass
        # Display text vs href mismatch (e.g., shows paypal.com but links elsewhere)
        if display and re.match(r"https?://", display):
            display_domain = re.search(r"https?://([^/]+)", display)
            href_domain = re.search(r"https?://([^/]+)", href)
            if display_domain and href_domain:
                if display_domain.group(1).lower() != href_domain.group(1).lower():
                    has_mismatch = 1

    return {
        "url_count": url_count,
        "has_ip_url": has_ip,
        "has_shortener": has_shortener,
        "has_href_mismatch": has_mismatch,
    }


# ── Single email parser ──────────────────────────────────────────────────
def parse_email(msg: email.message.Message, source: str, label: int) -> Optional[dict]:
    """Parse a single email.message.Message into a feature dict."""
    try:
        # Body text
        body_text = ""
        html_body = ""
        for part in msg.walk():
            ct = part.get_content_type()
            if ct == "text/plain":
                try:
                    body_text += part.get_content()
                except Exception:
                    payload = part.get_payload(decode=True)
                    if payload:
                        body_text += payload.decode("utf-8", errors="replace")
            elif ct == "text/html":
                try:
                    html_body += part.get_content()
                except Exception:
                    payload = part.get_payload(decode=True)
                    if payload:
                        html_body += payload.decode("utf-8", errors="replace")

        if not body_text and html_body:
            # Fallback: extract text from HTML
            try:
                soup = BeautifulSoup(html_body, "html.parser")
                body_text = soup.get_text(separator=" ", strip=True)
            except Exception:
                body_text = ""

        # Skip empty emails
        if not body_text.strip() and not html_body.strip():
            return None

        subject = msg.get("Subject", "") or ""
        from_addr = msg.get("From", "") or ""
        return_path = msg.get("Return-Path", "") or ""
        reply_to = msg.get("Reply-To", "") or ""

        from_domain = extract_domain(from_addr)
        return_path_domain = extract_domain(return_path)
        reply_to_domain = extract_domain(reply_to)
        received_headers = msg.get_all("Received") or []
        sender_ip = extract_ip_from_received(received_headers)

        # URL extraction
        urls = extract_urls_from_html(html_body) if html_body else extract_urls_from_text(body_text)
        uf = url_features(urls)

        # Stable email ID from content hash
        content_hash = hashlib.sha256(
            (subject + body_text[:500] + from_addr).encode("utf-8", errors="replace")
        ).hexdigest()[:16]
        email_id = f"{source}_{content_hash}"

        return {
            "email_id": email_id,
            "label": label,
            "source": source,
            "subject": subject[:500],
            "body_text": body_text[:10000],  # cap at 10K chars
            "from_domain": from_domain,
            "return_path_domain": return_path_domain,
            "reply_to_domain": reply_to_domain,
            "from_reply_to_mismatch": int(
                bool(reply_to_domain) and reply_to_domain != from_domain
            ),
            "from_return_path_mismatch": int(
                bool(return_path_domain) and return_path_domain != from_domain
            ),
            "received_hops": len(received_headers),
            "sender_ip": sender_ip,
            "urls_json": json.dumps(urls[:50]),  # cap at 50 URLs
            "url_count": uf["url_count"],
            "has_ip_url": uf["has_ip_url"],
            "has_shortener": uf["has_shortener"],
            "has_href_mismatch": uf["has_href_mismatch"],
            "body_length": len(body_text),
        }
    except Exception as e:
        log.debug(f"Failed to parse email: {e}")
        return None


def parse_file(filepath: Path, source: str, label: int) -> Optional[dict]:
    """Parse a single .eml or raw email file."""
    try:
        raw = filepath.read_bytes()
        msg = email.message_from_bytes(raw, policy=email.policy.default)
        return parse_email(msg, source, label)
    except Exception as e:
        log.debug(f"Failed to read {filepath}: {e}")
        return None


def parse_mbox(mbox_path: Path, source: str, label: int) -> list[dict]:
    """Parse all emails from an mbox file."""
    records = []
    try:
        mbox = mailbox.mbox(str(mbox_path))
        for msg in mbox:
            rec = parse_email(msg, source, label)
            if rec:
                records.append(rec)
    except Exception as e:
        log.warning(f"Failed to parse mbox {mbox_path}: {e}")
    return records


# ── Dataset-specific walkers ──────────────────────────────────────────────
def walk_spamassassin(base: Path) -> list[dict]:
    """Parse SpamAssassin directories (ham_*/spam_*)."""
    records = []
    if not base.exists():
        log.warning(f"SpamAssassin dir not found: {base}")
        return records

    for subdir in sorted(base.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith("."):
            continue
        name_lower = subdir.name.lower()
        if "ham" in name_lower:
            label, source = 0, "spamassassin_ham"
        elif "spam" in name_lower:
            label, source = 1, "spamassassin_spam"
        else:
            continue

        files = [f for f in subdir.iterdir() if f.is_file() and not f.name.startswith(".")]
        log.info(f"  {subdir.name}: {len(files)} files (label={label})")
        for fp in tqdm(files, desc=f"  {subdir.name}", leave=False):
            rec = parse_file(fp, source, label)
            if rec:
                records.append(rec)
    return records


def walk_nazario(base: Path) -> list[dict]:
    """Parse Nazario phishing corpus (mbox format, all phishing)."""
    records = []
    mbox_path = base / "phishing3.mbox"
    if not mbox_path.exists():
        log.warning(f"Nazario mbox not found: {mbox_path}")
        return records

    records = parse_mbox(mbox_path, "nazario", label=1)
    log.info(f"  Nazario: {len(records)} emails parsed from mbox")
    return records


def walk_phishing_pot(base: Path) -> list[dict]:
    """Parse Phishing Pot .eml files (all phishing)."""
    records = []
    if not base.exists():
        log.warning(f"Phishing Pot dir not found: {base}")
        return records

    eml_files = list(base.rglob("*.eml"))
    log.info(f"  Phishing Pot: {len(eml_files)} .eml files")
    for fp in tqdm(eml_files, desc="  Phishing Pot", leave=False):
        rec = parse_file(fp, "phishing_pot", label=1)
        if rec:
            records.append(rec)
    return records


def walk_enron_spam(base: Path) -> list[dict]:
    """Parse Enron-Spam AUEB directories (ham/spam per subset)."""
    records = []
    if not base.exists():
        log.warning(f"Enron-Spam dir not found: {base}")
        return records

    for subset_dir in sorted(base.iterdir()):
        if not subset_dir.is_dir() or subset_dir.name.startswith("."):
            continue
        for class_dir in sorted(subset_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            name_lower = class_dir.name.lower()
            if "ham" in name_lower:
                label, source = 0, f"enron_{subset_dir.name}_ham"
            elif "spam" in name_lower:
                label, source = 1, f"enron_{subset_dir.name}_spam"
            else:
                continue
            files = [f for f in class_dir.iterdir() if f.is_file() and not f.name.startswith(".")]
            log.info(f"  {subset_dir.name}/{class_dir.name}: {len(files)} files")
            for fp in tqdm(files, desc=f"  {class_dir.name}", leave=False):
                rec = parse_file(fp, source, label)
                if rec:
                    records.append(rec)
    return records


# ── Splitting ─────────────────────────────────────────────────────────────
def create_splits(df: pd.DataFrame, out_dir: Path, cal_size: int = 3000, seed: int = 42):
    """
    Split into train (60%), val (10%), cal (legitimate only), test (20%).
    Remaining ~10% of legitimate goes to calibration set.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Quality filters: remove empty/near-empty bodies
    before = len(df)
    df = df[df["body_length"] >= 50]
    log.info(f"Quality filter (body >= 50 chars): {before} → {len(df)} emails")

    # Deduplicate by email_id (content hash)
    before = len(df)
    df = df.drop_duplicates(subset="email_id", keep="first")
    log.info(f"Dedup by email_id: {before} → {len(df)} emails")

    # Content-level dedup: first 500 chars of body text (avoids collapsing
    # phishing variants that share greetings but differ in body)
    before = len(df)
    df["_body_prefix"] = df["body_text"].fillna("").str[:500]
    df = df.drop_duplicates(subset="_body_prefix", keep="first")
    df = df.drop(columns=["_body_prefix"])
    log.info(f"Dedup by body prefix (500 chars): {before} → {len(df)} emails")

    # First split: 80% train+val+cal, 20% test
    train_val_cal, test = train_test_split(
        df, test_size=0.20, random_state=seed, stratify=df["label"]
    )

    # From the 80%, separate legitimate for calibration first
    legit_pool = train_val_cal[train_val_cal["label"] == 0]
    phishing_pool = train_val_cal[train_val_cal["label"] == 1]
    mixed_pool = train_val_cal  # we'll re-split below

    # Reserve calibration set (legitimate only)
    actual_cal_size = min(cal_size, len(legit_pool) // 2)  # don't take more than half
    if actual_cal_size < 500:
        log.warning(f"Small calibration set: only {actual_cal_size} legitimate emails available")

    cal = legit_pool.sample(n=actual_cal_size, random_state=seed)
    remaining = train_val_cal.drop(cal.index)

    # Split remaining into train (75%) and val (25%) → overall ~60% train, ~10% val
    train, val = train_test_split(
        remaining, test_size=0.125, random_state=seed, stratify=remaining["label"]
    )
    # 0.125 of 80% = 10% of total

    # Save
    for name, split_df in [("train", train), ("val", val), ("cal", cal), ("test", test)]:
        path = out_dir / f"{name}.parquet"
        split_df.to_parquet(path, index=False)
        n_legit = (split_df["label"] == 0).sum()
        n_phish = (split_df["label"] == 1).sum()
        log.info(f"  {name}: {len(split_df)} emails ({n_legit} legit, {n_phish} phishing) → {path}")

    # Save summary
    summary = {
        "total": len(df),
        "train": len(train),
        "val": len(val),
        "cal": len(cal),
        "test": len(test),
        "sources": df["source"].value_counts().to_dict(),
    }
    (out_dir / "split_summary.json").write_text(json.dumps(summary, indent=2))
    log.info(f"Split summary → {out_dir / 'split_summary.json'}")


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Parse email datasets and create splits")
    parser.add_argument("--raw-dir", default="data/raw", help="Directory with downloaded datasets")
    parser.add_argument("--out-dir", default="data/splits", help="Output directory for splits")
    parser.add_argument("--cal-size", type=int, default=3000, help="Target calibration set size")
    parser.add_argument("--max-phishing", type=int, default=8000,
                        help="Max phishing emails to include (subsample large datasets)")
    parser.add_argument("--max-legit", type=int, default=8000,
                        help="Max legitimate emails to include")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    log.info("=" * 60)
    log.info("  Parsing email datasets")
    log.info("=" * 60)

    all_records = []

    # Parse each dataset
    log.info("SpamAssassin:")
    all_records.extend(walk_spamassassin(raw_dir / "spamassassin"))

    log.info("Nazario Phishing Corpus:")
    all_records.extend(walk_nazario(raw_dir / "nazario"))

    log.info("Phishing Pot:")
    all_records.extend(walk_phishing_pot(raw_dir / "phishing_pot"))

    log.info("Enron-Spam AUEB:")
    all_records.extend(walk_enron_spam(raw_dir / "enron-spam"))

    if not all_records:
        log.error("No emails parsed! Check that data/raw/ contains downloaded datasets.")
        log.error("Run: bash scripts/download_sources.sh")
        return

    df = pd.DataFrame(all_records)
    log.info(f"\nTotal parsed: {len(df)} emails")
    log.info(f"  Legitimate: {(df['label'] == 0).sum()}")
    log.info(f"  Phishing:   {(df['label'] == 1).sum()}")
    log.info(f"  Sources:    {df['source'].nunique()}")
    log.info(f"\nPer source:")
    for src, count in df["source"].value_counts().items():
        label_dist = df[df["source"] == src]["label"].value_counts().to_dict()
        log.info(f"  {src}: {count} (legit={label_dist.get(0, 0)}, phish={label_dist.get(1, 0)})")

    # Subsample if datasets are very large (balance classes roughly)
    legit = df[df["label"] == 0]
    phish = df[df["label"] == 1]

    if len(phish) > args.max_phishing:
        log.info(f"\nSubsampling phishing: {len(phish)} → {args.max_phishing}")
        phish = phish.sample(n=args.max_phishing, random_state=42)
    if len(legit) > args.max_legit:
        log.info(f"Subsampling legitimate: {len(legit)} → {args.max_legit}")
        legit = legit.sample(n=args.max_legit, random_state=42)

    df = pd.concat([legit, phish], ignore_index=True)
    log.info(f"\nFinal dataset: {len(df)} emails ({(df['label']==0).sum()} legit, {(df['label']==1).sum()} phish)")

    # Save full parsed dataset
    parsed_path = Path(args.raw_dir).parent / "parsed" / "all_emails.parquet"
    parsed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parsed_path, index=False)
    log.info(f"Full parsed dataset → {parsed_path}")

    # Create splits
    log.info(f"\nCreating splits:")
    create_splits(df, out_dir, cal_size=args.cal_size)

    log.info("\nDone! Next steps:")
    log.info("  1. python experiments/train_bert.py        (fine-tune BERT scorer)")
    log.info("  2. python experiments/train_scorers.py     (header + URL scorers)")
    log.info("  3. python experiments/calibrate_and_eval.py (conformal calibration + evaluation)")
    log.info("  4. python experiments/llm_baseline.py      (zero-shot GPT-4o + Claude baseline)")


if __name__ == "__main__":
    main()
