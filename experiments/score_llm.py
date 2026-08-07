#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas>=2.0",
#   "pyarrow>=14.0",
#   "openai>=1.30",
#   "anthropic>=0.25",
#   "python-dotenv>=1.0",
#   "tqdm>=4.65",
# ]
# ///
"""
Score emails with Claude Sonnet 4 and GPT-4o-mini for calibration comparison.

Usage:
    uv run experiments/score_llm.py --model claude --data-dir data/splits_v2
    uv run experiments/score_llm.py --model gpt --data-dir data/splits_v2
    uv run experiments/score_llm.py --model both --data-dir data/splits_v2

Outputs scores to results/llm_scores_{model}_{split}.json (incremental).
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

SYSTEM_PROMPT = (
    "You are a phishing detection system. Analyze the following email "
    "and respond with ONLY a number between 0 and 1 representing the probability "
    "that this email is a phishing attempt. 0 = definitely legitimate, 1 = "
    "definitely phishing. Respond with just the number, nothing else."
)


def parse_score(response_text: str) -> float:
    """Parse a float from the LLM response. Return 0.5 on failure."""
    text = response_text.strip()
    try:
        score = float(text)
        return max(0.0, min(1.0, score))
    except ValueError:
        # Try to extract first number
        import re
        match = re.search(r"(\d+\.?\d*)", text)
        if match:
            return max(0.0, min(1.0, float(match.group(1))))
        return 0.5


async def score_claude(emails: list[dict], results_path: Path, max_concurrent: int = 30):
    """Score emails with Claude Sonnet 4."""
    import anthropic

    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(max_concurrent)

    # Load existing results
    existing = {}
    if results_path.exists():
        with open(results_path) as f:
            for item in json.load(f):
                existing[item["email_id"]] = item

    results = list(existing.values())
    done_ids = set(existing.keys())
    todo = [e for e in emails if e["email_id"] not in done_ids]

    print(f"Claude: {len(done_ids)} already done, {len(todo)} remaining")

    async def score_one(email_data):
        async with semaphore:
            user_msg = f"Subject: {email_data.get('subject', '')}\n\nBody: {email_data['body_text'][:4000]}"
            t0 = time.time()
            try:
                response = await client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=10,
                    temperature=0,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_msg}],
                )
                raw = response.content[0].text
                score = parse_score(raw)
                latency = (time.time() - t0) * 1000
                return {
                    "email_id": email_data["email_id"],
                    "label": int(email_data["label"]),
                    "score": score,
                    "raw_response": raw,
                    "latency_ms": latency,
                    "error": None,
                }
            except Exception as e:
                return {
                    "email_id": email_data["email_id"],
                    "label": int(email_data["label"]),
                    "score": 0.5,
                    "raw_response": None,
                    "latency_ms": (time.time() - t0) * 1000,
                    "error": str(e),
                }

    # Process in batches with progress bar
    batch_size = 100
    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        batch_results = await asyncio.gather(*[score_one(e) for e in batch])
        results.extend(batch_results)

        # Save incrementally
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        done = len(results)
        total = len(emails)
        errors = sum(1 for r in results if r["error"])
        print(f"  Progress: {done}/{total} ({done/total:.0%}) | Errors: {errors}")

    return results


async def score_gpt(emails: list[dict], results_path: Path, max_concurrent: int = 5):
    """Score emails with GPT-4o-mini with rate limit handling."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(max_concurrent)

    # Load existing results — only keep successful ones
    existing = {}
    if results_path.exists():
        with open(results_path) as f:
            for item in json.load(f):
                if not item.get("error"):
                    existing[item["email_id"]] = item

    results = list(existing.values())
    done_ids = set(existing.keys())
    todo = [e for e in emails if e["email_id"] not in done_ids]

    print(f"GPT-4o-mini: {len(done_ids)} already done, {len(todo)} remaining", flush=True)

    async def score_one(email_data):
        async with semaphore:
            # Truncate to ~1500 chars to stay within TPM limits
            body = email_data['body_text'][:1500]
            user_msg = f"Subject: {email_data.get('subject', '')}\n\nBody: {body}"
            max_retries = 8
            for attempt in range(max_retries):
                t0 = time.time()
                try:
                    response = await client.chat.completions.create(
                        model="gpt-4o",
                        max_tokens=10,
                        temperature=0,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_msg},
                        ],
                    )
                    raw = response.choices[0].message.content
                    score = parse_score(raw)
                    latency = (time.time() - t0) * 1000
                    return {
                        "email_id": email_data["email_id"],
                        "label": int(email_data["label"]),
                        "score": score,
                        "raw_response": raw,
                        "latency_ms": latency,
                        "error": None,
                    }
                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str and attempt < max_retries - 1:
                        # Exponential backoff: 1, 2, 4, 8, 16, 32, 64
                        wait = min(2 ** attempt, 60)
                        await asyncio.sleep(wait)
                        continue
                    return {
                        "email_id": email_data["email_id"],
                        "label": int(email_data["label"]),
                        "score": 0.5,
                        "raw_response": None,
                        "latency_ms": (time.time() - t0) * 1000,
                        "error": err_str,
                    }

    # Process in small batches with inter-batch delay
    batch_size = 20
    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        batch_results = await asyncio.gather(*[score_one(e) for e in batch])
        results.extend(batch_results)

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        done = len(results)
        total = len(emails)
        errors = sum(1 for r in batch_results if r["error"])
        print(f"  Progress: {done}/{total} ({done/total:.0%}) | Batch errors: {errors}", flush=True)

        # Brief pause between batches to avoid rate limit bursts
        if errors > 0:
            await asyncio.sleep(2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Score emails with LLMs")
    parser.add_argument("--model", choices=["claude", "gpt", "both"], default="both")
    parser.add_argument("--data-dir", default="data/splits_v2")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--splits", nargs="+", default=["val", "cal", "test"],
                        help="Which splits to score")
    parser.add_argument("--max-concurrent", type=int, default=30)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)
    data_dir = Path(args.data_dir)

    # Load all requested splits
    all_emails = []
    for split in args.splits:
        df = pd.read_parquet(data_dir / f"{split}.parquet")
        for _, row in df.iterrows():
            all_emails.append({
                "email_id": row["email_id"],
                "label": int(row["label"]),
                "subject": row.get("subject", ""),
                "body_text": row.get("body_text", ""),
                "split": split,
            })

    print(f"Total emails to score: {len(all_emails)} across splits {args.splits}")

    if args.model in ("claude", "both"):
        print(f"\n{'='*60}")
        print("Scoring with Claude Sonnet 4")
        print(f"{'='*60}")
        results_path = results_dir / "llm_scores_claude_v2.json"
        asyncio.run(score_claude(all_emails, results_path, args.max_concurrent))

    if args.model in ("gpt", "both"):
        print(f"\n{'='*60}")
        print("Scoring with GPT-4o-mini")
        print(f"{'='*60}")
        results_path = results_dir / "llm_scores_gpt_v2.json"
        asyncio.run(score_gpt(all_emails, results_path, args.max_concurrent))

    print("\nDone! Results saved to results/llm_scores_*_v2.json")


if __name__ == "__main__":
    main()
