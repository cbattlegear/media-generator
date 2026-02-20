"""
Generate DALL-E style image prompts for director headshots using Ollama.

Fetches directors from the production API by ID range, sends each name to
a local Ollama model to produce an image prompt, and saves the result
to director_image_prompt/{director_id}.prompt.txt.

Usage:
    python generate_director_prompts.py --start 1 --end 50
    python generate_director_prompts.py --start 1 --end 50 --model llama3.1
    python generate_director_prompts.py --start 1 --end 50 --api-url http://localhost:8000
"""

import argparse
import os
import sys

import ollama
import requests

DEFAULT_API_URL = "https://api.battlecabbage.com"
DEFAULT_MODEL = os.getenv("LOCAL_MODEL_NAME", "gpt-oss")
OUTPUT_DIR = "director_image_prompt"


def fetch_directors_in_range(api_url: str, start_id: int, end_id: int) -> list[dict]:
    """Fetch directors from the API and filter to those within the ID range."""
    directors = []
    skip = 0
    limit = 200

    while True:
        resp = requests.get(
            f"{api_url}/directors", params={"skip": skip, "limit": limit}, timeout=30
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        directors.extend(d for d in batch if start_id <= d["director_id"] <= end_id)
        skip += limit
        # Stop early if we've gone past possible results
        if all(d["director_id"] > end_id for d in batch):
            break

    return sorted(directors, key=lambda d: d["director_id"])


def generate_prompt(director_name: str, model: str) -> str:
    """Call local Ollama to generate a DALL-E style headshot prompt for a director."""
    user_prompt = (
        f"Create a dalle style prompt to create headshot/profile picture based on "
        f"the satirical name {director_name}, only return the prompt and no other text."
    )
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response["message"]["content"].strip()


def main():
    parser = argparse.ArgumentParser(description="Generate director headshot prompts via Ollama")
    parser.add_argument("--start", type=int, required=True, help="Starting director ID (inclusive)")
    parser.add_argument("--end", type=int, required=True, help="Ending director ID (inclusive)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument(
        "--api-url", type=str, default=DEFAULT_API_URL, help="Media generator API base URL"
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Fetching directors {args.start}-{args.end} from {args.api_url}...")
    directors = fetch_directors_in_range(args.api_url, args.start, args.end)

    if not directors:
        print("No directors found in the specified range.")
        sys.exit(0)

    print(f"Found {len(directors)} directors. Generating prompts with model '{args.model}'...")

    for director in directors:
        director_id = director["director_id"]
        director_name = director["director"]
        out_path = os.path.join(OUTPUT_DIR, f"{director_id}.prompt.txt")

        if os.path.exists(out_path):
            print(f"  [{director_id}] {director_name} — skipped (already exists)")
            continue

        print(f"  [{director_id}] {director_name} — generating...", end=" ", flush=True)
        try:
            prompt_text = generate_prompt(director_name, args.model)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(prompt_text)
            print("done")
        except Exception as e:
            print(f"FAILED: {e}")

    print("Complete.")


if __name__ == "__main__":
    main()
