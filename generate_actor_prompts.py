"""
Generate DALL-E style image prompts for actor headshots using Ollama.

Fetches actors from the production API by ID range, sends each name to
a local Ollama model to produce an image prompt, and saves the result
to actor_image_prompt/{actor_id}.prompt.txt.

Usage:
    python generate_actor_prompts.py --start 1 --end 50
    python generate_actor_prompts.py --start 1 --end 50 --model llama3.1
    python generate_actor_prompts.py --start 1 --end 50 --api-url http://localhost:8000
"""

import argparse
import os
import sys

import ollama
import requests

DEFAULT_API_URL = "https://api.battlecabbage.com"
DEFAULT_MODEL = os.getenv("LOCAL_MODEL_NAME", "gpt-oss")
OUTPUT_DIR = "actor_image_prompt"


def fetch_actors_in_range(api_url: str, start_id: int, end_id: int) -> list[dict]:
    """Fetch actors from the API and filter to those within the ID range."""
    actors = []
    skip = 0
    limit = 200

    while True:
        resp = requests.get(f"{api_url}/actors", params={"skip": skip, "limit": limit}, timeout=30)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        actors.extend(a for a in batch if start_id <= a["actor_id"] <= end_id)
        skip += limit
        # Stop early if we've gone past possible results
        if all(a["actor_id"] > end_id for a in batch):
            break

    return sorted(actors, key=lambda a: a["actor_id"])


def generate_prompt(actor_name: str, model: str) -> str:
    """Call local Ollama to generate a DALL-E style headshot prompt for an actor."""
    user_prompt = (
        f"Create a dalle style prompt to create headshot/profile picture based on "
        f"the satirical name {actor_name}, only return the prompt and no other text."
    )
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response["message"]["content"].strip()


def main():
    parser = argparse.ArgumentParser(description="Generate actor headshot prompts via Ollama")
    parser.add_argument("--start", type=int, required=True, help="Starting actor ID (inclusive)")
    parser.add_argument("--end", type=int, required=True, help="Ending actor ID (inclusive)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument(
        "--api-url", type=str, default=DEFAULT_API_URL, help="Media generator API base URL"
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Fetching actors {args.start}-{args.end} from {args.api_url}...")
    actors = fetch_actors_in_range(args.api_url, args.start, args.end)

    if not actors:
        print("No actors found in the specified range.")
        sys.exit(0)

    print(f"Found {len(actors)} actors. Generating prompts with model '{args.model}'...")

    for actor in actors:
        actor_id = actor["actor_id"]
        actor_name = actor["actor"]
        out_path = os.path.join(OUTPUT_DIR, f"{actor_id}.prompt.txt")

        if os.path.exists(out_path):
            print(f"  [{actor_id}] {actor_name} — skipped (already exists)")
            continue

        print(f"  [{actor_id}] {actor_name} — generating...", end=" ", flush=True)
        try:
            prompt_text = generate_prompt(actor_name, args.model)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(prompt_text)
            print("done")
        except Exception as e:
            print(f"FAILED: {e}")

    print("Complete.")


if __name__ == "__main__":
    main()
