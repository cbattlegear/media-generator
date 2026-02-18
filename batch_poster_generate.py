#!/usr/bin/env python
"""
Batch Poster Generator

Generates movie poster images for movies that don't have them.
Queries the media-generator API for movies missing posters,
generates image prompts using the existing AI text model,
sends them to InvokeAI for image generation, and uploads results.

Usage:
    python batch_poster_generate.py
    python batch_poster_generate.py --limit 10 --verbose
    python batch_poster_generate.py --media-api http://localhost:8000 --invokeai http://localhost:9090
"""

import argparse
import json
import os
import random
import sys
import time
import traceback
from io import BytesIO

import requests
from dotenv import load_dotenv

# Add project root to path for lib imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.aoai_model import aoaiText
from lib.ollama_model import ollamaText
from lib.process_helper import processHelper

# Defaults
DEFAULT_MEDIA_API_URL = "http://localhost:8000"
DEFAULT_INVOKEAI_URL = "http://localhost:9090"
POLL_INTERVAL = 5
GENERATION_TIMEOUT = 300

# The static InvokeAI FLUX graph structure.
# Only the prompt and seed are injected via the "data" array.
INVOKEAI_GRAPH = {
    "id": "flux_graph:DHMtzyKWJZ",
    "nodes": {
        "positive_prompt:XSchZlBWDn": {
            "id": "positive_prompt:XSchZlBWDn",
            "type": "string",
            "is_intermediate": True,
            "use_cache": True
        },
        "seed:6h9cteOn1T": {
            "id": "seed:6h9cteOn1T",
            "type": "integer",
            "is_intermediate": True,
            "use_cache": True
        },
        "flux2_klein_model_loader:g6iI22rxsU": {
            "type": "flux2_klein_model_loader",
            "id": "flux2_klein_model_loader:g6iI22rxsU",
            "model": {
                "key": "c3ff1051-1fc8-415f-b876-b6788ec0397a",
                "hash": "blake3:c3ee838d71d99497db01fae6f304eafd9e734e935f3b783e968d50febb56be2c",
                "path": "c3ff1051-1fc8-415f-b876-b6788ec0397a/flux-2-klein-4b-Q4_K_M.gguf",
                "file_size": 2604311104,
                "name": "FLUX.2 Klein 4B (GGUF Q4)",
                "description": "FLUX.2 Klein 4B GGUF Q4_K_M quantized - runs on 6-8GB VRAM. Installs with VAE and Qwen3 4B encoder. ~2.6GB",
                "source": "https://huggingface.co/unsloth/FLUX.2-klein-4B-GGUF/resolve/main/flux-2-klein-4b-Q4_K_M.gguf",
                "source_type": "url",
                "source_api_response": None,
                "cover_image": None,
                "type": "main",
                "trigger_phrases": None,
                "default_settings": {
                    "vae": None,
                    "vae_precision": None,
                    "scheduler": None,
                    "steps": 4,
                    "cfg_scale": 1,
                    "cfg_rescale_multiplier": None,
                    "width": 1024,
                    "height": 1792,
                    "guidance": None,
                    "cpu_only": None
                },
                "config_path": None,
                "base": "flux2",
                "format": "gguf_quantized",
                "variant": "klein_4b"
            },
            "vae_model": {
                "key": "19babdab-c41c-4b45-a217-e4dd484a2fd2",
                "hash": "blake3:531855de70db993d0f6181f82cde27d15411d58b7ffa3b2fdce2b9434c0173c2",
                "name": "FLUX.2 VAE",
                "base": "flux2",
                "type": "vae"
            },
            "qwen3_encoder_model": {
                "key": "90034a7d-35d8-4517-a19d-90c3b99193b0",
                "hash": "blake3:af5840e6770dc99f678e69867949c8b9264835915eb82a990e940fa6e4fa6c81",
                "name": "FLUX.2 Klein Qwen3 4B Encoder",
                "base": "any",
                "type": "qwen3_encoder"
            },
            "is_intermediate": True,
            "use_cache": True
        },
        "flux2_klein_text_encoder:nmmeJl47fN": {
            "type": "flux2_klein_text_encoder",
            "id": "flux2_klein_text_encoder:nmmeJl47fN",
            "is_intermediate": True,
            "use_cache": True
        },
        "flux2_denoise:cUHCHzGwqi": {
            "type": "flux2_denoise",
            "id": "flux2_denoise:cUHCHzGwqi",
            "num_steps": 30,
            "is_intermediate": True,
            "use_cache": True,
            "denoising_start": 0,
            "denoising_end": 1,
            "width": 1024,
            "height": 1024
        },
        "core_metadata:iUZ4nb2S38": {
            "id": "core_metadata:iUZ4nb2S38",
            "type": "core_metadata",
            "is_intermediate": True,
            "use_cache": True,
            "model": {
                "key": "c3ff1051-1fc8-415f-b876-b6788ec0397a",
                "hash": "blake3:c3ee838d71d99497db01fae6f304eafd9e734e935f3b783e968d50febb56be2c",
                "name": "FLUX.2 Klein 4B (GGUF Q4)",
                "base": "flux2",
                "type": "main"
            },
            "steps": 30,
            "vae": {
                "key": "19babdab-c41c-4b45-a217-e4dd484a2fd2",
                "hash": "blake3:531855de70db993d0f6181f82cde27d15411d58b7ffa3b2fdce2b9434c0173c2",
                "name": "FLUX.2 VAE",
                "base": "flux2",
                "type": "vae"
            },
            "qwen3_encoder": {
                "key": "90034a7d-35d8-4517-a19d-90c3b99193b0",
                "hash": "blake3:af5840e6770dc99f678e69867949c8b9264835915eb82a990e940fa6e4fa6c81",
                "name": "FLUX.2 Klein Qwen3 4B Encoder",
                "base": "any",
                "type": "qwen3_encoder"
            },
            "width": 1024,
            "height": 1024,
            "generation_mode": "flux2_txt2img"
        },
        "canvas_output:Z16pWr3QXH": {
            "type": "flux2_vae_decode",
            "id": "canvas_output:Z16pWr3QXH",
            "is_intermediate": False,
            "use_cache": False
        }
    },
    "edges": [
        {"source": {"node_id": "flux2_klein_model_loader:g6iI22rxsU", "field": "qwen3_encoder"}, "destination": {"node_id": "flux2_klein_text_encoder:nmmeJl47fN", "field": "qwen3_encoder"}},
        {"source": {"node_id": "flux2_klein_model_loader:g6iI22rxsU", "field": "max_seq_len"}, "destination": {"node_id": "flux2_klein_text_encoder:nmmeJl47fN", "field": "max_seq_len"}},
        {"source": {"node_id": "flux2_klein_model_loader:g6iI22rxsU", "field": "transformer"}, "destination": {"node_id": "flux2_denoise:cUHCHzGwqi", "field": "transformer"}},
        {"source": {"node_id": "flux2_klein_model_loader:g6iI22rxsU", "field": "vae"}, "destination": {"node_id": "flux2_denoise:cUHCHzGwqi", "field": "vae"}},
        {"source": {"node_id": "flux2_klein_model_loader:g6iI22rxsU", "field": "vae"}, "destination": {"node_id": "canvas_output:Z16pWr3QXH", "field": "vae"}},
        {"source": {"node_id": "positive_prompt:XSchZlBWDn", "field": "value"}, "destination": {"node_id": "flux2_klein_text_encoder:nmmeJl47fN", "field": "prompt"}},
        {"source": {"node_id": "flux2_klein_text_encoder:nmmeJl47fN", "field": "conditioning"}, "destination": {"node_id": "flux2_denoise:cUHCHzGwqi", "field": "positive_text_conditioning"}},
        {"source": {"node_id": "seed:6h9cteOn1T", "field": "value"}, "destination": {"node_id": "flux2_denoise:cUHCHzGwqi", "field": "seed"}},
        {"source": {"node_id": "flux2_denoise:cUHCHzGwqi", "field": "latents"}, "destination": {"node_id": "canvas_output:Z16pWr3QXH", "field": "latents"}},
        {"source": {"node_id": "seed:6h9cteOn1T", "field": "value"}, "destination": {"node_id": "core_metadata:iUZ4nb2S38", "field": "seed"}},
        {"source": {"node_id": "positive_prompt:XSchZlBWDn", "field": "value"}, "destination": {"node_id": "core_metadata:iUZ4nb2S38", "field": "positive_prompt"}},
        {"source": {"node_id": "core_metadata:iUZ4nb2S38", "field": "metadata"}, "destination": {"node_id": "canvas_output:Z16pWr3QXH", "field": "metadata"}}
    ]
}


def log(message, level="info"):
    """Print a formatted log message."""
    prefix = {
        "info": "[INFO]", "success": "[OK]", "error": "[ERROR]",
        "warning": "[WARN]", "verbose": "[DEBUG]"
    }
    print(f"  {prefix.get(level, '[INFO]')} {message}")


def get_movies_missing_posters(api_url, limit=100):
    """Fetch movies that need poster generation from the media-generator API."""
    resp = requests.get(f"{api_url}/movies/missing-posters", params={"limit": limit})
    resp.raise_for_status()
    return resp.json()


def build_image_prompt(movie, templates_base, verbose=False):
    """
    Build an image generation prompt using the AI text model.

    Reuses the image_prompt templates from prompts.json and sends them
    to the configured text model (Azure OpenAI or Ollama) to produce
    a detailed image generation prompt for InvokeAI.
    """
    prompt_file_path = os.path.join(templates_base, "prompts.json")
    with open(prompt_file_path) as f:
        prompts_json = json.load(f)

    system_prompt = random.choice(prompts_json["image_prompt_system"])
    prompt_template = random.choice(prompts_json["image_prompt"])

    # Build replacement values from the movie API response
    replacements = {
        "title": movie.get("title", "Unknown"),
        "tagline": movie.get("tagline", ""),
        "description": movie.get("description", ""),
        "genres": movie.get("genre", ""),
        "mpaa_ratings": movie.get("mpaa_rating", "NR"),
        "eras": "Modern",
    }

    # Fill in template placeholders (same logic as lib/image.py)
    prompt = prompt_template
    start_index = prompt.find("{")
    while start_index != -1:
        end_index = prompt.find("}")
        key = prompt[start_index + 1:end_index]
        key_value = replacements.get(key, "NO VALUE")
        prompt = prompt.replace("{" + key + "}", str(key_value), 1)
        start_index = prompt.find("{")

    if verbose:
        log(f"Image prompt template filled: {prompt}", "verbose")

    # Call the AI text model to generate a detailed image prompt
    model_type = os.getenv("MODEL_TYPE", "").lower()
    if model_type == "azure_openai":
        text_model = aoaiText()
    else:
        text_model = ollamaText()

    text_model.user_prompt = prompt
    text_model.system_prompt = system_prompt

    completion = text_model.generateResponse()

    # Extract the image_prompt from the JSON response
    start = completion.find("{")
    end = completion.rfind("}") + 1
    if start != -1 and end > start:
        try:
            result = json.loads(completion[start:end])
            image_prompt = result.get("image_prompt", "")
            if image_prompt:
                return image_prompt
        except json.JSONDecodeError:
            pass

    log(f"Failed to parse image prompt from AI response, using template directly", "warning")
    return prompt


def build_invokeai_payload(prompt, seed=None):
    """Build the InvokeAI enqueue_batch request body."""
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    return {
        "prepend": False,
        "batch": {
            "graph": INVOKEAI_GRAPH,
            "runs": 1,
            "data": [
                [{"node_path": "seed:6h9cteOn1T", "field_name": "value", "items": [seed]}],
                [{"node_path": "positive_prompt:XSchZlBWDn", "field_name": "value", "items": [prompt]}]
            ],
            "origin": "generate",
            "destination": "generate"
        }
    }


def enqueue_generation(invokeai_url, prompt, seed=None):
    """Enqueue an image generation batch in InvokeAI."""
    payload = build_invokeai_payload(prompt, seed)
    resp = requests.post(
        f"{invokeai_url}/api/v1/queue/default/enqueue_batch",
        json=payload
    )
    resp.raise_for_status()
    return resp.json()


def wait_for_batch(invokeai_url, batch_id, timeout=GENERATION_TIMEOUT):
    """Poll InvokeAI batch status until complete or timeout."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        resp = requests.get(f"{invokeai_url}/api/v1/queue/default/b/{batch_id}/status")
        resp.raise_for_status()
        status = resp.json()

        completed = status.get("completed", 0)
        failed = status.get("failed", 0)
        canceled = status.get("canceled", 0)
        total = status.get("total", 0)

        if failed > 0 or canceled > 0:
            return {"success": False, "status": status}
        if completed >= total and total > 0:
            return {"success": True, "status": status}

        time.sleep(POLL_INTERVAL)

    return {"success": False, "status": {"error": "timeout"}}


def get_latest_image_name(invokeai_url):
    """Get the most recently generated image name from InvokeAI."""
    resp = requests.get(
        f"{invokeai_url}/api/v1/images/",
        params={"limit": 1, "offset": 0}
    )
    resp.raise_for_status()
    data = resp.json()

    items = data.get("items", [])
    if items:
        return items[0].get("image_name")
    return None


def download_image(invokeai_url, image_name):
    """Download a full-resolution image from InvokeAI."""
    resp = requests.get(f"{invokeai_url}/api/v1/images/i/{image_name}/full")
    resp.raise_for_status()
    return BytesIO(resp.content)


def upload_poster(api_url, movie_id, image_data, api_key):
    """Upload a poster image to the media-generator API."""
    files = {"file": ("poster.png", image_data, "image/png")}
    headers = {"X-Api-Key": api_key}
    resp = requests.put(f"{api_url}/movies/{movie_id}/poster", files=files, headers=headers)
    resp.raise_for_status()
    return resp.json()


def process_movie(movie, api_url, invokeai_url, templates_base, api_key, verbose=False):
    """Generate and upload a poster for a single movie."""
    movie_id = movie["movie_id"]
    title = movie["title"]
    log(f"Processing movie {movie_id}: '{title}'")

    # Step 1: Build the image prompt using the AI text model
    log(f"  Generating image prompt for '{title}'...")
    try:
        image_prompt = build_image_prompt(movie, templates_base, verbose)
    except Exception as e:
        log(f"  Failed to generate image prompt: {e}", "error")
        if verbose:
            log(traceback.format_exc(), "verbose")
        return False

    if verbose:
        log(f"  Image prompt: {image_prompt}", "verbose")

    # Step 2: Enqueue generation in InvokeAI
    log(f"  Enqueuing image generation in InvokeAI...")
    try:
        enqueue_result = enqueue_generation(invokeai_url, image_prompt)
        batch_id = enqueue_result.get("batch", {}).get("batch_id")
        if not batch_id:
            log(f"  No batch_id in enqueue response", "error")
            return False
        log(f"  Batch enqueued: {batch_id}")
    except Exception as e:
        log(f"  Failed to enqueue generation: {e}", "error")
        return False

    # Step 3: Wait for generation to complete
    log(f"  Waiting for image generation...")
    result = wait_for_batch(invokeai_url, batch_id)
    if not result["success"]:
        log(f"  Image generation failed: {result['status']}", "error")
        return False
    log(f"  Image generation complete")

    # Step 4: Download the generated image from InvokeAI
    try:
        image_name = get_latest_image_name(invokeai_url)
        if not image_name:
            log(f"  Could not find generated image in InvokeAI", "error")
            return False
        log(f"  Downloading image: {image_name}")
        image_data = download_image(invokeai_url, image_name)
    except Exception as e:
        log(f"  Failed to download image: {e}", "error")
        return False

    # Step 5: Upload poster to the media-generator API
    try:
        log(f"  Uploading poster for movie {movie_id}...")
        upload_poster(api_url, movie_id, image_data, api_key)
        log(f"  Poster uploaded for '{title}'", "success")
        return True
    except Exception as e:
        log(f"  Failed to upload poster: {e}", "error")
        return False


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Batch generate movie posters for movies missing them."
    )
    parser.add_argument(
        "--media-api", default=DEFAULT_MEDIA_API_URL,
        help=f"Media generator API URL (default: {DEFAULT_MEDIA_API_URL})"
    )
    parser.add_argument(
        "--invokeai", default=DEFAULT_INVOKEAI_URL,
        help=f"InvokeAI API URL (default: {DEFAULT_INVOKEAI_URL})"
    )
    parser.add_argument(
        "--limit", type=int, default=100,
        help="Maximum number of movies to process (default: 100)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--api-key", default=None,
        help="API key for the media-generator API (default: from API_KEY env var)"
    )
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("API_KEY", "")
    if not api_key:
        log("No API key provided. Set API_KEY in .env or pass --api-key", "error")
        return 1

    templates_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

    print(f"\n=== Batch Poster Generator ===")
    print(f"  Media API: {args.media_api}")
    print(f"  InvokeAI:  {args.invokeai}")
    print()

    # Fetch movies missing posters
    try:
        movies = get_movies_missing_posters(args.media_api, limit=args.limit)
    except Exception as e:
        log(f"Failed to fetch movies: {e}", "error")
        return 1

    if not movies:
        log("No movies missing posters. Nothing to do.", "success")
        return 0

    log(f"Found {len(movies)} movie(s) missing posters")
    print()

    success_count = 0
    fail_count = 0

    for movie in movies:
        try:
            if process_movie(movie, args.media_api, args.invokeai, templates_base, api_key, args.verbose):
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            log(f"Unexpected error processing movie {movie.get('movie_id')}: {e}", "error")
            fail_count += 1
        print()

    print(f"=== Complete: {success_count} succeeded, {fail_count} failed ===\n")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
