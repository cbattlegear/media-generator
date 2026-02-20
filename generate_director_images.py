"""
Generate director headshot images from prompts using InvokeAI.

Reads prompt files from director_image_prompt/{director_id}.prompt.txt,
sends each to a local InvokeAI instance running FLUX.2 Klein,
and saves the resulting image as director_image_prompt/{director_id}.png.

Usage:
    python generate_director_images.py
    python generate_director_images.py --invokeai http://localhost:9090
"""

import argparse
import glob
import os
import random
import re
import sys
import time
from io import BytesIO

import requests
from PIL import Image

DEFAULT_INVOKEAI_URL = "http://localhost:9090"
PROMPT_DIR = "director_image_prompt"
POLL_INTERVAL = 5
GENERATION_TIMEOUT = 300

# Headshot image dimensions (square for profile pictures)
IMAGE_WIDTH = 192
IMAGE_HEIGHT = 192

# Model names to look up from InvokeAI (same as batch poster generator)
MAIN_MODEL_NAME = "FLUX.2 Klein 4B (GGUF Q4)"
VAE_MODEL_NAME = "FLUX.2 VAE"
ENCODER_MODEL_NAME = "FLUX.2 Klein Qwen3 4B Encoder"


def _random_id(length=10):
    """Generate a random alphanumeric ID for graph node suffixes."""
    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(random.choice(chars) for _ in range(length))


def lookup_invokeai_models(invokeai_url):
    """
    Query InvokeAI for installed models and return the three required model records.

    Returns a dict with keys: 'main', 'vae', 'qwen3_encoder'.
    """
    resp = requests.get(f"{invokeai_url}/api/v2/models/")
    resp.raise_for_status()
    data = resp.json()
    models = data.get("models", data) if isinstance(data, dict) else data

    found = {}
    for model in models:
        name = model.get("name", "")
        if name == MAIN_MODEL_NAME:
            found["main"] = model
        elif name == VAE_MODEL_NAME:
            found["vae"] = model
        elif name == ENCODER_MODEL_NAME:
            found["qwen3_encoder"] = model

    missing = [
        n
        for role, n in [
            ("main", MAIN_MODEL_NAME),
            ("vae", VAE_MODEL_NAME),
            ("qwen3_encoder", ENCODER_MODEL_NAME),
        ]
        if role not in found
    ]
    if missing:
        raise RuntimeError(f"Required models not found in InvokeAI: {missing}")

    return found


def _model_ref(model, fields=("key", "hash", "name", "base", "type")):
    """Extract a minimal model reference dict for use in graph nodes."""
    return {k: model.get(k) for k in fields}


def build_invokeai_graph(models):
    """Build the InvokeAI FLUX graph for headshot generation."""
    main_model = models["main"]
    vae_model = models["vae"]
    encoder_model = models["qwen3_encoder"]

    graph_id = f"flux_graph:{_random_id()}"
    prompt_id = f"positive_prompt:{_random_id()}"
    seed_id = f"seed:{_random_id()}"
    loader_id = f"flux2_klein_model_loader:{_random_id()}"
    encoder_id = f"flux2_klein_text_encoder:{_random_id()}"
    denoise_id = f"flux2_denoise:{_random_id()}"
    metadata_id = f"core_metadata:{_random_id()}"
    output_id = f"canvas_output:{_random_id()}"

    main_model_full = {
        "key": main_model["key"],
        "hash": main_model["hash"],
        "path": main_model.get("path", ""),
        "file_size": main_model.get("file_size", 0),
        "name": main_model["name"],
        "description": main_model.get("description", ""),
        "source": main_model.get("source", ""),
        "source_type": main_model.get("source_type", "url"),
        "source_api_response": main_model.get("source_api_response"),
        "cover_image": main_model.get("cover_image"),
        "type": "main",
        "trigger_phrases": main_model.get("trigger_phrases"),
        "default_settings": main_model.get("default_settings", {}),
        "config_path": main_model.get("config_path"),
        "base": "flux2",
        "format": main_model.get("format", "gguf_quantized"),
        "variant": main_model.get("variant", "klein_4b"),
    }

    vae_ref = _model_ref(vae_model)
    encoder_ref = _model_ref(encoder_model)

    graph = {
        "id": graph_id,
        "nodes": {
            prompt_id: {
                "id": prompt_id, "type": "string",
                "is_intermediate": True, "use_cache": True,
            },
            seed_id: {
                "id": seed_id, "type": "integer",
                "is_intermediate": True, "use_cache": True,
            },
            loader_id: {
                "type": "flux2_klein_model_loader", "id": loader_id,
                "model": main_model_full,
                "vae_model": vae_ref,
                "qwen3_encoder_model": encoder_ref,
                "is_intermediate": True, "use_cache": True,
            },
            encoder_id: {
                "type": "flux2_klein_text_encoder", "id": encoder_id,
                "is_intermediate": True, "use_cache": True,
            },
            denoise_id: {
                "type": "flux2_denoise", "id": denoise_id,
                "num_steps": 30, "is_intermediate": True, "use_cache": True,
                "denoising_start": 0, "denoising_end": 1,
                "width": IMAGE_WIDTH, "height": IMAGE_HEIGHT,
            },
            metadata_id: {
                "id": metadata_id, "type": "core_metadata",
                "is_intermediate": True, "use_cache": True,
                "model": _model_ref(main_model),
                "steps": 30,
                "vae": vae_ref,
                "qwen3_encoder": encoder_ref,
                "width": IMAGE_WIDTH, "height": IMAGE_HEIGHT,
                "generation_mode": "flux2_txt2img",
            },
            output_id: {
                "type": "flux2_vae_decode", "id": output_id,
                "is_intermediate": False, "use_cache": False,
            },
        },
        "edges": [
            {"source": {"node_id": loader_id, "field": "qwen3_encoder"}, "destination": {"node_id": encoder_id, "field": "qwen3_encoder"}},
            {"source": {"node_id": loader_id, "field": "max_seq_len"}, "destination": {"node_id": encoder_id, "field": "max_seq_len"}},
            {"source": {"node_id": loader_id, "field": "transformer"}, "destination": {"node_id": denoise_id, "field": "transformer"}},
            {"source": {"node_id": loader_id, "field": "vae"}, "destination": {"node_id": denoise_id, "field": "vae"}},
            {"source": {"node_id": loader_id, "field": "vae"}, "destination": {"node_id": output_id, "field": "vae"}},
            {"source": {"node_id": prompt_id, "field": "value"}, "destination": {"node_id": encoder_id, "field": "prompt"}},
            {"source": {"node_id": encoder_id, "field": "conditioning"}, "destination": {"node_id": denoise_id, "field": "positive_text_conditioning"}},
            {"source": {"node_id": seed_id, "field": "value"}, "destination": {"node_id": denoise_id, "field": "seed"}},
            {"source": {"node_id": denoise_id, "field": "latents"}, "destination": {"node_id": output_id, "field": "latents"}},
            {"source": {"node_id": seed_id, "field": "value"}, "destination": {"node_id": metadata_id, "field": "seed"}},
            {"source": {"node_id": prompt_id, "field": "value"}, "destination": {"node_id": metadata_id, "field": "positive_prompt"}},
            {"source": {"node_id": metadata_id, "field": "metadata"}, "destination": {"node_id": output_id, "field": "metadata"}},
        ],
    }

    return graph, seed_id, prompt_id


def log(message, level="info"):
    """Print a formatted log message."""
    prefix = {
        "info": "[INFO]", "success": "[OK]", "error": "[ERROR]",
        "warning": "[WARN]",
    }
    print(f"  {prefix.get(level, '[INFO]')} {message}")


def enqueue_generation(invokeai_url, graph, seed_id, prompt_id, prompt, seed=None):
    """Enqueue an image generation batch in InvokeAI."""
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    payload = {
        "prepend": False,
        "batch": {
            "graph": graph,
            "runs": 1,
            "data": [
                [{"node_path": seed_id, "field_name": "value", "items": [seed]}],
                [{"node_path": prompt_id, "field_name": "value", "items": [prompt]}],
            ],
            "origin": "generate",
            "destination": "generate",
        },
    }
    resp = requests.post(
        f"{invokeai_url}/api/v1/queue/default/enqueue_batch", json=payload
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
        f"{invokeai_url}/api/v1/images/", params={"limit": 1, "offset": 0}
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


def find_prompt_files(prompt_dir):
    """Find all prompt files and return sorted list of (director_id, filepath) tuples."""
    pattern = os.path.join(prompt_dir, "*.prompt.txt")
    results = []
    for filepath in glob.glob(pattern):
        basename = os.path.basename(filepath)
        match = re.match(r"^(\d+)\.prompt\.txt$", basename)
        if match:
            results.append((int(match.group(1)), filepath))
    return sorted(results, key=lambda x: x[0])


def main():
    parser = argparse.ArgumentParser(
        description="Generate director headshot images from prompts using InvokeAI"
    )
    parser.add_argument(
        "--invokeai", default=DEFAULT_INVOKEAI_URL,
        help=f"InvokeAI API URL (default: {DEFAULT_INVOKEAI_URL})",
    )
    args = parser.parse_args()

    print(f"\n=== Director Headshot Generator ===")
    print(f"  InvokeAI:    {args.invokeai}")
    print(f"  Prompt dir:  {PROMPT_DIR}")
    print()

    # Look up model keys from InvokeAI
    log("Looking up models from InvokeAI...")
    try:
        models = lookup_invokeai_models(args.invokeai)
        graph, seed_id, prompt_id = build_invokeai_graph(models)
        log(f"Models found: {', '.join(m['name'] for m in models.values())}", "success")
    except Exception as e:
        log(f"Failed to look up models from InvokeAI: {e}", "error")
        return 1

    # Find prompt files
    prompt_files = find_prompt_files(PROMPT_DIR)
    if not prompt_files:
        log(f"No prompt files found in {PROMPT_DIR}/")
        return 0

    log(f"Found {len(prompt_files)} prompt files")
    print()

    success_count = 0
    fail_count = 0
    skip_count = 0

    for director_id, prompt_path in prompt_files:
        output_path = os.path.join(PROMPT_DIR, f"{director_id}.png")

        if os.path.exists(output_path):
            log(f"[{director_id}] skipped (image already exists)")
            skip_count += 1
            continue

        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_text = f.read().strip()

        if not prompt_text:
            log(f"[{director_id}] skipped (empty prompt file)", "warning")
            skip_count += 1
            continue

        log(f"[{director_id}] Generating headshot...")

        # Enqueue generation
        try:
            enqueue_result = enqueue_generation(
                args.invokeai, graph, seed_id, prompt_id, prompt_text
            )
            batch_id = enqueue_result.get("batch", {}).get("batch_id")
            if not batch_id:
                log(f"[{director_id}] No batch_id in enqueue response", "error")
                fail_count += 1
                continue
        except Exception as e:
            log(f"[{director_id}] Failed to enqueue: {e}", "error")
            fail_count += 1
            continue

        # Wait for completion
        result = wait_for_batch(args.invokeai, batch_id)
        if not result["success"]:
            log(f"[{director_id}] Generation failed: {result['status']}", "error")
            fail_count += 1
            continue

        # Download and save
        try:
            image_name = get_latest_image_name(args.invokeai)
            if not image_name:
                log(f"[{director_id}] Could not find generated image", "error")
                fail_count += 1
                continue

            image_data = download_image(args.invokeai, image_name)
            with Image.open(image_data) as img:
                img.save(output_path, "PNG")

            log(f"[{director_id}] Saved {output_path}", "success")
            success_count += 1
        except Exception as e:
            log(f"[{director_id}] Failed to download/save: {e}", "error")
            fail_count += 1

    print()
    print(
        f"=== Complete: {success_count} generated, {skip_count} skipped, "
        f"{fail_count} failed ==="
    )
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
