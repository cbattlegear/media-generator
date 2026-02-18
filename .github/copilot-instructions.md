# Copilot Instructions for media-generator

## Architecture

This is an AI-powered fake movie generator that produces complete movie metadata (title, tagline, description, MPAA rating, critic reviews) and poster images using Azure OpenAI or local Ollama models. It is part of the larger [Battlecabbage Media](https://github.com/Battlecabbage-Media) project.

### Two model backends

The `MODEL_TYPE` environment variable (`azure_openai` or `local`) selects which backend is used. Each backend has a parallel class hierarchy:

- **Azure OpenAI**: `lib/aoai_model.py` — `aoaiText`, `aoaiImage`, `aoaiVision` (GPT-4, DALL-E 3, GPT-4 Vision)
- **Local/Ollama**: `lib/ollama_model.py` — `ollamaText`, `ollamaImage` (ComfyUI/StableDiffusion), `ollamaVision` (llama3.2-vision)

All model classes share the same interface: `system_prompt`, `user_prompt`, `generateResponse()` (text/vision) or `generateImage()` (image).

### Generation pipeline (lib/)

1. **Prompt building** — `media.generateObjectPrompt()` loads `templates/prompts.json`, picks a random prompt template, and fills `{placeholders}` with random values from template JSON files (e.g., `templates/genres.json`, `templates/actors.json`). The filled values are tracked in `object_prompt_list`.
2. **Movie generation** — `media.generateObject()` sends the prompt to the text model, parses JSON from the completion to populate title, tagline, description, etc.
3. **Critic review** — `criticReview` builds a review prompt from movie details and gets a JSON response with `critic_score`, `critic_review`, `critic_tone`.
4. **Image generation** — `image.generateImagePrompt()` asks the text model for a detailed DALL-E/SD prompt (including font selection from system fonts), then `image.generateImage()` sends it to the image model with up to 5 retries.
5. **Image processing** — `image.processImage()` sends the generated image to the vision model to determine title placement (top/middle/bottom), font color, and whether the image already has text. If no text, it composites the title onto the poster using Pillow.

### Key components

- **`lib/generator.py`** — `MediaGenerator` class orchestrates the full pipeline. Returns `GenerationResult` / `GenerationStats` dataclasses.
- **`lib/process_helper.py`** — Shared utilities: logging (with colored console output), JSON extraction from AI completions, output path generation (`outputs/YYYY/MM/DD/`), process ID creation.
- **`api/`** — FastAPI REST API (`api/main.py`) with SQLAlchemy ORM models (`api/models.py`). Stores movies in SQL Server (schema in `db-init/init-db.sql`). Uses `slowapi` for rate limiting and `X-Api-Key` header auth for write operations.
- **`batch_poster_generate.py`** — Standalone batch script that queries the API for movies missing posters, generates prompts via the text model, sends them to InvokeAI (FLUX model) for image generation, and uploads results back to the API.

### Database

SQL Server with tables: `movies`, `genres`, `actors`, `directors`, `criticreviews`, `actorstomoviesjoin`, `directorstomoviesjoin`. Schema is initialized by `db-init/init-db.sql`. Docker Compose (`docker-compose.yml`) runs SQL Server + the API.

### Image handling

- Poster images are stored in the `/images/` directory.
- **Thumbnails**: Each poster has a `_thumb.webp` variant at 512×896 quality 80 (e.g., `movie_42.png` original → `movie_42_thumb.webp` thumbnail). The `poster_url` DB field points to the thumbnail.
- The `PUT /movies/{movie_id}/poster` API endpoint accepts a required `file` (original) and optional `thumbnail` upload. When a thumbnail is provided, `poster_url` points to it.
- The batch poster job (`batch_poster_generate.py`) creates thumbnails automatically with Pillow before uploading both files.
- `convert_thumbnails.py` is a one-time migration script that converts existing images to thumbnails and updates the DB. Can run standalone or via `Dockerfile.convert`.

## Build & Run

```bash
# Install dependencies
pip install -r requirements.txt

# Or install as a package (includes dev tools)
pip install -e ".[dev]"

# CLI usage
python media_generator.py --count 5 --verbose
python media_generator.py --dry-run --no-image --json  # metadata only, no files saved
python media_generator.py -c 1 -j  # single movie as JSON to stdout

# Run the API
uvicorn api.main:app --reload

# Docker (API + SQL Server)
docker compose up

# Batch poster generation (requires running API + InvokeAI)
python batch_poster_generate.py --limit 10 --verbose

# One-time thumbnail migration (converts existing images, updates DB)
python convert_thumbnails.py
python convert_thumbnails.py --dry-run  # preview only
```

## Linting & Formatting

```bash
# Formatting (line length 100)
black .

# Linting (line length 100)
ruff check .

# Type checking
mypy .
```

Configured in `pyproject.toml`: line length is 100, target Python 3.9+.

## Conventions

- **Class naming**: Domain classes use lowercase names (`media`, `image`, `criticReview`, `processHelper`) — this is intentional, not PEP 8-style. The `MediaGenerator` wrapper class and dataclasses use PascalCase.
- **camelCase methods**: Internal library methods use camelCase (`generateObject`, `buildCriticPrompt`, `saveMediaObject`). The `MediaGenerator` public API uses snake_case (`generate_single`, `generate_batch`).
- **JSON extraction pattern**: AI model responses are expected to contain JSON. `processHelper.extractJson()` finds the first `{...}` block in the completion text. All generation methods follow the pattern: call model → extract JSON → validate required fields → populate object.
- **Template system**: Prompt templates in `templates/prompts.json` use `{key}` placeholders (not Python f-strings). These are resolved by `media.parseTemplate()` which picks random values from corresponding `templates/{key}.json` files.
- **Private fields**: Fields prefixed with `_` (e.g., `_process`, `_prompt_file_path`) are excluded from serialization in `to_json()` methods.
- **Environment config**: All AI model configuration comes from environment variables (see `.example.env`). The `MODEL_TYPE` variable switches between `azure_openai` and `local` backends throughout the codebase.
