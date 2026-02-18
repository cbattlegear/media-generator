#!/usr/bin/env python
"""
One-time migration script to convert existing poster images to webp thumbnails.

Connects to SQL Server, finds movies with poster_url pointing to an image file,
converts each original to a 512x896 webp thumbnail with _thumb suffix,
and updates the DB poster_url to point to the thumbnail.

Originals are kept alongside the thumbnails.

Usage:
    python convert_thumbnails.py
    python convert_thumbnails.py --dry-run
    python convert_thumbnails.py --images-dir /path/to/images

Environment variables (or .env file):
    DB_SERVER, DB_NAME, DB_USER, DB_PASSWORD, DB_DRIVER
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

THUMB_WIDTH = 512
THUMB_HEIGHT = 896
THUMB_QUALITY = 80


def get_db_engine():
    """Create a SQLAlchemy engine from environment variables."""
    server = os.getenv("DB_SERVER", "localhost")
    database = os.getenv("DB_NAME", "BattleCabbageVideo")
    username = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    driver = os.getenv("DB_DRIVER", "ODBC Driver 18 for SQL Server")

    if username and password:
        connection_url = URL.create(
            "mssql+pyodbc",
            username=username,
            password=password,
            host=server,
            database=database,
            query={"driver": driver, "TrustServerCertificate": "yes"},
        )
    else:
        connection_url = URL.create(
            "mssql+pyodbc",
            host=server,
            database=database,
            query={
                "driver": driver,
                "Trusted_Connection": "yes",
                "TrustServerCertificate": "yes",
            },
        )

    return create_engine(connection_url)


def create_thumbnail(source_path: Path, thumb_path: Path) -> bool:
    """Create a 512x896 webp thumbnail from the source image."""
    try:
        with Image.open(source_path) as img:
            img = img.convert("RGB")
            img = img.resize((THUMB_WIDTH, THUMB_HEIGHT), Image.LANCZOS)
            img.save(thumb_path, "WEBP", quality=THUMB_QUALITY)
        return True
    except Exception as e:
        print(f"  [ERROR] Failed to create thumbnail: {e}")
        return False


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Convert existing poster images to webp thumbnails."
    )
    parser.add_argument(
        "--images-dir",
        default=None,
        help="Path to images directory (default: ./images)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    args = parser.parse_args()

    images_dir = Path(
        args.images_dir
        or os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
    )

    if not images_dir.exists():
        print(f"[ERROR] Images directory not found: {images_dir}")
        return 1

    print(f"=== Thumbnail Migration ===")
    print(f"  Images dir: {images_dir}")
    print(f"  Dry run: {args.dry_run}")
    print()

    engine = get_db_engine()

    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT movie_id, poster_url FROM movies "
                "WHERE poster_url IS NOT NULL AND poster_url != 'movie_poster_url.jpeg'"
            )
        ).fetchall()

    print(f"  Found {len(rows)} movie(s) with poster images")
    print()

    success_count = 0
    skip_count = 0
    fail_count = 0

    for movie_id, poster_url in rows:
        # poster_url may be "/images/movie_42.png" or "/images/subdir/movie_42.png"
        # Strip the leading "/images/" to get the relative path within images_dir
        rel_path = poster_url.replace("/images/", "", 1) if poster_url.startswith("/images/") else os.path.basename(poster_url)
        source_path = images_dir / rel_path

        # Build thumbnail path preserving subdirectory structure
        source_p = Path(rel_path)
        thumb_filename = f"{source_p.stem}_thumb.webp"
        thumb_rel = str(source_p.parent / thumb_filename) if source_p.parent != Path(".") else thumb_filename
        thumb_path = images_dir / thumb_rel
        thumb_url = f"/images/{thumb_rel.replace(os.sep, '/')}"

        # Skip if already pointing to a thumbnail
        if "_thumb.webp" in poster_url:
            skip_count += 1
            continue

        if not source_path.exists():
            print(f"  [SKIP] movie {movie_id}: source not found ({rel_path})")
            skip_count += 1
            continue

        if args.dry_run:
            print(f"  [DRY] movie {movie_id}: {rel_path} -> {thumb_rel}")
            success_count += 1
            continue

        # Create thumbnail
        print(f"  Converting movie {movie_id}: {rel_path} -> {thumb_rel}")
        if not create_thumbnail(source_path, thumb_path):
            fail_count += 1
            continue

        # Update DB
        try:
            with engine.begin() as conn:
                conn.execute(
                    text("UPDATE movies SET poster_url = :url WHERE movie_id = :id"),
                    {"url": thumb_url, "id": movie_id},
                )
            success_count += 1
        except Exception as e:
            print(f"  [ERROR] Failed to update DB for movie {movie_id}: {e}")
            fail_count += 1

    print()
    print(f"=== Complete: {success_count} converted, {skip_count} skipped, {fail_count} failed ===")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
