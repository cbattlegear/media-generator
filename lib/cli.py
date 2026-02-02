"""
Command-line interface for the media-generator library.

This module provides the CLI frontend for generating AI-powered media objects.
It can be run directly or through the installed entry point.

Usage:
    python -m lib.cli --count 5 --verbose
    media-generator --count 5 --verbose  (if installed via pip)
"""

import argparse
import json
import os
import sys
from typing import Optional

from lib.generator import MediaGenerator, GenerationResult


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        prog="media-generator",
        description="Generate AI-powered fake media objects including movie titles, "
                    "descriptions, critic reviews, and poster images.",
        epilog="Example: %(prog)s --count 5 --verbose"
    )
    
    parser.add_argument(
        "-c", "--count",
        type=int,
        default=None,
        help="Number of media objects to generate (default: 1, or GENERATE_COUNT env var)"
    )
    
    parser.add_argument(
        "-d", "--dry-run",
        action="store_true",
        help="Dry run mode - generate without saving files to disk"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed progress including prompts and completions"
    )
    
    parser.add_argument(
        "--no-image",
        action="store_true",
        help="Skip image generation (only generate metadata and critic review)"
    )
    
    parser.add_argument(
        "-j", "--json",
        action="store_true",
        help="Output generated media as JSON to stdout instead of saving to files"
    )
    
    parser.add_argument(
        "-w", "--working-dir",
        type=str,
        default=None,
        help="Working directory containing templates (default: current directory)"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0"
    )
    
    return parser


def progress_callback(current: int, total: int, result: GenerationResult) -> None:
    """Callback for progress updates during batch generation."""
    status = "✓" if result.success else "✗"
    title = result.title or "Unknown"
    if result.success:
        print(f"  [{current}/{total}] {status} {title}")
    else:
        print(f"  [{current}/{total}] {status} {title} - {result.error}")


def main(args: Optional[list] = None) -> int:
    """
    Main entry point for the CLI.
    
    Args:
        args: Command line arguments (defaults to sys.argv if None).
    
    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = create_parser()
    parsed_args = parser.parse_args(args)
    
    # Determine count: CLI arg > env var > default of 1
    count = parsed_args.count
    if count is None:
        env_count = os.environ.get("GENERATE_COUNT")
        if env_count and env_count.isdigit():
            count = int(env_count)
        else:
            count = 1
    
    # Validate count
    if count < 1:
        parser.error("Count must be at least 1")
        return 1
    
    try:
        # Create the generator
        generator = MediaGenerator(
            working_dir=parsed_args.working_dir,
            verbose=parsed_args.verbose,
            dry_run=parsed_args.dry_run,
            skip_image=parsed_args.no_image
        )
        
        if parsed_args.dry_run:
            print("Dry run mode enabled - generated media objects will not be saved", file=sys.stderr)
        
        if parsed_args.no_image:
            print("Image generation disabled - only metadata and reviews will be created", file=sys.stderr)
        
        # Determine if we should save to files
        save_to_file = not parsed_args.json and not parsed_args.dry_run
        
        # Run generation
        if count == 1:
            result = generator.generate_single(save=save_to_file)
            if result.success:
                if parsed_args.json:
                    # Output JSON to stdout
                    if result.media_object:
                        print(json.dumps(result.media_object.to_json(), indent=2))
                else:
                    print(f"\n✓ Successfully generated: {result.title}")
                    if result.generation_time:
                        print(f"  Generation time: {result.generation_time}")
                return 0
            else:
                print(f"\n✗ Generation failed: {result.error}", file=sys.stderr)
                return 1
        else:
            # For batch generation with JSON output, collect all results
            results = []
            
            def json_progress_callback(current: int, total: int, res: GenerationResult) -> None:
                if res.success and res.media_object:
                    results.append(res.media_object.to_json())
                if not parsed_args.json:
                    progress_callback(current, total, res)
            
            stats = generator.generate_batch(
                count=count,
                save=save_to_file,
                on_progress=json_progress_callback if parsed_args.json or not parsed_args.verbose else None
            )
            
            if parsed_args.json:
                # Output all results as JSON array
                print(json.dumps(results, indent=2))
            else:
                print(f"\n{'=' * 50}")
                print("Generation Complete")
                print(f"{'=' * 50}")
                print(f"  Requested:  {stats.total_requested}")
                print(f"  Successful: {stats.success_count}")
                if stats.total_failures > 0:
                    print(f"  Failed:     {stats.total_failures}")
                    print(f"    - Prompt failures: {stats.completion_fail_count}")
                    print(f"    - Image failures:  {stats.image_fail_count}")
                    print(f"    - Save failures:   {stats.save_fail_count}")
                if stats.total_time:
                    print(f"  Total time: {stats.total_time}")
            
            return 0 if stats.success_count == stats.total_requested else 1
            
    except KeyboardInterrupt:
        print("\n\nGeneration cancelled by user.")
        return 130
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if parsed_args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
