#!/usr/bin/env python
"""
Media Generator - CLI Entry Point

This is the main entry point for the media generator CLI.
The actual implementation is in lib.cli, and the library functionality
is in lib.generator.

Usage:
    python media_generator.py --count 5 --verbose
    python media_generator.py -c 3 -d  # dry run

For library usage, import from the lib package:
    from lib import MediaGenerator, GenerationResult, GenerationStats
    
    generator = MediaGenerator(verbose=True)
    result = generator.generate_single()
"""

import sys
from lib.cli import main

if __name__ == "__main__":
    sys.exit(main())
