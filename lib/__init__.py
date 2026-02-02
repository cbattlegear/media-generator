"""
media-generator library

A Python library for generating AI-powered fake media objects including
movie titles, descriptions, critic reviews, and poster images.

Example usage:
    ```python
    from lib import MediaGenerator, GenerationResult, GenerationStats
    
    # Create a generator instance
    generator = MediaGenerator(verbose=True)
    
    # Generate a single media object
    result = generator.generate_single()
    if result.success:
        print(f"Generated: {result.title}")
    
    # Generate multiple media objects
    stats = generator.generate_batch(count=5)
    print(f"Successfully generated {stats.success_count} items")
    ```

Classes:
    MediaGenerator: Main class for generating media objects.
    GenerationResult: Result of a single generation attempt.
    GenerationStats: Statistics for batch generation runs.
    
For more details, see the MediaGenerator class documentation.
"""

from lib.generator import MediaGenerator, GenerationResult, GenerationStats
from lib.media import media
from lib.image import image
from lib.critic_review import criticReview
from lib.process_helper import processHelper

__all__ = [
    # Main API
    "MediaGenerator",
    "GenerationResult", 
    "GenerationStats",
    # Lower-level components (for advanced usage)
    "media",
    "image",
    "criticReview",
    "processHelper",
]

__version__ = "1.0.0"
