"""
MediaGenerator - Core library class for generating AI-powered media objects.

This module provides the main MediaGenerator class that handles the complete
workflow of generating media objects including titles, descriptions, critic
reviews, and poster images.
"""

import os
import datetime
from typing import Optional, Callable
from dataclasses import dataclass
from dotenv import load_dotenv

from lib.process_helper import processHelper
from lib.media import media
from lib.image import image
from lib.critic_review import criticReview


@dataclass
class GenerationResult:
    """Result of a single media generation attempt."""
    success: bool
    media_object: Optional[media] = None
    title: str = ""
    error: Optional[str] = None
    generation_time: Optional[datetime.timedelta] = None


@dataclass
class GenerationStats:
    """Statistics for a batch generation run."""
    total_requested: int = 0
    success_count: int = 0
    completion_fail_count: int = 0
    image_fail_count: int = 0
    save_fail_count: int = 0
    total_time: Optional[datetime.timedelta] = None

    @property
    def total_failures(self) -> int:
        return self.completion_fail_count + self.image_fail_count + self.save_fail_count


class MediaGenerator:
    """
    Main class for generating AI-powered media objects.
    
    This class provides the core functionality to generate fake movie/media
    information including titles, descriptions, critic reviews, and poster images
    using AI models (Azure OpenAI or local Ollama models).
    
    Example usage:
        ```python
        from lib.generator import MediaGenerator
        
        generator = MediaGenerator()
        
        # Generate a single media object
        result = generator.generate_single()
        if result.success:
            print(f"Generated: {result.title}")
        
        # Generate multiple media objects
        stats = generator.generate_batch(count=5)
        print(f"Successfully generated {stats.success_count} items")
        ```
    
    Attributes:
        verbose: Whether to output detailed progress information.
        dry_run: If True, generates but doesn't save to disk.
        templates_base: Path to the templates directory.
        prompt_file_path: Path to the prompts.json file.
    """
    
    def __init__(
        self,
        working_dir: Optional[str] = None,
        verbose: bool = False,
        dry_run: bool = False,
        skip_image: bool = False,
        message_callback: Optional[Callable[[str, str], None]] = None
    ):
        """
        Initialize the MediaGenerator.
        
        Args:
            working_dir: Working directory containing templates. Defaults to current directory.
            verbose: Enable verbose output for debugging.
            dry_run: Generate without saving files to disk.
            skip_image: Skip image generation (only generate metadata and reviews).
            message_callback: Optional callback for log messages. Receives (message, level).
        """
        self.verbose = verbose
        self.dry_run = dry_run
        self.skip_image = skip_image
        self._message_callback = message_callback
        
        # Set up working directory and paths
        self.working_dir = working_dir or os.getcwd()
        self.templates_base = os.path.join(self.working_dir, "templates")
        self.prompt_file_path = os.path.join(self.templates_base, "prompts.json")
        
        # Initialize process helper
        self._process = processHelper()
        
        # Load environment
        self._load_environment()
    
    def _load_environment(self) -> None:
        """Load and validate environment variables."""
        load_dotenv()
        
        self._process.envCheck("MODEL_TYPE")
        model_type = os.getenv("MODEL_TYPE", "").lower()
        
        if model_type != "azure_openai":
            self._process.envCheck("AZURE_OPENAI_TEXT_ENDPOINT_KEY")
            self._process.envCheck("AZURE_OPENAI_TEXT_API_VERSION")
            self._process.envCheck("AZURE_OPENAI_TEXT_ENDPOINT")
            self._process.envCheck("AZURE_OPENAI_TEXT_DEPLOYMENT_NAME")
            self._process.envCheck("AZURE_OPENAI_IMAGE_ENDPOINT_KEY")
            self._process.envCheck("AZURE_OPENAI_IMAGE_API_VERSION")
            self._process.envCheck("AZURE_OPENAI_IMAGE_ENDPOINT")
            self._process.envCheck("AZURE_OPENAI_IMAGE_DEPLOYMENT_NAME")
            self._process.envCheck("AZURE_OPENAI_VISION_ENDPOINT")
            self._process.envCheck("AZURE_OPENAI_VISION_ENDPOINT_KEY")
            self._process.envCheck("AZURE_OPENAI_VISION_DEPLOYMENT_NAME")
            self._process.envCheck("AZURE_OPENAI_VISION_API_VERSION")
        elif model_type != "local":
            self._process.envCheck("LOCAL_MODEL_NAME")
    
    def _validate_setup(self) -> bool:
        """Validate that required files exist."""
        if not os.path.exists(self.prompt_file_path):
            self._output_message(
                f"Error opening prompt file: {self.prompt_file_path}. Check that it exists in templates!",
                "error"
            )
            return False
        return True
    
    def _output_message(self, message: str, level: str) -> None:
        """Output a message through the callback or process helper."""
        if self._message_callback:
            self._message_callback(message, level)
        self._process.outputMessage(message, level)
    
    def generate_single(self, save: bool = True) -> GenerationResult:
        """
        Generate a single media object.
        
        Args:
            save: Whether to save the generated media to disk. 
                  Overridden by dry_run setting.
        
        Returns:
            GenerationResult with success status and generated media object.
        """
        if not self._validate_setup():
            return GenerationResult(success=False, error="Setup validation failed")
        
        object_start_time = datetime.datetime.now()
        self._process.createProcessId()
        
        # Create media object
        media_object = media(
            self._process,
            self.prompt_file_path,
            self.templates_base,
            self.verbose
        )
        
        # Build the prompt
        self._output_message("Building object prompt", "")
        if not media_object.generateObjectPrompt():
            return GenerationResult(
                success=False,
                error="Failed to generate object prompt"
            )
        
        if self.verbose:
            import json
            self._output_message(f"Object prompt:\n {media_object.movie_prompt}", "verbose")
            self._output_message(f"Template list:\n {json.dumps(media_object.object_prompt_list, indent=4)}", "verbose")
        
        # Generate the media object
        self._output_message("Submitting object prompt for completion", "")
        if not media_object.generateObject():
            return GenerationResult(
                success=False,
                error="Failed to generate media object from prompt"
            )
        
        if self.verbose:
            import json
            self._output_message(f"Object completion:\n {json.dumps(media_object.to_json(), indent=4)}", "verbose")
        
        self._output_message(f"Generated media object '{media_object.title}'", "")
        
        # Create critic review
        self._output_message(f"Creating critic review for '{media_object.title}'", "")
        review = criticReview(media_object, self.verbose)
        
        if not review.buildCriticPrompt():
            return GenerationResult(
                success=False,
                title=media_object.title,
                error="Failed to build critic prompt"
            )
        
        if self.verbose:
            self._output_message(f"Critic prompt:\n {review.prompt}", "verbose")
        
        if not review.generateCriticReview():
            return GenerationResult(
                success=False,
                title=media_object.title,
                error="Failed to generate critic review"
            )
        
        if self.verbose:
            self._output_message(f"Critic review:\n {media_object.reviews}", "verbose")
        
        media_object.reviews.append(review.to_json())
        self._output_message(f"Critic review created for '{media_object.title}'", "")
        
        # Generate image (unless skipped)
        image_object = None
        if not self.skip_image:
            self._output_message(f"Creating image for '{media_object.title}'", "")
            image_object = image(media_object)
            
            self._output_message(f"Generating image prompt for '{media_object.title}'", "")
            if not image_object.generateImagePrompt():
                return GenerationResult(
                    success=False,
                    title=media_object.title,
                    media_object=media_object,
                    error="Failed to generate image prompt"
                )
            
            if self.verbose:
                self._output_message(f"Image prompt:\n{image_object.poster_prompt.get('image_prompt', '')}", "verbose")
            
            self._output_message(f"Generating image for '{media_object.title}' from prompt", "")
            if not image_object.generateImage():
                return GenerationResult(
                    success=False,
                    title=media_object.title,
                    media_object=media_object,
                    error="Failed to generate image"
                )
            
            # Process the image
            if not image_object.processImage():
                return GenerationResult(
                    success=False,
                    title=media_object.title,
                    media_object=media_object,
                    error="Failed to process image"
                )
            
            self._output_message(f"Image created for '{media_object.title}'", "")
            media_object.image_generation_time = datetime.datetime.now()
        
        media_object.create_time = datetime.datetime.now()
        
        # Save if requested and not in dry run mode
        should_save = save and not self.dry_run
        if should_save:
            if not media_object.saveMediaObject():
                return GenerationResult(
                    success=False,
                    title=media_object.title,
                    media_object=media_object,
                    error="Failed to save media object"
                )
            
            if image_object and not image_object.saveImage():
                self._output_message(f"Error saving image for '{media_object.title}', cleaning up", "error")
                media_object.objectCleanup()
                return GenerationResult(
                    success=False,
                    title=media_object.title,
                    media_object=media_object,
                    error="Failed to save image"
                )
        
        generation_time = datetime.datetime.now() - object_start_time
        self._output_message(
            f"Media created: '{media_object.title}', generate time: {generation_time}",
            "success"
        )
        
        return GenerationResult(
            success=True,
            title=media_object.title,
            media_object=media_object,
            generation_time=generation_time
        )
    
    def generate_batch(
        self,
        count: int = 1,
        save: bool = True,
        on_progress: Optional[Callable[[int, int, GenerationResult], None]] = None
    ) -> GenerationStats:
        """
        Generate multiple media objects.
        
        Args:
            count: Number of media objects to generate.
            save: Whether to save generated media to disk.
            on_progress: Optional callback called after each generation.
                         Receives (current_index, total_count, result).
        
        Returns:
            GenerationStats with counts and timing information.
        """
        if not self._validate_setup():
            return GenerationStats(total_requested=count)
        
        start_time = datetime.datetime.now()
        stats = GenerationStats(total_requested=count)
        
        self._output_message(
            f"Starting creation of {count} media object{'s' if count > 1 else ''}",
            ""
        )
        
        for i in range(count):
            self._output_message(f"Creating media object: {i + 1} of {count}", "info")
            
            result = self.generate_single(save=save)
            
            if result.success:
                stats.success_count += 1
            else:
                # Categorize the failure
                if result.error and "image" in result.error.lower():
                    stats.image_fail_count += 1
                elif result.error and "save" in result.error.lower():
                    stats.save_fail_count += 1
                else:
                    stats.completion_fail_count += 1
            
            if on_progress:
                on_progress(i + 1, count, result)
        
        stats.total_time = datetime.datetime.now() - start_time
        
        message_level = "success" if stats.success_count == count else "warning"
        self._output_message(
            f"Finished generating {stats.success_count} media object{'s' if stats.success_count != 1 else ''} "
            f"of {count}, Total Time: {stats.total_time}",
            message_level
        )
        
        if stats.success_count < count:
            self._output_message(
                f"Prompt Completion failures: {stats.completion_fail_count}\n"
                f"Image Generate Failures: {stats.image_fail_count}\n"
                f"Save Failures: {stats.save_fail_count}",
                "info"
            )
        
        return stats
