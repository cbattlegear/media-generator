"""
Media Generator API Package

This package provides a FastAPI-based REST API for the media-generator library.
"""

from api.main import app
from api.models import MovieModel, GenreModel, ActorModel, DirectorModel, CriticReviewModel

__all__ = ["app", "MovieModel", "GenreModel", "ActorModel", "DirectorModel", "CriticReviewModel"]
