"""
FastAPI Web API for the media-generator library.

Provides REST endpoints for generating AI-powered media objects
and storing them in a SQL Server database.

Usage:
    uvicorn api.main:app --reload
    
Or run directly:
    python -m api.main
"""

import random
import sys
import os
from contextlib import asynccontextmanager
from datetime import date
from typing import Optional, List
from decimal import Decimal

from fastapi import FastAPI, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import text, func

# Add parent directory to path so we can import lib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from lib import MediaGenerator
from api.models import (
    MovieModel, GenreModel, ActorModel, DirectorModel, CriticReviewModel,
    create_db_engine, get_session_factory,
    get_or_create_genre, get_or_create_actor, get_or_create_director
)


# Pydantic models for API request/response
class GenerateRequest(BaseModel):
    """Request model for generating media."""
    count: int = Field(default=1, ge=1, le=10, description="Number of media objects to generate (1-10)")
    skip_image: bool = Field(default=False, description="Skip image generation")
    verbose: bool = Field(default=False, description="Enable verbose logging")


class ReviewResponse(BaseModel):
    """Response model for a critic review."""
    critic_review_id: Optional[int] = None
    critic_review: Optional[str] = None
    critic_score: Optional[float] = None

    class Config:
        from_attributes = True


class ActorResponse(BaseModel):
    """Response model for an actor."""
    actor_id: int
    actor: str

    class Config:
        from_attributes = True


class DirectorResponse(BaseModel):
    """Response model for a director."""
    director_id: int
    director: str

    class Config:
        from_attributes = True


class MovieResponse(BaseModel):
    """Response model for a movie."""
    movie_id: Optional[int] = None
    external_id: str
    title: str
    tagline: Optional[str] = None
    mpaa_rating: Optional[str] = None
    description: Optional[str] = None
    popularity_score: Optional[float] = None
    genre: Optional[str] = None
    poster_url: Optional[str] = None
    release_date: Optional[date] = None
    actors: List[ActorResponse] = []
    directors: List[DirectorResponse] = []
    reviews: List[ReviewResponse] = []

    class Config:
        from_attributes = True


class GenerateResponse(BaseModel):
    """Response model for generation results."""
    success: bool
    message: str
    generated_count: int
    movies: List[MovieResponse] = []


class StatsResponse(BaseModel):
    """Response model for database statistics."""
    total_movies: int
    total_reviews: int
    total_actors: int
    total_directors: int
    genres: dict
    ratings: dict


# Global database engine and session factory
engine = None
SessionFactory = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler - setup and teardown."""
    global engine, SessionFactory
    
    # Startup
    print("Initializing database connection...")
    engine = create_db_engine(echo=False)
    # Don't create tables - using existing schema
    SessionFactory = get_session_factory(engine)
    print("Database initialized successfully.")
    
    yield
    
    # Shutdown
    if engine:
        engine.dispose()
        print("Database connection closed.")


app = FastAPI(
    title="Media Generator API",
    description="Generate AI-powered fake media objects including movie titles, descriptions, critic reviews, and poster images.",
    version="1.0.0",
    lifespan=lifespan
)


def get_db() -> Session:
    """Dependency to get database session."""
    db = SessionFactory()
    try:
        yield db
    finally:
        db.close()


def generate_random_release_date() -> date:
    """Generate a random release date within the last 50 years."""
    today = date.today()
    days_back = random.randint(0, 365 * 50)  # Up to 50 years back
    return date.fromordinal(today.toordinal() - days_back)


def save_movie_to_db(media_object, db: Session) -> MovieModel:
    """Save a generated media object to the database using the existing schema."""
    
    # Get or create genre
    genre_name = media_object.genre or "Unknown"
    genre = get_or_create_genre(db, genre_name)
    
    # Create the movie record
    movie_record = MovieModel(
        external_id=media_object.media_id[:30],  # Truncate to fit schema
        title=media_object.title[:100],  # Truncate to fit schema
        tagline=(media_object.tagline or "")[:500],
        description=(media_object.description or "")[:2000],
        mpaa_rating=(media_object.mpaa_rating or "NR")[:5],
        popularity_score=Decimal(str(media_object.popularity_score)) if media_object.popularity_score else None,
        genre_id=genre.genre_id,
        poster_url=getattr(media_object, 'poster_url', None),
        release_date=date.today(),
    )
    
    db.add(movie_record)
    db.flush()  # Get the movie_id
    
    # Add actors from the prompt list
    actors_list = media_object.object_prompt_list.get("actors", [])
    for actor_name in actors_list:
        if actor_name:
            actor = get_or_create_actor(db, actor_name[:500])
            if actor not in movie_record.actors:
                movie_record.actors.append(actor)
    
    # Add directors from the prompt list
    directors_list = media_object.object_prompt_list.get("directors", [])
    for director_name in directors_list:
        if director_name:
            director = get_or_create_director(db, director_name[:500])
            if director not in movie_record.directors:
                movie_record.directors.append(director)
    
    # Add critic reviews
    for review_data in media_object.reviews:
        review_text = review_data.get("review", "")[:4000]
        review_record = CriticReviewModel(
            movie_id=movie_record.movie_id,
            critic_score=Decimal(str(review_data.get("score", 0))) if review_data.get("score") else None,
            critic_review=review_text or "No review provided",
        )
        db.add(review_record)
    
    db.commit()
    db.refresh(movie_record)
    
    return movie_record


def movie_to_response(movie: MovieModel) -> MovieResponse:
    """Convert a MovieModel to a MovieResponse."""
    return MovieResponse(
        movie_id=movie.movie_id,
        external_id=movie.external_id,
        title=movie.title,
        tagline=movie.tagline,
        mpaa_rating=movie.mpaa_rating,
        description=movie.description,
        popularity_score=float(movie.popularity_score) if movie.popularity_score else None,
        genre=movie.genre_rel.genre if movie.genre_rel else None,
        poster_url=movie.poster_url,
        release_date=movie.release_date,
        actors=[ActorResponse(actor_id=a.actor_id, actor=a.actor) for a in movie.actors],
        directors=[DirectorResponse(director_id=d.director_id, director=d.director) for d in movie.directors],
        reviews=[
            ReviewResponse(
                critic_review_id=r.critic_review_id,
                critic_review=r.critic_review,
                critic_score=float(r.critic_score) if r.critic_score else None
            ) for r in movie.reviews
        ]
    )


@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "media-generator-api", "version": "1.0.0"}


@app.get("/health", tags=["Health"])
async def health_check(db: Session = Depends(get_db)):
    """Detailed health check including database connectivity."""
    try:
        # Test database connection
        db.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    return {
        "status": "ok",
        "database": db_status,
        "model_type": os.getenv("MODEL_TYPE", "not set")
    }


@app.post("/generate", response_model=GenerateResponse, tags=["Generation"])
async def generate_media(
    request: GenerateRequest,
    db: Session = Depends(get_db)
):
    """
    Generate new media objects.
    
    Generates AI-powered fake movie/media information and saves to the database.
    Returns the generated media objects as JSON.
    """
    try:
        generator = MediaGenerator(
            verbose=request.verbose,
            dry_run=True,  # Don't save files, we save to DB instead
            skip_image=request.skip_image
        )
        
        generated_movies = []
        success_count = 0
        
        for _ in range(request.count):
            result = generator.generate_single(save=False)
            
            if result.success and result.media_object:
                # Save to database
                movie_record = save_movie_to_db(result.media_object, db)
                
                # Build response
                movie_response = movie_to_response(movie_record)
                generated_movies.append(movie_response)
                success_count += 1
        
        return GenerateResponse(
            success=success_count > 0,
            message=f"Generated {success_count} of {request.count} movies",
            generated_count=success_count,
            movies=generated_movies
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.get("/movies", response_model=List[MovieResponse], tags=["Movies"])
async def list_movies(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of records to return"),
    genre: Optional[str] = Query(None, description="Filter by genre"),
    db: Session = Depends(get_db)
):
    """
    List all movies from the database.
    
    Supports pagination and filtering by genre.
    """
    query = db.query(MovieModel)
    
    if genre:
        query = query.join(GenreModel).filter(GenreModel.genre.ilike(f"%{genre}%"))
    
    movies = query.order_by(MovieModel.movie_id.desc()).offset(skip).limit(limit).all()
    
    return [movie_to_response(m) for m in movies]


@app.get("/movies/{movie_id}", response_model=MovieResponse, tags=["Movies"])
async def get_movie(movie_id: int, db: Session = Depends(get_db)):
    """
    Get a specific movie by ID.
    """
    movie = db.query(MovieModel).filter(MovieModel.movie_id == movie_id).first()
    
    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found")
    
    return movie_to_response(movie)


@app.get("/genres", response_model=List[dict], tags=["Lookup"])
async def list_genres(db: Session = Depends(get_db)):
    """List all genres."""
    genres = db.query(GenreModel).order_by(GenreModel.genre).all()
    return [{"genre_id": g.genre_id, "genre": g.genre} for g in genres]


@app.get("/actors", response_model=List[ActorResponse], tags=["Lookup"])
async def list_actors(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db)
):
    """List all actors with pagination."""
    actors = db.query(ActorModel).order_by(ActorModel.actor).offset(skip).limit(limit).all()
    return [ActorResponse(actor_id=a.actor_id, actor=a.actor) for a in actors]


@app.get("/directors", response_model=List[DirectorResponse], tags=["Lookup"])
async def list_directors(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db)
):
    """List all directors with pagination."""
    directors = db.query(DirectorModel).order_by(DirectorModel.director).offset(skip).limit(limit).all()
    return [DirectorResponse(director_id=d.director_id, director=d.director) for d in directors]


@app.get("/stats", response_model=StatsResponse, tags=["Statistics"])
async def get_stats(db: Session = Depends(get_db)):
    """
    Get statistics about the movie database.
    """
    total_movies = db.query(func.count(MovieModel.movie_id)).scalar() or 0
    total_reviews = db.query(func.count(CriticReviewModel.critic_review_id)).scalar() or 0
    total_actors = db.query(func.count(ActorModel.actor_id)).scalar() or 0
    total_directors = db.query(func.count(DirectorModel.director_id)).scalar() or 0
    
    # Genre distribution
    genre_counts = db.query(
        GenreModel.genre, 
        func.count(MovieModel.movie_id)
    ).join(MovieModel).group_by(GenreModel.genre).all()
    genres = {g[0]: g[1] for g in genre_counts}
    
    # Rating distribution
    rating_counts = db.query(
        MovieModel.mpaa_rating,
        func.count(MovieModel.movie_id)
    ).group_by(MovieModel.mpaa_rating).all()
    ratings = {r[0] or "NR": r[1] for r in rating_counts}
    
    return StatsResponse(
        total_movies=total_movies,
        total_reviews=total_reviews,
        total_actors=total_actors,
        total_directors=total_directors,
        genres=genres,
        ratings=ratings
    )


@app.get("/movies/top-rated", response_model=List[MovieResponse], tags=["Movies"])
async def get_top_rated_movies(db: Session = Depends(get_db)):
    """
    Get the top 5 highest rated movies by average critic score.
    """
    # Subquery to get average critic score per movie
    avg_scores = db.query(
        CriticReviewModel.movie_id,
        func.avg(CriticReviewModel.critic_score).label("avg_score")
    ).group_by(CriticReviewModel.movie_id).subquery()
    
    # Join with movies and order by average score descending
    top_movies = db.query(MovieModel).join(
        avg_scores, MovieModel.movie_id == avg_scores.c.movie_id
    ).order_by(avg_scores.c.avg_score.desc()).limit(5).all()
    
    return [movie_to_response(m) for m in top_movies]


@app.get("/movies/worst-rated", response_model=List[MovieResponse], tags=["Movies"])
async def get_worst_rated_movies(db: Session = Depends(get_db)):
    """
    Get the top 5 lowest rated movies by average critic score.
    """
    # Subquery to get average critic score per movie
    avg_scores = db.query(
        CriticReviewModel.movie_id,
        func.avg(CriticReviewModel.critic_score).label("avg_score")
    ).group_by(CriticReviewModel.movie_id).subquery()
    
    # Join with movies and order by average score ascending
    worst_movies = db.query(MovieModel).join(
        avg_scores, MovieModel.movie_id == avg_scores.c.movie_id
    ).order_by(avg_scores.c.avg_score.asc()).limit(5).all()
    
    return [movie_to_response(m) for m in worst_movies]


@app.get("/movies/recent", response_model=List[MovieResponse], tags=["Movies"])
async def get_recent_movies(db: Session = Depends(get_db)):
    """
    Get the top 5 most recently released movies.
    """
    recent_movies = db.query(MovieModel).order_by(
        MovieModel.release_date.desc()
    ).limit(5).all()
    
    return [movie_to_response(m) for m in recent_movies]


@app.get("/genres/top", response_model=List[dict], tags=["Lookup"])
async def get_top_genres(db: Session = Depends(get_db)):
    """
    Get the top 5 genres by movie count.
    """
    top_genres = db.query(
        GenreModel.genre_id,
        GenreModel.genre,
        func.count(MovieModel.movie_id).label("movie_count")
    ).join(MovieModel).group_by(
        GenreModel.genre_id, GenreModel.genre
    ).order_by(func.count(MovieModel.movie_id).desc()).limit(5).all()
    
    return [{"genre_id": g.genre_id, "genre": g.genre, "movie_count": g.movie_count} for g in top_genres]


@app.get("/actors/top", response_model=List[dict], tags=["Lookup"])
async def get_top_actors(db: Session = Depends(get_db)):
    """
    Get the top 5 actors by movie count.
    """
    top_actors = db.query(
        ActorModel.actor_id,
        ActorModel.actor,
        func.count(MovieModel.movie_id).label("movie_count")
    ).join(ActorModel.movies).group_by(
        ActorModel.actor_id, ActorModel.actor
    ).order_by(func.count(MovieModel.movie_id).desc()).limit(5).all()
    
    return [{"actor_id": a.actor_id, "actor": a.actor, "movie_count": a.movie_count} for a in top_actors]


@app.get("/directors/top", response_model=List[dict], tags=["Lookup"])
async def get_top_directors(db: Session = Depends(get_db)):
    """
    Get the top 5 directors by movie count.
    """
    top_directors = db.query(
        DirectorModel.director_id,
        DirectorModel.director,
        func.count(MovieModel.movie_id).label("movie_count")
    ).join(DirectorModel.movies).group_by(
        DirectorModel.director_id, DirectorModel.director
    ).order_by(func.count(MovieModel.movie_id).desc()).limit(5).all()
    
    return [{"director_id": d.director_id, "director": d.director, "movie_count": d.movie_count} for d in top_directors]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
