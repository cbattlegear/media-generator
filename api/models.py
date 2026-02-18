"""
Database models for the media-generator API.

Maps to an existing SQL Server database schema with tables:
- movies, genres, actors, directors, criticreviews
- actorstomoviesjoin, directorstomoviesjoin
- posterqueue
"""

from datetime import date, datetime
from decimal import Decimal
from sqlalchemy import (
    create_engine, Column, Integer, String, Date, DateTime,
    DECIMAL, ForeignKey, Table
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.engine import URL
import os

Base = declarative_base()


# Many-to-many association tables
actors_to_movies = Table(
    'actorstomoviesjoin',
    Base.metadata,
    Column('actor_id', Integer, ForeignKey('actors.actor_id'), primary_key=True),
    Column('movie_id', Integer, ForeignKey('movies.movie_id'), primary_key=True)
)

directors_to_movies = Table(
    'directorstomoviesjoin',
    Base.metadata,
    Column('director_id', Integer, ForeignKey('directors.director_id'), primary_key=True),
    Column('movie_id', Integer, ForeignKey('movies.movie_id'), primary_key=True)
)


class GenreModel(Base):
    """SQLAlchemy model for genres table."""
    __tablename__ = "genres"

    genre_id = Column(Integer, primary_key=True, autoincrement=True)
    genre = Column(String(100), nullable=False)

    # Relationship
    movies = relationship("MovieModel", back_populates="genre_rel")

    def __repr__(self):
        return f"<Genre(genre_id={self.genre_id}, genre='{self.genre}')>"


class ActorModel(Base):
    """SQLAlchemy model for actors table."""
    __tablename__ = "actors"

    actor_id = Column(Integer, primary_key=True, autoincrement=True)
    actor = Column(String(500), nullable=False)

    # Many-to-many relationship
    movies = relationship("MovieModel", secondary=actors_to_movies, back_populates="actors")

    def __repr__(self):
        return f"<Actor(actor_id={self.actor_id}, actor='{self.actor}')>"


class DirectorModel(Base):
    """SQLAlchemy model for directors table."""
    __tablename__ = "directors"

    director_id = Column(Integer, primary_key=True, autoincrement=True)
    director = Column(String(500), nullable=False)

    # Many-to-many relationship
    movies = relationship("MovieModel", secondary=directors_to_movies, back_populates="directors")

    def __repr__(self):
        return f"<Director(director_id={self.director_id}, director='{self.director}')>"


class MovieModel(Base):
    """SQLAlchemy model for movies table."""
    __tablename__ = "movies"

    movie_id = Column(Integer, primary_key=True, autoincrement=True)
    external_id = Column(String(30), nullable=False)
    title = Column(String(100), nullable=False)
    tagline = Column(String(500), nullable=False)
    description = Column(String(2000), nullable=False)
    mpaa_rating = Column(String(5), nullable=False)
    popularity_score = Column(DECIMAL(18, 2), nullable=True)
    genre_id = Column(Integer, ForeignKey('genres.genre_id'), nullable=False)
    poster_url = Column(String(500), nullable=True)
    release_date = Column(Date, nullable=False)

    # Relationships
    genre_rel = relationship("GenreModel", back_populates="movies")
    actors = relationship("ActorModel", secondary=actors_to_movies, back_populates="movies")
    directors = relationship("DirectorModel", secondary=directors_to_movies, back_populates="movies")
    reviews = relationship("CriticReviewModel", back_populates="movie", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Movie(movie_id={self.movie_id}, title='{self.title}')>"


class CriticReviewModel(Base):
    """SQLAlchemy model for criticreviews table."""
    __tablename__ = "criticreviews"

    critic_review_id = Column(Integer, primary_key=True, autoincrement=True)
    movie_id = Column(Integer, ForeignKey('movies.movie_id'), nullable=False)
    critic_score = Column(DECIMAL(18, 2), nullable=True)
    critic_review = Column(String(4000), nullable=False)

    # Relationship
    movie = relationship("MovieModel", back_populates="reviews")

    def __repr__(self):
        return f"<CriticReview(critic_review_id={self.critic_review_id}, critic_score={self.critic_score})>"


class PosterQueueModel(Base):
    """SQLAlchemy model for posterqueue table."""
    __tablename__ = "posterqueue"

    queue_id = Column(Integer, primary_key=True, autoincrement=True)
    movie_id = Column(Integer, ForeignKey('movies.movie_id'), nullable=False, unique=True)
    status = Column(String(20), nullable=False, default="pending")
    attempt_count = Column(Integer, nullable=False, default=0)
    max_attempts = Column(Integer, nullable=False, default=3)
    claimed_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationship
    movie = relationship("MovieModel")

    def __repr__(self):
        return f"<PosterQueue(queue_id={self.queue_id}, movie_id={self.movie_id}, status='{self.status}')>"


def get_database_url() -> str:
    """
    Build the SQL Server connection URL from environment variables.
    
    Required environment variables:
        - DB_SERVER: SQL Server hostname
        - DB_NAME: Database name
        - DB_USER: Database username (optional for Windows auth)
        - DB_PASSWORD: Database password (optional for Windows auth)
        - DB_DRIVER: ODBC driver name (default: "ODBC Driver 18 for SQL Server")
    """
    server = os.getenv("DB_SERVER", "localhost")
    database = os.getenv("DB_NAME", "media_generator")
    username = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    driver = os.getenv("DB_DRIVER", "ODBC Driver 18 for SQL Server")
    
    # Check if we should use trusted connection (Windows auth)
    if username and password:
        connection_url = URL.create(
            "mssql+pyodbc",
            username=username,
            password=password,
            host=server,
            database=database,
            query={"driver": driver, "TrustServerCertificate": "yes"}
        )
    else:
        # Use trusted connection (Windows authentication)
        connection_url = URL.create(
            "mssql+pyodbc",
            host=server,
            database=database,
            query={"driver": driver, "Trusted_Connection": "yes", "TrustServerCertificate": "yes"}
        )
    
    return connection_url


def create_db_engine(echo: bool = False):
    """Create and return a SQLAlchemy engine."""
    return create_engine(get_database_url(), echo=echo)


def get_session_factory(engine) -> sessionmaker:
    """Create and return a session factory."""
    return sessionmaker(bind=engine)


def get_or_create_genre(db, genre_name: str) -> GenreModel:
    """Get an existing genre or create a new one."""
    genre = db.query(GenreModel).filter(GenreModel.genre == genre_name).first()
    if not genre:
        genre = GenreModel(genre=genre_name)
        db.add(genre)
        db.flush()
    return genre


def get_or_create_actor(db, actor_name: str) -> ActorModel:
    """Get an existing actor or create a new one."""
    actor = db.query(ActorModel).filter(ActorModel.actor == actor_name).first()
    if not actor:
        actor = ActorModel(actor=actor_name)
        db.add(actor)
        db.flush()
    return actor


def get_or_create_director(db, director_name: str) -> DirectorModel:
    """Get an existing director or create a new one."""
    director = db.query(DirectorModel).filter(DirectorModel.director == director_name).first()
    if not director:
        director = DirectorModel(director=director_name)
        db.add(director)
        db.flush()
    return director
