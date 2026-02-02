-- Initialize the BattleCabbageVideo database with the required schema
-- This script runs automatically when the container starts for the first time

USE master;
GO

-- Create database if it doesn't exist
IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = N'BattleCabbageVideo')
BEGIN
    CREATE DATABASE BattleCabbageVideo;
END
GO

USE BattleCabbageVideo;
GO

-- Create genres table
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='genres' AND xtype='U')
BEGIN
    CREATE TABLE [dbo].[genres](
        [genre_id] [int] IDENTITY(1,1) NOT NULL,
        [genre] [nvarchar](100) NOT NULL,
        CONSTRAINT [PK_genres] PRIMARY KEY CLUSTERED ([genre_id] ASC)
    );
END
GO

-- Create movies table
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='movies' AND xtype='U')
BEGIN
    CREATE TABLE [dbo].[movies](
        [movie_id] [int] IDENTITY(1,1) NOT NULL,
        [external_id] [nvarchar](30) NOT NULL,
        [title] [nvarchar](100) NOT NULL,
        [tagline] [nvarchar](500) NOT NULL,
        [description] [nvarchar](2000) NOT NULL,
        [mpaa_rating] [nvarchar](5) NOT NULL,
        [popularity_score] [decimal](18, 2) NULL,
        [genre_id] [int] NOT NULL,
        [poster_url] [nvarchar](500) NULL,
        [release_date] [date] NOT NULL,
        CONSTRAINT [PK_movies] PRIMARY KEY CLUSTERED ([movie_id] ASC),
        CONSTRAINT [FK_movies_genres] FOREIGN KEY([genre_id]) REFERENCES [dbo].[genres] ([genre_id])
    );
END
GO

-- Create actors table
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='actors' AND xtype='U')
BEGIN
    CREATE TABLE [dbo].[actors](
        [actor_id] [int] IDENTITY(1,1) NOT NULL,
        [actor] [nvarchar](500) NOT NULL,
        CONSTRAINT [PK_actors] PRIMARY KEY CLUSTERED ([actor_id] ASC)
    );
END
GO

-- Create directors table
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='directors' AND xtype='U')
BEGIN
    CREATE TABLE [dbo].[directors](
        [director_id] [int] IDENTITY(1,1) NOT NULL,
        [director] [nvarchar](500) NOT NULL,
        CONSTRAINT [PK_directors] PRIMARY KEY CLUSTERED ([director_id] ASC)
    );
END
GO

-- Create actors to movies join table
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='actorstomoviesjoin' AND xtype='U')
BEGIN
    CREATE TABLE [dbo].[actorstomoviesjoin](
        [actor_id] [int] NOT NULL,
        [movie_id] [int] NOT NULL,
        CONSTRAINT [PK_actorstomoviesjoin] PRIMARY KEY CLUSTERED ([actor_id] ASC, [movie_id] ASC),
        CONSTRAINT [FK_actorstomoviesjoin_actors] FOREIGN KEY([actor_id]) REFERENCES [dbo].[actors] ([actor_id]),
        CONSTRAINT [FK_actorstomoviesjoin_movies] FOREIGN KEY([movie_id]) REFERENCES [dbo].[movies] ([movie_id])
    );
END
GO

-- Create directors to movies join table
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='directorstomoviesjoin' AND xtype='U')
BEGIN
    CREATE TABLE [dbo].[directorstomoviesjoin](
        [director_id] [int] NOT NULL,
        [movie_id] [int] NOT NULL,
        CONSTRAINT [PK_directorstomoviesjoin] PRIMARY KEY CLUSTERED ([director_id] ASC, [movie_id] ASC),
        CONSTRAINT [FK_directorstomoviesjoin_directors] FOREIGN KEY([director_id]) REFERENCES [dbo].[directors] ([director_id]),
        CONSTRAINT [FK_directorstomoviesjoin_movies] FOREIGN KEY([movie_id]) REFERENCES [dbo].[movies] ([movie_id])
    );
END
GO

-- Create critic reviews table
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='criticreviews' AND xtype='U')
BEGIN
    CREATE TABLE [dbo].[criticreviews](
        [critic_review_id] [int] IDENTITY(1,1) NOT NULL,
        [movie_id] [int] NOT NULL,
        [critic_score] [decimal](18, 2) NULL,
        [critic_review] [nvarchar](4000) NOT NULL,
        CONSTRAINT [PK_criticreviews] PRIMARY KEY CLUSTERED ([critic_review_id] ASC),
        CONSTRAINT [FK_criticreviews_movies] FOREIGN KEY([movie_id]) REFERENCES [dbo].[movies] ([movie_id])
    );
END
GO

PRINT 'Database schema initialization complete.';
GO
