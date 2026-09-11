-- Run against the existing movie database before deploying trailer hosting.
IF COL_LENGTH(N'dbo.movies', N'trailer_url') IS NULL
BEGIN
    ALTER TABLE [dbo].[movies] ADD [trailer_url] [nvarchar](500) NULL;
END
GO
