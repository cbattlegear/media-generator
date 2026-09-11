import io
import subprocess
from datetime import date
from pathlib import Path

import anyio
import pytest
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from starlette import formparsers
from starlette.datastructures import Headers

from api import main, trailers
from api.models import Base, GenreModel, MovieModel

AUTH = {"X-Api-Key": "trailer-test-key"}


@pytest.fixture(scope="session")
def mp4(tmp_path_factory):
    path = tmp_path_factory.mktemp("video") / "trailer.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=16x16:r=10",
            "-t",
            "0.2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    return path.read_bytes()


@pytest.fixture
def database(tmp_path):
    engine = create_engine(
        "sqlite:///" + (tmp_path / "movies.db").as_posix(),
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    with factory() as db:
        genre = GenreModel(genre="Adventure")
        db.add(genre)
        db.flush()
        db.add(
            MovieModel(
                external_id="test-movie",
                title="Test Movie",
                tagline="An adventure",
                description="A test movie.",
                mpaa_rating="PG",
                genre_id=genre.genre_id,
                poster_url="/images/movie_1.png",
                release_date=date(2026, 1, 1),
            )
        )
        db.commit()
    yield factory
    engine.dispose()


@pytest.fixture
def client(tmp_path, monkeypatch, database):
    public = tmp_path / "trailers" / "public"
    public.mkdir(parents=True)
    monkeypatch.setattr(trailers, "PUBLIC_DIR", public)
    monkeypatch.setattr(trailers, "STAGING_DIR", public.parent / ".uploads")
    monkeypatch.setenv("API_KEYS", AUTH["X-Api-Key"])

    mount = next(route for route in main.app.routes if route.path == "/trailers")
    monkeypatch.setattr(mount.app, "directory", str(public))
    monkeypatch.setattr(mount.app, "all_directories", [str(public)])

    def get_db():
        with database() as db:
            yield db

    monkeypatch.setitem(main.app.dependency_overrides, main.get_db, get_db)
    # Skip production lifespan setup: this client uses only the temporary database.
    test_client = TestClient(main.app)
    yield test_client
    test_client.close()


def upload(client, data, movie_id=1, content_type="video/mp4", filename="trailer.mp4"):
    return client.put(
        f"/movies/{movie_id}/trailer",
        headers=AUTH,
        files={"file": (filename, data, content_type)},
    )


def assert_no_files():
    assert list(trailers.PUBLIC_DIR.iterdir()) == []
    assert not trailers.STAGING_DIR.exists() or list(trailers.STAGING_DIR.iterdir()) == []


def test_upload_persists_url_and_preserves_movie_fields(client, database, mp4):
    assert client.get("/movies/1").json()["trailer_url"] is None
    response = upload(client, mp4)
    assert response.status_code == 200, response.text
    movie = response.json()
    url = movie["trailer_url"]
    assert url.startswith("/trailers/movie_1_") and url.endswith(".mp4")
    assert movie["poster_url"] == "/images/movie_1.png"
    assert movie["title"] == "Test Movie"
    assert movie["genre"] == "Adventure"
    with database() as db:
        assert db.get(MovieModel, 1).trailer_url == url
    assert client.get("/movies/1").json()["trailer_url"] == url
    assert client.get("/movies").json()[0]["trailer_url"] == url
    assert client.get("/genres/1/movies").json()[0]["trailer_url"] == url
    assert (trailers.PUBLIC_DIR / Path(url).name).read_bytes() == mp4
    assert list(trailers.STAGING_DIR.iterdir()) == []


def test_public_playback_head_and_ranges(client, mp4):
    url = upload(client, mp4).json()["trailer_url"]
    response = client.get(url)
    assert response.status_code == 200
    assert response.content == mp4
    assert response.headers["content-type"] == "video/mp4"
    assert response.headers["accept-ranges"] == "bytes"

    head = client.head(url)
    assert head.status_code == 200
    assert head.content == b""
    assert int(head.headers["content-length"]) == len(mp4)

    partial = client.get(url, headers={"Range": "bytes=16-47"})
    assert partial.status_code == 206
    assert partial.content == mp4[16:48]
    assert partial.headers["content-range"] == f"bytes 16-47/{len(mp4)}"
    assert partial.headers["content-length"] == "32"

    suffix = client.get(url, headers={"Range": "bytes=-32"})
    assert suffix.status_code == 206
    assert suffix.content == mp4[-32:]

    invalid = client.get(url, headers={"Range": f"bytes={len(mp4)}-"})
    assert invalid.status_code == 416
    assert invalid.headers["content-range"] == f"bytes */{len(mp4)}"

    stale = client.get(url, headers={"Range": "bytes=0-15", "If-Range": '"stale"'})
    assert stale.status_code == 200
    assert stale.content == mp4


def test_replacements_keep_old_urls_readable(client, mp4):
    first = upload(client, mp4).json()["trailer_url"]
    second = upload(client, mp4).json()["trailer_url"]
    assert first != second
    assert client.get("/movies/1").json()["trailer_url"] == second
    assert client.get(first).content == mp4
    assert client.get(second).content == mp4

    invalid = upload(client, b"not a video")
    assert invalid.status_code == 422
    assert client.get("/movies/1").json()["trailer_url"] == second
    assert len(list(trailers.PUBLIC_DIR.iterdir())) == 2


@pytest.mark.parametrize("headers,status", [({}, 422), ({"X-Api-Key": "invalid"}, 401)])
def test_upload_requires_authentication(client, mp4, headers, status):
    response = client.put(
        "/movies/1/trailer", headers=headers, files={"file": ("trailer.mp4", mp4, "video/mp4")}
    )
    assert response.status_code == status
    assert_no_files()


def test_missing_movie(client, mp4):
    assert upload(client, mp4, movie_id=999).status_code == 404
    assert_no_files()


def test_missing_file(client):
    assert client.put("/movies/1/trailer", headers=AUTH).status_code == 422
    assert_no_files()


@pytest.mark.parametrize("content_type", ["video/webm", "image/png", "application/octet-stream"])
def test_unsupported_content_type(client, mp4, content_type):
    assert upload(client, mp4, content_type=content_type).status_code == 415
    assert_no_files()


@pytest.mark.parametrize("data", [b"", b"not a video", b"\0\0\0\x20ftypisom" + b"\0" * 32])
def test_empty_spoofed_and_corrupt_mp4_rejected(client, data):
    assert upload(client, data).status_code == 422
    assert client.get("/movies/1").json()["trailer_url"] is None
    assert_no_files()


def test_audio_only_mp4_is_rejected(client, tmp_path):
    path = tmp_path / "audio.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=8000:cl=mono",
            "-t",
            "0.1",
            "-c:a",
            "aac",
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    assert upload(client, path.read_bytes()).status_code == 422
    assert_no_files()


def test_file_size_limit(client, mp4, monkeypatch):
    monkeypatch.setattr(trailers, "MAX_SIZE_BYTES", len(mp4) - 1)
    assert upload(client, mp4).status_code == 413
    assert_no_files()


def test_exact_file_size_limit_is_allowed(client, mp4, monkeypatch):
    monkeypatch.setattr(trailers, "MAX_SIZE_BYTES", len(mp4))
    monkeypatch.setattr(trailers, "CHUNK_SIZE", 17)
    assert upload(client, mp4).status_code == 200


def test_unknown_file_size_is_bounded_while_copying(client, mp4, monkeypatch):
    monkeypatch.setattr(trailers, "MAX_SIZE_BYTES", 64)
    monkeypatch.setattr(trailers, "CHUNK_SIZE", 17)
    file = UploadFile(file=io.BytesIO(mp4), headers=Headers({"content-type": "video/mp4"}))
    with pytest.raises(HTTPException) as error, trailers.store_trailer(file, 1):
        pytest.fail("Oversized trailer was published")
    assert error.value.status_code == 413
    assert_no_files()


def test_request_content_length_limit(client, mp4, monkeypatch):
    monkeypatch.setattr(trailers, "MAX_SIZE_BYTES", 64)
    monkeypatch.setattr(trailers, "MULTIPART_OVERHEAD_BYTES", 64)
    assert upload(client, mp4).status_code == 413
    assert_no_files()


@pytest.mark.parametrize("content_length", [None, "1"])
def test_request_limit_without_trustworthy_content_length(client, monkeypatch, content_length):
    monkeypatch.setattr(trailers, "MAX_SIZE_BYTES", 128)
    monkeypatch.setattr(trailers, "MULTIPART_OVERHEAD_BYTES", 128)
    headers = {**AUTH, "Content-Type": "multipart/form-data; boundary=trailer-test"}
    if content_length is not None:
        headers["Content-Length"] = content_length

    def chunks():
        yield (
            b"--trailer-test\r\n"
            b'Content-Disposition: form-data; name="file"; filename="trailer.mp4"\r\n'
            b"Content-Type: video/mp4\r\n\r\n"
        )
        yield b"x" * 300
        yield b"\r\n--trailer-test--\r\n"

    response = client.put("/movies/1/trailer", headers=headers, content=chunks())
    assert response.status_code == 413
    assert_no_files()


def test_request_limit_closes_spooled_files_between_network_chunks(client, monkeypatch):
    monkeypatch.setattr(trailers, "MAX_SIZE_BYTES", 256)
    monkeypatch.setattr(trailers, "MULTIPART_OVERHEAD_BYTES", 256)
    created = []
    original = formparsers.SpooledTemporaryFile

    def track_file(*args, **kwargs):
        file = original(*args, **kwargs)
        created.append(file)
        return file

    monkeypatch.setattr(formparsers, "SpooledTemporaryFile", track_file)
    chunks = iter(
        [
            b"--trailer-test\r\n"
            b'Content-Disposition: form-data; name="file"; filename="trailer.mp4"\r\n'
            b"Content-Type: video/mp4\r\n\r\n" + b"x" * 200,
            b"x" * 400,
        ]
    )
    sent = []

    async def receive():
        return {"type": "http.request", "body": next(chunks), "more_body": True}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "PUT",
        "scheme": "http",
        "path": "/movies/1/trailer",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"content-type", b"multipart/form-data; boundary=trailer-test"),
            (b"x-api-key", AUTH["X-Api-Key"].encode()),
        ],
        "client": ("127.0.0.1", 1234),
        "server": ("testserver", 80),
    }
    anyio.run(main.app, scope, receive, send)
    assert sent[0]["status"] == 413
    assert created and all(file.closed for file in created)
    assert_no_files()


def test_staging_and_unvalidated_files_are_not_public(client, mp4, monkeypatch):
    validate = trailers.validate_mp4

    def observe_staging(path):
        assert path.parent == trailers.STAGING_DIR
        assert list(trailers.PUBLIC_DIR.iterdir()) == []
        assert client.get(f"/trailers/.uploads/{path.name}").status_code == 404
        assert client.get(f"/trailers/%2e%2e/.uploads/{path.name}").status_code == 404
        validate(path)

    monkeypatch.setattr(trailers, "validate_mp4", observe_staging)
    assert upload(client, mp4).status_code == 200


def test_client_filename_is_ignored(client, mp4):
    response = upload(client, mp4, filename="../../escape.mp4")
    assert response.status_code == 200
    assert Path(response.json()["trailer_url"]).name.startswith("movie_1_")
    assert not (trailers.PUBLIC_DIR.parent / "escape.mp4").exists()


def test_database_failure_preserves_previous_trailer(client, mp4, monkeypatch):
    previous = upload(client, mp4).json()["trailer_url"]

    def fail_commit(_db):
        raise SQLAlchemyError("simulated database failure")

    monkeypatch.setattr(Session, "commit", fail_commit)
    response = upload(client, mp4)
    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to save trailer URL"
    assert client.get("/movies/1").json()["trailer_url"] == previous
    assert client.get(previous).content == mp4
    assert len(list(trailers.PUBLIC_DIR.iterdir())) == 1
    assert list(trailers.STAGING_DIR.iterdir()) == []


def test_publish_failure_cleans_staging(client, mp4, monkeypatch):
    def fail_replace(*_args):
        raise OSError("simulated storage failure")

    monkeypatch.setattr(trailers.os, "replace", fail_replace)
    response = upload(client, mp4)
    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to store trailer"
    assert client.get("/movies/1").json()["trailer_url"] is None
    assert_no_files()


@pytest.mark.parametrize(
    "failure,status",
    [
        (FileNotFoundError("ffprobe unavailable"), 503),
        (subprocess.TimeoutExpired("ffprobe", 30), 422),
    ],
)
def test_validation_tool_failures_clean_up(client, mp4, monkeypatch, failure, status):
    def fail_probe(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(trailers.subprocess, "run", fail_probe)
    assert upload(client, mp4).status_code == status
    assert_no_files()


def test_openapi_documents_upload_and_movie_url(client):
    schema = client.get("/openapi.json").json()
    upload_schema = schema["paths"]["/movies/{movie_id}/trailer"]["put"]
    assert "multipart/form-data" in upload_schema["requestBody"]["content"]
    assert any(parameter["name"] == "x-api-key" for parameter in upload_schema["parameters"])
    assert "trailer_url" in schema["components"]["schemas"]["MovieResponse"]["properties"]
