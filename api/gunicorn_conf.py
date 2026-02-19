"""
Gunicorn configuration for production deployment.

Uses UvicornWorker to run FastAPI with multiple worker processes.
All settings are configurable via environment variables.

Usage:
    gunicorn -c api/gunicorn_conf.py api.main:app
"""

import multiprocessing
import os

# Server socket
host = os.getenv("HOST", "0.0.0.0")
port = os.getenv("PORT", "8000")
bind = f"{host}:{port}"

# Worker processes
_default_workers = 2 * multiprocessing.cpu_count() + 1
workers = int(os.getenv("WEB_WORKERS", _default_workers))
worker_class = "uvicorn.workers.UvicornWorker"

# Worker lifecycle
timeout = int(os.getenv("WORKER_TIMEOUT", "120"))
keepalive = int(os.getenv("KEEP_ALIVE", "5"))
max_requests = int(os.getenv("MAX_REQUESTS", "1000"))
max_requests_jitter = int(os.getenv("MAX_REQUESTS_JITTER", "50"))
graceful_timeout = int(os.getenv("GRACEFUL_TIMEOUT", "30"))

# Load app before forking workers to save memory
preload_app = True

# Logging
accesslog = "-"
errorlog = "-"
loglevel = os.getenv("LOG_LEVEL", "info").lower()
