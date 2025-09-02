#!/bin/sh
exec uv run uvicorn tableai_backend.app.main:app --host 0.0.0.0 --port ${PORT} --reload