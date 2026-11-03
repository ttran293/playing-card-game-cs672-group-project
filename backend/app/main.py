"""HTTP entry point. Detection remains available through its CLI module."""

from fastapi import FastAPI

app = FastAPI(title="Playing Card Analysis")


@app.get("/api/health")
async def health():
    return {"status": "ok"}
