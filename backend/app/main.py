from fastapi import FastAPI

from .routes import detection, health

app = FastAPI(title="Playing Card Analysis")
app.include_router(health.router, prefix="/api")
app.include_router(detection.router, prefix="/api")
