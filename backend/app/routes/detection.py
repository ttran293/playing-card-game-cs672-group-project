import logging
from threading import Lock

from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from ..services.analysis import analyze_image

router = APIRouter()
analysis_lock = Lock()
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


async def read_upload(request):
    contents = bytearray()
    async for chunk in request.stream():
        if len(contents) + len(chunk) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Choose an image smaller than 20 MB.")
        contents.extend(chunk)

    if not contents:
        raise HTTPException(400, "Choose an image first.")

    return bytes(contents)


@router.post("/detect")
async def detect(request: Request):
    contents = await read_upload(request)

    # Only one request uses the model at a time.
    if not analysis_lock.acquire(blocking=False):
        raise HTTPException(409, "Another image is being analyzed. Try again shortly.")

    try:
        # Keep slow model work off the API event loop.
        return await run_in_threadpool(analyze_image, contents)
    except ValueError as error:
        raise HTTPException(400, str(error))
    except Exception:
        logging.exception("Image analysis failed")
        raise HTTPException(500, "Detection failed. Check the backend logs and Roboflow API key.")
    finally:
        analysis_lock.release()
