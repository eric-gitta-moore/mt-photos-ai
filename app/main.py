from dotenv import load_dotenv
from fastapi import FastAPI
import uvicorn
import asyncio
from PIL import ImageFile
from app.utils.util import restart_timer
from app.config import http_port
from app.routers import ping, clip, ocr, face, index, restart

ImageFile.LOAD_TRUNCATED_IMAGES = True

load_dotenv()
app = FastAPI()
app.include_router(index.router)
app.include_router(restart.router)
app.include_router(ping.router)
app.include_router(clip.router)
app.include_router(ocr.router)
app.include_router(face.router)

restart_task = None
restart_lock = asyncio.Lock()


@app.on_event("shutdown")
async def on_shutdown():
    if restart_task and not restart_task.done():
        restart_task.cancel()
        try:
            await restart_task
        except asyncio.CancelledError:
            pass


@app.middleware("http")
async def activity_monitor(request, call_next):
    global restart_task

    async with restart_lock:
        if restart_task and not restart_task.done():
            restart_task.cancel()

        restart_task = asyncio.create_task(restart_timer())

    response = await call_next(request)
    return response


if __name__ == "__main__":
    uvicorn.run("main:app", host=None, port=http_port)
