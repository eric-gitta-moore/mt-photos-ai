from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from app.config import device

router = APIRouter(
    prefix="",
    tags=["ping"],
)

@router.get("/ping")
async def ping() -> PlainTextResponse:
    return PlainTextResponse("pong")

@router.post("/check")
async def check_req():
    return {
        "result": "pass",
        "title": "mt-photos-ai服务",
        "help": "https://mtmt.tech/docs/advanced/ocr_api",
        "device": device,
    }
