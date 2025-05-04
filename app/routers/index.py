from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from app.config import device

router = APIRouter(
    prefix="",
    tags=["index"],
)

@router.get("/", response_class=HTMLResponse)
async def top_info():
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>MT Photos AI Server</title>
    <style>p{{text-align: center;}}</style>
</head>
<body>
<p style="font-weight: 600;">MT Photos智能识别服务</p>
<p>服务状态： 运行中</p>
<p>推理设备： {device}</p>
<p>使用方法： <a href="https://mtmt.tech/docs/advanced/ocr_api">https://mtmt.tech/docs/advanced/ocr_api</a></p>
</body>
</html>"""
    return html_content
