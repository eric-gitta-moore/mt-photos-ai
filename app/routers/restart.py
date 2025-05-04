from fastapi import APIRouter, Depends
from app.dependencies import verify_header
from app.utils.util import restart_program

router = APIRouter(
    prefix="/",
    tags=["restart"],
    dependencies=[Depends(verify_header)],
)


@router.post("/restart")
async def check_req():
    # cuda版本 OCR没有显存未释放问题，这边可以关闭重启
    return {"result": "unsupported"}
    # restart_program()


@router.post("/restart_v2")
async def check_req():
    # 预留触发服务重启接口-自动释放内存
    restart_program()
    return {"result": "pass"}
