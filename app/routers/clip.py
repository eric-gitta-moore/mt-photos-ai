import os
from fastapi import APIRouter, Depends, File, UploadFile
from PIL import Image
from io import BytesIO
from app.dependencies import verify_header
from app.scheme import ClipTxtRequest
from app.config import device
import cn_clip.clip as clip

router = APIRouter(
    prefix="/clip",
    tags=["clip"],
    dependencies=[Depends(verify_header)],
)

clip_model_name = os.getenv("CLIP_MODEL")
clip_processor = None
clip_model = None


def load_clip_model():
    global clip_processor
    global clip_model
    if clip_processor is None:
        model, preprocess = clip.load_from_name(
            clip_model_name, device=device, download_root="/app/.cache/clip"
        )
        model.eval()
        clip_model = model
        clip_processor = preprocess


@router.post("/img")
async def clip_process_image(file: UploadFile = File(...)):
    load_clip_model()
    image_bytes = await file.read()
    try:
        image = clip_processor(Image.open(BytesIO(image_bytes))).unsqueeze(0).to(device)
        image_features = clip_model.encode_image(image)
        return {"result": ["{:.16f}".format(vec) for vec in image_features[0]]}
    except Exception as e:
        print(e)
        return {"result": [], "msg": str(e)}


@router.post("/txt")
async def clip_process_txt(request: ClipTxtRequest):
    load_clip_model()
    text = clip.tokenize([request.text]).to(device)
    text_features = clip_model.encode_text(text)
    return {"result": ["{:.16f}".format(vec) for vec in text_features[0]]}
