import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, UploadFile
from app.dependencies import verify_header
from app.utils.util import predict
from app.utils.util import to_fixed
from rapidocr import EngineType, LangDet, ModelType, OCRVersion, RapidOCR
from app.config import device

router = APIRouter(
    tags=["ocr"],
    dependencies=[Depends(verify_header)],
)

ocr_model = None


def load_ocr_model():
    global ocr_model
    if ocr_model is None:
        with_torch = True if device == "cuda" else False
        engine_type = EngineType.TORCH if with_torch else EngineType.ONNXRUNTIME
        ocr_model = RapidOCR(
            params={
                "Det.engine_type": engine_type,
                # "Det.ocr_version": OCRVersion.PPOCRV5, pytorch 没有 v5 版本
                "Cls.engine_type": engine_type,
                # "Cls.ocr_version": OCRVersion.PPOCRV5,
                "Rec.engine_type": engine_type,
                # "Rec.ocr_version": OCRVersion.PPOCRV5,
                "EngineConfig.torch.use_cuda": with_torch,
                # "Global.lang_det": "ch_server",
                # "Global.lang_rec": "ch_server",
                # "EngineConfig.torch.gpu_id": 0,  # 指定 GPU id
            }
        )
        # https://rapidai.github.io/RapidOCRDocs/main/install_usage/rapidocr/usage/


@router.post("/ocr")
async def process_image(file: UploadFile = File(...)):
    load_ocr_model()
    image_bytes = await file.read()
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        height, width, _ = img.shape
        if width > 10000 or height > 10000:
            return {"result": [], "msg": "height or width out of range"}

        _result = await predict(ocr_model, img)
        result = convert_rapidocr_to_json(_result)
        del img
        del _result
        return {"result": result}
    except Exception as e:
        print(e)
        return {"result": [], "msg": str(e)}


def convert_rapidocr_to_json(rapidocr_output):

    if rapidocr_output.txts is None:
        return {"texts": [], "scores": [], "boxes": []}

    texts = list(rapidocr_output.txts)
    scores = [f"{score:.2f}" for score in rapidocr_output.scores]
    boxes_coords = rapidocr_output.boxes

    boxes = []
    for box in boxes_coords:
        xs = [point[0] for point in box]
        ys = [point[1] for point in box]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        width = x_max - x_min
        height = y_max - y_min

        boxes.append(
            {
                "x": to_fixed(x_min),
                "y": to_fixed(y_min),
                "width": to_fixed(width),
                "height": to_fixed(height),
            }
        )

    output = {"texts": texts, "scores": scores, "boxes": boxes}

    return output
