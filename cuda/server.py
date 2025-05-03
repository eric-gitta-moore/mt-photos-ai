import logging
from dotenv import load_dotenv
import os
import sys
from fastapi import Depends, FastAPI, File, UploadFile, HTTPException, Header
from fastapi.responses import HTMLResponse
import uvicorn
import numpy as np
import cv2
import asyncio

# from paddleocr import PaddleOCR
import torch
from PIL import Image, ImageFile
from io import BytesIO
from pydantic import BaseModel
from rapidocr import RapidOCR  # Paddle的cuda镜像太大，改用torch，RapidOCR支持torch
import cn_clip.clip as clip
import insightface
from insightface.utils import storage
from insightface.app import FaceAnalysis

ImageFile.LOAD_TRUNCATED_IMAGES = True

on_linux = sys.platform.startswith("linux")

load_dotenv()
app = FastAPI()

api_auth_key = os.getenv("API_AUTH_KEY", "mt_photos_ai_extra")
http_port = int(os.getenv("HTTP_PORT", "8060"))
server_restart_time = int(os.getenv("SERVER_RESTART_TIME", "300"))
env_auto_load_txt_modal = (
    os.getenv("AUTO_LOAD_TXT_MODAL", "off") == "on"
)  # 是否自动加载 CLIP 文本模型，开启可以优化第一次搜索时的响应速度，文本模型占用 700 多 m 内存

clip_model_name = os.getenv("CLIP_MODEL")


ocr_model = None
clip_processor = None
clip_model = None

restart_task = None
restart_lock = asyncio.Lock()

device = "cuda" if torch.cuda.is_available() else "cpu"


class ClipTxtRequest(BaseModel):
    text: str


def load_ocr_model():
    global ocr_model
    if ocr_model is None:
        ocr_model = RapidOCR(
            params={
                "Global.with_torch": True,
                "EngineConfig.torch.use_cuda": torch.cuda.is_available(),  # 使用 torch GPU 版推理
                "Global.lang_det": "ch_server",
                "Global.lang_rec": "ch_server",
                # "EngineConfig.torch.gpu_id": 0,  # 指定 GPU id
            }
        )
        # https://rapidai.github.io/RapidOCRDocs/main/install_usage/rapidocr/usage/


def load_clip_model():
    global clip_processor
    global clip_model
    if clip_processor is None:
        model, preprocess = clip.load_from_name(clip_model_name, device=device, download_root="/app/.cache/clip")
        model.eval()
        clip_model = model
        clip_processor = preprocess


@app.on_event("startup")
async def startup_event():
    if env_auto_load_txt_modal:
        load_clip_model()


@app.on_event("shutdown")
async def on_shutdown():
    if restart_task and not restart_task.done():
        restart_task.cancel()
        try:
            await restart_task
        except asyncio.CancelledError:
            pass


async def restart_timer():
    await asyncio.sleep(server_restart_time)
    restart_program()


@app.middleware("http")
async def activity_monitor(request, call_next):
    global restart_task

    async with restart_lock:
        if restart_task and not restart_task.done():
            restart_task.cancel()

        restart_task = asyncio.create_task(restart_timer())

    response = await call_next(request)
    return response


async def verify_header(api_key: str = Header(...)):
    # 在这里编写验证逻辑，例如检查 api_key 是否有效
    if api_key != api_auth_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key


def to_fixed(num):
    return str(round(num, 2))


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


@app.get("/", response_class=HTMLResponse)
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


@app.post("/check")
async def check_req(api_key: str = Depends(verify_header)):
    return {
        "result": "pass",
        "title": "mt-photos-ai服务",
        "help": "https://mtmt.tech/docs/advanced/ocr_api",
        "device": device,
    }


@app.post("/restart")
async def check_req(api_key: str = Depends(verify_header)):
    # cuda版本 OCR没有显存未释放问题，这边可以关闭重启
    return {"result": "unsupported"}
    # restart_program()


@app.post("/restart_v2")
async def check_req(api_key: str = Depends(verify_header)):
    # 预留触发服务重启接口-自动释放内存
    restart_program()
    return {"result": "pass"}


@app.post("/ocr")
async def process_image(
    file: UploadFile = File(...), api_key: str = Depends(verify_header)
):
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


@app.post("/clip/img")
async def clip_process_image(
    file: UploadFile = File(...), api_key: str = Depends(verify_header)
):
    load_clip_model()
    image_bytes = await file.read()
    try:
        image = clip_processor(Image.open(BytesIO(image_bytes))).unsqueeze(0).to(device)
        image_features = clip_model.encode_image(image)
        return {"result": ["{:.16f}".format(vec) for vec in image_features[0]]}
    except Exception as e:
        print(e)
        return {"result": [], "msg": str(e)}


@app.post("/clip/txt")
async def clip_process_txt(
    request: ClipTxtRequest, api_key: str = Depends(verify_header)
):
    load_clip_model()
    text = clip.tokenize([request.text]).to(device)
    text_features = clip_model.encode_text(text)
    return {"result": ["{:.16f}".format(vec) for vec in text_features[0]]}




detector_backend = os.getenv("DETECTOR_BACKEND", "insightface")
# 人脸检测及特征提取模型
models = [
    "antelopev2",
    "buffalo_l",
    "buffalo_m",
    "buffalo_s",
]
recognition_model = os.getenv("RECOGNITION_MODEL", "buffalo_l")
detection_thresh = float(os.getenv("DETECTION_THRESH", "0.65"))
# 设置下载模型URL
storage.BASE_REPO_URL = 'https://github.com/kqstone/mt-photos-insightface-unofficial/releases/download/models'
# 初始化人脸识别器
faceAnalysis = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], allowed_modules=['detection', 'recognition'], name=recognition_model)
faceAnalysis.prepare(ctx_id=0, det_thresh=detection_thresh, det_size=(640, 640))


@app.post("/represent")
async def process_image(file: UploadFile = File(...), api_key: str = Depends(verify_header)):
    content_type = file.content_type

    image_bytes = await file.read()
    try:
        img = None
        if content_type == 'image/gif':
            # Use Pillow to read the first frame of the GIF file
            with Image.open(BytesIO(image_bytes)) as img:
                if img.is_animated:
                    img.seek(0)  # Seek to the first frame of the GIF
                frame = img.convert('RGB')  # Convert to RGB mode
                np_arr = np.array(frame)  # Convert to NumPy array
                img = cv2.cvtColor(np_arr, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
        if img is None:
            # Use OpenCV for other image types
            np_arr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            err = f"The uploaded file {file.filename} is not a valid image format or is corrupted."
            print(err)
            return {'result': [], 'msg': str(err)}

        height, width, _ = img.shape
        if width > 10000 or height > 10000:
            return {'result': [], 'msg': 'height or width out of range'}

        data = {"detector_backend": detector_backend, "recognition_model": recognition_model}
        embedding_objs = await predict(_represent, img)
        # embedding_objs = DeepFace.represent(
        #     img_path=img,
        #     detector_backend=detector_backend,
        #     model_name=recognition_model,
        #     enforce_detection=True,  # 强制检测，如果为true会报错, 设置为False时可以针对整张照片进行特征识别
        #     align=True,
        # )
        #enforce_detection=True 时，未识别到人脸的错误信息
        #1
        # Face could not be detected in numpy array.Please confirm that the picture is a face photo or consider to set enforce_detection param to False.
        #2
        # Exception while extracting faces from numpy array.Consider to set enforce_detection arg to False.
        del img
        data["result"] = embedding_objs
        # logging.info("detector_backend: %s", detector_backend)
        # logging.info("recognition_model: %s", recognition_model)
        logging.info("detected_img: %s", file.filename)
        logging.info("img_type: %s", content_type)
        logging.info("detected_persons: %d", len(embedding_objs))
        for embedding_obj in embedding_objs:
            logging.info("facial_area: %s", str(embedding_obj["facial_area"]))
            logging.info("face_confidence: %f", embedding_obj["face_confidence"])
        return data
    except Exception as e:
        if 'set enforce_detection' in str(e):
            return {'result': []}
        print(e)
        return {'result': [], 'msg': str(e)}

def _represent(img):
  faces = faceAnalysis.get(img)
  results = []
  for face in faces:
    resp_obj = {}
    embedding = face.normed_embedding.astype(float)
    resp_obj["embedding"] = embedding.tolist()
    # print(len(resp_obj["embedding"]))
    box = face.bbox
    resp_obj["facial_area"] = {"x" : int(box[0]), "y" : int(box[1]), "w" : int(box[2] - box[0]), "h" : int(box[3] - box[1])}
    resp_obj["face_confidence"] = face.det_score.astype(float) 
    results.append(resp_obj)
  return results


async def predict(predict_func, inputs):
    return await asyncio.get_running_loop().run_in_executor(None, predict_func, inputs)


def restart_program():
    print("restart_program")
    python = sys.executable
    os.execl(python, python, *sys.argv)


if __name__ == "__main__":
    uvicorn.run("server:app", host=None, port=http_port)
