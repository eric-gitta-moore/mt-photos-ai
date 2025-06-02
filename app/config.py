import os
import sys
import torch

api_auth_key = os.getenv("API_AUTH_KEY", "mt_photos_ai_extra")
device = os.getenv("DEVICE", "cpu")
onnx_providers = os.getenv("ONNX_PROVIDERS", "CUDAExecutionProvider,CPUExecutionProvider").split(",")
on_linux = sys.platform.startswith("linux")
http_port = int(os.getenv("HTTP_PORT", "8060"))
server_restart_time = int(os.getenv("SERVER_RESTART_TIME", "300"))


# 废弃 默认关闭
env_auto_load_txt_modal = (
    os.getenv("AUTO_LOAD_TXT_MODAL", "off") == "on"
)  # 是否自动加载 CLIP 文本模型，开启可以优化第一次搜索时的响应速度，文本模型占用 700 多 m 内存
