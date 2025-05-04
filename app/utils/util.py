import os
import sys
import asyncio
from app.config import server_restart_time


def restart_program():
    print("restart_program")
    python = sys.executable
    os.execl(python, python, *sys.argv)


async def predict(predict_func, inputs):
    return await asyncio.get_running_loop().run_in_executor(None, predict_func, inputs)


def to_fixed(num):
    return str(round(num, 2))


async def restart_timer():
    await asyncio.sleep(server_restart_time)
    restart_program()
