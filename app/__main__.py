import signal
import subprocess
import logging
import os

logging.basicConfig(level=logging.INFO)
logging.info("Initializing MT Photos AI")

try:
    with subprocess.Popen(
        [
            "python",
            os.path.join(os.path.dirname(__file__), "main.py"),
        ],
    ) as cmd:
        cmd.wait()
except KeyboardInterrupt:
    cmd.send_signal(signal.SIGINT)
exit(cmd.returncode)
