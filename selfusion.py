import logging
import uvicorn
import threading

from selfusion_utils.transformation import Transformation
from selfusion_utils.args import get_args

args, unknown = get_args()

logging.basicConfig(encoding='utf-8', level=logging.DEBUG)

def start_api():
    uvicorn.run("selfusion_utils.prompt:app", host="127.0.0.1", port=8000)


if __name__ == "__main__":
    api_thread = threading.Thread(target=start_api, daemon=True)
    api_thread.start()

    Transformation().run()
