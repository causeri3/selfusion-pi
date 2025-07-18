from PIL import Image
import requests
import io
import numpy as np
import logging

from selfusion_utils.args import get_args
from selfusion_utils.prompt import prompt_store

args, unknown = get_args()

def request_sdxlturbo(selfie,
                      result_container,
                      #prompt:str = args.prompt,
                      amount_pics:int = args.amount_pics,
                      num_inference_steps:int = args.num_inference_steps,
                      strength_min:float =args.strength_min,
                      strength_max:float = args.strength_max,
                      guidance_scale: int = args.guidance_scale,
                      # 10 min, but thread kills it anyway earlier with LOADING_DURATION_SEC
                      timeout=600):
    data = {
        'prompt': prompt_store.prompt,
        'amount_pics': amount_pics,
        'num_inference_steps': num_inference_steps,
        'strength_min': strength_min,
        'strength_max': strength_max,
        'guidance_scale': guidance_scale
    }
    try:
        files = _selfie_to_file_data(selfie)

        response = requests.post(
            "http://localhost:8000/sdxlturbo",
                                  files=files,
                                  data=data,
                                  timeout=timeout)
        buffer = io.BytesIO(response.content)
        result_container['images'] = np.load(buffer)
        result_container['success'] = True
    except Exception as e:
        logging.warning(f"SDXL Turbo failed: {e}")
        result_container['success'] = False


def _selfie_to_file_data(selfie):
    selfie_rgb = Image.fromarray(selfie[..., ::-1])
    buffer = io.BytesIO()
    selfie_rgb.save(buffer, format="PNG")
    buffer.seek(0)
    return {'file': ('selfie.png', buffer, 'image/png')}
