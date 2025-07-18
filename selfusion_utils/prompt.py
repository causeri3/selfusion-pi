from fastapi import FastAPI, Body

from selfusion_utils.args import get_args

args, unknown = get_args()

class PromptStore:
    def __init__(self, initial_prompt=args.prompt):
        self.prompt = initial_prompt

app = FastAPI()
prompt_store = PromptStore()

@app.post("/set_prompt")
async def set_prompt(prompt: str = Body(..., embed=True)):
    prompt_store.prompt = prompt
    return {"status": "success", "new_prompt": prompt_store.prompt}

@app.get("/get_prompt")
def get_prompt():
    return {"prompt": prompt_store.prompt}