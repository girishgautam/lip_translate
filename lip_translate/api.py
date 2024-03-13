from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from lip_translate.load_checkpoints import *
import numpy as np
import json



app = FastAPI()

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get("/home/")
def home():
    return {'message': "Hello"}

@app.post("/predict/")
async def numpy_test(test: Request):
    data = await test.json()

    first_frame = np.array(np.array(json.loads(data)))


    loaded_data = np.expand_dims(first_frame ,axis=-1)
    num_chunks = loaded_data.shape[0] // 75
    complete_chunks_only = loaded_data[:num_chunks * 75]

    list = []
    for chunk_start in range(0, complete_chunks_only.shape[0], 75):
        chunk = complete_chunks_only[chunk_start:chunk_start+75]
        predicted_text = load_checkpoints(chunk)
        list.append(predicted_text)
    result = ' '.join(list)
    return {"message": result}
