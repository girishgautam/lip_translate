from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from load_checkpoints import *
import numpy as np
import json, os
import time



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
async def numpy_test(request: Request):
    data = await request.json()  # Assuming this is a list of lists for frames.


    frames_storage.extend(data)  # Extend the storage with the new frames.

    if len(frames_storage) >= 75:
        predictions = []
        # Process in chunks of 75 frames
        while len(frames_storage) >= 75:
            chunk = frames_storage[:75]  # Extract the first 75-frame chunk
            del frames_storage[:75]  # Remove these frames from storage

            # Convert chunk to NumPy array for processing
            vid_frames_grey = np.expand_dims(np.array(chunk), axis=-1)

            # Process the chunk with your model here
            predicted_text = load_checkpoints(vid_frames_grey)  # Placeholder for model processing
            predictions.append(predicted_text)

        return {"message": " ".join(predictions)}
    else:
        return {"message": "Accumulating frames, please continue sending."}


loaded_data = np.load('/home/girishj/code/girishgautam/lip_translate/mathilda_test.npz', allow_pickle=True)

def test_function(data):
    # Assume data is a list of frames with each frame being a (30, 70) array

    # Save the incoming frames to an npz file
    timestamp = time.time()
    filename = f'./frames_{timestamp}.npz'
    np.savez_compressed(filename, frames=np.array(data))

    # Check if there are enough files to start processing
    if len([name for name in os.listdir('./') if name.endswith('.npz')]) * 5 >= 75:  # Adjust based on actual frames per file

        frames_storage = []
        for file in os.listdir('./'):
            if file.endswith('.npz'):
                with np.load(file) as data:
                    frames_storage.append(data['frames'])  # Accumulate frame arrays

        # Concatenate along the first axis (frame axis), assuming each is already (30, 70)
        frames_storage_array = np.concatenate(frames_storage, axis=0)

        predictions = []
        # Ensure there are enough frames and process in chunks of 75 frames
        while len(frames_storage_array) >= 75:
            # Select the first 75 frames
            chunk = frames_storage_array[:75]
            # Drop the processed frames
            frames_storage_array = frames_storage_array[75:]

            # Ensure the chunk shape is (75, 30, 70) before adding the channel dimension
            vid_frames_grey = np.expand_dims(chunk, axis=-1)  # Now should be (75, 30, 70, 1)
            print(vid_frames_grey.shape)

            # Process the chunk with your model here
            predicted_text = load_checkpoints(vid_frames_grey)
            print(predicted_text)# Placeholder for actual model processing
            predictions.append(predicted_text)
            

        # Cleanup: Delete the npz files to avoid reprocessing
        for file in os.listdir('./'):
                 if file.endswith('.npz'):
                    os.remove(file)

        return {"message": " ".join(predictions)}
    else:
        return {"message": "Accumulating frames, please continue sending."}


if __name__=='__main__':
    test_function(loaded_data['arr_0'])
