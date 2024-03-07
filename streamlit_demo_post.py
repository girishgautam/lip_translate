import streamlit as st
from PIL import Image
import requests
import os
import cv2
import numpy as np
#GET Request with existing API

VIDEO_PATH = "data/raw_data/s1/testt/"

video_list = os.listdir(VIDEO_PATH)
frames = []
frames_colour = []

for vid_name in video_list: # Iterate on video files
    print(f"Processing video: {vid_name}")
    if vid_name.endswith(".mpg"):
        vid_path = VIDEO_PATH + vid_name
        vid = cv2.VideoCapture(vid_path)

        while(True):
            success, frame = vid.read()
                # Read frame
            if not success:
                break
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)   # Convert image into grayscale
            frames.append(gray)                  # Add image to the frame buffer
            frames_colour.append(frame)
        vid.release()

file_path = 'data/raw_data/s1/testt/bbaf5a.mpg'

os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

st.write('WE DID IT THIS IS THE CORRECT THING BELOW!')
v_file = open('test_video.mp4', 'rb')
v_bytes = v_file.read()
st.video(v_bytes)

frames_array = np.array(frames)
st.write(frames_array.shape)


fastapi_url = "http://localhost:8000/predict/"

#for frame in frames:
frame_bytes = frames_array.tobytes()
st.write('I made the frames into bytes but now I cannot display them!')
st.video(frame_bytes)
response = requests.post(fastapi_url, data=frame_bytes)
st.write(response.ok)
