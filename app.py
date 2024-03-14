import streamlit as st
import os

import tensorflow as tf
import numpy as np
from lip_translate.initiate_model import initiate_model, predict_video


data = np.load('/home/girishj/code/girishgautam/lip_translate/lip_translate/raw_data/zipped_vids_2000_3.npz')

# Convert to a Python dictionary
data_vids= {key: data[key] for key in data.files}


tes_vid_url = data_vids['bwwz1n']

col1, col2 = st.columns(2)


with col1:
        st.info('The video below displays the converted video in mp4 format')
        os.system(f'ffmpeg -i {tes_vid_url} -vcodec libx264 test_video.mp4 -y')

        # Rendering inside of the app
        video = open('test_video.mp4', 'rb')
        video_bytes = video.read()
        st.video(video_bytes)


with col2:
        st.info('This is all the machine learning model sees when making a prediction')

        model = initiate_model()
        checkpoint_dir = 'model_mathilda_2000_12mar'

# If you want to load weights from a specific epoch
        epoch_number = 100  # for example, to load from checkpoint_epoch-06
        checkpoint_path = f"{checkpoint_dir}/checkpoint_epoch-{epoch_number:02d}"

        model.load_weights(checkpoint_path)

        predicted_text = predict_video(model, tes_vid_url)



        # Convert prediction to text

        st.text(predicted_text)
