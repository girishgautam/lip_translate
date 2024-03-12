from tensorflow.keras.models import load_model
from lip_translate.initiate_model import initiate_model, predict_video
import numpy as np

# # Load the model
model=initiate_model()

checkpoint_dir = 'model_mathilda_2000_12mar'

# If you want to load weights from a specific epoch
epoch_number = 100  # for example, to load from checkpoint_epoch-06
checkpoint_path = f"{checkpoint_dir}/checkpoint_epoch-{epoch_number:02d}"

model.load_weights(checkpoint_path)


data = np.load('/home/girishj/code/girishgautam/lip_translate/zipped_vids_2000_3.npz')

# Convert to a Python dictionary
data_vids= {key: data[key] for key in data.files}


tes_vid_url = data_vids['bwwz1n']

predicted_text = predict_video(model, tes_vid_url)
print(predicted_text)
