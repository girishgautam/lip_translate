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

# Test data not seen by model_mathilda_2000_12mar
#zipped_vids_5000_0-200.npz - 'srwm6p', 'brby5a', 'lbaq1a', 'lrix2n', 'swbb9s', 'lgwk5p', 'bwwn7p', 'sbie9p', 'pbwc1s',
# 'pwwp1n', 'brbe6s', 'pbahza', 'sbwm1p', 'prit7n', 'pbio4n', 'sbbl5s', 'swig9a', 'bwag5p', 'sgbi9n', 'bgatzn', 'prwn4p',
# 'prbi8p', 'lriv5p', 'bbio5n', 'sbizzs', 'swam8s', 'pwwj7s', 'sran8p', 'lwic7n', 'pgwi2p', 'sgba5n', 'swbg7n', 'sbalzs'.

#zipped_vids_5000_800-1100.npz - 'swbi6p', 'sbby6a', 'pgih4a', 'bbwe1p', 'pbwc7s', 'pgwi7a', 'lrin5n', 'lbij4n', 'lwbc3p',
# 'lrbj1p', 'swbv3p', 'lway9n', 'lbah4n', 'srwn5n', 'pwib8n', 'bgwf4p', 'swau3a', 'bgaz8a', 'pwbb5n', 'pbba2n', 'lrbo9s',
# 'pgwo5p', 'lwbk6n', 'srau3s', 'sbiq8a', 'pwio1p', 'prim5s', 'srif4s', 'pbau2s', 'bwae1a', 'lrwczn', 'pwwp7n', 'lbwc5p',

data = np.load('/home/girishj/code/girishgautam/lip_translate/zipped_vids_5000_0-200.npz')

# Convert to a Python dictionary
data_vids= {key: data[key] for key in data.files}


tes_vid_url = data_vids['pgwi2p']

predicted_text = predict_video(model, tes_vid_url)
print(predicted_text)
