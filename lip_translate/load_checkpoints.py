from tensorflow.keras.models import load_model
from lip_translate.initiate_model import initiate_model, predict_video

# # Load the model
model=initiate_model()

checkpoint_dir = 'models_checkpoints'

# If you want to load weights from a specific epoch
epoch_number = 80  # for example, to load from checkpoint_epoch-06
checkpoint_path = f"{checkpoint_dir}/checkpoint_epoch-{epoch_number:02d}"

model.load_weights(checkpoint_path)

tes_vid_url = 'url XXXXXXXXXXXXX'

predicted_text = predict_video(model, tes_vid_url)
