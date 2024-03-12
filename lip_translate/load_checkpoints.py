from tensorflow.keras.models import load_model
from lip_translate.initiate_model import initiate_model

# # Load the model
model = load_model('models/checkpoint')

model=initiate_model()

model.load_weights('models - checkpoint 96')
