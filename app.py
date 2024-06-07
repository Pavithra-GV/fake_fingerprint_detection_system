# IMPORT
import os
import tensorflow as tf
import keras
import cv2
import numpy as np
from keras.models import load_model
import gradio as gr

# Load the pre-trained model
model = load_model('<insert path of the saved model>')

# Prediction function
def predict_image(image):
    resize = tf.image.resize(image, (256, 256))
    yhat = model.predict(np.expand_dims(resize / 255, 0))
    if yhat > 0.5:
        return 'Predicted as real'
    else:
        return 'Predicted as fake'

# Gradio interface
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="numpy"),
    outputs="text",
    title="Fake Finger Detection",
    description="Upload an image to check if it is a real or fake finger."
)

# Launch the interface
interface.launch(share=True)
