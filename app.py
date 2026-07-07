import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import numpy as np

st.set_page_config(
    page_title='Solar Panel Defect Classifier',
    page_icon='☀️',
    layout='centered'
)

st.title("☀️ Solar Panel Defect Classifier")
st.write("Upload an Image of a Solar Panel to Detect Defects Using our Model")


CLASSES=[
    'Bird-drop',
    'Clean',
    'Dusty',
    'Electrical-damage',
    'Physical-damage',
    'Snow-Coverd'
]

uploaded_file=st.file_uploader('Upload a Solar Panel Image..',type=['jpg','jpeg','png'])