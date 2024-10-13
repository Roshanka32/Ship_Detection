import streamlit as st
import pickle
import cv2
import numpy as np
import tensorflow as tf
import keras
from PIL import Image as im

# load SSD Model
loaded_model = keras.saving.load_model('D:/ACADEMIC/SEMESTER TWO/DESERTATION/Ship_Detection_Model/Ship_Classification/SSD/Models/ssd_model.keras')


# generate UI using streamlit

st.markdown("<h1 style='text-align: center;'>Ship Detection using SSD</h1>", unsafe_allow_html=True)
st.header('',divider='rainbow')


Uploaded_img = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if Uploaded_img:
    
    image = im.open(Uploaded_img)  
    st.image(image, use_column_width=True)
    
    # Convert RGBA mode to RGB mode if needed
    if image.mode == 'RGBA':
        image = image.convert('RGB')
        
    #preprocess image
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = loaded_model.predict(img_array)
    

    if prediction >= 0.5:
        st.subheader(':green[Ship Detected...]')
        
    else:
        st.subheader(':red[No Ship Detected!]')
