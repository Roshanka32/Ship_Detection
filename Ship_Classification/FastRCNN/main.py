import streamlit as st
import pickle
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import keras
import io


# Load the saved Fast R-CNN model from the specified path
saved_model =  keras.saving.load_model('D:/ACADEMIC/SEMESTER TWO/DESERTATION/Ship_Detection_Model/Ship_Classification/FastRCNN/Models/fastRCNN_Model.keras') 


# generate UI using streamlit

st.markdown("<h1 style='text-align: center;'>Ship Detection using FastRCNN</h1>", unsafe_allow_html=True)
st.header('',divider='rainbow')

Uploaded_img = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
        
        
if Uploaded_img:
    
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(Uploaded_img.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    st.image(img, use_column_width=True)

    # Resize the image
    img = cv2.resize(img, (224, 224))

    # Normalize the image
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict the class of the image
    prediction = saved_model.predict(img)

    if prediction[0][0] > 0.5:
        st.subheader(':green[Ship Detected...]')
        
    else:
        st.subheader(':red[No Ship Detected!]')
