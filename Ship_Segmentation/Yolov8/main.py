import streamlit as st
from ultralytics import YOLO
from PIL import Image as im
import numpy as np
from io import BytesIO

# Load best model weights
model = YOLO("D:/04_Personal_Files/Python/Ship_Detection_Model/Ship_Segmentation/runs/segment/train9/weights/best.pt")

# Generate UI using Streamlit
st.markdown("<h1 style='text-align: center;'>Ship Segmentation using Yolov8</h1>", unsafe_allow_html=True)

img = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if img:
    # Open the uploaded image
    image = im.open(img)
    
    # Convert RGBA mode to RGB mode if needed
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    # Set image height and width
    image = image.resize((640, 640))

    # Convert image to bytes
    img_bytes = BytesIO()
    image.save(img_bytes, format='jpeg')
    img_bytes.seek(0)
    
    # Get model prediction
    results = model(image)

    # Get the segmented result image with labels and bounding boxes
    result_image = results[0].plot(labels=True, boxes=True)  

    # Convert the result image from array to PIL image
    result_pil_image = im.fromarray(result_image)
    
    # Display the original and predicted images
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.image(result_pil_image, caption='Segmented Image', use_column_width=True)