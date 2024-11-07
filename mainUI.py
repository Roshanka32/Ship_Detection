import streamlit as st
from ultralytics import YOLO
from PIL import Image as im
import numpy as np
from io import BytesIO

# Load classification and segmentation model weights
classification_model = YOLO("D:/ACADEMIC/SEMESTER TWO/DESERTATION/Ship_Detection_Model/Ship_Classification/YOLOv8/runs/classify/train2/weights/best.pt")
segmentation_model = YOLO("D:/ACADEMIC/SEMESTER TWO/DESERTATION/Ship_Detection_Model/Ship_Segmentation/runs/segment/train2/weights/best.pt")

# Generate UI using Streamlit
st.markdown("<h1 style='text-align: center;'>Ship Detection and Segmentation using Yolov8</h1>", unsafe_allow_html=True)

img = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if img:
    # Open the uploaded image
    image = im.open(img)
    
    # Convert RGBA mode to RGB mode if needed
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    # Set image height and width
    image = image.resize((500, 500))

    # Convert image to bytes
    img_bytes = BytesIO()
    image.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    # Get classification model prediction
    classification_results = classification_model(image)
    res_probs = classification_results[0].probs.top1
    
    # Check if ship is detected
    if res_probs == 1:
        st.subheader(':green[Ship Detected...]')
        
        # Pass the same image to the segmentation model
        segmentation_results = segmentation_model(image)

        # Get the segmented result image with labels and bounding boxes
        result_image = segmentation_results[0].plot(labels=True, boxes=True)  # Plotting the results with labels and bounding boxes

        # Convert the result image from array to PIL image
        result_pil_image = im.fromarray(result_image)
        
        # Display the original and predicted images
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.image(result_pil_image, caption='Segmented Image', use_column_width=True)
    else:
        st.subheader(':red[No Ship Detected!]')