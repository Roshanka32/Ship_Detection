import streamlit as st
from ultralytics import YOLO
from PIL import Image as im
import numpy as np
from io import BytesIO

# Load classification and segmentation model weights
classification_model = YOLO(r"D:\ACADEMIC\SEMESTER TWO\DESERTATION\Ship_Detection_Model\Ship_Classification\YOLOv8\runs\classify\train2\weights\best.pt")
segmentation_model = YOLO(r"D:\ACADEMIC\SEMESTER TWO\DESERTATION\Ship_Detection_Model\Ship_Segmentation\Yolov8\results\Trained_Results\weights\best.pt") 


# Generate UI using Streamlit
st.markdown("<h1 style='text-align: center; margin-bottom: -80px'>Ship Detection and Segmentation using Yolov8 </h1>", unsafe_allow_html=True)
st.header('', divider='rainbow')

img = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if img:
    # Open the uploaded image
    image = im.open(img)
    
    # Convert RGBA mode to RGB mode if needed
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    # Set image height and width
    image = image.resize((300, 300))
    st.subheader(':blue[Classification Result]')
    st.image(image)

    # Convert image to bytes
    img_bytes = BytesIO()
    image.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    # Get classification model prediction
    classification_results = classification_model(image)
    res_probs = classification_results[0].probs.top1
    
    # Check if ship is detected
    if res_probs == 1:

        st.success("**Prediction  :** Ship Detected")
        
        # Yolov8 Segmentation >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        
        st.subheader(':blue[Segmentation Result]')
        
        # Pass the same image to the segmentation model
        segmentation_results = segmentation_model(image)

        # Get the segmented result image with labels and bounding boxes
        result_image = segmentation_results[0].plot(labels=True, boxes=True)

        # Convert the result image from array to PIL image
        result_pil_image = im.fromarray(result_image)
        
        result_pil_image = result_pil_image.resize((300, 300))
        
        # Display the predicted images
        st.image(result_pil_image, caption='Yolo v8 Segmented Image')
        
        
    else:
        st.error("**Prediction  :** No Ship Detected..!")