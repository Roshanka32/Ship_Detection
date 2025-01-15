import streamlit as st
from ultralytics import YOLO
from PIL import Image as im
from io import BytesIO

# load best model weights
model = YOLO(r"D:\ACADEMIC\SEMESTER TWO\DESERTATION\Ship_Detection_Model\Ship_Classification\YOLOv8\runs\classify\train2\weights\best.pt")
# generate UI using streamlit

st.markdown("<h1 style='text-align: center;'>Ship Detection using Yolov8</h1>", unsafe_allow_html=True)
st.header('',divider='rainbow')

img = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if img:
    image = im.open(img)  
    
    # Convert RGBA mode to RGB mode if needed
    if image.mode == 'RGBA':
        image = image.convert('RGB')
        
    #set image height and weight.........
    image = image.resize((500, 500))

    # Convert image to bytes
    img_bytes = BytesIO()
    image.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    # get model prediction
    results = model(image)
    
    img_array = results[0].orig_img 
    img_data = im.fromarray(img_array) 
    img_data
    
    res_probs = results[0].probs.top1
    
    # print prob
    if res_probs == 1:
        st.subheader(':green[Ship Detected...]')
        
    else:
        st.subheader(':red[No Ship Detected!]')

    # st.text(results)