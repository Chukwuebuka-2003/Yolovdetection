import streamlit as st
import torch
from PIL import Image
import numpy as np

# Load YOLOv5 model
model = torch.hub.load("ultralytics/yolov5", "yolov5s")

st.title("Object Detection with YOLOv5")
st.write("Upload an image to detect objects")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Detecting...")

    # Perform inference
    results = model(img)

    # Render results on the image
    results.render()  # this updates results.imgs with boxes and labels

    # Convert the image to display it in Streamlit
    detected_img = Image.fromarray(np.squeeze(results.render()))

    st.image(detected_img, caption="Detected Image", use_column_width=True)

    # Get the detected objects
    detected_objects = results.pandas().xyxy[0]
    st.write(detected_objects[["name", "confidence"]])

# Optional: Save uploaded image and detected image
if st.button("Save Results"):
    img.save("uploaded_image.jpg")
    detected_img.save("detected_image.jpg")
    st.write("Images saved!")
