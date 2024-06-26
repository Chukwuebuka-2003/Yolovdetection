import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image, ImageDraw
import numpy as np


# Load Faster R-CNN model
@st.cache(allow_output_mutation=True)
def load_model():
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model


model = load_model()

# Define the transforms
transform = transforms.Compose([transforms.ToTensor()])

st.title("Object Detection with Faster R-CNN")
st.write("Upload an image to detect objects")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display the uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Detecting...")

    # Transform the image
    img_tensor = transform(img).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        predictions = model(img_tensor)

    # Draw bounding boxes on the image
    draw = ImageDraw.Draw(img)
    for element in predictions[0]["boxes"]:
        draw.rectangle(
            ((element[0], element[1]), (element[2], element[3])), outline="red", width=3
        )

    # Convert boxes and labels to displayable format
    boxes = predictions[0]["boxes"].cpu().numpy()
    scores = predictions[0]["scores"].cpu().numpy()
    labels = predictions[0]["labels"].cpu().numpy()

    # Display the image with bounding boxes
    st.image(img, caption="Detected Image", use_column_width=True)

    # Display detected objects
    detected_objects = []
    for box, score, label in zip(boxes, scores, labels):
        if score > 0.5:  # Filter out low-confidence predictions
            detected_objects.append({"box": box, "score": score, "label": label})

    st.write("Detected Objects:")
    for obj in detected_objects:
        st.write(f"Label: {obj['label']}, Score: {obj['score']:.2f}")

# Optional: Save uploaded image and detected image
if uploaded_file and st.button("Save Results"):
    img.save("uploaded_image.jpg")
    img.save("detected_image.jpg")
    st.write("Images saved!")
