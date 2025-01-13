import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import requests
from bs4 import BeautifulSoup

# Load YOLO models
models = {
    "leaf": YOLO("./models/leaf.pt", task='detect'),
    "flower": YOLO("./models/flower.pt", task='detect'),
    "fruit": YOLO("./models/fruit.pt", task='detect'),
}

# Load Flower Classifier Model
flower_classifier = load_model("./models/flower_classifier_model.h5")

# Flower Class Labels
class_labels = [
    'bougainvillea', 'daisy', 'dandelion', 'frangipani', 'hibiscus', 
    'rose', 'sunflower', 'tulips', 'zinnia'
]

# Preprocessing function for flower classification
def preprocess_image(image):
    image = cv2.resize(image, (150, 150))
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

# Fetch description for the flower from Wikipedia
def get_plant_description(flower_name):
    search_url = f"https://en.wikipedia.org/wiki/{flower_name.replace(' ', '_')}"
    try:
        response = requests.get(search_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract paragraphs and filter out empty or irrelevant ones
        paragraphs = soup.find_all('p', recursive=True)
        full_description = ""
        
        for paragraph in paragraphs:
            text = paragraph.get_text(strip=True)
            if text and not text.lower().startswith("see also") and len(text) > 50:  # Filter short or irrelevant content
                full_description = text
                break
        
        return full_description if full_description else "No detailed description found for this plant."
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Streamlit UI styles
st.markdown(
    """
    <style>
        body { background-color: #212529; color: white; }
        h1, h3 { font-weight: bold; }
        .daily-tip { background: linear-gradient(45deg, #2D6A4F, #40916C); color: white; padding: 1rem; }
    </style>
    """, 
    unsafe_allow_html=True
)

# App Header
st.markdown("<h1>Hello, Plant Lover!</h1>", unsafe_allow_html=True)

# Live Video Detection Section
st.markdown("<h3>Live Plant Detection</h3>", unsafe_allow_html=True)

# Video capture toggle
video_capture = None
if st.button("Start Video Capture"):
    video_capture = cv2.VideoCapture(0)

if st.button("Stop Video Capture") and video_capture is not None:
    video_capture.release()
    cv2.destroyAllWindows()
    video_capture = None
    st.write("Video Capture Stopped.")

if video_capture and video_capture.isOpened():
    frame_placeholder = st.empty()  # Container for video frames
    ret, frame = video_capture.read()

    if not ret:
        st.error("Error: Unable to read the video feed.")
    else:
        # YOLO detection and bounding box drawing
        for name, model in models.items():
            results = model(frame)
            if results and results[0].boxes:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].numpy())
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    label = f"{name} ({conf:.2f})"
                    color = (0, 255, 0) if name == "leaf" else (255, 0, 0) if name == "flower" else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)

# Image Upload and Classification
st.markdown("<h3>Gallery</h3>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Perform object detection
    detected_image = image.copy()
    for name, model in models.items():
        results = model(detected_image)
        if results and results[0].boxes:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].numpy())
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                label = f"{name} ({conf:.2f})"
                color = (0, 255, 0) if name == "leaf" else (255, 0, 0) if name == "flower" else (0, 0, 255)
                cv2.rectangle(detected_image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(detected_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    st.image(detected_image, caption="Detected Objects", use_column_width=True)

    # Flower Classification
    preprocessed_image = preprocess_image(image)
    prediction = flower_classifier.predict(preprocessed_image)
    predicted_class = np.argmax(prediction, axis=1)
    predicted_label = class_labels[predicted_class[0]]

    st.markdown(f"### Predicted Flower: {predicted_label}", unsafe_allow_html=True)
    st.markdown(f"Confidence: {prediction[0][predicted_class[0]]:.2f}", unsafe_allow_html=True)

    # Display plant description
    if st.button('Get Description'):
        description = get_plant_description(predicted_label)
        st.markdown(f"### About {predicted_label}:", unsafe_allow_html=True)
        st.markdown(f"{description}", unsafe_allow_html=True)

# Daily Plant Tip
st.markdown(
    """
    <div class="daily-tip">
        <h5>Daily Plant Tip</h5>
        <p>Most indoor plants grow best in indirect sunlight and moderate humidity.</p>
    </div>
    """, 
    unsafe_allow_html=True
)
