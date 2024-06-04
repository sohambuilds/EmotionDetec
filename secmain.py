import cv2
import numpy as np
import gradio as gr
from keras._tf_keras.keras.models import model_from_json
from tensorflow import keras  # Assuming TensorFlow backend for your model

# Function to load the model (assuming saved as HDF5)
def load_model():
    json_file = open("emotiondetector.json", "r")
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights("emotiondetectorweights.weights.h5")
    return model
  

# Function to extract features from an image
def extract_features(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image = cv2.resize(image, (48, 48))  # Resize to model input size
    feature = np.array(image).reshape(1, 48, 48, 1)  # Reshape for model input
    return feature / 255.0  # Normalize pixel values

# Function to detect emotions in an image
def detect_emotion(image):
    model = load_model()  # Load the model on each inference
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, 1.3, 5)

    if len(faces) > 0:
        (p, q, r, s) = faces[0]
        face_image = image[q:q + s, p:p + r]
        face_features = extract_features(face_image)
        prediction = model.predict(face_features)
        prediction_label = labels[prediction.argmax()]
        return prediction_label
    else:
        return "No face detected"  # Handle no face case

# Define emotion labels (modify based on your model's output)
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
          4: 'neutral', 5: 'sad', 6: 'surprise'}

# Gradio interface definition
interface = gr.Interface(
    fn=detect_emotion,
    inputs="image",
    outputs="text",
    description="Emotion Detection with Your Model",
    allow_flagging=True  # Allow reporting issues
)

# Launch the Gradio app on Hugging Face Spaces
interface.launch(share=True)  # Set "share=True" for public deployment

