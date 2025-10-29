import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# --- Load model and class labels ---
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"
model = tf.keras.models.load_model(model_path)
class_indices = json.load(open(f"{working_dir}/class_indices.json"))

# --- Translations ---
translations = {
    "English": {
        "title": "Plant Disease Classifier",
        "upload_prompt": "Upload an image...",
        "button": "Classify",
        "prediction": "Prediction",
        "select_language": "Select Language"
    },
    "Hindi": {
        "title": "पौध रोग वर्गीकरण",
        "upload_prompt": "एक छवि अपलोड करें...",
        "button": "वर्गीकृत करें",
        "prediction": "अनुमानित रोग",
        "select_language": "भाषा चुनें"
    },
    "Telugu": {
        "title": "మొక్కల వ్యాధి వర్గీకరణ",
        "upload_prompt": "చిత్రాన్ని అప్‌లోడ్ చేయండి...",
        "button": "వర్గీకరించు",
        "prediction": "అంచనా వ్యాధి",
        "select_language": "భాషను ఎంచుకోండి"
    }
}

# --- Language Selector ---
lang = st.selectbox("🌐 " + translations["English"]["select_language"], list(translations.keys()))
T = translations[lang]  # Current translation dictionary

# --- UI Title ---
st.title(T["title"])

# --- File Upload ---
uploaded_image = st.file_uploader(T["upload_prompt"], type=["jpg", "jpeg", "png"])

# --- Image Processing and Prediction ---
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# --- Display and Predict ---
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button(T["button"]):
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f'{T["prediction"]}: {str(prediction)}')
