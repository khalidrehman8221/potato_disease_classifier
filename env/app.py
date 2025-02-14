import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# Load the trained model
MODEL_PATH = "E:\Work\ML End-End\Potato_project\env\model_potato.h5"  # Make sure to have a .h5 model
model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels (Modify based on your dataset)
CLASS_NAMES = ['Healthy', 'Late Blight', 'Early Blight']

# Streamlit UI
st.title(":leaves: Potato Disease Classifier :leaves:")
st.write("Upload an image of a potato leaf to classify its condition.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)


    # Preprocess the image
    img = image.resize((224, 224))  # Ensure the image is 224x224 (Modify based on your model input size)
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    # Display results
    st.write(f"### Prediction: {predicted_class}")
    st.write(f"### Confidence: {confidence:.2f}%")
