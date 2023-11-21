import streamlit as st
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.models import load_model

# Load the Keras skin type prediction model (replace 'your_model.h5' with the actual model file)
model = load_model('keras_model.h5')

# Function to preprocess and classify skin type
def classify_skin_type(image):
    # Preprocess the image (adjust based on your model requirements)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0  # Normalize pixel values to be between 0 and 1

    # Make a prediction
    prediction = model.predict(np.expand_dims(image, axis=0))

    # Convert the prediction to a class label
    skin_type_label = "Normal" if prediction[0, 0] > 0.5 else "Oily"

    return skin_type_label

# Function to process the uploaded image
def process_uploaded_image(uploaded_image):
    # Convert the uploaded file to a NumPy array
    image_np = np.array(uploaded_image)

    # Ensure the image has three channels (RGB)
    if len(image_np.shape) < 3 or image_np.shape[-1] < 3:
        st.warning("Image has insufficient channels. Unable to process.")
        return None

    # Display the uploaded image
    st.image(image_np, caption="Uploaded Image", use_column_width=True)

    # Classify the skin type
    skin_type_label = classify_skin_type(image_np)

    return skin_type_label

# Main Streamlit app
def main():
    st.title("Skin Type Prediction App")

    # Sidebar
    st.sidebar.header("User Options")

    # Option to upload an image
    uploaded_image = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Option to use webcam
    use_webcam = st.sidebar.checkbox("Use Webcam")

    if use_webcam:
        st.warning("Using the webcam is not supported in this demo.")

    # Main content
    if uploaded_image:
        # Process the uploaded image
        skin_type_label = process_uploaded_image(uploaded_image)

        if skin_type_label is not None:
            # Display the result
            st.success(f"Predicted Skin Type: {skin_type_label}")

# Run the app
if __name__ == "__main__":
    main()
