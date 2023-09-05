import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import urllib.request
import os

st.title("Shoe Classification")
st.write("Predict the shoe that is being represented in the image.")

# Define a function to load the model
@st.cache(allow_output_mutation=True)
def load_shoe_model():
    model_path = 'model.h5'  # Update this path to the correct location of your 'model.h5' file
    if not os.path.isfile(model_path):
        urllib.request.urlretrieve('https://github.com/Pravein-m/dl3/blob/5949d91f1adf00eb97f2d98ff0f1b9d0a0147c62/model.h5', model_path)
    return load_model(model_path)

model = load_shoe_model()
class_labels = ['Adidas', 'Nike']

# The rest of your code remains the same.


uploaded_file = st.file_uploader("Upload an image of a shoe:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image1 = Image.open(uploaded_file)
    image1 = image1.resize((224, 224))  # Resize the image to match the model input size
    img_array = image.img_to_array(image1)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    predictions = model.predict(img_array)
    label = class_labels[np.argmax(predictions)]

    st.write("### Prediction Result")
    if st.button("Predict"):
        st.image(image1, caption="Uploaded Image", use_column_width=True)
        st.markdown(f"<h2 style='text-align: center;'>Image of {label}</h2>", unsafe_allow_html=True)
else:
    st.write("Please upload a file or choose a sample image.")

# Provide an option to use a sample image
sample_img_choice = st.button("Use Sample Image")
if sample_img_choice:
    sample_image_path = "sample_image.jpg"  # Provide the path to your sample image
    sample_image = Image.open(sample_image_path)
    sample_image = sample_image.resize((224, 224))
    sample_img_array = image.img_to_array(sample_image)
    sample_img_array = np.expand_dims(sample_img_array, axis=0)
    sample_img_array = sample_img_array / 255.0
    sample_predictions = model.predict(sample_img_array)
    sample_label = class_labels[np.argmax(sample_predictions)]

    st.write("### Prediction Result for Sample Image")
    st.image(sample_image, caption="Sample Image", use_column_width=True)
    st.markdown(f"<h2 style='text-align: center;'>{sample_label}</h2>", unsafe_allow_html=True)
