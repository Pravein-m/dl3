
import streamlit as st
import os
from tensorflow import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.preprocessing import image
from PIL import Image
import warnings
warnings.filterwarnings('ignore')
import pathlib
from keras.models import load_model
import matplotlib.image as mpimg


# Define the prediction function
def prediction():
    
    st.write("Predict the Rice image that is being represented in the image")
    
    # Define the input fields
    model = load_model("model.h5")


    
    uploaded_file = st.file_uploader(
        "Upload an image of a Rice Image:", type="jpg"
    )
    predictions=-1
    if uploaded_file is not None:
        image1 = Image.open(uploaded_file)
        image1=image.smart_resize(image1,(64,64))
        img_array = image.img_to_array(image1)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array/255.0
        predictions = model.predict(img_array)
    st.write("### Prediction Result")
    if st.button("Predict"): 
        labels = {"Adidas","Nike"}
        if prediction!=-1:
            if uploaded_file is not None:
                image1 = Image.open(uploaded_file)
                st.image(image1, caption="Uploaded Image", use_column_width=True)
                st.markdown(
                    f"<h2 style='text-align: center;'>Image of {labels[np.argmax(predictions)]}</h2>",
                    unsafe_allow_html=True,
                )
            else:
                st.write("Please upload file or choose sample image.")

def main():
    st.set_page_config(page_title="Rice Image Classification", page_icon=":heart:")
    st.markdown("<h1 style='text-align: center; color: white;'>Rice Image Classification</h1>", unsafe_allow_html=True)
# Create the tab layout
    tabs = ["Home", "Classification"]
    page = st.sidebar.selectbox("Select a page", tabs)

# Show the appropriate page based on the user selection
    if page == "Home":
        home()
    elif page == "Classification":
        prediction()
main()
