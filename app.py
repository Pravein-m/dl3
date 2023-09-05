import streamlit as st
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Define the prediction function
def prediction():
    st.write("Predict the Rice image that is being represented in the image")
    
    # Define the input fields
    model_path = "model.h5"  # Update with the correct path to your model file
    
    if not os.path.isfile(model_path):
        st.write("Model file not found. Please upload the model file or provide the correct path.")
        return
    
    model = load_model(model_path)
    
    uploaded_file = st.file_uploader("Upload an image of a Rice Image:", type="jpg")
    
    if uploaded_file is not None:
        image1 = Image.open(uploaded_file)
        image1 = image1.resize((64, 64))  # Resize the image to match the model input size
        img_array = image.img_to_array(image1)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        predictions = model.predict(img_array)
        
        labels = ["Adidas", "Nike"]
        
        st.write("### Prediction Result")
        if st.button("Predict"): 
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

if __name__ == "__main__":
    main()
