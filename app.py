import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
#from utils import predict_label
from PIL import Image
import numpy as np
st.title("Shoe Classification")

st.write("Predict the shoe that is being represented in the image.")
import urllib.request
def load_model():
    if not os.path.isfile('model.h5'):
        urllib.request.urlretrieve('https://github.com/Pravein-m/dl3/blob/5949d91f1adf00eb97f2d98ff0f1b9d0a0147c62/model.h5', 'model.h5')
    return tensorflow.keras.models.load_model('model.h5')

# model = load_model("model.h5")
l=['Adidas','Nike']

# model = load_model("model.h5")

uploaded_file = st.file_uploader(
    "Upload an image of a seed :", type="jpg"
)
predictions=-1
if uploaded_file is not None:
    image1 = Image.open(uploaded_file)
    image1=image.smart_resize(image1,(224,224))
    img_array = image.img_to_array(image1)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array/255.0
    predictions = load_model().predict(img_array)
    label=l[np.argmax(predictions)]


st.write("### Prediction Result")
if st.button("Predict"):
    if uploaded_file is not None:
        image1 = Image.open(uploaded_file)
        st.image(image1, caption="Uploaded Image", use_column_width=True)
        st.markdown(
            f"<h2 style='text-align: center;'>Image of {label}</h2>",
            unsafe_allow_html=True,
        )
    else:
        st.write("Please upload file or choose sample image.")


st.write("If you would not like to upload an image, you can use the sample image instead:")
sample_img_choice = st.button("Use Sample Image")

if sample_img_choice:
    image1 = Image.open("")
    image1=image.smart_resize(image1,(224,224))
    img_array = image.img_to_array(image1)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array/255.0
    predictions = load_model().predict(img_array)
    label=l[np.argmax(predictions)]
    st.markdown(
        f"<h2 style='text-align: center;'>{label}</h2>",
        unsafe_allow_html=True,
    )
