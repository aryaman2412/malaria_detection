import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

st.write("""
            # Malaria Cell Classification
        """)

model_path = r"C:\Users\USER\Desktop\malaria-detection\model_vgg19.h5"

upload_file = st.sidebar.file_uploader("Upload Cell Images", type=["png", "jpg", "jpeg"])

Generate_pred = st.sidebar.button("Predict")

model = tf.keras.models.load_model(model_path)

def import_n_pred(image_data, model):
    size = (224, 224)
    # Use Image.Resampling.LANCZOS instead of Image.ANTIALIAS
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img = np.asarray(image)
    reshape = img[np.newaxis, ...]
    pred = model.predict(reshape)
    return pred

if Generate_pred:
    if upload_file is not None:
        image = Image.open(upload_file)
        with st.expander('Cell Image', expanded=True):
            st.image(image, use_column_width=True)
        pred = import_n_pred(image, model)
        labels = ["MALARIA POSITIVE", "MALARIA NEGATIVE"]
        st.title("Result: {}".format(labels[np.argmax(pred)]))
    else:
        st.error("Please upload an image file.")
