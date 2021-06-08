import tensorflow as tf
from skimage.transform import resize
import time
from PIL import Image, ImageOps
from PIL import Image
from io import BytesIO
import io
import numpy as np
from PIL import Image
import numpy as np
from numpy import asarray
import streamlit as st
from tensorflow.keras import preprocessing

model = tf.keras.models.load_model('CNN-CT')


st.write("""
# COVID-19 identification App
This App analyzes a **chest CT-image** and identifies the presence of **COVID-19**!
""")

#SIDEBAR OF THE WEB APP. THIS TAKES THE INPUT OF THE USER(ETF AND WEIGHTS)
CT_Image =  st.file_uploader(
    'Select a CT-scan')

if CT_Image is not None:
    u_img = Image.open(CT_Image)
    show = st.image(u_img, use_column_width=True)
    show.image(u_img, 'Uploaded Image', use_column_width=True)

    u_img = ImageOps.grayscale(u_img)
    image = np.asarray(u_img) / 255
    my_image = resize(image, (150, 150)).reshape(-1, 150,150,1)




if st.sidebar.button("Click Here to Classify"):

    if CT_Image is None:

        st.sidebar.write("Please upload an Image to Classify")

    else:
        with st.spinner('Classifying ...'):
            Category = ["COVID", "Non-COVID"]
            pred = model.predict((my_image))

            if pred[0][0] > 0.5:
                Cat = "Non-COVID"
            else:
                Cat = "COVID"


            time.sleep(2)
            st.sidebar.success('Done!')

            st.sidebar.header("Algorithm Predicts: ")
            if Cat == "COVID":
                st.sidebar.write("The lung is classified as **COVID-19 affected**")
            elif Cat == "Non-COVID":
                st.sidebar.write("The lung is classified as **Healthy**")
            else:
                st.sidebar.write("ciao")
