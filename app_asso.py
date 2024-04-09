# ui
import streamlit as st

# paths, folders/files
import os, sys, random, re
from io import BytesIO

# math, dataframes
import numpy as np
import pandas as pd
# NN
import tensorflow as tf
from tensorflow.keras.metrics import Accuracy, Precision, Recall, AUC
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import pickle

model_name = 'NASNetLarge'

from tensorflow.keras.applications.nasnet import NASNetLarge
from tensorflow.keras.applications.nasnet import preprocess_input
size_wh = 331
target_size=(size_wh, size_wh)


# Define a function to load the model
@st.cache_resource
def load_model():
    # Load the pre-trained model
    pickle_path_model = './NASNetLarge_10_lay_11.1_pc_model.pkl'
    model = pickle.load(open(pickle_path_model, 'rb'))

    return model

# Load the model
model = load_model()


# Define a function to load the label encoder
@st.cache_resource
def load_encoder():
    with open('./pickle/label_encoder.pkl', 'rb') as encoder:
        label_encoder = pickle.load(encoder)

    return label_encoder

# Load the label encoder
label_encoder = load_encoder()

# Define a function to load the dict
@st.cache_resource
def load_dico_fr():
    with open('./pickle/dico_fr.pkl', 'rb') as dico_fr:
        dico_fr = pickle.load(dico_fr)

    return dico_fr

# Load the label encoder
dico_fr = load_dico_fr()


# Function to handle file upload
def handle_uploaded_file(uploaded_file):
    # Read the uploaded file as bytes
    file_bytes = uploaded_file.getvalue()

    # Load the image from bytes
    image = load_img(BytesIO(file_bytes))
    # Display the uploaded image
    st.image(image, caption='Uploaded Image')

    image = load_img(BytesIO(file_bytes), target_size=target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    image = np.asarray(image)

    # prediction = model.predict(image)
    prediction = model.predict(image)
    prediction_dict = {str(i): float(prediction[0][i]) for i in range(0, 120)}

    # Sort the dictionary based on predicted probabilities in descending order
    sorted_dict = dict(sorted(prediction_dict.items(), key=lambda item: item[1], reverse=True))

    for i, breed, proba in enumerate(list(sorted_dict.items())[:2]):
        answer = dico_fr[str(breed)]
        if i == 0: # première réponse en bleu
            st.write(f"Model prediction: :blue[{answer}] ({int(proba*100)}%)")
        else: # defaut color
            st.write(f"Model prediction: {answer} ({int(proba*100)}%)")


# Main function
def main():
    st.title("Drag and Drop Image Uploader")

    # Drag and drop area
    st.write("Drag and drop an image file here:")
    uploaded_file = st.file_uploader(label="", type=["jpg", "jpeg", "png"], accept_multiple_files=False, key="fileUploader")

    # Handle file upload
    if uploaded_file is not None:
        handle_uploaded_file(uploaded_file)

# Run the app
main()



