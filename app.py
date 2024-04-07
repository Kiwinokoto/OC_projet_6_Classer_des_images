# ui
import streamlit as st

# paths, folders/files
import os, sys, random, re

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
    pickle_path_model = './pickle/pretrained/120_classes/NASNetLarge_15_lay_11.0_pc_model.pkl'
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


# Define a function to load the data
@st.cache_data
def load_data():
    # Load your data here
    data = pd.read_csv('./pickle/leftover_data.csv', sep=',')
    return data

# Load the data
data = load_data()

nb_dogs = data.shape[0]

st.write("### Click the button to pick a dog randomly!")
st.write(f"### {nb_dogs} different images possible")

# x = st.text_input("Movie", "Star Wars")

if st.button("Click Me"):
    random_index = random.randint(0, nb_dogs)
    st.write(f"You picked dog nb `{random_index}` in the dataset")

    dog_image_path = data['photo_path'][random_index]
    breed = data['breed'][random_index]
    st.image(dog_image_path, caption=f"{breed}")

    image = load_img(dog_image_path, target_size=target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    image = np.asarray(image)


    # prediction = model.predict(image)
    prediction_labels = model.predict(image)
    predicted_label = np.argmax(prediction_labels)
    # load and cache label_encoder first
    predicted_breed = label_encoder.inverse_transform(predicted_label)


    # Display the predicted label
    st.write(f"Predicted breed: {predicted_breed}")

    # st.write(f"Model predicts : `{prediction}`")