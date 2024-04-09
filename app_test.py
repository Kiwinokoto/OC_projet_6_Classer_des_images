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


# Define a function to load the data
@st.cache_data
def load_data():
    # Load your data here
    data = pd.read_csv('./pickle/leftover_data.csv', sep=',')
    return data

# Load the data
data = load_data()

nb_dogs = data.shape[0]

# st.write("### Click the button to pick a dog randomly!")
# st.write(f"### {nb_dogs} different images possible")

# x = st.text_input("Movie", "Star Wars")

if st.button("Click Me"):
    random_index = random.randint(0, nb_dogs)
    st.write(f"You picked dog nb `{random_index}` in the dataset")

    dog_image_path = data['photo_path'][random_index]
    breed = data['breed'][random_index]

    data["target"] = label_encoder.transform(data["breed"])
    # target = data['target'][random_index]
    st.image(dog_image_path, caption=f"{breed}")

    image = load_img(dog_image_path, target_size=target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    image = np.asarray(image)

    dico = {}
    for index, classe in enumerate(list(label_encoder.classes_)):
        # st.write(f"{index}, {classe}")
        dico[str(index)] = classe

    # prediction = model.predict(image)
    prediction = model.predict(image)
    prediction_dict = {str(i): float(prediction[0][i]) for i in range(0, 120)}

    # Sort the dictionary based on predicted probabilities in descending order
    sorted_dict = dict(sorted(prediction_dict.items(), key=lambda item: item[1], reverse=True))

    for breed, proba in list(sorted_dict.items())[:2]:
        st.write(f"Model prediction: {dico[str(breed)]} ({int(proba*100)}%)")
