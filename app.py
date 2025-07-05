import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load model
model = load_model("fruit_classifier_mobilenetv2.h5")

# Define class labels (same order as during training)
class_labels = [
    'Apple Golden 1', 'Apple Red 1', 'Apple Red Delicious 1',
    'Banana 1', 'Banana Red 1',
    'Orange 1', 'Strawberry 1', 'Pineapple 1', 'Mango 1', 'Mango Red 1',
    'Grape Blue 1', 'Grape White 1',
    'Pomegranate 1', 'Papaya 1', 'Watermelon 1', 'Guava 1',
    'Pear 1', 'Pear Red 1', 'Peach 1', 'Plum 1', 'Kiwi 1',
    'Lemon 1', 'Lychee 1', 'Blueberry 1', 'Cherry 1', 'Cherry Rainier 1',
    'Dates 1', 'Fig 1', 'Cocos 1', 'Blackberrie 1', 'Raspberry 1',
    'Mulberry 1', 'Gooseberry 1', 'Clementine 1', 'Mandarine 1',
    'Grapefruit Pink 1', 'Pomelo Sweetie 1', 'Apricot 1'
]

st.title("🍎 Fruit Classifier")
st.markdown("Upload a fruit image and I will classify it!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    pred = model.predict(img_array)
    class_index = np.argmax(pred)
    predicted_label = class_labels[class_index]
    confidence = np.max(pred) * 100

    st.success(f"Predicted: **{predicted_label}** ({confidence:.2f}%)")
