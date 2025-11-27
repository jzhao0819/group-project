import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Page configuration
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿", 
    layout="centered"
)

st.title("🌿 Plant Disease Detector")
st.write("Upload a plant leaf image for AI-powered disease detection")

# Load model and class names
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('plant_dieases_CNN_f.keras')

@st.cache_data
def load_class_names():
    with open('class_names.json', 'r') as f:
        return json.load(f)

# Prediction function - customized for 64x64 input and 38 classes
def predict_disease(image):
    model = load_model()
    class_names = load_class_names()
    
    # Preprocess image for YOUR model (64x64 pixels)
    img = image.resize((64, 64))
    img_array = np.array(img) / 255.0
    
    # Handle different image formats
    if len(img_array.shape) == 2:  # Grayscale to RGB
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:  # RGBA to RGB
        img_array = img_array[:, :, :3]
    
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction using YOUR model
    predictions = model.predict(img_array, verbose=0)[0]
    predicted_idx = np.argmax(predictions)
    confidence = np.max(predictions)
    predicted_class = class_names[predicted_idx]
    
    return predicted_class, confidence, predictions

# File uploader
uploaded_file = st.file_uploader(
    "Choose a plant leaf image (JPG, PNG, JPEG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, use_column_width=True)
    
    # Analyze button
    if st.button("🔍 Analyze Disease", type="primary"):
        with st.spinner("AI is analyzing..."):
            disease, confidence, all_predictions = predict_disease(image)
        
        # Display results
        formatted_disease = disease.replace("___", " ").replace("_", " ").title()
        
        if "healthy" in disease.lower():
            st.success(f"✅ **Healthy Plant**")
            st.balloons()
        else:
            st.warning(f"⚠️ **Disease Detected**")
        
        st.markdown(f"**Diagnosis:** {formatted_disease}")
        st.markdown(f"**Confidence:** {confidence:.1%}")
        
        # Show Top 3 predictions
        class_names = load_class_names()
        top_3_idx = np.argsort(all_predictions)[-3:][::-1]
        
        st.markdown("**Top Predictions:**")
        for i, idx in enumerate(top_3_idx):
            cls_name = class_names[idx].replace("___", " ").replace("_", " ").title()
            conf = all_predictions[idx]
            st.write(f"{i+1}. {cls_name} ({conf:.1%})")

# Footer
st.markdown("---")
st.markdown("*AI Plant Disease Detection System*")
