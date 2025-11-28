import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# ----------------------- PAGE CONFIG ----------------------- #
st.set_page_config(
    page_title="Plant Doctor 🌿",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------- SESSION STATE --------------------- #
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# ----------------------- STYLING --------------------------- #
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    
    .upload-area {
        border: 3px dashed #3CB371;
        border-radius: 15px;
        padding: 3rem 1.5rem;
        text-align: center;
        background: #F0FFF0;
        margin: 1.5rem 0;
        transition: all 0.3s ease;
    }
    
    .diagnosis-card {
        background: linear-gradient(135deg, #ffffff, #f8fff8);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1.2rem 0;
        box-shadow: 0 4px 12px rgba(46, 139, 87, 0.15);
        border: 1px solid #e0f0e0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #FFF3CD, #FFEAA7);
        border: 2px solid #FFA500;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------- LOAD RESOURCES -------------------- #
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('plant_dieases_CNN_f.keras')

@st.cache_data
def load_class_names():
    with open('class_names.json', 'r') as f:
        return json.load(f)

# ----------------------- PREDICTION FUNCTION --------------- #
def predict_image(image, model, class_names):
    try:
        img = image.resize((64, 64))
        img_array = np.array(img) / 255.0
        
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
            
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = model.predict(img_array, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        predicted_class = class_names[predicted_idx]
        
        return predicted_class, confidence, None
        
    except Exception as e:
        return None, None, str(e)

def display_fallback_advice(plant_name, disease):
    formatted_disease = disease.replace("_", " ").title()
    st.info(f"""
    **🌱 Recommended Treatment for {formatted_disease}**
    
    ### 🚨 Immediate Actions
    - Remove affected leaves immediately to prevent spread
    - Isolate plant if possible to protect others
    - Clean tools thoroughly after handling infected plants
    
    ### 💧 Watering & Environment
    - Water at the base only, avoid wetting leaves
    - Improve air circulation around the plant
    - Ensure proper drainage to prevent root issues
    
    ### 🛡️ Treatment Plan
    - Apply appropriate organic or chemical treatment
    - Monitor plant recovery daily
    - Adjust sunlight exposure as needed
    """)

# ----------------------- MAIN APP -------------------------- #
def main():
    st.markdown('<h1 class="main-header">🌿 Plant Doctor</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align: center; color: #666; margin-bottom: 2rem;">'
        'Upload a plant leaf photo for instant AI-powered diagnosis and care advice.'
        '</p>',
        unsafe_allow_html=True
    )

    model = load_model()
    class_names = load_class_names()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📸 Upload Plant Image")
        
        uploaded_file = st.file_uploader(
            "Choose plant leaf image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Your Plant Leaf", width=400)

                if st.button("🔍 Analyze Plant Health", type="primary"):
                    with st.spinner("Analyzing your plant..."):
                        disease, confidence, error = predict_image(image, model, class_names)

                    if error:
                        st.error(f"Analysis failed: {error}")
                    else:
                        st.session_state.prediction_history.append(disease)
                        if len(st.session_state.prediction_history) > 5:
                            st.session_state.prediction_history.pop(0)

                        # Diagnosis Results
                        st.subheader("📋 Diagnosis Results")
                        formatted_disease = disease.replace("_", " ").title()

                        if "healthy" in disease.lower():
                            status_emoji = "✅"
                            status_text = "Healthy Plant"
                            status_color = "#2E8B57"
                        else:
                            status_emoji = "⚠️"
                            status_text = "Needs Attention"
                            status_color = "#FFA500"

                        st.markdown(f"""
                        <div class="diagnosis-card">
                            <div style="text-align: center; margin-bottom: 1.2rem;">
                                <div style="font-size: 2.5rem; margin-bottom: 0.8rem;">{status_emoji}</div>
                                <span style="background: {status_color}; color: white; padding: 0.4rem 0.8rem;
                                             border-radius: 15px; font-weight: 600;">
                                    {status_text}
                                </span>
                            </div>
                            <h3 style="color: {status_color}; text-align: center; margin-bottom: 0.8rem;">
                                {formatted_disease}
                            </h3>
                            <div style="text-align: center;">
                                <p style="font-size: 1.1rem; color: #666; margin-bottom: 0.5rem;">Confidence Level</p>
                                <h2 style="color: {status_color}; font-size: 2rem; margin: 0.3rem 0;">
                                    {confidence:.1%}
                                </h2>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Confidence Warnings
                        if confidence < 0.4:
                            st.markdown("""
                            <div class="warning-box">
                                <h4>⚠️ Low Confidence Warning</h4>
                                <p>The model is not very confident about this diagnosis.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        elif confidence < 0.75:
                            st.markdown("""
                            <div class="warning-box">
                                <h4>⚠️ Moderate Confidence</h4>
                                <p>This prediction has moderate confidence. You may want to:</p>
                                <ul>
                                    <li>Get a second opinion from a plant expert</li>
                                    <li>Upload additional images from different angles</li>
                                    <li>Monitor the plant for new symptoms</li>
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)

                        # User Feedback
                        st.subheader("🤔 Prediction Accuracy")
                        feedback = st.radio(
                            "Does this prediction seem correct?",
                            ["Yes, looks accurate", "No, this seems wrong", "Unsure"],
                            index=0
                        )

                        # Care Instructions
                        st.subheader("💡 Care Instructions")
                        plant_name = disease.split("_")[0] if "_" in disease else "plant"
                        st.warning("⚠️ Using fallback care advice (AI service issue).")
                        display_fallback_advice(plant_name, disease)

            except Exception as e:
                st.error(f"Error processing image: {e}")

    with col2:
        st.subheader("System Status")
        st.success("✅ Model Active - Plant diagnosis ready")
        st.info(f"**Plant Types Supported:** {len(class_names)}")

if __name__ == "__main__":
    main()
