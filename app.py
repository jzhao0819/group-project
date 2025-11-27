# Create app.py that matches your other model's output exactly
from google.colab import files

matching_app_content = '''import streamlit as st
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
    
    .error-box {
        background: linear-gradient(135deg, #F8D7DA, #F5C6CB);
        border: 2px solid #DC3545;
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

def debug_model_predictions(image, model, class_names):
    img = image.resize((64, 64))
    img_array = np.array(img) / 255.0
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)[0]
    top_5_indices = np.argsort(prediction)[-5:][::-1]

    st.write("🔍 **Debug - Top 5 Predictions:**")
    for rank, idx in enumerate(top_5_indices, start=1):
        st.write(f"{rank}. {class_names[idx]} - {prediction[idx]:.3f} ({prediction[idx]*100:.1f}%)")

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
    img_size = (64, 64)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📸 Upload Plant Image")
        
        uploaded_file = st.file_uploader(
            "Drag and drop your file here or click to browse",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG • Max 200MB",
            label_visibility="collapsed"
        )

        if uploaded_file is None:
            st.markdown("""
            <div class="upload-area">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🌿</div>
                <h3 style="color: #2E8B57; margin-bottom: 0.5rem;">Drag & Drop Your Plant Leaf Here</h3>
                <p style="color: #666; margin-bottom: 0.5rem;">or click the area above to browse files</p>
                <p style="color: #888; font-size: 0.9rem; margin: 0;">JPG, PNG, JPEG • Max 200MB</p>
            </div>
            """, unsafe_allow_html=True)

        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)

                st.success("✅ **File uploaded successfully!**")
                st.image(image, caption="📷 Your Plant Leaf", width=400)

                if st.button("🔍 Analyze Plant Health", type="primary", use_container_width=True):
                    with st.spinner("🔬 Analyzing your plant..."):
                        disease, confidence, error = predict_image(image, model, class_names)

                    if error:
                        st.error(f"Analysis failed: {error}")
                    else:
                        # Track for bias detection
                        st.session_state.prediction_history.append(disease)
                        if len(st.session_state.prediction_history) > 5:
                            st.session_state.prediction_history.pop(0)

                        # ----------------- DIAGNOSIS CARD ----------------- #
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

                        # ----------------- CONFIDENCE WARNINGS ------------- #
                        if confidence < 0.4:
                            st.markdown("""
                            <div class="warning-box">
                                <h4>⚠️ Low Confidence Warning</h4>
                                <p>The model is not very confident about this diagnosis. This may be due to:</p>
                                <ul>
                                    <li>Poor image quality</li>
                                    <li>Unusual angle or lighting</li>
                                    <li>Plant type underrepresented in training data</li>
                                    <li>Multiple diseases present</li>
                                </ul>
                                <p><strong>Recommendation:</strong> Try a clearer, well-lit image focusing on the leaf.</p>
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
                        else:
                            st.success("**✅ High Confidence** – Diagnosis is likely reliable.")

                        # ----------------- DEBUG EXPANDER ------------------ #
                        with st.expander("🔍 Debug Information"):
                            st.write(f"Predicted class: {disease}")
                            st.write(f"Raw confidence: {confidence}")
                            st.write(f"Model input size: {img_size}")
                            st.write(f"Available classes: {len(class_names)}")
                            st.write(f"Recent predictions: {st.session_state.prediction_history}")
                            debug_model_predictions(image, model, class_names)

                        # ----------------- USER FEEDBACK ------------------- #
                        st.markdown("---")
                        st.subheader("🤔 Prediction Accuracy")
                        feedback = st.radio(
                            "Does this prediction seem correct?",
                            ["Yes, looks accurate", "No, this seems wrong", "Unsure"],
                            index=0
                        )
                        if feedback == "No, this seems wrong":
                            st.warning(
                                "Thank you for your feedback! In a future version, "
                                "we could store this to improve the model."
                            )

                        # ----------------- CARE INSTRUCTIONS --------------- #
                        st.markdown("---")
                        st.subheader("💡 Care Instructions")

                        plant_name = disease.split("_")[0] if "_" in disease else "plant"
                        
                        st.warning("⚠️ Using fallback care advice (AI service issue).")
                        display_fallback_advice(plant_name, disease)

            except Exception as e:
                st.error(f"❌ Error processing image: {e}")

    with col2:
        st.subheader("System Status")
        st.markdown("""
        <div style="background: white; border-radius: 12px; padding: 1.2rem; margin: 0.8rem 0; 
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1); border-left: 4px solid #3CB371;">
            <h4 style="color: #2E8B57; margin-bottom: 0.3rem;">✅ Model Active</h4>
            <p style="color: #666; margin: 0; font-size: 0.9rem;">Plant diagnosis ready</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #2E8B57, #228B22);
                    color: white; border-radius: 10px; padding: 1.2rem;
                    text-align: center; margin: 0.8rem 0;">
            <h3 style="margin: 0; font-size: 1.8rem;">{len(class_names)}</h3>
            <p style="margin: 0; opacity: 0.9; font-size: 0.9rem;">Plant Types Supported</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":

    main()
'''

# Create and download the matching app.py
with open('app.py', 'w') as f:
    f.write(matching_app_content)

print("✅ Matching app.py created!")
print("📥 Downloading...")
files.download('app.py')
