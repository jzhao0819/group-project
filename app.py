# streamlit_app.py - UPDATED VERSION USING LOCAL KERAS MODEL
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import chatbot_helper

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
    
    .status-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #3CB371;
    }
    
    .diagnosis-card {
        background: linear-gradient(135deg, #ffffff, #f8fff8);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1.2rem 0;
        box-shadow: 0 4px 12px rgba(46, 139, 87, 0.15);
        border: 1px solid #e0f0e0;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #2E8B57, #228B22);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.7rem 1.5rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
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

# ----------------------- HELPERS --------------------------- #
def check_openai_setup():
    """Check if OpenAI advice helper is configured."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False
    try:
        test_advice = chatbot_helper.generate_advice("tomato", "healthy")
        # If we get a normal sentence back, assume OK
        if "OpenAI" not in test_advice and "API key" not in test_advice:
            return True
        return False
    except Exception:
        return False

# Updated paths - looking in current directory
MODEL_PATH = "plant_dieases_CNN_f.keras"
CLASS_NAMES_PATH = "class_names.json"

@st.cache_resource
def load_model():
    """Load the trained Keras model from local file."""
    if not os.path.exists(MODEL_PATH):
        st.sidebar.error("❌ Model file not found in current directory.")
        st.sidebar.write(f"Looking for: {os.path.abspath(MODEL_PATH)}")
        st.sidebar.write("Please make sure the model file is in the same directory as this script.")
        return None

    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        st.sidebar.success("✅ CNN model loaded (64x64 input)")
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {e}")
        return None

@st.cache_data
def load_class_names():
    """Load class names from json file."""
    try:
        with open(CLASS_NAMES_PATH, "r") as f:
            class_names = json.load(f)
        st.sidebar.info(f"✅ Loaded {len(class_names)} plant classes")
        return class_names
    except Exception as e:
        st.sidebar.warning(f"Could not load {CLASS_NAMES_PATH}, using default placeholder labels.")
        return [
            "Apple_healthy",
            "Apple_apple_scab",
            "Tomato_healthy", 
            "Tomato_early_blight",
            "Tomato_late_blight"
        ]

def predict_image(image, model, class_names, img_size):
    """Run model prediction on a PIL.Image and return (class, confidence, error_msg)."""
    try:
        img = image.resize(img_size)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array, verbose=0)[0]
        predicted_index = int(np.argmax(prediction))
        predicted_class = class_names[predicted_index]
        confidence = float(np.max(prediction))

        return predicted_class, confidence, None
    except Exception as e:
        return None, None, str(e)

def debug_model_predictions(image, model, class_names, img_size):
    """Show top-5 predictions for debugging."""
    img = image.resize(img_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)[0]
    top_5_indices = np.argsort(prediction)[-5:][::-1]

    st.write("🔍 **Debug - Top 5 Predictions:**")
    for rank, idx in enumerate(top_5_indices, start=1):
        st.write(f"{rank}. {class_names[idx]} - {prediction[idx]:.3f} ({prediction[idx]*100:.1f}%)")

def get_plant_advice(plant_name, disease):
    """Try to get advice from chatbot_helper, fall back if error."""
    try:
        return chatbot_helper.generate_advice(plant_name, disease)
    except Exception as e:
        if "rate_limit" in str(e).lower() or "429" in str(e):
            return "AI service rate limit reached. Using standard care advice instead."
        return "AI advice currently unavailable. Using standard care advice instead."

def display_fallback_advice(plant_name, disease):
    """Static care guide if AI advice is unavailable."""
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

# ----------------------- LOAD RESOURCES -------------------- #
model = load_model()
class_names = load_class_names()
openai_ready = check_openai_setup()

# Automatically infer image size from the model if possible
if model is not None and hasattr(model, "input_shape") and len(model.input_shape) == 4:
    img_size = (model.input_shape[1], model.input_shape[2])
else:
    img_size = (64, 64)  # fallback

# ----------------------- SIDEBAR DEBUG --------------------- #
with st.sidebar:
    st.header("🔧 Debug Info")
    st.write(f"Model loaded: {model is not None}")
    st.write(f"Number of classes: {len(class_names)}")
    st.write(f"OpenAI ready: {openai_ready}")
    if class_names:
        st.write("Sample classes:", class_names[:5])
    st.write("Model file path:", os.path.abspath(MODEL_PATH))
    st.write("Class names path:", os.path.abspath(CLASS_NAMES_PATH))
    
    # Show current directory contents for debugging
    st.write("Current directory files:")
    current_files = [f for f in os.listdir('.') if os.path.isfile(f)]
    st.write(current_files[:10])  # Show first 10 files

# ----------------------- MAIN HEADER ----------------------- #
st.markdown('<h1 class="main-header">🌿 Plant Doctor</h1>', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align: center; color: #666; margin-bottom: 2rem;">'
    'Upload a plant leaf photo for instant AI-powered diagnosis and care advice.'
    '</p>',
    unsafe_allow_html=True
)

# If model is missing, show error and stop
if model is None:
    st.error(f"""
    ## 🔧 Service Temporarily Unavailable
    
    The model file **{MODEL_PATH}** is not available in the current directory.
    
    Please make sure:
    - The file exists in the same directory as `streamlit_app.py`
    - The filename is exactly: `{MODEL_PATH}`
    - The file matches the `{CLASS_NAMES_PATH}` label order
    
    **Current directory:** {os.getcwd()}
    **Looking for:** {os.path.abspath(MODEL_PATH)}
    """)
    st.stop()

# ----------------------- LAYOUT ---------------------------- #
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📸 Upload Plant Image")
    st.write("**Choose a plant leaf image**")

    uploaded_file = st.file_uploader(
        "Drag and drop your file here or click to browse",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG • Max 200MB",
        label_visibility="collapsed"
    )

    # Nice empty state
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
            st.write(f"**Filename:** {uploaded_file.name}")

            # Preview
            st.image(image, caption="📷 Your Plant Leaf", width=400)

            # File info
            file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
            st.write(
                f"**Image Details:** {image.size[0]} × {image.size[1]} pixels • {file_size_mb:.1f} MB"
            )

            # Analyze button
            if st.button("🔍 Analyze Plant Health", type="primary", use_container_width=True):
                with st.spinner("🔬 Analyzing your plant..."):
                    disease, confidence, error = predict_image(
                        image, model, class_names, img_size
                    )

                if error:
                    st.error(f"""
                    ## ❌ Analysis Failed
                    **Error:** {error}

                    Please try a different image or check the model configuration.
                    """)
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

                    # ----------------- CORN BIAS CHECK ----------------- #
                    corn_count = sum(
                        1 for p in st.session_state.prediction_history
                        if "corn" in p.lower()
                    )
                    if corn_count >= 3:
                        st.markdown("""
                        <div class="error-box">
                            <h4>🚨 Potential Model Bias Detected</h4>
                            <p>The model has predicted <strong>corn-related</strong> classes several times in a row.</p>
                            <p>This may indicate:</p>
                            <ul>
                                <li>Training data imbalance</li>
                                <li>Limited performance on non-corn plants</li>
                                <li>Need for future retraining or fine-tuning</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)

                    # ----------------- DEBUG EXPANDER ------------------ #
                    with st.expander("🔍 Debug Information"):
                        st.write(f"Predicted class: {disease}")
                        st.write(f"Raw confidence: {confidence}")
                        st.write(f"Model input size: {img_size}")
                        st.write(f"Available classes: {len(class_names)}")
                        st.write(f"Recent predictions: {st.session_state.prediction_history}")
                        debug_model_predictions(image, model, class_names, img_size)

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

                    if openai_ready:
                        with st.spinner("🤖 Generating personalized care advice..."):
                            advice = get_plant_advice(plant_name, disease)

                        if any(key in advice for key in ["OpenAI", "API key", "rate limit"]):
                            st.warning("⚠️ Using fallback care advice (AI service issue).")
                            display_fallback_advice(plant_name, disease)
                        else:
                            st.success("✅ AI-Generated Personalized Advice")
                            st.info(advice)
                    else:
                        st.warning("⚠️ Using standard care advice (AI not configured).")
                        display_fallback_advice(plant_name, disease)

        except Exception as e:
            st.error(f"❌ Error processing image: {e}")

with col2:
    # Sidebar / status info
    st.subheader("System Status")
    st.write("Real-time service monitoring")

    st.markdown("""
    <div class="status-card">
        <h4 style="color: #2E8B57; margin-bottom: 0.3rem;">✅ Model Active</h4>
        <p style="color: #666; margin: 0; font-size: 0.9rem;">Plant diagnosis ready</p>
    </div>
    """, unsafe_allow_html=True)

    if openai_ready:
        st.markdown("""
        <div class="status-card">
            <h4 style="color: #2E8B57; margin-bottom: 0.3rem;">✅ AI Advice Ready</h4>
            <p style="color: #666; margin: 0; font-size: 0.9rem;">Personalized care tips enabled</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-card">
            <h4 style="color: #FFA500; margin-bottom: 0.3rem;">⚠️ Basic Advice Mode</h4>
            <p style="color: #666; margin: 0; font-size: 0.9rem;">Standard care tips only</p>
        </div>
        """, unsafe_allow_html=True)

    # Plant types metric
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #2E8B57, #228B22);
                color: white; border-radius: 10px; padding: 1.2rem;
                text-align: center; margin: 0.8rem 0;">
        <h3 style="margin: 0; font-size: 1.8rem;">{len(class_names)}</h3>
        <p style="margin: 0; opacity: 0.9; font-size: 0.9rem;">Plant Types Supported</p>
    </div>
    """, unsafe_allow_html=True)

    # Tips
    st.subheader("💡 Tips for Best Results")
    tips = [
        "Use clear, well-lit photos",
        "Focus on the affected leaves",
        "Include a plain, non-distracting background",
        "Take multiple angles if you're unsure",
        "Monitor your plant regularly for changes"
    ]
    for tip in tips:
        st.markdown(f"""
        <div style="background: white; border-radius: 10px; padding: 1rem;
                    margin: 0.6rem 0; box-shadow: 0 2px 6px rgba(0,0,0,0.08);
                    border-left: 3px solid #3CB371;">
            <p style="margin: 0; color: #555; font-size: 0.9rem;">• {tip}</p>
        </div>
        """, unsafe_allow_html=True)

# ----------------------- FOOTER ---------------------------- #
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1.5rem 0;">
    <p style="margin: 0; font-size: 0.9rem;">
        <strong>AI-powered plant health analysis</strong> • Keep your plants thriving 🌱
    </p>
</div>
""", unsafe_allow_html=True)
