import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import streamlit as st
from keras.models import load_model
from pathlib import Path

# Constants
IMAGE_SIZE = (224, 224)
ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png"]
MODEL_BASE_PATH = Path("D:/Corrosion detection/Corrosion Detection/models")

# Class mappings
OBJECT_CLASSES = {
    0: "Flanges",
    1: "Pipeline"
}

CORROSION_CLASSES_1st = {
    0: "No Corrosion Pipeline",
    1: "Pitting",
    2: "Uniform Corrosion"
}

CORROSION_CLASSES_2nd = {
    0: "Crevice Corrosion",
    1: "No Corrosion Flanges"
}

@st.cache_resource
def load_models():
    """Load models with caching to prevent reloading on every run"""
    try:
        models = {
            'object': load_model(MODEL_BASE_PATH / 'Object/model_checkpoint.h5'),
            'first': load_model(MODEL_BASE_PATH / '1st/model_1st_checkpoint.h5'),
            'second': load_model(MODEL_BASE_PATH / '2nd/model_2nd_checkpoint.h5')
        }
        return models, None
    except Exception as e:
        return None, f"Error loading models: {str(e)}"

def preprocess_image(image):
    """Preprocess image for model input"""
    image_resized = cv2.resize(image, IMAGE_SIZE)
    image_normalized = image_resized / 255.0
    return np.expand_dims(image_normalized, axis=0)

def get_severity_color(confidence, is_corrosion):
    """Return color based on confidence and whether corrosion is detected"""
    if not is_corrosion:  # No corrosion detected
        return "green"
    if confidence > 90:
        return "red"
    elif confidence > 80:
        return "orange"
    return "yellow"

def main():
    st.set_page_config(
        page_title="Corrosion Detection System",
        page_icon="🔍",
        layout="wide"
    )

    st.title("🔍 Corrosion Detection For Oil & Gas Grids")
    st.write("""
    This application uses advanced computer vision and deep learning techniques to detect 
    and classify corrosion in oil and gas infrastructure. Upload an image to begin analysis.
    """)

    # Load models
    models, error = load_models()
    if error:
        st.error(error)
        return

    # File uploader with progress bar
    with st.container():
        uploaded_file = st.file_uploader(
            "Upload an image for inspection",
            type=ALLOWED_EXTENSIONS,
            help="Supported formats: JPG, JPEG, PNG"
        )

    if uploaded_file:
        try:
            # Display uploaded image
            col1, col2 = st.columns(2)
            with col1:
                st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

            # Process image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            image_input = preprocess_image(image)

            with st.spinner("Analyzing image..."):
                # Object detection
                object_prediction = models['object'].predict(image_input, verbose=0)
                predicted_object = np.argmax(object_prediction)
                object_confidence = np.max(object_prediction) * 100

                # Corrosion detection based on object type
                if predicted_object == 1:  # Pipeline
                    model = models['first']
                    corrosion_classes = CORROSION_CLASSES_1st
                else:  # Flanges
                    model = models['second']
                    corrosion_classes = CORROSION_CLASSES_2nd

                corrosion_prediction = model.predict(image_input, verbose=0)
                corrosion_class = np.argmax(corrosion_prediction[0])
                corrosion_confidence = np.max(corrosion_prediction[0]) * 100

            # Display results
            with col2:
                st.subheader("Analysis Results")
                st.write("---")
                
                # Object detection results
                st.write("**Object Detection**")
                st.write(f"Detected Object: {OBJECT_CLASSES[predicted_object]}")
                st.progress(object_confidence / 100)
                st.write(f"Confidence: {object_confidence:.1f}%")
                
                st.write("---")
                
                # Corrosion detection results
                st.write("**Corrosion Analysis**")
                corrosion_result = corrosion_classes[corrosion_class]
                is_corrosion = not ("No Corrosion" in corrosion_result)
                severity_color = get_severity_color(corrosion_confidence, is_corrosion)
                
                st.markdown(f"""
                <div style='padding: 10px; border-radius: 5px; background-color: {severity_color}; color: white;'>
                    Classification: {corrosion_result}
                </div>
                """, unsafe_allow_html=True)
                
                st.progress(corrosion_confidence / 100)
                st.write(f"Confidence: {corrosion_confidence:.1f}%")

                # Warning messages based on confidence and corrosion type
                if corrosion_confidence < 70:
                    st.warning("⚠️ Low confidence detection. Consider getting expert verification.")
                
                if is_corrosion and corrosion_confidence > 80:
                    st.error("❗ Significant corrosion detected. Immediate inspection recommended.")

        except Exception as e:
            st.error(f"An error occurred during processing: {str(e)}")
            st.write("Please try uploading a different image or contact support if the issue persists.")

    # Add information about the system
    with st.expander("ℹ️ About this system"):
        st.write("""
        This corrosion detection system uses deep learning models trained on hundreds of images
        of gas infrastructure. The system:
        1. First identifies whether the component is a Pipeline or Flanges
        2. Then applies a specialized corrosion detection model based on the component type:
           - For Pipelines: Detects Uniform Corrosion, No Corrosion, or Pitting
           - For Flanges: Detects Crevice Corrosion or No Corrosion
        3. Provides confidence scores to help inform decision making
        
        For best results, ensure:
        - Images are well-lit and in focus
        - The component of interest is clearly visible
        - Images are taken from a consistent angle
        """)

if __name__ == "__main__":
    main()
