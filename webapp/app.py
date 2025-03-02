import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from PIL import Image
import cv2
import base64

# Page Configuration 
st.set_page_config(page_title="Brain Tumor Classification", layout="wide")

# Load Model
model = load_model("models/trained.h5")

# Class Labels
classes = ['no_tumor', 'pituitary_tumor', 'meningioma_tumor', 'glioma_tumor']

def preprocess_image(image):
    """ Convert image to RGB, resize, normalize, and add batch dimension """
    image = image.convert("RGB")
    image = np.array(image)  
    image = cv2.resize(image, (255, 255)) 
    image = image / 255.0 
    image = np.expand_dims(image, axis=0) 
    return image

def generate_gradcam(image, model, layer_name="block5_conv3"):
    """ Generate Grad-CAM Visualization """

    img_array = np.array(image.convert("RGB")) / 255.0  
    img_array = cv2.resize(img_array, (255, 255)) 
    img_array = np.expand_dims(img_array, axis=0) 

    img_array = np.array(img_array, dtype=np.float32)

    grad_model = tf.keras.models.Model(
        inputs=model.input, 
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False) 
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)  

    heatmap = cv2.resize(heatmap, (255, 255))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return Image.fromarray(heatmap)


def download_report(pred_class, confidence):
    """Generate and return a downloadable report"""
    report_text = f""" 
    Brain Tumor Classification Report
    -------------------------------
    Prediction : {pred_class}
    Confidence : {confidence:.2f}%
    """
    b64 = base64.b64encode(report_text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="report.txt">Download Report</a>'
    return href

# Sidebar for Image Upload
st.sidebar.title("Upload MRI Scan")
uploaded_file = st.sidebar.file_uploader("Choose an MRI image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Scan", use_column_width=True)

    if st.button("Classify Image"):
        input_tensor = preprocess_image(image)
        output = model.predict(input_tensor)
        pred_idx = np.argmax(output, axis=1)[0]
        confidence = output[0][pred_idx] * 100
        pred_class = classes[pred_idx]

        # Display Result
        col1, col2 = st.columns([2, 1])
        with col1:
            st.success(f"Prediction: {pred_class}")
            st.info(f"Confidence: {confidence:.2f}%")
        with col2:
            gradcam_image = generate_gradcam(image, model)
            st.image(gradcam_image, caption="Grad-CAM Visualization", use_column_width=True)
        
        # Download Report
        st.markdown(download_report(pred_class, confidence), unsafe_allow_html=True)
