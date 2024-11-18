import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image

KEY = st.secrets["roboFlow"]["api_key"] # API key from secrets

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key= KEY
)

MODEL_ID = "dog-and-cats/1" 

st.title("CLIP Crop Detection")
st.write("Upload an image to run inference with the RoboFlow model.")

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file) # Upload
    st.image(image, caption="Uploaded Image", use_container_width=True) # Display

    # Save the image temporarily
    temp_path = "temp_image.jpg"
    image.save(temp_path)

    st.write("Running inference...")
    result = CLIENT.infer(temp_path, model_id=MODEL_ID)


    st.write("Detection Results:")

    if "predictions" in result:
        simplified_results = [
            {
                "class": pred["class"],
                "confidence": pred["confidence"]
            }
            for pred in result["predictions"]
        ]
        st.json(simplified_results)
    else:
        st.write("No predictions found.")



    #st.json(result)
