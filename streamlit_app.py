import streamlit as st
from PIL import Image
import numpy as np

from src.clip_utils import load_clip_model, get_text_features, get_image_features, compute_similarity
from src.llama_utils import process_user_input, process_hidden_prompt, process_silent_instruction
from src.classes import get_candidate_captions

from groq import Groq

client = Groq()

def main():
    st.title("CLIP Crop & Disease Detection")

    # Initialize session state variables
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
            
    # Upload image
    model, preprocess, device = load_clip_model()  # Load the model once, outside the cache
    candidate_captions = get_candidate_captions()
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)  # Upload and show image
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Get features for captions and image then compute similarities
        text_features = get_text_features(candidate_captions, device, model)
        image_features = get_image_features(image, device, model, preprocess)
        similarities = compute_similarity(image_features, text_features)
        
        # Get the top 3 most similar captions (for debugging)
        top_indices = np.argsort(similarities.cpu().numpy())[::-1][:3] 
        top_captions = [candidate_captions[idx] for idx in top_indices]
        top_confidences = [similarities[idx].item() * 100 for idx in top_indices]  # Convert to percentage
        
        best_caption = top_captions[0]
        best_class = best_caption.split(":")[0]
        confidence = top_confidences[0]

        # Prompt LLM for description of image
        if len(st.session_state.chat_history) == 0:
            #print("Best caption: ", best_caption)
            prompt = (
                f"You have been provided a picture of a {best_caption}."
                f"You should say what it is, and be open to answering questions about it."
            )
            process_hidden_prompt(st, prompt)

    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        elif message["role"] == "assistant":
            st.chat_message("assistant").write(message["content"])

    # input field for users message:
    user_prompt = st.chat_input("Ask LLAMA...")
 
    if user_prompt:
        print("Text prompt")
        process_user_input(st, user_prompt) 

if __name__ == "__main__":
    main()  