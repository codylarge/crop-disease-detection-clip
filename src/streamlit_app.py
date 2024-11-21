import streamlit as st
from PIL import Image
import numpy as np
from clip_utils import load_clip_model, get_text_features, get_image_features, compute_similarity
from classes import get_candidate_captions

def main():
    st.title("CLIP Crop Detection")
    
    model, preprocess, device = load_clip_model()  # Load the model once, outside the cache
    candidate_captions = get_candidate_captions()

    # Upload image
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
        confidence = top_confidences[0]
        st.write(f"Predicted: {best_caption} with confidence: {confidence:.1f}%")


if __name__ == "__main__":
    main()
