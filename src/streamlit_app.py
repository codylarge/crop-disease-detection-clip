import streamlit as st
import clip
import torch
from PIL import Image
import numpy as np

# Cache to only load model once
@st.cache_resource
def load_clip_model(device):
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess

def get_text_features(captions, device, model):
    text_tokens = clip.tokenize(captions).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
    return text_features

def get_image_features(image, device, model, preprocess):
    image_input = preprocess(image).unsqueeze(0).to(device)  # Preprocess image and move to device
    with torch.no_grad():
        image_features = model.encode_image(image_input)  # Get image features
    return image_features

def compute_similarity(image_features, text_features):
    # Normalize the features to unit vectors
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    similarity = (image_features @ text_features.T).squeeze(0) # Cosine Similarity (Dot Product)
    return similarity

def main():
    st.title("CLIP Model with Streamlit")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load_clip_model(device)  # Load the model once, outside the cache

    class_names = ["corn", "apple", "soybeans", "tomato", "strawberry", "orange", "grapes", "watermelon", "banana", "peach"]
    candidate_captions = [f"A picture of {cls}" for cls in class_names]

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)  # Upload and show image
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        text_features = get_text_features(candidate_captions, device, model)

        image_features = get_image_features(image, device, model, preprocess)

        similarities = compute_similarity(image_features, text_features)
        
        # Find the best matching class based on cosine similarity
        best_match_idx = np.argmax(similarities.cpu().numpy())
        best_caption = candidate_captions[best_match_idx]
        confidence = similarities[best_match_idx].item() * 100 # Confidence as percentage
        st.write(f"Predicted: {best_caption} with confidence: {confidence:.1f}%")

if __name__ == "__main__":
    main()
