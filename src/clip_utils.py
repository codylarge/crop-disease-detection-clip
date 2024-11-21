import streamlit as st
import clip
import torch

# Cache to only load model once
@st.cache_resource
def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

# Return vector embeddings for given text (caption)
def get_text_features(captions, device, model):
    text_tokens = clip.tokenize(captions).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
    return text_features

# Return vector embeddings for given image
def get_image_features(image, device, model, preprocess):
    image_input = preprocess(image).unsqueeze(0).to(device)  # Preprocess image and move to device
    with torch.no_grad():
        image_features = model.encode_image(image_input)  # Get image features
    return image_features

# Compute Cosine Similarity of Image & Text features
def compute_similarity(image_features, text_features):
    # Normalize the features to unit vectors
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    similarity = (image_features @ text_features.T).squeeze(0) # Cosine Similarity (Dot Product)
    return similarity