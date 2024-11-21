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