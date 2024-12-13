classes = [
    "corn", "wheat", "rice", "cotton", "watermelon", "tomato", "carrot", "pineapple", "peach",  "sunflower"
]

disease_classes = [
    "blight", "rust", "powdery mildew", "downy mildew", "aphids", "leaf spot",
]

def get_candidate_captions():
    candidate_captions = [f"{cls}" for cls in classes]
    candidate_captions.extend([f"Disease {cls}" for cls in disease_classes])
    return candidate_captions