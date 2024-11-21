classes = [
    "corn", "wheat", "rice", "soybeans", "barley", "oats", "cotton", 
    "potato", "tomato", "sunflower", "canola", "guava", 
    "orange", "apple", "banana", "grapes", "pineapple", "papaya", 
    "peach", "watermelon", "strawberry", "blueberry", "kiwi", "melon", 
    "cucumber", "lettuce", "carrot", "spinach", "chili pepper", "pumpkin"
]

disease_classes = [
    "blight", "rust", "powdery mildew", "downy mildew", "aphids", "fusarium wilt", 
    "leaf spot", "bacterial wilt", "early blight", "late blight", "aphid infestation", 
    "damping off", "verticillium wilt", "corn smut", "tomato mosaic virus", 
    "black rot", "clubroot", "stem canker", "flooding damage", "gray mold"
]

def get_candidate_captions():
    candidate_captions = [f"A picture of {cls}" for cls in classes]
    candidate_captions.extend([f"Disease {cls}" for cls in disease_classes])
    return candidate_captions