import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import json
from pathlib import Path

# Configure Streamlit
st.set_page_config(
    page_title="Irish Landmark Recognition",
    page_icon="Castle",  # Using text instead of emoji
    layout="wide",
    initial_sidebar_state="expanded"
)

# Model class
class IrishLandmarkClassifier(nn.Module):
    def __init__(self, num_classes=6, model_name='resnet18'):
        super(IrishLandmarkClassifier, self).__init__()

        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=False)
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

@st.cache_resource
def load_model_and_config():
    """Load model and configuration"""
    try:
        checkpoint = torch.load("models/irish_landmarks_resnet18.pth", map_location="cpu")

        model = IrishLandmarkClassifier()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        return model, checkpoint.get('class_names', [])
    except:
        return None, ['Cliffs of Moher', "Giant's Causeway", 'Ring of Kerry', 
                     'Dublin Castle', 'Killarney National Park', 'Rock of Cashel']

def preprocess_image(image):
    """Preprocess image for prediction"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return transform(image).unsqueeze(0)

def predict_landmark(model, image_tensor):
    """Make prediction"""
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted = torch.max(probabilities, 0)

    return predicted.item(), confidence.item(), probabilities.numpy()

# Main app
def main():
    st.title(" Irish Landmark Recognition")
    st.markdown("Upload a photo of an Irish landmark and let AI identify it!")

    model, class_names = load_model_and_config()

    uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if model is not None:
            image_tensor = preprocess_image(image)
            predicted_idx, confidence, probabilities = predict_landmark(model, image_tensor)

            predicted_landmark = class_names[predicted_idx]

            st.success(f"Predicted: {predicted_landmark}")
            st.info(f"Confidence: {confidence:.1%}")

            # Confidence chart
            confidence_df = pd.DataFrame({
                'Landmark': class_names,
                'Confidence': probabilities * 100
            })

            fig = px.bar(confidence_df, x='Landmark', y='Confidence', 
                        title='Confidence Scores')
            st.plotly_chart(fig)

if __name__ == "__main__":
    main()
