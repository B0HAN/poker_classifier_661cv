import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import timm
from torch.utils.data import Dataset
import numpy as np

# Define the model class (as you defined it in your training script)
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Corrected number of inputs to the fully connected layer
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)  # Correctly flattened
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Load the model
model = SimpleCNN(53)
model.load_state_dict(torch.load('playing_card_classifier_simpleCNN.pth'))
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

def predict(image):
    image = Image.open(image).convert('RGB')
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_probs, top_idxs = torch.topk(probabilities, 5)  # Get top 5 probabilities
    top_probs = top_probs.numpy().flatten()
    top_classes = [class_names[idx] for idx in top_idxs.numpy().flatten()]
    return top_probs, top_classes


target_to_class = {0: 'ace of clubs', 1: 'ace of diamonds', 2: 'ace of hearts', 3: 'ace of spades', 4: 'eight of clubs', 
                   5: 'eight of diamonds', 6: 'eight of hearts', 7: 'eight of spades', 8: 'five of clubs', 9: 'five of diamonds', 
                   10: 'five of hearts', 11: 'five of spades', 12: 'four of clubs', 13: 'four of diamonds', 14: 'four of hearts', 
                   15: 'four of spades', 16: 'jack of clubs', 17: 'jack of diamonds', 18: 'jack of hearts', 19: 'jack of spades', 
                   20: 'joker', 21: 'king of clubs', 22: 'king of diamonds', 23: 'king of hearts', 24: 'king of spades', 
                   25: 'nine of clubs', 26: 'nine of diamonds', 27: 'nine of hearts', 28: 'nine of spades', 
                   29: 'queen of clubs', 30: 'queen of diamonds', 31: 'queen of hearts', 32: 'queen of spades', 
                   33: 'seven of clubs', 34: 'seven of diamonds', 35: 'seven of hearts', 36: 'seven of spades', 
                   37: 'six of clubs', 38: 'six of diamonds', 39: 'six of hearts', 40: 'six of spades', 
                   41: 'ten of clubs', 42: 'ten of diamonds', 43: 'ten of hearts', 44: 'ten of spades', 
                   45: 'three of clubs', 46: 'three of diamonds', 47: 'three of hearts', 48: 'three of spades', 
                   49: 'two of clubs', 50: 'two of diamonds', 51: 'two of hearts', 52: 'two of spades'}

# Convert to list for easy indexing
class_names = [None]*len(target_to_class)  # Initialize list with None to ensure all positions are filled
for idx, name in target_to_class.items():
    class_names[idx] = name

# Streamlit app interface
st.title('Poker Card Classifier by Team Sizzle')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")
    top_probs, top_classes = predict(uploaded_file)  # Adjust function call
    st.write("Top predictions:")
    for i, (prob, cls) in enumerate(zip(top_probs, top_classes)):
        st.write(f"{i+1}: {cls} with a probability of {prob:.2%}")
