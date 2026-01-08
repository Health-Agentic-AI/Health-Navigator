import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

def classify_colon(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.1),
        nn.Linear(512, 9)
    )    
    
    model.load_state_dict(torch.load('training/models/model_epoch_13.pt'))
    model = model.to(device)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(), # convert to tensor
        transforms.Normalize((0.7405, 0.5330, 0.7058), (0.1237, 0.1768, 0.1244))
    ])

    img = Image.open(image_path)
    img_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension

    # Inference
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)

    class_names = [
        "Adipose",
        "Background",
        "Debris",
        "Lymphocytes",
        "Mucus",
        "Smooth Muscle",
        "Normal Colon Mucosa",
        "Cancer-associated Stroma",
        "Colorectal Adenocarcinoma Epithelium"
    ]


    predicted_class = output.argmax(dim=1).item()
    return class_names[predicted_class]