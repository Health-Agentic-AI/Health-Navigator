import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


def classify_chest_xray(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define image transformations
    model = models.resnet18(weights=None) # not pretrained
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) # Change input channels from 3 to 1 (grayscale images)
    model.fc = nn.Linear(512, 14)  # final fully connected layer

    
    model.load_state_dict(torch.load('models/model_epoch_66.pt'))

    model = model.to(device) # move the model to the GPU

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load and process single image
    img = Image.open(image_path)
    img_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension

    # Inference
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)

    class_names = [
        'Atelectasis',
        'Cardiomegaly',
        'Effusion',
        'Infiltration',
        'Mass',
        'Nodule',
        'Pneumonia',
        'Pneumothorax',
        'Consolidation',
        'Edema',
        'Emphysema',
        'Fibrosis',
        'Pleural_Thickening',
        'Hernia'
    ]

    probabilities = torch.sigmoid(output).squeeze()  # Apply sigmoid
    threshold = 0.5  # Adjust based on your needs
    predicted_labels = (probabilities > threshold).cpu().numpy()

    # Return all positive predictions
    results = {class_names[i]: probabilities[i].item() 
            for i, pred in enumerate(predicted_labels) if pred}
    return results
