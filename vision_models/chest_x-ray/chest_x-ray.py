import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def classify_chest_xray(image_path):
    # Define image transformations
    model = models.resnet18(weights=None) # not pretrained
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) # Change input channels from 3 to 1 (grayscale images)
    model.fc = nn.Linear(512, 14)  # final fully connected layer

    
    model.load_state_dict(torch.load('models/model_epoch_66.pt'))

    model = model.to(device) # move the model to the GPU

    from torchvision import transforms
    from PIL import Image

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

    predicted_class = output.argmax(dim=1).item()
    return class_names[predicted_class]
