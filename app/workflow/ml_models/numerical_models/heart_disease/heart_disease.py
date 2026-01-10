import torch
import torch.nn as nn
import pickle
import pandas as pd

class HeartDiseaseNN(nn.Module):
    def __init__(self, input_size=19):
        super(HeartDiseaseNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(32, 16)
        self.bn4 = nn.BatchNorm1d(16)
        self.dropout4 = nn.Dropout(0.2)
        self.fc5 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.dropout1(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(self.relu(self.bn2(self.fc2(x))))
        x = self.dropout3(self.relu(self.bn3(self.fc3(x))))
        x = self.dropout4(self.relu(self.bn4(self.fc4(x))))
        return self.fc5(x)

# Load model and scaler once at module level
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = HeartDiseaseNN(input_size=19).to(device)
checkpoint = torch.load('training/results/saved_models/model_epoch_85.pth', 
                       map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load scaler
with open('training/results/saved_models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Features that need scaling
CONTINUOUS_COLS = ['BMI', 'MentHlth', 'PhysHlth', 'GenHlth', 'Age', 'Education', 'Income']

import pandas as pd

def predict_heart_disease(patient_data, threshold=0.40):
    feature_order = [
        'HighBP', 'HighChol', 'BMI', 'Smoker', 'Stroke', 'Diabetes',
        'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump',
        'AnyHealthcare', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk',
        'Sex', 'Age', 'Education', 'Income'
    ]
    
    missing = [f for f in feature_order if f not in patient_data]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    
    # Create DataFrame
    df = pd.DataFrame([patient_data])
    
    # Scale continuous features
    continuous_cols = ['BMI', 'MentHlth', 'PhysHlth', 'GenHlth', 'Age', 'Education', 'Income']
    df_scaled = df.copy()
    df_scaled[continuous_cols] = scaler.transform(df[continuous_cols])
    
    # Get values in correct order
    features = df_scaled[feature_order].values[0]
    input_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor).squeeze()
        probability = torch.sigmoid(output).cpu().item()
        prediction = 1 if probability >= threshold else 0
    
    return {'prediction': prediction, 'probability': round(probability, 4)}