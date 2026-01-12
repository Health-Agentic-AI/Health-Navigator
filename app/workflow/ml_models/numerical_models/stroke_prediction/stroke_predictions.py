import torch
import torch.nn as nn
import pickle
import pandas as pd

# ========================
# Neural Network
# ========================
class StrokeNN(nn.Module):
    def __init__(self, input_size=16):
        super(StrokeNN, self).__init__()
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

# ========================
# Load model, scaler, and features
# ========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
checkpoint = torch.load('results/stroke_model.pth', map_location=device)
model = StrokeNN(input_size=checkpoint['input_size']).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load scaler
with open('results/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load feature names
with open('results/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

print("✓ Stroke model loaded successfully!\n")

# ========================
# Prediction Function
# ========================
def predict_stroke_risk(patient_data, threshold=0.5):
    df = pd.DataFrame([patient_data])
    features = df[feature_names].values[0]
    features_scaled = scaler.transform([features])
    input_tensor = torch.FloatTensor(features_scaled).to(device)
    
    with torch.no_grad():
        output = model(input_tensor).squeeze()
        probability = torch.sigmoid(output).cpu().item()
        prediction = 1 if probability >= threshold else 0
    
    return {'prediction': prediction, 'probability': round(probability, 4)}

# ========================
# Get user input interactively
# ========================
if __name__ == "__main__":
    print("STROKE PREDICTION - USER INPUT")
    print("="*50)
    
    patient_data = {}
    for feature in feature_names:
        while True:
            val = input(f"Enter value for {feature}: ")
            try:
                val = float(val)
                patient_data[feature] = val
                break
            except ValueError:
                print("Please enter a valid number.")

    result = predict_stroke_risk(patient_data)
    print("\nRESULT")
    print("="*50)
    print(f"Prediction: {'At Risk' if result['prediction'] == 1 else 'Not At Risk'}")
    print(f"Probability: {result['probability']} ({result['probability']*100:.1f}%)")
    print("="*50)
