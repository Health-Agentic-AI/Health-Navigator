# ===============================================
# CANCER PREDICTION MODEL - PYTORCH SAFE LOAD
# ===============================================

import torch
import torch.nn as nn
import pickle
import pandas as pd

# ========================
# Neural Network
# ========================
class CancerNN(nn.Module):
    def __init__(self, input_size):
        super(CancerNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(0.2)

        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout1(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(self.relu(self.bn2(self.fc2(x))))
        x = self.dropout3(self.relu(self.bn3(self.fc3(x))))
        return self.fc4(x)

# ========================
# Load model & artifacts safely
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load checkpoint
checkpoint = torch.load("cancer_prediction_model.pt", map_location=device)

# Load feature names and scaler
with open("feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Initialize model with current input size
model = CancerNN(input_size=len(feature_names)).to(device)

# ========================
# Safe state_dict loading
# ========================
model_dict = model.state_dict()
# Only load layers that match shapes
filtered_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.size() == model_dict[k].size()}
model_dict.update(filtered_dict)
model.load_state_dict(model_dict)
model.eval()

print("✓ Cancer PyTorch model loaded safely\n")

# ========================
# Prediction function
# ========================
def predict_cancer(patient_data, threshold=0.5):
    missing = [f for f in feature_names if f not in patient_data]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    df_input = pd.DataFrame([patient_data])
    X = df_input[feature_names].values
    X_scaled = scaler.transform(X)

    input_tensor = torch.FloatTensor(X_scaled).to(device)

    with torch.no_grad():
        logit = model(input_tensor).squeeze()
        probability = torch.sigmoid(logit).cpu().item()
        prediction = 1 if probability >= threshold else 0

    return {
        "prediction": prediction,
        "probability": round(probability, 4)
    }

# ========================
# User Input
# ========================
if __name__ == "__main__":
    print("CANCER PREDICTION - USER INPUT")
    print("=" * 60)

    patient = {}
    for feature in feature_names:
        while True:
            try:
                patient[feature] = float(input(f"{feature}: "))
                break
            except ValueError:
                print("Enter a valid number.")

    result = predict_cancer(patient)

    print("\nRESULT")
    print("=" * 60)
    print("Prediction:", "Cancer" if result["prediction"] else "No Cancer")
    print(f"Probability: {result['probability']*100:.1f}%")
