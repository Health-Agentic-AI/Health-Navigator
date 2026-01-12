"""
Run this to check what's in your existing pickle files
"""
import pickle
import os

def check_pickle_file(filepath):
    """Check contents of a pickle file"""
    if not os.path.exists(filepath):
        print(f"❌ NOT FOUND: {filepath}")
        return None
    
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"✅ FOUND: {filepath}")
        return data
    except Exception as e:
        print(f"⚠️  ERROR loading {filepath}: {e}")
        return None

print("="*60)
print("CHECKING EXISTING PICKLE FILES")
print("="*60)

# Check feature_names.pkl
print("\n1️⃣ Feature Names:")
print("-" * 60)
features = check_pickle_file("training/feature_names.pkl")
if features:
    print(f"   Number of features: {len(features)}")
    print(f"   Features: {features}")

# Check scaler.pkl
print("\n2️⃣ Scaler:")
print("-" * 60)
scaler = check_pickle_file("training/scaler.pkl")
if scaler:
    print(f"   Type: {type(scaler).__name__}")
    print(f"   Mean shape: {scaler.mean_.shape if hasattr(scaler, 'mean_') else 'N/A'}")
    print(f"   Scale shape: {scaler.scale_.shape if hasattr(scaler, 'scale_') else 'N/A'}")

# Check best_params.pkl
print("\n3️⃣ Best Parameters:")
print("-" * 60)
params = check_pickle_file("training/best_params.pkl")
if params:
    print(f"   Parameters: {params}")

# Check if model file exists
print("\n4️⃣ Model File:")
print("-" * 60)
if os.path.exists("training/cancer_prediction_model.pt"):
    print("✅ FOUND: training/cancer_prediction_model.pt")
    import torch
    try:
        checkpoint = torch.load("training/cancer_prediction_model.pt", map_location='cpu')
        print(f"   Model keys: {list(checkpoint.keys())[:5]}... (showing first 5)")
        
        # Try to infer input size from first layer
        if 'fc1.weight' in checkpoint:
            input_size = checkpoint['fc1.weight'].shape[1]
            print(f"   ✓ Detected input size: {input_size}")
    except Exception as e:
        print(f"   ⚠️  Error loading: {e}")
else:
    print("❌ NOT FOUND: training/cancer_prediction_model.pt")
    print("   ⚠️  YOU NEED TO TRAIN THE MODEL!")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)

if features and scaler and os.path.exists("training/cancer_prediction_model.pt"):
    print("✅ All required files exist! You can use Cancer_prediction.py")
    print("\nNext step: Test inference with Cancer_prediction.py")
else:
    print("⚠️  Missing files. You need to:")
    if not os.path.exists("training/cancer_prediction_model.pt"):
        print("   1. Train the model using cancer_prediction.ipynb")
    if not features:
        print("   2. Generate feature_names.pkl")
    if not scaler:
        print("   3. Generate scaler.pkl")
    print("\n👉 See Step 3 below for training instructions")