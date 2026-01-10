"""
Train Cancer Prediction Model
This script trains a Random Forest model on the cancer prediction dataset
and saves it for later use.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

def load_data(filepath='cancer_predictions.csv'):
    """Load the cancer prediction dataset"""
    print("=" * 60)
    print("LOADING DATASET")
    print("=" * 60)
    
    df = pd.read_csv(filepath)
    print(f"\n✓ Loaded {len(df)} patient records")
    print(f"✓ Features: {df.shape[1]} columns")
    
    return df

def explore_data(df):
    """Explore and display dataset information"""
    print("\n" + "=" * 60)
    print("DATA EXPLORATION")
    print("=" * 60)
    
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nDataset Info:")
    print(df.info())
    
    print("\nStatistical Summary:")
    print(df.describe())
    
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    print("\nTarget Variable Distribution:")
    print(df['Diagnosis'].value_counts())
    print(f"Cancer cases: {df['Diagnosis'].sum()}")
    print(f"No cancer cases: {len(df) - df['Diagnosis'].sum()}")

def prepare_data(df, test_size=0.2, random_state=42):
    """Prepare data for training"""
    print("\n" + "=" * 60)
    print("PREPARING DATA")
    print("=" * 60)
    
    # Separate features and target
    X = df.drop('Diagnosis', axis=1)
    y = df['Diagnosis']
    
    print(f"\n✓ Features (X): {X.shape}")
    print(f"✓ Target (y): {y.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"\n✓ Training set: {len(X_train)} samples ({(1-test_size)*100:.0f}%)")
    print(f"✓ Testing set: {len(X_test)} samples ({test_size*100:.0f}%)")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n✓ Features scaled using StandardScaler")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_model(X_train, y_train, n_estimators=100, random_state=42):
    """Train Random Forest model"""
    print("\n" + "=" * 60)
    print("TRAINING MODEL")
    print("=" * 60)
    
    print(f"\n🔄 Training Random Forest with {n_estimators} trees...")
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1  # Use all CPU cores
    )
    
    model.fit(X_train, y_train)
    
    print("✅ Training completed!")
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n📊 Accuracy: {accuracy*100:.2f}%")
    
    # Classification report
    print("\n" + "-" * 60)
    print("Classification Report:")
    print("-" * 60)
    print(classification_report(y_test, y_pred, target_names=['No Cancer', 'Cancer']))
    
    # Confusion matrix
    print("-" * 60)
    print("Confusion Matrix:")
    print("-" * 60)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(f"\nTrue Negatives: {cm[0][0]} | False Positives: {cm[0][1]}")
    print(f"False Negatives: {cm[1][0]} | True Positives: {cm[1][1]}")
    
    return accuracy, y_pred

def save_model(model, scaler, model_path='trained_model.pkl', scaler_path='scaler.pkl'):
    """Save trained model and scaler"""
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)
    
    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n✓ Model saved to: {model_path}")
    
    # Save scaler
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Scaler saved to: {scaler_path}")
    
    # Get file sizes
    model_size = os.path.getsize(model_path) / 1024  # KB
    scaler_size = os.path.getsize(scaler_path) / 1024  # KB
    
    print(f"\n📦 Model size: {model_size:.2f} KB")
    print(f"📦 Scaler size: {scaler_size:.2f} KB")

def main():
    """Main training pipeline"""
    print("\n" + "=" * 60)
    print("CANCER PREDICTION MODEL TRAINING")
    print("=" * 60)
    
    # Load data
    df = load_data('cancer_predictions.csv')
    
    # Explore data
    explore_data(df)
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    
    # Train model
    model = train_model(X_train, y_train, n_estimators=100)
    
    # Evaluate model
    accuracy, predictions = evaluate_model(model, X_test, y_test)
    
    # Save model and scaler
    save_model(model, scaler)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\n✅ Final Accuracy: {accuracy*100:.2f}%")
    print("✅ Model and scaler saved successfully")
    print("\n💡 You can now use the trained model with predict.py or chatbot.py")
    print("=" * 60)

if __name__ == "__main__":
    main()