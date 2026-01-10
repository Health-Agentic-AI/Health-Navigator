import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPClassifier
import joblib
import json
from datetime import datetime

# Load dataset
df = pd.read_csv('stroke_risk_dataset_v2.csv') # Update ur Path File here So u can Run

# Encode gender
le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])

# Separate features and targets
X = df.drop(['at_risk', 'stroke_risk_percentage'], axis=1)
y_clf = df['at_risk']
y_reg = df['stroke_risk_percentage']

# Train-test split
X_train, X_test, y_clf_train, y_clf_test = train_test_split(X, y_clf, test_size=0.2, random_state=42, stratify=y_clf)
_, _, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)

print("=" * 60)
print("TRAINING FINAL MODELS")
print("=" * 60)

# Neural Network for Classification
print("\n1. Neural Network (Classification)...")
nn_clf = MLPClassifier(random_state=42, max_iter=500, hidden_layer_sizes=(100, 50))
nn_clf.fit(X_train, y_clf_train)

y_pred = nn_clf.predict(X_test)
y_pred_proba = nn_clf.predict_proba(X_test)[:, 1]

nn_results = {
    'accuracy': float(accuracy_score(y_clf_test, y_pred)),
    'f1_score': float(f1_score(y_clf_test, y_pred)),
    'roc_auc': float(roc_auc_score(y_clf_test, y_pred_proba))
}

print(f"Accuracy: {nn_results['accuracy']:.4f}")
print(f"F1 Score: {nn_results['f1_score']:.4f}")
print(f"ROC-AUC: {nn_results['roc_auc']:.4f}")

# ElasticNet for Regression
print("\n2. ElasticNet (Regression)...")
elasticnet_reg = ElasticNet(random_state=42)
elasticnet_reg.fit(X_train, y_reg_train)

y_pred = elasticnet_reg.predict(X_test)

en_results = {
    'mse': float(mean_squared_error(y_reg_test, y_pred)),
    'rmse': float(np.sqrt(mean_squared_error(y_reg_test, y_pred))),
    'mae': float(mean_absolute_error(y_reg_test, y_pred)),
    'r2_score': float(r2_score(y_reg_test, y_pred))
}

print(f"RMSE: {en_results['rmse']:.4f}")
print(f"MAE: {en_results['mae']:.4f}")
print(f"R² Score: {en_results['r2_score']:.4f}")

# Save models
joblib.dump(nn_clf, 'neural_network_classifier.pkl')
joblib.dump(elasticnet_reg, 'elasticnet_regressor.pkl')

print("\n" + "=" * 60)
print("MODELS SAVED")
print("=" * 60)
print("✓ neural_network_classifier.pkl")
print("✓ elasticnet_regressor.pkl")

# Save results
final_results = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'neural_network_classification': nn_results,
    'elasticnet_regression': en_results,
    'dataset_info': {
        'total_samples': len(df),
        'train_size': len(X_train),
        'test_size': len(X_test),
        'features': list(X.columns)
    }
}

with open('final_models_results.json', 'w') as f:
    json.dump(final_results, f, indent=4)

print("✓ final_models_results.json")

# Example predictions
print("\n" + "=" * 60)
print("EXAMPLE PREDICTIONS")
print("=" * 60)

sample = X_test.iloc[0:1]
clf_pred = nn_clf.predict(sample)[0]
clf_prob = nn_clf.predict_proba(sample)[0][1]
reg_pred = elasticnet_reg.predict(sample)[0]

print(f"\nSample Patient:")
print(f"  Classification: {'At Risk' if clf_pred == 1 else 'Not At Risk'} (probability: {clf_prob:.2%})")
print(f"  Regression: {reg_pred:.2f}% stroke risk")