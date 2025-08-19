# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load data
DATA_PATH = '../data/match_data.csv'
MODEL_PATH = '../models/match_predictor.pkl'

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

# Basic data checks
if df.isnull().sum().sum() > 0:
    print("Warning: Dataset contains missing values. Filling with zeros.")
    df.fillna(0, inplace=True)

# Feature engineering
df['home_advantage'] = df['venue'].apply(lambda x: 1 if x.lower() == 'home' else 0)

# Features and target
features = ['home_advantage', 'injuries', 'yellow_cards']
if not all(f in df.columns for f in features + ['result']):
    raise ValueError("Missing required columns in the dataset.")

X = df[features]
y = df['result']  # 0 = lose, 1 = draw, 2 = win

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(model, MODEL_PATH)

print(f"✅ Model trained and saved to {MODEL_PATH}")
