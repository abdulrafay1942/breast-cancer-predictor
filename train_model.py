import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load data
df = pd.read_csv("brca.csv")
df.drop(columns=["Unnamed: 0"], inplace=True)
df['y'] = df['y'].map({'B': 0, 'M': 1})

X = df.drop('y', axis=1)
y = df['y']

# Save feature names
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Train
model = LogisticRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Model Accuracy: {acc*100:.2f}%")

# Save model + scaler + feature names
joblib.dump(model,         "model.pkl")
joblib.dump(scaler,        "scaler.pkl")
joblib.dump(feature_names, "features.pkl")
print("Saved: model.pkl, scaler.pkl, features.pkl")
