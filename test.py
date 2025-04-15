from deepfraud import NeuralNetwork
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

scaler = joblib.load("scaler.pkl")

df = pd.read_csv("creditcard.csv")

X = df.drop(columns=["fraud"]).values.tolist()
y = df["fraud"].values.tolist() 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test = scaler.transform(X_test)

#1. load weights and baises
model = NeuralNetwork(X_train, y_train, [4, 8])
model.load_config("output")

# 2. Prdecit
prediction = model.predict(X_test)
for i in range(100):
    print(f"Input: {X_test[i][:5]}... => Prediction: {prediction[i]:.4f} | Actual: {y_test[i]}")
