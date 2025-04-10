from deepfraud import NeuralNetwork
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("creditcard.csv")

X = df.drop(columns=["Class"]).values.tolist()
y = df["Class"].values.tolist() 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).tolist()
X_test = scaler.transform(X_test).tolist()

X_train = X_train[:1000]
y_train = y_train[:1000]

# 1. Load dataset
model = NeuralNetwork(X_train, y_train, [4, 8]) 
model.train(1000)

predictions = model.predict(X_test)

for i in range(10):
    print(f"Input: {X_test[i][:5]}... => Prediction: {predictions[i]:.4f} | Actual: {y_test[i]}")
