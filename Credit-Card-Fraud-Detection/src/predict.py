import joblib
import numpy as np

def predict_transaction(transaction):
    model = joblib.load("random_forest.pkl")

    transaction = np.array(transaction).reshape(1, -1)
    prediction = model.predict(transaction)

    if prediction[0] == 1:
        return "⚠️ Fraud Transaction"
    else:
        return "✅ Genuine Transaction"
