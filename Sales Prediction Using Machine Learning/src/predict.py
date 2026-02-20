import numpy as np

def make_prediction(model, tv, radio, newspaper):
    """
    Predict sales based on custom input
    """
    input_data = np.array([[tv, radio, newspaper]])
    prediction = model.predict(input_data)
    return prediction[0]
