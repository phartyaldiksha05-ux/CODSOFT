from src.data_preprocessing import preprocess_data
from src.train_model import train_models
from src.evaluate_model import evaluate

# Step 1: Preprocess Data
X, y = preprocess_data("data/creditcard.csv")

# Step 2: Train Models
X_test, y_test, lr_model, rf_model = train_models(X, y)

# Step 3: Evaluate Models
evaluate(lr_model, X_test, y_test, "Logistic Regression")
evaluate(rf_model, X_test, y_test, "Random Forest")
