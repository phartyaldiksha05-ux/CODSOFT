# =========================================================
# SALES PREDICTION PROJECT - MAIN FILE
# =========================================================

import os
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import load_data, show_basic_info
from src.data_preprocessing import split_data
from src.model import train_linear_regression, train_random_forest
from src.evaluate import evaluate_model
from src.predict import make_prediction


# ----------------------------
# 1. Load Dataset
# ----------------------------

data_path = os.path.join("data", "advertising.csv")
data = load_data(data_path)

show_basic_info(data)


# ----------------------------
# 2. EDA - Correlation Heatmap
# ----------------------------

plt.figure(figsize=(6,4))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


# ----------------------------
# 3. Split Data
# ----------------------------

X_train, X_test, y_train, y_test = split_data(data)


# ----------------------------
# 4. Train Models
# ----------------------------

lr_model = train_linear_regression(X_train, y_train)
rf_model = train_random_forest(X_train, y_train)


# ----------------------------
# 5. Evaluate Models
# ----------------------------

print("\n===== Linear Regression =====")
mae_lr, mse_lr, rmse_lr, r2_lr = evaluate_model(lr_model, X_test, y_test)
print("MAE:", mae_lr)
print("MSE:", mse_lr)
print("RMSE:", rmse_lr)
print("R2:", r2_lr)

print("\n===== Random Forest =====")
mae_rf, mse_rf, rmse_rf, r2_rf = evaluate_model(rf_model, X_test, y_test)
print("MAE:", mae_rf)
print("MSE:", mse_rf)
print("RMSE:", rmse_rf)
print("R2:", r2_rf)


# ----------------------------
# 6. Compare Models
# ----------------------------

if r2_lr > r2_rf:
    best_model = lr_model
    print("\nLinear Regression performs better.")
else:
    best_model = rf_model
    print("\nRandom Forest performs better.")


# ----------------------------
# 7. Example Prediction
# ----------------------------

predicted_sales = make_prediction(best_model, 200, 40, 50)
print("\nPredicted Sales for (TV=200, Radio=40, Newspaper=50):")
print(predicted_sales)
