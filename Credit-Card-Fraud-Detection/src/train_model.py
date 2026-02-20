from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_models(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)

    # Random Forest
    rf = RandomForestClassifier(
    n_estimators=20,      # reduced trees
    max_depth=10,         # limit tree depth
    n_jobs=-1,            # use all CPU cores
    random_state=42
)

    rf.fit(X_train, y_train)

    # Save models
    joblib.dump(lr, "logistic_model.pkl")
    joblib.dump(rf, "random_forest.pkl")

    return X_test, y_test, lr, rf
