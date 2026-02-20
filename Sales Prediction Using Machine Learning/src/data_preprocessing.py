from sklearn.model_selection import train_test_split

def split_data(data):
    """
    Separate features and target
    Perform train-test split
    """
    X = data[["TV", "Radio", "Newspaper"]]
    y = data["Sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test
