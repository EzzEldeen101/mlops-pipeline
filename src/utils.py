import pickle


def preprocess_data(data):
    return data


def extract_features(data):
    return data


def save_model(model, filename):
    with open(filename, "wb") as f:
        pickle.dump(model, f)


def load_model(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def evaluate_model(model, X_test, y_test):
    return model.score(X_test, y_test)