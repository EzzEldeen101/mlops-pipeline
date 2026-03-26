import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris


class MachineLearningModel:
    def __init__(self):
        self.model = LogisticRegression()
        self.is_trained = False

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def evaluate(self, X_test, y_test):
        return self.model.score(X_test, y_test)

    def predict(self, X):
        return self.model.predict(X)


def load_data():
    data = load_iris()
    return data.data, data.target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print("Model environment ready!")
        return

    X, y = load_data()
    model = MachineLearningModel()
    model.train(X, y)
    print("Training completed.")


if __name__ == "__main__":
    main()