import unittest
import numpy as np
from src.model import MachineLearningModel


class TestModel(unittest.TestCase):

    def setUp(self):
        self.model = MachineLearningModel()

        # simple dummy dataset
        self.X_train = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
        self.y_train = np.array([0, 1, 1, 0])

    def test_model_training(self):
        self.model.train(self.X_train, self.y_train)
        self.assertTrue(self.model.is_trained)

    def test_model_prediction(self):
        self.model.train(self.X_train, self.y_train)
        prediction = self.model.predict([[1, 1]])
        self.assertIsNotNone(prediction)

    def test_model_evaluation(self):
        self.model.train(self.X_train, self.y_train)
        score = self.model.evaluate(self.X_train, self.y_train)
        self.assertGreaterEqual(score, 0)


if __name__ == "__main__":
    unittest.main()