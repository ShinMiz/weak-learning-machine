"""
Tests for weak learning machine.
弱学習機のテスト
"""
import numpy as np
import unittest
from weak_learner import DecisionStump, WeakLearner
from adaboost import AdaBoost


class TestDecisionStump(unittest.TestCase):
    """Test cases for Decision Stump"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        self.y_train = np.array([1, 1, -1, -1, -1])
        
    def test_fit(self):
        """Test fitting a decision stump"""
        stump = DecisionStump()
        stump.fit(self.X_train, self.y_train)
        
        self.assertIsNotNone(stump.feature_index)
        self.assertIsNotNone(stump.threshold)
        self.assertIn(stump.polarity, [1, -1])
        
    def test_predict(self):
        """Test prediction with decision stump"""
        stump = DecisionStump()
        stump.fit(self.X_train, self.y_train)
        
        predictions = stump.predict(self.X_train)
        
        self.assertEqual(len(predictions), len(self.y_train))
        self.assertTrue(all(p in [-1, 1] for p in predictions))
        
    def test_weighted_fit(self):
        """Test fitting with sample weights"""
        stump = DecisionStump()
        weights = np.array([0.3, 0.3, 0.1, 0.1, 0.2])
        stump.fit(self.X_train, self.y_train, weights)
        
        self.assertIsNotNone(stump.feature_index)


class TestAdaBoost(unittest.TestCase):
    """Test cases for AdaBoost"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.X_train = np.random.randn(50, 2)
        self.y_train = np.where(self.X_train[:, 0] + self.X_train[:, 1] > 0, 1, -1)
        
    def test_fit(self):
        """Test fitting AdaBoost"""
        clf = AdaBoost(n_estimators=10)
        clf.fit(self.X_train, self.y_train)
        
        self.assertGreater(len(clf.learners), 0)
        self.assertEqual(len(clf.learners), len(clf.alphas))
        
    def test_predict(self):
        """Test prediction with AdaBoost"""
        clf = AdaBoost(n_estimators=10)
        clf.fit(self.X_train, self.y_train)
        
        predictions = clf.predict(self.X_train)
        
        self.assertEqual(len(predictions), len(self.y_train))
        self.assertTrue(all(p in [-1, 1] for p in predictions))
        
    def test_score(self):
        """Test accuracy scoring"""
        clf = AdaBoost(n_estimators=20)
        clf.fit(self.X_train, self.y_train)
        
        accuracy = clf.score(self.X_train, self.y_train)
        
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)
        # With proper training, we should get decent accuracy
        self.assertGreater(accuracy, 0.6)
        
    def test_multiple_estimators(self):
        """Test with different numbers of estimators"""
        for n in [5, 10, 20]:
            clf = AdaBoost(n_estimators=n)
            clf.fit(self.X_train, self.y_train)
            self.assertGreater(len(clf.learners), 0)


if __name__ == '__main__':
    unittest.main()
