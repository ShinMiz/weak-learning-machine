"""
AdaBoost Algorithm
AdaBoostアルゴリズム
"""
import numpy as np
from weak_learner import DecisionStump


class AdaBoost:
    """
    AdaBoost classifier using weak learners.
    弱学習器を使用したAdaBoost分類器
    """
    
    def __init__(self, n_estimators=50, weak_learner_class=DecisionStump):
        """
        Initialize AdaBoost.
        
        Args:
            n_estimators: Number of weak learners to train
            weak_learner_class: Class to use for weak learners
        """
        self.n_estimators = n_estimators
        self.weak_learner_class = weak_learner_class
        self.learners = []
        self.alphas = []
        
    def fit(self, X, y):
        """
        Train the AdaBoost classifier.
        
        Args:
            X: Training data features (n_samples, n_features)
            y: Training data labels (n_samples,), should be -1 or 1
        """
        n_samples = X.shape[0]
        
        # Initialize weights uniformly
        weights = np.ones(n_samples) / n_samples
        
        self.learners = []
        self.alphas = []
        
        for t in range(self.n_estimators):
            # Train a weak learner
            learner = self.weak_learner_class()
            learner.fit(X, y, weights)
            
            # Get predictions
            predictions = learner.predict(X)
            
            # Calculate error
            error = np.sum(weights[y != predictions])
            
            # Avoid division by zero
            if error == 0:
                error = 1e-10
            if error >= 0.5:
                continue
            
            # Calculate alpha (learner weight)
            alpha = 0.5 * np.log((1 - error) / error)
            
            # Update weights
            weights *= np.exp(-alpha * y * predictions)
            weights /= np.sum(weights)
            
            # Store learner and its weight
            self.learners.append(learner)
            self.alphas.append(alpha)
        
        return self
    
    def predict(self, X):
        """
        Make predictions using the trained AdaBoost classifier.
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            predictions: Predicted labels (n_samples,)
        """
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)
        
        for alpha, learner in zip(self.alphas, self.learners):
            predictions += alpha * learner.predict(X)
        
        return np.sign(predictions)
    
    def score(self, X, y):
        """
        Calculate accuracy on given data.
        
        Args:
            X: Input features
            y: True labels
            
        Returns:
            accuracy: Accuracy score
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)
