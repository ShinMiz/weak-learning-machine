"""
Weak Learner Base Class
弱学習器の基底クラス
"""
import numpy as np


class WeakLearner:
    """
    Base class for weak learners.
    弱学習器の基底クラス
    """
    
    def fit(self, X, y, sample_weights=None):
        """
        Train the weak learner.
        
        Args:
            X: Training data features
            y: Training data labels
            sample_weights: Weights for each sample
        """
        raise NotImplementedError("Subclasses must implement fit method")
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        raise NotImplementedError("Subclasses must implement predict method")


class DecisionStump(WeakLearner):
    """
    Decision Stump - a weak learner that makes decisions based on a single feature.
    決定株 - 単一の特徴に基づいて決定を行う弱学習器
    """
    
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.polarity = 1
        
    def fit(self, X, y, sample_weights=None):
        """
        Train the decision stump.
        
        Args:
            X: Training data features (n_samples, n_features)
            y: Training data labels (n_samples,)
            sample_weights: Weights for each sample (n_samples,)
        """
        n_samples, n_features = X.shape
        
        if sample_weights is None:
            sample_weights = np.ones(n_samples) / n_samples
        
        # Normalize weights
        sample_weights = sample_weights / np.sum(sample_weights)
        
        min_error = float('inf')
        
        # Try each feature
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            
            # Try each unique value as threshold
            for threshold in unique_values:
                # Try both polarities
                for polarity in [1, -1]:
                    predictions = np.ones(n_samples)
                    predictions[polarity * feature_values < polarity * threshold] = -1
                    
                    # Calculate weighted error
                    error = np.sum(sample_weights[y != predictions])
                    
                    if error < min_error:
                        min_error = error
                        self.feature_index = feature_idx
                        self.threshold = threshold
                        self.polarity = polarity
        
        return self
    
    def predict(self, X):
        """
        Make predictions using the trained decision stump.
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            predictions: Predicted labels (n_samples,)
        """
        n_samples = X.shape[0]
        predictions = np.ones(n_samples)
        
        mask = self.polarity * X[:, self.feature_index] < self.polarity * self.threshold
        predictions[mask] = -1
        
        return predictions
