import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
from xgboost_distribution import XGBDistribution

class BaseRegressor:
    """Base class for regression models"""
    
    def __init__(self, name="base", **kwargs):
        self.name = name
        self.model = None
        self.is_fitted = False
        self.metrics = {}
        
    def fit(self, X_train, y_train):
        """Train the model on the given data"""
        raise NotImplementedError("Subclass must implement abstract method")
    
    def predict(self, X):
        """Make predictions using the trained model"""
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet")
        return self.model.predict(X)
    
    def predict_with_uncertainty(self, X):
        """Predict with uncertainty estimates"""
        raise NotImplementedError("Subclass must implement abstract method")
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model on test data and return metrics"""
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet")
        
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        self.metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        return self.metrics
    
    def save_model(self, output_dir):
        """Save the model to disk"""
        if not self.is_fitted:
            raise ValueError("Cannot save model that has not been fitted")
        
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, f"{self.name}_model.joblib")
        joblib.dump(self.model, model_path)
        
        # Save metrics
        if self.metrics:
            metrics_path = os.path.join(output_dir, f"{self.name}_metrics.joblib")
            joblib.dump(self.metrics, metrics_path)
        
        return model_path
    
    def load_model(self, model_path):
        """Load a saved model from disk"""
        self.model = joblib.load(model_path)
        self.is_fitted = True
        return self


class RFRegressor(BaseRegressor):
    """Random Forest Regressor"""
    
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features=1.0, random_state=42, **kwargs):
        super().__init__(name="rf", **kwargs)
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'random_state': random_state
        }
        self.model = RandomForestRegressor(**self.params)
    
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Predict using mean of tree predictions"""
        all_preds = np.stack([tree.predict(X) for tree in self.model.estimators_], axis=0)
        return np.mean(all_preds, axis=0)
    
    def predict_with_uncertainty(self, X):
        """Predict with uncertainty using standard deviation across trees"""
        all_preds = np.stack([tree.predict(X) for tree in self.model.estimators_], axis=0)
        mean_preds = np.mean(all_preds, axis=0)
        std_preds = np.std(all_preds, axis=0)
        return mean_preds, std_preds


class XGBDistRegressor(BaseRegressor):
    """XGBoost Distribution Regressor"""
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=6,
                 subsample=0.8, colsample_bytree=0.8, random_state=42,
                 tree_method='hist', **kwargs):
        super().__init__(name="xgbd", **kwargs)
        self.params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'random_state': random_state,
            'tree_method': tree_method
        }
        self.model = XGBDistribution(**self.params)
    
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train, verbose=False)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Predict using loc from XGBDistribution"""
        predictions = self.model.predict(X)
        return predictions.loc
    
    def predict_with_uncertainty(self, X):
        """Predict with uncertainty using loc and scale from XGBDistribution"""
        predictions = self.model.predict(X)
        return predictions.loc, predictions.scale


def create_regressor(name, **kwargs):
    """Create a regressor by name"""
    regressors = {
        'rf': RFRegressor,
        'xgbd': XGBDistRegressor
    }
    
    if name not in regressors:
        raise ValueError(f"Unknown regressor: {name}. Available options: {list(regressors.keys())}")
    
    return regressors[name](**kwargs)
