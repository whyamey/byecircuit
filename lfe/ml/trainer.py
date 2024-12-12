import numpy as np
from dataclasses import dataclass
from ..utils.conversion import sigmoid


@dataclass
class TrainingConfig:
    """Configuration for training logistic regression"""
    learning_rate = 0.01
    iterations = 1000
    prior_variance = 1.0
    batch_size = None
    verbose = True
    early_stopping_rounds = None
    early_stopping_tol = 1e-4


class BayesianLogisticTrainer:
    """Bayesian Logistic Regression trainer"""

    def __init__(self, n_features, config=None):
        """Initialize trainer"""
        self.n_features = n_features
        self.config = config or TrainingConfig()

        self.weights = np.zeros(n_features)
        self.feature_names = None
        self.training_history = []

    def set_feature_names(self, names):
        """Set feature names for interpretability"""
        if len(names) != self.n_features:
            raise ValueError("Number of names must match number of features")
        self.feature_names = names

    def _compute_loss(self, X, y):
        """Compute loss and gradient"""

        logits = X @ self.weights
        probs = sigmoid(logits)

        eps = 1e-10 
        nll = -np.mean(
            y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps)
        )

        l2_reg = 0.5 * np.sum(self.weights ** 2) / self.config.prior_variance
        loss = nll + l2_reg

        grad_likelihood = X.T @ (probs - y) / len(y)
        grad_prior = self.weights / self.config.prior_variance
        gradient = grad_likelihood + grad_prior

        return loss, gradient

    def _should_stop_early(self, current_loss):
        """Check early stopping criteria"""
        if not self.config.early_stopping_rounds:
            return False

        if len(self.training_history) < self.config.early_stopping_rounds:
            return False

        recent_losses = [h['loss'] for h in
                        self.training_history[-self.config.early_stopping_rounds:]]

        loss_diff = abs(recent_losses[0] - recent_losses[-1])

        return loss_diff < self.config.early_stopping_tol

    def train(self, X, y, X_val=None, y_val=None):
        """Train the model"""
        if self.config.verbose:
            print("\nTraining Progress:")

        n_samples = len(X)
        batch_size = self.config.batch_size or n_samples

        for i in range(self.config.iterations):
            if batch_size < n_samples:
                idx = np.random.choice(n_samples, batch_size, replace=False)
                X_batch, y_batch = X[idx], y[idx]
            else:
                X_batch, y_batch = X, y

            loss, gradient = self._compute_loss(X_batch, y_batch)

            self.weights -= self.config.learning_rate * gradient

            val_loss = None
            val_acc = None
            if X_val is not None and y_val is not None:
                val_loss, _ = self._compute_loss(X_val, y_val)
                val_acc = self.evaluate_accuracy(X_val, y_val)

            history = {
                'iteration': i,
                'loss': loss,
                'val_loss': val_loss,
                'val_accuracy': val_acc
            }
            self.training_history.append(history)

            if self.config.verbose and i % 100 == 0:
                status = f"Iteration {i}, Loss: {loss:.4f}"
                if val_loss is not None:
                    status += f", Val Loss: {val_loss:.4f}"
                if val_acc is not None:
                    status += f", Val Acc: {val_acc:.2%}"
                print(status)

            if self._should_stop_early(loss):
                if self.config.verbose:
                    print("Early stopping triggered")
                break

        if self.config.verbose:
            print("\nFinal Model Weights:")
            if self.feature_names:
                for feat, weight in zip(self.feature_names, self.weights):
                    print(f"{feat}: {weight:.4f}")
            else:
                print(self.weights)

        return self.weights

    def predict_proba(self, X):
        """Predict class probabilities"""
        logits = X @ self.weights

        return sigmoid(logits)

    def predict(self, X, threshold=0.5):
        """Predict classes"""
        probs = self.predict_proba(X)
        
        return (probs > threshold).astype(int)

    def evaluate_accuracy(self, X, y, threshold=0.5):
        """Calculate prediction accuracy"""
        predictions = self.predict(X, threshold)

        return (predictions == y).mean()

    def get_training_summary(self):
        """Get summary of training process"""
        if not self.training_history:
            return {}

        final_metrics = self.training_history[-1]
        best_val_acc = max(h['val_accuracy'] for h in self.training_history
                          if h['val_accuracy'] is not None)

        return {
            'iterations_completed': len(self.training_history),
            'final_loss': final_metrics['loss'],
            'final_val_loss': final_metrics.get('val_loss'),
            'final_val_accuracy': final_metrics.get('val_accuracy'),
            'best_val_accuracy': best_val_acc,
            'convergence_achieved': self._should_stop_early(final_metrics['loss']),
            'model_summary': {
                'n_features': self.n_features,
                'feature_names': self.feature_names,
                'non_zero_weights': np.sum(np.abs(self.weights) > 1e-5),
                'largest_weight': float(np.max(np.abs(self.weights))),
                'average_weight': float(np.mean(np.abs(self.weights)))
            },
            'training_config': {
                'learning_rate': self.config.learning_rate,
                'iterations': self.config.iterations,
                'prior_variance': self.config.prior_variance,
                'batch_size': self.config.batch_size,
                'early_stopping_rounds': self.config.early_stopping_rounds,
                'early_stopping_tol': self.config.early_stopping_tol
            }
        }

    def get_feature_importances(self):
        """Get feature importance scores based on weight magnitudes"""
        importances = np.abs(self.weights)
        normalized_imp = importances / np.sum(importances)

        if self.feature_names:
            return {name: float(imp) for name, imp in
                   zip(self.feature_names, normalized_imp)}
        else:
            return {f'feature_{i}': float(imp) for i, imp in
                   enumerate(normalized_imp)}

    def serialize_weights(self):
        """Serialize model weights for storage/transmission"""
        return np.array(self.weights).tobytes()

    @classmethod
    def deserialize_weights(cls, data, n_features):
        """Deserialize model weights"""
        return np.frombuffer(data).reshape(n_features)


class StockBayesianLogisticTrainer:
    """Stock Bayesian Logistic Regression trainer"""
    def __init__(self, n_features, learning_rate=0.01,
                 iterations=1000, prior_variance=1.0):
        self.n_features = n_features
        self.lr = learning_rate
        self.iterations = iterations
        self.prior_variance = prior_variance
        self.weights = np.zeros(n_features)
        self.feature_names = None

    def set_feature_names(self, names):
        """Set feature names for printing"""
        self.feature_names = names

    def train(self, X, y):
        """Train the model"""
        print("\nTraining Progress:")
        for i in range(self.iterations):
            logits = X @ self.weights
            probs = sigmoid(logits)

            if i % 200 == 0:
                loss = -np.mean(y * np.log(probs + 1e-10) + (1-y) * np.log(1-probs + 1e-10))
                print(f"Iteration {i}, Loss: {loss:.4f}")

            grad_likelihood = X.T @ (probs - y)
            grad_prior = self.weights / self.prior_variance
            gradient = grad_likelihood + grad_prior
            self.weights -= self.lr * gradient

        print("\nFinal Model Weights:")
        if self.feature_names:
            for feat, weight in zip(self.feature_names, self.weights):
                print(f"{feat}: {weight:.4f}")
        else:
            print(self.weights)

        return self.weights

    def evaluate_accuracy(self, X, y):
        """Calculate accuracy on a dataset"""
        logits = X @ self.weights
        probs = sigmoid(logits)
        predictions = (probs > 0.5).astype(int)
        accuracy = (predictions == y).mean()

        return accuracy
