import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from ..core.params import LFEParams
from ..utils.conversion import convert_to_fixed_point

def prepare_binary_iris_data(test_size=0.2, random_state=None):
    """Prepare iris dataset for binary classification"""
    iris = load_iris()
    
    X = iris.data[:100]  # Only first two classes
    y = iris.target[:100]
    
    scaler = StandardScaler()
    
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': iris.feature_names,
        'class_names': iris.target_names[:2],
        'scaler': scaler
    }

def prepare_iris_data_sim():
    """Prepare iris dataset for binary classification"""
    
    iris = load_iris()
    
    X = iris.data[:100]
    y = iris.target[:100]
    
    scaler = StandardScaler()
    
    X = scaler.fit_transform(X)
    
    print("\nDataset Information:")
    print("Features used:", iris.feature_names)
    print("Classes:", iris.target_names[:2])
    print("Number of samples per class:", np.bincount(y))
    
    return X, y, scaler, iris.feature_names, iris.target_names[:2]

def prepare_data_for_lfe(X, params):
    """Convert data to fixed-point representation for LFE"""
    return convert_to_fixed_point(X, params.fixed_point_bits)

def create_test_batch(X, batch_size):
    """Create batch of test samples"""
    indices = np.random.choice(len(X), batch_size, replace=False)
    
    return X[indices]

def calculate_metrics(y_true, y_pred):
    """Calculate various classification metrics"""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    }

def create_confusion_matrix_str(y_true, y_pred, class_names=None):
    """Create formatted confusion matrix string"""
    if class_names is None:
        class_names = ('Class 0', 'Class 1')
    cm = np.zeros((2, 2), dtype=int)
    for i in range(len(y_true)):
        cm[y_true[i], y_pred[i]] += 1
    
    result = "\nConfusion Matrix:\n"
    result += "                  Predicted\n"
    result += f"                  {class_names[0]:<8} {class_names[1]:<8}\n"
    result += f"Actual {class_names[0]:<8} {cm[0,0]:^8d} {cm[0,1]:^8d}\n"
    result += f"      {class_names[1]:<8} {cm[1,0]:^8d} {cm[1,1]:^8d}"
    
    return result

def save_predictions(y_pred, probas, feature_values, feature_names=None, filepath="predictions.txt"):
    """Save predictions with feature values"""
    
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(feature_values.shape[1])]
    
    with open(filepath, 'w') as f:
        header = ','.join(feature_names + ['prediction', 'probability'])
        f.write(header + '\n')
        
        for i in range(len(y_pred)):
            features = ','.join(f'{val:.4f}' for val in feature_values[i])
            f.write(f'{features},{y_pred[i]},{probas[i]:.4f}\n')
