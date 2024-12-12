from lfe.core.scheme import LegacyLFEScheme
from lfe.core.params import LFEParams
from lfe.core.circuit import SmallLFECircuit
from lfe.ml.trainer import StockBayesianLogisticTrainer
from lfe.ml.data import prepare_iris_data_sim
from lfe.utils.conversion import convert_to_fixed_point, sigmoid

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def run_demo():
    """Run demo for the Legacy LFE (this is what the compiler hopefully outputs)"""

    try:
        params = LFEParams()
        lfe = LegacyLFEScheme(params)

        print("Loading and preparing Iris dataset...")
        X, y, scaler, feature_names, class_names = prepare_iris_data_sim()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=None
        )

        print("\nInitializing model training...")
        trainer = StockBayesianLogisticTrainer(n_features=4)
        trainer.set_feature_names(feature_names)
        weights = trainer.train(X_train, y_train)

        train_acc = trainer.evaluate_accuracy(X_train, y_train)
        test_acc = trainer.evaluate_accuracy(X_test, y_test)
        print(f"\nTraining Accuracy: {train_acc:.2%}")
        print(f"Test Accuracy: {test_acc:.2%}")

        print("\nCreating LFE circuit...")
        circuit = SmallLFECircuit(params)
        feature_wires = [circuit.add_input_gate(params.fixed_point_bits) for _ in range(4)]

        product_wires = []
        fixed_weights = convert_to_fixed_point(weights, params.fixed_point_bits)
        for fw, weight in zip(feature_wires, fixed_weights):
            mult_wires = circuit.add_multiplication_gate(fw, int(weight))
            product_wires.append(mult_wires)

        digest, key = lfe.setup(circuit)
        compressed = lfe.compress(circuit, fixed_weights)

        print("\nTesting individual predictions...")
        for i in range(min(5, len(X_test))):
            test_features = X_test[i]
            true_class = y_test[i]

            print(f"\nTest Sample {i+1}:")
            print("Features:")
            for feat_name, feat_val in zip(feature_names, test_features):
                print(f"{feat_name}: {feat_val:.4f}")
            print(f"True class: {class_names[true_class]}")

            test_x = convert_to_fixed_point(test_features, params.fixed_point_bits)
            print(f"Fixed point values: {test_x}")
            test_inputs = [int(val).to_bytes(16, 'big', signed=True) for val in test_x]

            result = lfe.evaluate(circuit, compressed, test_inputs)
            float_result = result / (1 << params.fixed_point_bits)

            prob = sigmoid(float_result)
            pred_class = 1 if prob > 0.5 else 0
            print(f"Raw model output: {float_result:.4f}")
            print(f"Probability of {class_names[1]}: {prob:.4f}")
            print(f"Predicted class: {class_names[pred_class]} (confidence: {max(prob, 1-prob):.2%})")
            print(f"Prediction {'correct' if pred_class == true_class else 'incorrect'}!")

        y_pred = (sigmoid(X_test @ trainer.weights) > 0.5).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print("                  Predicted")
        print("                  Setosa  Versicolor")
        print(f"Actual Setosa      {cm[0][0]:^6d}  {cm[0][1]:^10d}")
        print(f"      Versicolor   {cm[1][0]:^6d}  {cm[1][1]:^10d}")

    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    run_demo()
