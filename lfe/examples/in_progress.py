import numpy as np
from time import time

from lfe.core.params import LFEParams
from lfe.core.circuit import SecureLFECircuit
from lfe.core.scheme import LFEScheme

from lfe.ml.data import prepare_binary_iris_data, prepare_data_for_lfe, calculate_metrics, create_confusion_matrix_str
from lfe.ml.trainer import BayesianLogisticTrainer, TrainingConfig

from lfe.utils.conversion import convert_to_fixed_point

def run_demo():
    """Run experiment for the full in-progress circuit"""

    print("\nLFE Bayesian Logistic Regression Demo")

    params = LFEParams(
        fixed_point_bits=12,
        sigmoid_pieces=8
    )

    print("Loading Iris dataset...")
    data = prepare_binary_iris_data(test_size=0.2, random_state=42)
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    feature_names = data['feature_names']
    class_names = data['class_names']

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {feature_names}")
    print(f"Classes: {class_names}")

    print("\nTraining logistic regression model...")
    config = TrainingConfig()

    trainer = BayesianLogisticTrainer(n_features=X_train.shape[1], config=config)
    trainer.set_feature_names(feature_names)
    weights = trainer.train(X_train, y_train)

    print("\nConverting to fixed-point representation...")
    weights_fixed = convert_to_fixed_point(weights, params.fixed_point_bits)
    X_test_fixed = prepare_data_for_lfe(X_test, params)

    print("\nFixed-point weights:")
    for name, w in zip(feature_names, weights_fixed):
        print(f"{name}: {w}")

    print("\nCreating LFE circuit...")
    lfe = LFEScheme(params)
    circuit = SecureLFECircuit(n_features=len(feature_names), params=params)

    output_wire = circuit.create_logistic_circuit(weights_fixed)

    stats = circuit.get_circuit_stats()
    print("\nCircuit statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")

    print("\nGenerating circuit commitment...")
    digest, key = lfe.setup(circuit)

    print("Compressing circuit...")
    compressed = lfe.compress(circuit, weights_fixed)

    print("\nGenerating and verifying weight proofs...")
    proof = lfe.prove_weights_valid(weights_fixed)
    if not lfe.verify_weights_proof(proof, digest):
        raise ValueError("Weight proof verification failed!")
    print("Weight proofs verified successfully!")

    print("\nBenchmarking inference...")
    n_test = min(5, len(X_test))
    times = []
    predictions = []

    for i in range(n_test):
        print(f"\nTest sample {i+1}:")
        test_x = X_test_fixed[i]
        true_y = y_test[i]

        print("Features:")
        for name, val in zip(feature_names, X_test[i]):
            print(f"{name}: {val:.4f}")
        print(f"True class: {class_names[true_y]}")

        start = time()
        try:
            result, _ = lfe.evaluate(circuit, compressed, test_x, benchmark=True)
            eval_time = time() - start
            times.append(eval_time)

            pred_class = 1 if result > 0.5 else 0
            predictions.append(pred_class)

            print(f"Predicted probability: {result:.4f}")
            print(f"Predicted class: {class_names[pred_class]}")
            print(f"Inference time: {eval_time:.3f}s")
            print(f"Prediction {'correct' if pred_class == true_y else 'incorrect'}!")

        except Exception as e:
            print(f"Error during inference: {str(e)}")

    if times:
        avg_time = np.mean(times)
        std_time = np.std(times)
        print(f"\nAverage inference time: {avg_time:.3f} ± {std_time:.3f} seconds")

    if predictions:
        metrics = calculate_metrics(y_test[:len(predictions)], predictions)
        print("\nTest Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        print(create_confusion_matrix_str(
            y_test[:len(predictions)],
            predictions,
            class_names
        ))

if __name__ == "__main__":
    try:
        run_demo()
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise
