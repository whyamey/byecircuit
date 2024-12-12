import os
import pickle
import time

import numpy as np

from ..crypto.commitment import HashBasedCommitment
from ..crypto.proof import ZKProofSystem

class LFEScheme:
    """Main LFE scheme implementation for secure logistic regression"""

    def __init__(self, params):
        """Initialize LFE scheme"""
        self.params = params
        self.commitment = HashBasedCommitment(params)
        self.proof_system = ZKProofSystem(params)

    def setup(self, circuit):
        """Generate digest and secret key for the circuit"""
        circuit_bytes = pickle.dumps(circuit)
        digest, key, _ = self.commitment.commit(circuit_bytes)
        return digest, key

    def compress(self, circuit, weights):
        """Compress circuit for evaluation"""
        compressed = {
            "wire_labels": circuit.wire_labels,
            "gates": []
        }

        for gate in circuit.gates:
            gate_data = {
                "type": gate.type,
                "input_wires": gate.input_wires.copy(),
                "output_wire": gate.output_wire,
                "garbled_table": [{
                    'ciphertext': entry['ciphertext'],
                    'tag': entry['tag'],
                    'input_val': entry.get('input_val', None),
                    'input_vals': entry.get('input_vals', None),
                    'output_val': entry['output_val']
                } for entry in gate.garbled_table],
                "nonce": gate.nonce
            }

            if gate.type == "MUL":
                gate_data["weight"] = gate.weight
            elif gate.type == "CMP":
                gate_data["threshold"] = gate.threshold

            compressed["gates"].append(gate_data)

        return pickle.dumps(compressed)

    def prove_weights_valid(self, weights):
        """Generate ZK proof for weight validity"""
        return self.proof_system.prove_weights_valid(weights)

    def verify_weights_proof(self, proof, commitment):
        """Verify ZK proof of weight validity"""
        return self.proof_system.verify_weights_proof(proof, commitment)

    def evaluate(self, circuit, compressed_circuit, inputs, benchmark=False):
        """Evaluate the compressed circuit"""
        start_time = time.time()

        print("\nInput values:")
        for i, val in enumerate(inputs):
            sign = "-" if val < 0 else "+"
            abs_val = abs(val)
            print(f"Input {i}: {val} ({sign}{abs_val})")

        try:
            circuit_data = pickle.loads(compressed_circuit)
            wire_values = {}
            wire_labels = {}

            for i, input_val in enumerate(inputs):
                wire_id = i
                sign_bit = 1 if input_val < 0 else 0
                abs_val = abs(input_val)
                bit = abs_val & 1
                wire_labels[wire_id] = circuit_data['wire_labels'][wire_id][bit]
                wire_values[wire_id] = bit | (sign_bit << (circuit.params.fixed_point_bits - 1))

            for i, gate_data in enumerate(circuit_data["gates"]):
                input_wire_labels = [wire_labels[w] for w in gate_data["input_wires"]]
                input_vals = [wire_values[w] for w in gate_data["input_wires"]]

                print(f"\nGate {i} ({gate_data['type']}) inputs: {input_vals}")

                from .gates import SecureGate
                gate = SecureGate(
                    gate_data["type"],
                    circuit.params.fixed_point_bits,
                    circuit.params.fixed_point_bits,
                )
                gate.garbled_table = gate_data["garbled_table"]
                gate.nonce = gate_data["nonce"]

                if gate_data["type"] == "MUL":
                    gate.weight = gate_data["weight"]
                elif gate_data["type"] == "CMP":
                    gate.threshold = gate_data["threshold"]

                output_label, output_val = gate.evaluate(input_wire_labels)
                wire_labels[gate_data["output_wire"]] = output_label
                wire_values[gate_data["output_wire"]] = output_val

                print(f"Gate {i} output: {output_val}")

            final_wire = circuit_data["gates"][-1]["output_wire"]
            final_val = wire_values[final_wire]

            result = 0.99 if final_val else 0.01

            eval_time = time.time() - start_time

            if benchmark:
                print(f"\nEvaluation completed in {eval_time:.2f} seconds")
                print(f"Final value: {final_val}")
                print(f"Result probability: {result:.4f}")

            return result, eval_time

        except Exception as e:
            print(f"Error in evaluation: {str(e)}")
            raise

    def benchmark_inference(self, circuit, compressed_circuit, test_inputs, num_trials=5):
        """Benchmark circuit inference"""
        times = []
        results = []

        for i, inputs in enumerate(test_inputs):
            print(f"\nBenchmarking input {i+1}/{len(test_inputs)}")
            trial_times = []
            trial_results = []

            for j in range(num_trials):
                print(f"Trial {j+1}/{num_trials}")
                try:
                    result, eval_time = self.evaluate(
                        circuit, compressed_circuit, inputs, benchmark=True
                    )
                    trial_times.append(eval_time)
                    trial_results.append(result)
                except Exception as e:
                    print(f"Error in trial {j+1}: {str(e)}")
                    continue

            if trial_times:
                times.extend(trial_times)
                results.extend(trial_results)

        if times:
            avg_time = np.mean(times)
            std_time = np.std(times)
            print(f"\nAverage inference time: {avg_time:.2f} ± {std_time:.2f} seconds")

            return {
                'mean_time': avg_time,
                'std_time': std_time,
                'n_successful': len(results),
                'n_total': len(test_inputs) * num_trials
            }
        else:
            return {
                'mean_time': None,
                'std_time': None,
                'n_successful': 0,
                'n_total': len(test_inputs) * num_trials
            }

class LegacyLFEScheme:
    """Legacy implementation of LFE scheme"""

    def __init__(self, params):
        """Initialize legacy LFE scheme"""
        self.params = params
        self.commitment = HashBasedCommitment(params)

        self.proof_salt = os.urandom(32)
        self.challenges = self._generate_challenges()

    def _generate_challenges(self):
        """Generate random challenges for ZK proofs"""
        challenges = []
        for _ in range(self.params.commitment_rounds):
            challenge = os.urandom(32)
            challenges.append(challenge)
        return challenges

    def setup(self, circuit):
        """Generate digest and secret key for the circuit"""
        circuit_bytes = pickle.dumps(circuit)
        digest, key, _ = self.commitment.commit(circuit_bytes)
        return digest, key

    def compress(self, circuit, weights):
        """Compress circuit for evaluation"""
        compressed = []

        for gate in circuit.gates:
            if gate.type == "MUL":
                input_label = circuit.wire_labels[gate.input_wires[0]][0]
                output_label = circuit.wire_labels[gate.output_wire][0]

                ct, tag = gate.garble(input_label, output_label)

                gate_data = {
                    'input_wire': gate.input_wires[0],
                    'output_wire': gate.output_wire,
                    'weight': int(gate.weight),
                    'ct': ct,
                    'tag': tag,
                    'nonce': gate.nonce
                }
                compressed.append(gate_data)

        return pickle.dumps(compressed)

    def evaluate(self, circuit, compressed_circuit, inputs):
        """Evaluate the compressed circuit with input values"""
        circuit_data = pickle.loads(compressed_circuit)
        wire_values = {}

        for i, input_val in enumerate(inputs):
            val = int.from_bytes(input_val, byteorder='big', signed=True)
            base_wire = i * 8
            for j in range(8):
                wire_values[base_wire + j] = val

        result = 0
        for gate_data in circuit_data:
            try:
                input_wire = gate_data['input_wire']

                if input_wire not in wire_values:
                    print(f"Warning: Missing value for wire {input_wire}")
                    continue

                input_val = wire_values[input_wire]
                weight = gate_data['weight']

                product = (input_val * weight) >> circuit.params.fixed_point_bits
                wire_values[gate_data['output_wire']] = product

                if gate_data['output_wire'] % 8 == 0:
                    result += product

            except Exception as e:
                print(f"Gate evaluation error for wire {input_wire}: {str(e)}")
                continue

        return result

    def _commit_weight(self, weight, randomness):
        """Create a Pedersen-style commitment to a weight value"""
        h = hashes.Hash(hashes.SHA3_256())
        h.update(str(weight).encode() + randomness)
        return h.finalize()

    def _prove_response(self, weight, randomness, challenge):
        """Generate response for the ZK proof"""
        h = hashes.Hash(hashes.SHA3_256())
        h.update(str(weight).encode() + randomness + challenge)
        return h.finalize()

    def _verify_response(self, commitment, response, challenge):
        """Verify a response in the ZK proof"""
        h = hashes.Hash(hashes.SHA3_256())
        h.update(commitment + challenge)
        expected = h.finalize()

        return hmac.HMAC.compare_digest(response, expected)

    def prove_weights_valid(self, weights):
        """Generate ZK proof that weights are well-formed"""
        proof = b""

        # Simplified Sigma protocol
        for i, w in enumerate(weights):
            r = os.urandom(32)
            commitment = self._commit_weight(w, r)

            e = self.challenges[i % len(self.challenges)]

            response = self._prove_response(w, r, e)

            proof += commitment + response

        return proof

    def verify_weights_proof(self, proof, commitment):
        """Verify ZK proof of weights validity"""
        try:
            chunks = [proof[i:i+64] for i in range(0, len(proof), 64)]
            print("Debug: Number of proof chunks:", len(chunks))
            print("Debug: Commitment size:", len(commitment))

            for i, (comm, resp) in enumerate(zip(chunks[::2], chunks[1::2])):
                e = self.challenges[i % len(self.challenges)]
                result = self._verify_response(comm, resp, e)
                print(f"Debug: Chunk {i} verification result:", result)
                if not result:
                    return False
            return True
        except Exception as e:
            print(f"Debug: Exception in verify_weights_proof: {str(e)}")
            return False
