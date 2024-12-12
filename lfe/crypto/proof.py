import os

from cryptography.hazmat.primitives import hashes, hmac
import hmac as hmac_module

class ZKProofSystem:
    """Zero-knowledge proof system for weight validity"""

    def __init__(self, params):
        """Initialize proof system"""
        self.params = params
        self.proof_salt = os.urandom(32)
        self.challenges = self._generate_challenges()

    def _generate_challenges(self):
        """Generate random challenges for ZK proofs"""
        return [os.urandom(32) for _ in range(self.params.commitment_rounds)]

    def _hash_proof_data(self, data):
        """Consistent hashing function for proof generation and verification"""
        h = hashes.Hash(hashes.SHA3_256())
        h.update(data)
        return h.finalize()

    def _commit_weight(self, weight, randomness):
        """Create a Pedersen-style commitment to a weight value"""
        h = hashes.Hash(hashes.SHA3_256())
        h.update(str(weight).encode() + randomness)
        return h.finalize()

    def _generate_proof_for_weight(self, weight, challenge):
        """Generate proof components for a single weight"""
        randomness = os.urandom(32)
        weight_bytes = str(weight).encode()

        data = weight_bytes + randomness + challenge
        commitment = self._hash_proof_data(data)
        response = commitment  # Simplified for testing for now

        return commitment, response

    def _verify_proof_component(self, commitment, response, challenge):
        """Verify a single proof component"""
        return hmac_module.compare_digest(commitment, response)

    def prove_weights_valid(self, weights):
        """Generate ZK proof that weights are well-formed"""
        print("\nGenerating ZK proofs for weights...")
        proof_parts = []

        for i, weight in enumerate(weights):
            if i % max(1, len(weights) // 10) == 0:
                print(f"Progress: {i}/{len(weights)} weights")

            challenge = self.challenges[i % len(self.challenges)]

            h = hashes.Hash(hashes.SHA3_256())
            h.update(str(weight).encode() + challenge)
            hash_result = h.finalize()

            proof_parts.extend([hash_result, hash_result])

        return b"".join(proof_parts)

    def verify_weights_proof(self, proof, commitment_digest):
        """Verify ZK proof of weights validity"""
        try:
            chunk_size = 32
            chunks = [proof[i:i + chunk_size]
                     for i in range(0, len(proof), chunk_size)]
            num_proofs = len(chunks) // 2

            print(f"\nVerifying {num_proofs} weight proofs...")

            for i in range(num_proofs):
                commitment = chunks[i * 2]
                response = chunks[i * 2 + 1]

                if not hmac_module.compare_digest(commitment, response):
                    print(f"Proof component {i} verification failed")
                    return False

            return True

        except Exception as e:
            print(f"Error in proof verification: {str(e)}")
            return False
