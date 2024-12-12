import os
import pickle

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.backends import default_backend

from .params import PiecewiseApprox
from .gates import SecureGate, GarbledGate
from ..utils.conversion import create_sigmoid_approx

class SecureLFECircuit:
    """Circuit implementation for BLR with integrity verification"""

    def __init__(self, n_features, params):
        self.n_features = n_features
        self.params = params
        self.gates = []
        self.wire_labels = {}

        breakpoints, coeffs = create_sigmoid_approx(params.sigmoid_pieces)
        self.sigmoid_approx = PiecewiseApprox(breakpoints.tolist(), coeffs.tolist())

        self._setup_security()

    def _setup_security(self):
        """Initialize security components"""
        self.circuit_hash = None
        self.proof_salt = os.urandom(32)
        self.crs = self._generate_crs()

    def _generate_crs(self):
        """Generate common reference string"""
        entropy = os.urandom(64)
        kdf = HKDF(
            algorithm=hashes.SHA3_256(),
            length=self.params.security_parameter,
            salt=None,
            info=b"LFE-CRS-Generation",
            backend=default_backend(),
        )
        return kdf.derive(entropy)

    def verify_circuit_integrity(self):
        """Verify circuit hasn't been tampered with"""
        if self.circuit_hash is None:
            return False

        current_hash = hashes.Hash(hashes.SHA3_256())
        for gate in self.gates:
            gate_data = pickle.dumps((gate.type, gate.input_wires, gate.output_wire))
            current_hash.update(gate_data)
        return current_hash.finalize() == self.circuit_hash

    def add_input_gate(self, bits):
        """Add input gate with specified bit width"""
        wire_id = len(self.wire_labels)
        self.wire_labels[wire_id] = [os.urandom(16) for _ in range(1 << bits)]
        return wire_id

    def add_gate(self, gate, input_wires):
        """Add gate to circuit"""
        for wire in input_wires:
            if wire not in self.wire_labels:
                raise ValueError(f"Wire {wire} not found in wire labels")

        gate.input_wires = input_wires.copy()
        output_wire = len(self.wire_labels)
        gate.output_wire = output_wire

        self.wire_labels[output_wire] = [os.urandom(16) for _ in range(2)]

        input_labels = []
        for wire in input_wires:
            input_labels.append(self.wire_labels[wire])

        gate.garble(input_labels, self.wire_labels[output_wire])
        self.gates.append(gate)

        return output_wire

    def add_multiplication_gate(self, input_wire, weight):
        """Add multiplication gate"""
        print(f"\nAdding MUL gate with weight {weight}")
        gate = SecureGate("MUL", self.params.fixed_point_bits, 1)
        gate.weight = weight
        return self.add_gate(gate, [input_wire])

    def add_addition_gate(self, input_wires):
        """Add addition gate"""
        print(f"\nAdding ADD gate for {len(input_wires)} inputs")
        gate = SecureGate("ADD", self.params.fixed_point_bits, 1)
        return self.add_gate(gate, input_wires)

    def add_activation_circuit(self, input_wire):
        """Add sigmoid activation using comparison gates"""
        print("\nAdding activation circuit...")

        cmp_gate = SecureGate("CMP", self.params.fixed_point_bits, 1)
        cmp_gate.threshold = 0
        return self.add_gate(cmp_gate, [input_wire])

    def create_logistic_circuit(self, weights):
        """Create full logistic regression circuit"""
        if len(weights) != self.n_features:
            raise ValueError("Number of weights must match number of features")

        input_wires = []
        for _ in range(self.n_features):
            wire = self.add_input_gate(self.params.fixed_point_bits)
            input_wires.append(wire)

        product_wires = []
        for in_wire, weight in zip(input_wires, weights):
            wire = self.add_multiplication_gate(in_wire, weight)
            product_wires.append(wire)

        sum_wire = self.add_addition_gate(product_wires)

        return self.add_activation_circuit(sum_wire)

    def get_circuit_stats(self):
        """Get statistics about the circuit"""
        gate_types = {}
        for gate in self.gates:
            gate_types[gate.type] = gate_types.get(gate.type, 0) + 1

        return {
            'total_gates': len(self.gates),
            'gate_types': gate_types,
            'total_wires': len(self.wire_labels),
            'input_wires': self.n_features,
            'fixed_point_bits': self.params.fixed_point_bits,
        }

class SmallLFECircuit:
    """Circuit implementation for Mini BLR LFE"""

    def __init__(self, params):
        self.params = params
        self.gates = []
        self.wire_labels = {}

        self.crs = self._generate_crs()

        self.aux_info = {}

        self._setup_zk_system()

    def _generate_crs(self):
        """Generate common reference string"""
        entropy = os.urandom(64)
        kdf = HKDF(
            algorithm=hashes.SHA3_256(),
            length=self.params.security_parameter,
            salt=None,
            info=b"LFE-CRS-Generation",
            backend=default_backend()
        )
        return kdf.derive(entropy)

    def _setup_zk_system(self):
        """Setup zero-knowledge proving system"""
        self.zk_generators = []
        for _ in range(self.params.commitment_rounds):
            g = os.urandom(32)
            self.zk_generators.append(g)

    def verify_circuit_integrity(self):
        """Verify circuit hasn't been tampered with"""
        current_hash = hashes.Hash(hashes.SHA3_256())

        for gate in self.gates:
            gate_data = pickle.dumps((gate.type, gate.input_wires, gate.output_wire))
            current_hash.update(gate_data)

        return current_hash.finalize() == self.circuit_hash

    def check_overflow(self, x, y):
        """Check for overflow in fixed-point multiplication"""
        product = x * y
        max_val = (1 << (2 * self.params.fixed_point_bits)) - 1
        return product > max_val

    def add_input_gate(self, bit_width):
        """Add input gate with specified bit width"""
        wire_ids = []
        for _ in range(bit_width):
            wire_id = len(self.wire_labels)
            self.wire_labels[wire_id] = (os.urandom(16), os.urandom(16))
            wire_ids.append(wire_id)
        return wire_ids

    def add_multiplication_gate(self, input1_wires, weight):
        """Add a multiplication gate with a constant weight"""
        output_wires = []
        for i in range(self.params.fixed_point_bits):
            gate = GarbledGate("MUL", self.params)
            gate.input_wires = [input1_wires[i]]
            gate.output_wire = len(self.wire_labels)
            gate.weight = weight
            self.wire_labels[gate.output_wire] = (os.urandom(16), os.urandom(16))
            self.gates.append(gate)
            output_wires.append(gate.output_wire)
        return output_wires
