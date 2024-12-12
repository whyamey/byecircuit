import os

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

class SecureGate:
    """Base class for secure gate implementation"""

    def __init__(self, gate_type, input_bits, output_bits):
        """Initialize gate with gate type and bit widths"""
        self.type = gate_type
        self.input_bits = input_bits
        self.output_bits = output_bits
        self.nonce = os.urandom(12)
        self.garbled_table = []
        self.input_wires = []
        self.output_wire = None
        self.weight = None
        self.threshold = None

    def _compute_mul_output(self, input_val):
        """Compute multiplication gate output with sign handling"""
        signed_input = (input_val - (1 << (self.input_bits - 1))
                       if input_val >= (1 << (self.input_bits - 1))
                       else input_val)
        if self.weight is None:
            raise ValueError("Weight not set for MUL gate")
        result = signed_input * self.weight
        return (result >> self.input_bits) & 1

    def _compute_add_output(self, input_vals):
        """Compute addition gate output"""
        result = sum(input_vals)
        mask = (1 << self.output_bits) - 1
        return result & mask

    def _compute_cmp_output(self, input_val):
        """Compute comparison gate output"""
        if self.threshold is None:
            raise ValueError("Threshold not set for CMP gate")
        signed_val = (input_val - (1 << (self.input_bits - 1))
                     if input_val >= (1 << (self.input_bits - 1))
                     else input_val)
        return 1 if signed_val >= self.threshold else 0

    def garble(self, input_labels, output_labels):
        """Garble the gate"""
        print(f"Garbling {self.type} gate with {len(input_labels)} inputs")

        if self.type == "MUL":
            self._garble_mul(input_labels[0], output_labels)
        elif self.type == "ADD":
            self._garble_add(input_labels, output_labels)
        elif self.type == "CMP":
            self._garble_cmp(input_labels[0], output_labels)
        else:
            raise ValueError(f"Unknown gate type: {self.type}")

    def _garble_mul(self, input_label, output_labels):
        """Garble multiplication gate"""
        table_size = 2
        self.garbled_table = []
        print(f"Creating {table_size} entries for MUL gate (weight={self.weight})")

        sign_mask = 1 << (self.input_bits - 1)
        value_mask = sign_mask - 1

        for i in range(table_size):
            key = input_label[i][:16]
            cipher = Cipher(
                algorithms.AES(key),
                modes.GCM(self.nonce),
                backend=default_backend()
            )
            encryptor = cipher.encryptor()

            input_sign = -1 if i & sign_mask else 1
            input_val = i & value_mask
            weight_sign = -1 if self.weight & sign_mask else 1
            weight_val = self.weight & value_mask

            result = (input_val * weight_val * input_sign * weight_sign)
            output_val = 1 if result >= 0 else 0

            print(f"  MUL: input={input_val}({input_sign}), weight={weight_val}({weight_sign}), result={result}, output={output_val}")
            output_label = output_labels[output_val]

            ciphertext = encryptor.update(output_label) + encryptor.finalize()
            self.garbled_table.append({
                'ciphertext': ciphertext,
                'tag': encryptor.tag,
                'input_val': i,
                'output_val': output_val,
                'debug_result': result
            })

    def _garble_add(self, input_labels, output_labels):
        """Garble addition gate"""
        table_size = 2 ** len(input_labels)
        self.garbled_table = []
        print(f"Creating {table_size} entries for ADD gate")

        sign_mask = 1 << (self.input_bits - 1)
        value_mask = sign_mask - 1

        for i in range(table_size):
            input_bits = [(i >> j) & 1 for j in range(len(input_labels))]
            curr_labels = [input_labels[j][bit] for j, bit in enumerate(input_bits)]

            # XOR all input labels for key
            key = curr_labels[0][:16]
            for label in curr_labels[1:]:
                key = bytes(a ^ b for a, b in zip(key, label[:16]))

            cipher = Cipher(
                algorithms.AES(key),
                modes.GCM(self.nonce),
                backend=default_backend()
            )
            encryptor = cipher.encryptor()

            # Calculate sum handling signs
            total = 0
            for j, bit in enumerate(input_bits):
                if bit:
                    val = 1 << j
                    if j == len(input_bits) - 1:
                        val = -val
                    total += val

            output_val = 1 if total >= 0 else 0
            print(f"  ADD: inputs={input_bits}, sum={total}, output={output_val}")

            output_label = output_labels[output_val]
            ciphertext = encryptor.update(output_label) + encryptor.finalize()
            self.garbled_table.append({
                'ciphertext': ciphertext,
                'tag': encryptor.tag,
                'input_vals': input_bits,
                'output_val': output_val,
                'debug_sum': total
            })

    def _garble_cmp(self, input_label, output_labels):
        """Garble comparison gate"""
        table_size = 2
        self.garbled_table = []
        print(f"Creating {table_size} entries for CMP gate (threshold={self.threshold})")

        for i in range(table_size):
            key = input_label[i][:16]
            cipher = Cipher(
                algorithms.AES(key),
                modes.GCM(self.nonce),
                backend=default_backend()
            )
            encryptor = cipher.encryptor()

            val = i << self.input_bits
            output_val = 1 if val >= self.threshold else 0

            print(f"  CMP: input={val}, threshold={self.threshold}, output={output_val}")

            output_label = output_labels[output_val]
            ciphertext = encryptor.update(output_label) + encryptor.finalize()
            self.garbled_table.append({
                'ciphertext': ciphertext,
                'tag': encryptor.tag,
                'input_val': i,
                'output_val': output_val
            })

    def evaluate(self, input_labels):
        """Evaluate the gate with given input labels"""
        print(f"\nDebug gate evaluation for {self.type} gate:")
        print(f"Input labels: {[label.hex()[:16] for label in input_labels]}")
        print(f"Number of gate table entries: {len(self.garbled_table)}")

        # Generate decryption key based on input labels
        if len(input_labels) == 1:
            key = input_labels[0][:16]
        else:
            key = input_labels[0][:16]
            for label in input_labels[1:]:
                key = bytes(a ^ b for a, b in zip(key, label[:16]))

        # Try each table entry until one decrypts successfully
        for i, entry in enumerate(self.garbled_table):
            try:
                cipher = Cipher(
                    algorithms.AES(key),
                    modes.GCM(self.nonce, entry['tag']),
                    backend=default_backend()
                )
                decryptor = cipher.decryptor()
                output_label = decryptor.update(entry['ciphertext'])
                decryptor.finalize()

                output_val = entry['output_val']
                print(f"Successfully decrypted entry {i} (output value: {output_val})")
                return output_label, output_val

            except Exception as e:
                print(f"Failed to decrypt entry {i}")
                continue

        raise ValueError(f"No valid decryption found for {self.type} gate")

class GarbledGate:
    def __init__(self, gate_type, params):
        """Initialize garbled gate with gate type and parameters"""
        self.type = gate_type
        self.input_wires = []
        self.output_wire = None
        self.params = params
        self.nonce = os.urandom(12)

    def garble(self, input_label, output_label):
        """Garble the input and output labels"""
        cipher = Cipher(
            algorithms.AES(input_label),
            modes.GCM(self.nonce),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(output_label) + encryptor.finalize()
        return ciphertext, encryptor.tag

    def evaluate_with_carry(self, input0, input1, carry_in=False):
        """Evaluate gate with carry bit handling"""
        cipher = Cipher(
            algorithms.AES(input0),
            modes.GCM(self.nonce, self.table[1][0]),  # Use stored auth tag
            backend=default_backend()
        )
        decryptor = cipher.decryptor()

        decryptor.authenticate_additional_data(str(carry_in).encode())

        try:
            output = decryptor.update(self.table[0][0]) + decryptor.finalize()
            carry_out = (int.from_bytes(input0, 'big') +
                        int.from_bytes(input1, 'big') +
                        carry_in) >> self.params.fixed_point_bits
            return output, bool(carry_out)
        except:
            raise ValueError("Authentication failed - possible tampering detected")
