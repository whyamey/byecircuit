from dataclasses import dataclass

@dataclass
class LFEParams:
    """Parameters for the LFE scheme configuration"""

    security_parameter: int = 128
    fixed_point_bits: int = 8
    circuit_depth: int = 16
    commitment_rounds: int = 40
    hash_function: str = "SHA3-256"
    aead_mode: str = "GCM"
    kdf_info: bytes = b"LFE-LogisticRegression-v1"
    sigmoid_pieces: int = 8

    def __post_init__(self):
        """Validate parameter constraints"""
        if self.security_parameter < 128:
            raise ValueError("Security parameter must be at least 128 bits")
        if self.commitment_rounds < 40:
            raise ValueError("Need at least 40 commitment rounds for 2^-40 soundness")
        if self.sigmoid_pieces < 4:
            raise ValueError("Need at least 4 pieces for sigmoid approximation")

@dataclass
class PiecewiseApprox:
    """Piecewise polynomial approximation of sigmoid function"""

    breakpoints: list
    coefficients: list

    def evaluate(self, x):
        """Evaluate the piecewise approximation at a point"""
        piece_index = 0
        for i, bp in enumerate(self.breakpoints[1:]):
            if x <= bp:
                break
            piece_index = i

        coeffs = self.coefficients[piece_index]
        result = coeffs[-1]
        x_shift = x - self.breakpoints[piece_index]

        for c in coeffs[-2::-1]:
            result = result * x_shift + c

        return result
