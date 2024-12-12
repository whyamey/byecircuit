import numpy as np

def sigmoid(x):
    """Compute sigmoid function with numerical stability"""
    return 1 / (1 + np.exp(-np.clip(x, -100, 100)))

def convert_to_fixed_point(values, bits):
    """Convert floating point values to fixed point representation"""
    scale = 1 << bits
    if isinstance(values, list):
        values = np.array(values)

    if len(values.shape) > 1:
        rows, cols = values.shape
        result = np.zeros((rows, cols), dtype=np.int32)
        for i in range(rows):
            for j in range(cols):
                result[i, j] = int(values[i, j] * scale)
        return result
    else:
        return np.array([int(v * scale) for v in values], dtype=np.int32)

def fixed_to_float(values, bits):
    """Convert fixed point values back to floating point"""
    scale = 1 << bits
    if isinstance(values, list):
        values = np.array(values)
    return values.astype(float) / scale

def create_sigmoid_approx(num_pieces=8):
    """Create piecewise polynomial approximation of sigmoid"""
    breakpoints = np.linspace(-6, 6, num_pieces + 1)
    coefficients = []

    for i in range(num_pieces):
        x_piece = np.linspace(breakpoints[i], breakpoints[i + 1], 100)
        y_piece = sigmoid(x_piece)
        coeffs = np.polyfit(x_piece - breakpoints[i], y_piece, 3)
        coefficients.append(coeffs)

    return breakpoints, np.array(coefficients)
