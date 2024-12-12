import os

from cryptography.hazmat.primitives import hashes, hmac
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.backends import default_backend

class HashBasedCommitment:
    """Hash-based commitment scheme for circuit integrity"""

    def __init__(self, params, key=None):
        """Initialize commitment scheme"""
        self.params = params
        self.nonce = os.urandom(16)

        if key is None:
            key = os.urandom(32)

        # Derive key using HKDF as better practice to
        kdf = HKDF(
            algorithm=hashes.SHA3_256(),
            length=32,
            salt=self.nonce,
            info=self.params.kdf_info,
            backend=default_backend(),
        )
        self.key = kdf.derive(key)

    def commit(self, message):
        """Create commitment to a message"""
        # Add randomness to prevent replay attacks, should be a group but will do for now
        r = os.urandom(32)

        # SHA3 has hardware implementation
        h = hashes.Hash(hashes.SHA3_256())
        h.update(self.key + r + message)
        commitment = h.finalize()

        h = hmac.HMAC(self.key, hashes.SHA3_256())
        h.update(commitment + r)
        tag = h.finalize()

        return commitment, tag, r

    def verify(self, message, commitment, tag, randomness):
        """Verify a commitment"""
        h = hashes.Hash(hashes.SHA3_256())
        h.update(self.key + randomness + message)
        expected_commitment = h.finalize()

        if not hmac.HMAC.compare_digest(commitment, expected_commitment):
            return False

        h = hmac.HMAC(self.key, hashes.SHA3_256())
        h.update(commitment + randomness)
        expected_tag = h.finalize()

        return hmac.HMAC.compare_digest(tag, expected_tag)
