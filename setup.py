from setuptools import setup, find_packages

setup(
    name="lfe",
    version="0.1.0",
    description="Compiling bayesian models to circuis for LFE",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scikit-learn>=0.24.0",
        "cryptography>=3.4.7"
    ],
)
