from setuptools import setup, find_packages

setup(
    name="nyaya-env",
    version="1.0.0",
    author="Swarup",
    description="India's First Judicial System RL Environment",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=0.29.0",
        "numpy>=1.24.0",
        "stable-baselines3>=2.1.0",
        "matplotlib>=3.7.0",
    ],
    python_requires=">=3.9",
)
