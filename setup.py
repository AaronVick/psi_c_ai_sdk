from setuptools import setup, find_packages

setup(
    name="psi_c_ai_sdk",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "networkx>=2.6.0",
        "sentence-transformers>=2.2.0",
        "scikit-learn>=1.0.0",
    ],
    author="ΨC-AI Team",
    author_email="info@psi-c-ai.org",
    description="ΨC-AI SDK: A cognitive framework for building self-reflective AI systems",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/psi-c-ai/psi-c-ai-sdk",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
) 