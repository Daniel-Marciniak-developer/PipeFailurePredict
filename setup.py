from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="pipefailurepredict",
    version="0.1.0",
    author="WaterPrime Team",
    author_email="contact@waterprime.example.com",
    description="A package for predicting water pipe failures",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/PipeFailurePredict--WaterPrime",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "flow-simulation=FlowAlgorithm.main:main",
            "visualize-network=FlowAlgorithm.visualize_network:main",
        ],
    },
)
