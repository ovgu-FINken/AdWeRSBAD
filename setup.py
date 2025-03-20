#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="adwersbad",
    version="0.1",
    packages=find_packages(),
    package_data={
        "": ["mydatabase.ini"],
    },
    data_files=["mydatabase.ini"],
    description="AdWeRSBAD: Adverse Weather and Road Scenarios Benchmark for Autonomous Driving",
    install_requires=[
        "torch",
        "torchvision",
        "torchaudio",
        "psycopg",
        "psycopg-binary",
        "psycopg-pool",
        "tqdm",
        "distinctipy",
    ],
    author="Dominik Weikert",
    author_email="dominik.weikert@ovgu.de",
    python_requires=">=3.8",
)
