"""
Setup configuration for GammaNeutral package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="gamma-neutral",
    version="0.1.0",
    author="jbarrerobuch",
    description="A gamma neutral trading strategy for crypto markets using options and perpetual futures",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jbarrerobuch/GammaNeutral",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    keywords="trading, options, futures, gamma-neutral, crypto, cryptocurrency, hedging, derivatives",
)
