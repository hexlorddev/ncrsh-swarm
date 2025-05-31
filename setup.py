#!/usr/bin/env python3
"""
ncrsh-Swarm: A Neural Network Framework That Self-Clones Across Systems
========================================================================

A revolutionary distributed neural network framework that combines P2P networking 
with cooperative learning. Like BitTorrent + Transformers.

Features:
- Self-replicating neural network nodes
- P2P discovery and mesh networking
- Distributed training protocols
- Cooperative gradient sharing
- Fault-tolerant swarm intelligence
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ncrsh-swarm",
    version="0.1.0",
    author="Dineth Nethsara",
    author_email="hexlorddev@gmail.com",
    description="A Neural Network Framework That Self-Clones Across Systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hexlorddev/ncrsh-swarm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "asyncio-dgram>=2.1.0",
        "aiohttp>=3.8.0",
        "cryptography>=3.4.0",
        "msgpack>=1.0.0",
        "psutil>=5.8.0",
        "networkx>=2.6.0",
        "websockets>=10.0",
        "uvloop>=0.17.0; sys_platform != 'win32'",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.910",
        ],
        "viz": [
            "matplotlib>=3.5.0",
            "plotly>=5.0.0",
            "dash>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ncrsh-swarm=ncrsh_swarm.cli:main",
            "swarm-node=ncrsh_swarm.node:main",
            "swarm-dashboard=ncrsh_swarm.dashboard:main",
        ],
    },
)