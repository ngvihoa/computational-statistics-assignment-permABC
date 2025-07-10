"""
Setup script for permABC package.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    """Read README.md file if it exists."""
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Permutation-enhanced Approximate Bayesian Computation"

# Read requirements
def read_requirements():
    """Read requirements from requirements.txt if it exists."""
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return [
        'jax>=0.4.0',
        'jaxlib>=0.4.0', 
        'numpy>=1.20.0',
        'scipy>=1.7.0',
        'matplotlib>=3.3.0',
        'seaborn>=0.11.0',
        'tqdm>=4.60.0'
    ]

setup(
    name="permabc",
    version="0.1.0",
    author="Antoine Luciano",
    author_email="luciano@ceremade.dauphine.fr",
    description="Permutation-enhanced Approximate Bayesian Computation",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/permABC",
    packages=find_packages(exclude=["experiments*", "tests*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.10",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "optional": [
            "pymc>=5.0.0",
            "psutil>=5.8.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "permabc-figures=experiments.scripts.comparison.run_all_figures:main",
        ],
    },
    include_package_data=True,
    package_data={
        "permabc": ["*.txt", "*.md"],
    },
    zip_safe=False,
)