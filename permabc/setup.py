"""Setup script for permABC package."""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "Permutation-based Approximate Bayesian Computation"

# Read requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(req_path):
        with open(req_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return []

setup(
    name="permabc",
    version="0.1.0",
    description="Permutation-based Approximate Bayesian Computation",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    
    # Author information
    author="Your Name",  # TODO: Remplacer par votre nom
    author_email="your.email@institution.edu",  # TODO: Remplacer par votre email
    
    # URLs
    url="https://github.com/yourusername/permABC",  # TODO: Remplacer par votre repo
    project_urls={
        "Bug Reports": "https://github.com/yourusername/permABC/issues",
        "Source": "https://github.com/yourusername/permABC",
        "Documentation": "https://permabc.readthedocs.io/",  # Si vous créez une doc
    },
    
    # Package configuration
    packages=find_packages(exclude=["tests*", "experiments*", "docs*"]),
    include_package_data=True,
    
    # Dependencies
    install_requires=read_requirements(),
    
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black>=21.0",
            "flake8>=3.8",
            "isort>=5.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme",
            "numpydoc",
            "matplotlib",  # Pour les plots dans la doc
        ],
        "notebooks": [
            "jupyter>=1.0",
            "ipykernel",
            "seaborn",  # Pour de beaux plots
        ],
        "optimization": [
            "jax>=0.3.0",
            "jaxlib>=0.3.0",
        ],
    },
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Package metadata
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",  # TODO: Choisir votre licence
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    
    # Keywords for discoverability
    keywords="bayesian computation, abc, permutation, statistics, inference",
    
    # Entry points (si vous voulez des commandes CLI)
    entry_points={
        "console_scripts": [
            # "permabc-run=permabc.cli:main",  # Optionnel: interface ligne de commande
        ],
    },
    
    # Package data
    package_data={
        "permabc": ["data/*.csv", "*.txt"],  # Si vous avez des données dans le package
    },
    
    # Zip safe
    zip_safe=False,
)