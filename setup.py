"""
URI-Transformer - Universal Reality Interface
Setup Configuration
"""

from setuptools import setup, find_packages
from pathlib import Path

readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="uri-transformer",
    version="1.0.0",
    author="BruinGrowly",
    description="Revolutionary AI architecture preserving semantic meaning through 4D coordinate system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BruinGrowly/URI_Transformer",
    project_urls={
        "Bug Tracker": "https://github.com/BruinGrowly/URI_Transformer/issues",
        "Documentation": "https://github.com/BruinGrowly/URI_Transformer/tree/main/docs",
        "Source Code": "https://github.com/BruinGrowly/URI_Transformer",
        "Semantic Substrate": "https://github.com/BruinGrowly/Semantic-Substrate-Engine",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*", "tools", "tools.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'sympy>=1.9',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
            'pylint>=2.12.0',
            'black>=22.0.0',
        ],
        'viz': [
            'matplotlib>=3.4.0',
        ],
    },
    keywords=[
        'transformer',
        'semantic-analysis',
        'nlp',
        'artificial-intelligence',
        'semantic-preservation',
        'uri-transformer',
        'meaning-preservation',
        'ai-safety',
        'ethical-ai',
    ],
    license="MIT",
    include_package_data=True,
    zip_safe=False,
)
