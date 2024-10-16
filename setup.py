from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()
setup(
    "identity_cluster",
    version="0.1.0",
    packages=find_packages(),
    install_requires = required,
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    github_url="https://github.com/phospheneai/pkg-identity-clustering",
    author="Sanjith Kumar @ Phospheneai",
    author_email="sanjith.kumar@phospheneai.com",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.11'
)