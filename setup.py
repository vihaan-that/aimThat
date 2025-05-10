from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="aimThat",
    version="0.1.0",
    author="NoScope9000 Team",
    author_email="team@example.com",
    description="Machine learning component of the NoScope9000 sniper shot prediction system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/NoScope9000-ML-Analysis",
    project_urls={
        "Unity Component": "https://github.com/AbhiramKothagundu/NoScope9000",
        "Bug Tracker": "https://github.com/yourusername/NoScope9000-ML-Analysis/issues",
    },
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
    ],
    include_package_data=True,
)
