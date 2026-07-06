from setuptools import setup, find_packages

setup(
    name="docking-engines",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "docking_engines": ["engines/*"],  
    },
    entry_points={
        "console_scripts": [
            "docking = docking_engines.cli:main",  # Global command: `docking`
        ],
    },
    author="Ilkwon Cho",
    author_email="ilkwonc@andrew.cmu.edu",
    description="A collection of molecular docking engines",
    #url="https://github.com/yourusername/docking_engines",
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
    ],
)
