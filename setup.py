import setuptools
import os

# Read the long description from README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="xtype",
    version="0.5.1",
    author="bitagoras",
    author_email="bitagoras@users.noreply.github.com",
    description="A Python library for serializing and deserializing data structures using the xtype binary format",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bitagoras/xtype-python",
    project_urls={
        "Bug Tracker": "https://github.com/bitagoras/xtype-python/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries",
        "Topic :: System :: Archiving",
    ],
    package_dir={"": "lib"},
    py_modules=["xtype"],
    packages=setuptools.find_packages(where="lib"),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.15.0",
    ],
)
