#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="robot",
    version="0.0.1",
    description="Describe Your Cool Project",
    author="I Wayan Wiprayoga Wisesa",
    author_email="iwawiwi@gmail.com",
    url="https://github.com/iwawiwi/literate-octo-robot",  # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    install_requires=["pytorch-lightning", "hydra-core"],
    packages=find_packages(),
)
