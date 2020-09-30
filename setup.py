#!/usr/bin/env python3

from distutils.core import setup
from distutils.cmd import Command

module_name = "arcnn"
version = "1.0.0"

setup(
    name=module_name,
    version=version,
    url="https://github.com/shizacat/ARCNN-pytorch.git",
    author="Matveev Alexey",
    author_email="a.matveev",
    description="""PyTorch implementation ARCNN""",
    packages=["arcnn", "arcnn.model"],
    package_dir={"arcnn": "arcnn"},
    install_requires=[],
    # scripts=[],
)
