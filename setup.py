#!/usr/bin/env python

# -*- encoding: utf-8

'''
_____.___._______________  __.____ __________    _________   ___ ___    _____  .___ 
\__  |   |\_   _____/    |/ _|    |   \      \   \_   ___ \ /   |   \  /  _  \ |   |
 /   |   | |    __)_|      < |    |   /   |   \  /    \  \//    ~    \/  /_\  \|   |
 \____   | |        \    |  \|    |  /    |    \ \     \___\    Y    /    |    \   |
 / ______|/_______  /____|__ \______/\____|__  /  \______  /\___|_  /\____|__  /___|
 \/               \/        \/               \/          \/       \/         \/     
 
@author: Yekun Chai
@license: CYK
@email: chaiyekun@gmail.com
@file: setup.py
@time: 1/15/21 3:35 PM 
@desc： 
               
'''

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="embedding4bert",
    version="0.0.4",
    author="cyk1337",
    author_email="chaiyekun@gmail.com",
    description="A package for extracting word representations from BERT/XLNet",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cyk1337/embedding4bert",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "transformers",
        "torch",
        "sentencepiece",
    ]
)