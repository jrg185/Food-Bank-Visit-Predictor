#!/bin/bash
# Upgrade pip and setuptools to ensure distutils is available
pip install --upgrade pip setuptools

# Install the necessary dependencies from requirements.txt
pip install -r requirements.txt

