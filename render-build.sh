#!/bin/bash

# Install dependencies required for dlib
apt-get update && apt-get install -y cmake gcc g++ make libboost-all-dev

# Install Python dependencies
pip install -r requirements.txt
