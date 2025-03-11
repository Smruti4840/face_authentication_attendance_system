#!/usr/bin/env bash
# Install system dependencies
apt-get update && apt-get install -y cmake g++ libopenblas-dev liblapack-dev libx11-dev libgtk2.0-dev

# Install Python dependencies
pip install -r requirements.txt
