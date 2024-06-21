FROM nvcr.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y git python3 python3-pip

# Install tmu Dependencies
RUN pip3 install cffi numpy pandas scikit-learn tqdm requests xxhash

# Upgrade pip
RUN pip3 install --upgrade pip