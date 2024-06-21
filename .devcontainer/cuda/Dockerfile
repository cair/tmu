FROM nvcr.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Run GIT
RUN apt-get update && apt-get install -y git python3 python3-pip

# Install Dependencies
RUN pip3 install cffi numpy pandas scikit-learn tqdm requests xxhash