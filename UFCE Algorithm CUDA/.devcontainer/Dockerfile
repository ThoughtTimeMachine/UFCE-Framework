# Base Image: NVIDIA CUDA 12.4 (Supports RTX 4070 Ti / sm_89)
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install core C++ build tools (No Python needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    cmake \
    gdb \
    && rm -rf /var/lib/apt/lists/*

# Set workspace
WORKDIR /workspace