#!/bin/bash

# Configuration
IMAGE="nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc6"
WORKSPACE_DIR="$(pwd)"

echo "Launching Docker container with image: $IMAGE"
echo "Mounting $WORKSPACE_DIR to /workspace"

# Check if the user has access to the docker daemon
if docker ps >/dev/null 2>&1; then
    echo "Docker daemon access detected. Using Docker..."
    docker run -it --rm \
        --gpus all \
        --ipc=host \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        --shm-size=16g \
        -v "$WORKSPACE_DIR":/workspace \
        -w /workspace \
        "$IMAGE" \
        /bin/bash
else
    echo "No Docker daemon access detected. Falling back to Singularity..."
    # Singularity will handle caching the image in ~/.singularity/cache

    # Create the temporary working directory for Singularity to use
    WORKDIR="/tmp/singularity_workdir_$$"
    mkdir -p "$WORKDIR"

    singularity shell --nv \
        --bind "$WORKSPACE_DIR":/workspace \
        --pwd /workspace \
        --containall \
        --no-home \
        --workdir "$WORKDIR" \
        docker://"$IMAGE"

    # Clean up the workdir after exiting
    rm -rf "$WORKDIR"
fi
