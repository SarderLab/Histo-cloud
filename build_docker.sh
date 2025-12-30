#!/bin/bash
# Build Docker image for HistomicsTK with TensorFlow 2.x

set -e

# Configuration
IMAGE_NAME="histomicstk-tf2"
IMAGE_TAG="latest"
DOCKERFILE="Dockerfile"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Building HistomicsTK Docker Image${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo -e "Image: ${YELLOW}${IMAGE_NAME}:${IMAGE_TAG}${NC}"
echo -e "Dockerfile: ${YELLOW}${DOCKERFILE}${NC}"
echo ""

# Check if Dockerfile exists
if [ ! -f "$DOCKERFILE" ]; then
    echo -e "${RED}Error: Dockerfile not found!${NC}"
    exit 1
fi

# Build the Docker image
echo -e "${GREEN}Starting Docker build...${NC}"
echo ""

docker build \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    -f "${DOCKERFILE}" \
    . 2>&1 | tee docker_build.log

# Check if build was successful
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo -e "${GREEN}======================================${NC}"
    echo -e "${GREEN}Docker build completed successfully!${NC}"
    echo -e "${GREEN}======================================${NC}"
    echo ""
    echo -e "Image: ${YELLOW}${IMAGE_NAME}:${IMAGE_TAG}${NC}"
    echo ""
    echo "You can run the container with:"
    echo -e "${YELLOW}docker run --gpus all -it ${IMAGE_NAME}:${IMAGE_TAG} /bin/bash${NC}"
    echo ""
    echo "Build log saved to: docker_build.log"
else
    echo ""
    echo -e "${RED}======================================${NC}"
    echo -e "${RED}Docker build failed!${NC}"
    echo -e "${RED}======================================${NC}"
    echo ""
    echo "Check docker_build.log for details"
    exit 1
fi
