#!/bin/bash
# Redactify Docker Setup and Run Script
# This script helps with setting up and running the Redactify application using Docker

# Set colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print section header
print_header() {
  echo -e "\n${GREEN}==== $1 ====${NC}\n"
}

# Function to print error
print_error() {
  echo -e "\n${RED}ERROR: $1${NC}\n"
}

# Function to print warning
print_warning() {
  echo -e "\n${YELLOW}WARNING: $1${NC}\n"
}

# Function to check if Docker is running
check_docker() {
  if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running or not installed. Please start Docker and try again."
    exit 1
  fi
}

# Function to check if Docker Compose is installed
check_docker_compose() {
  if ! docker-compose --version > /dev/null 2>&1; then
    print_error "Docker Compose is not installed. Please install it and try again."
    exit 1
  fi
}

# Function to show help
show_help() {
  echo "Redactify Docker Setup and Run Script"
  echo 
  echo "Usage: ./redactify.sh [command]"
  echo 
  echo "Commands:"
  echo "  build         Build the Redactify Docker image (CPU version)"
  echo "  build-gpu     Build the Redactify Docker image (GPU version)"
  echo "  run           Run the Redactify stack using Docker Compose"
  echo "  run-detached  Run the Redactify stack in detached mode"
  echo "  stop          Stop the Redactify containers"
  echo "  clean         Remove all Redactify containers and images"
  echo "  status        Check the status of Redactify containers"
  echo "  logs          View logs for all containers (or specify service name)"
  echo "  help          Show this help message"
  echo
  echo "Examples:"
  echo "  ./redactify.sh build         # Build the CPU Docker image"
  echo "  ./redactify.sh run           # Run all services"
  echo "  ./redactify.sh logs web      # View logs for the web service"
}

# Check if we're in the right directory
if [ ! -d "Redactify" ] || [ ! -d "docker" ]; then
  print_error "This script must be run from the Redactify project root directory."
  exit 1
fi

# First argument is the command
command=$1
shift

# Additional arguments
service_name=$1

# Check if Docker is running before executing commands
check_docker

case $command in
  build)
    print_header "Building Redactify CPU Docker image"
    echo "This may take some time as it downloads and installs dependencies..."
    docker build -f Redactify/Dockerfile -t redactify-cpu .
    if [ $? -eq 0 ]; then
      echo -e "\n${GREEN}Successfully built Redactify CPU image.${NC}"
      echo -e "You can now run Redactify using: ./redactify.sh run"
    else
      print_error "Failed to build Docker image."
    fi
    ;;
    
  build-gpu)
    print_header "Building Redactify GPU Docker image"
    if ! command -v nvidia-smi &> /dev/null; then
      print_warning "NVIDIA drivers may not be installed or properly configured."
      echo "Continuing with build, but GPU support may not work."
    fi
    
    echo "This may take some time as it downloads and installs dependencies..."
    docker build -f Redactify/Dockerfile.gpu -t redactify-gpu .
    if [ $? -eq 0 ]; then
      echo -e "\n${GREEN}Successfully built Redactify GPU image.${NC}"
      echo -e "You need to create a docker-compose.override.yml file for GPU support."
      echo -e "See the DOCKER_SETUP.md file for instructions."
    else
      print_error "Failed to build Docker image."
    fi
    ;;
    
  run)
    print_header "Running Redactify services"
    check_docker_compose
    
    # Check if Redis is already running on port 6379
    if nc -z localhost 6379 >/dev/null 2>&1; then
      print_warning "Port 6379 (Redis) is already in use."
      echo "This may cause conflicts with the Redis container."
      echo "Consider stopping any existing Redis server or modifying port mappings in docker/docker-compose.yml."
      read -p "Do you want to continue anyway? (y/N): " continue_anyway
      if [[ ! $continue_anyway =~ ^[Yy]$ ]]; then
        echo "Operation cancelled."
        exit 1
      fi
    fi
    
    docker-compose -f docker/docker-compose.yml up
    ;;
    
  run-detached)
    print_header "Running Redactify services in detached mode"
    check_docker_compose
    
    # Check if Redis is already running on port 6379
    if nc -z localhost 6379 >/dev/null 2>&1; then
      print_warning "Port 6379 (Redis) is already in use."
      echo "This may cause conflicts with the Redis container."
      echo "Consider stopping any existing Redis server or modifying port mappings in docker/docker-compose.yml."
      read -p "Do you want to continue anyway? (y/N): " continue_anyway
      if [[ ! $continue_anyway =~ ^[Yy]$ ]]; then
        echo "Operation cancelled."
        exit 1
      fi
    fi
    
    docker-compose -f docker/docker-compose.yml up -d
    if [ $? -eq 0 ]; then
      echo -e "\n${GREEN}Redactify services are now running in the background.${NC}"
      echo -e "Access the web interface at: http://localhost:5000"
      echo -e "To view logs: ./redactify.sh logs"
      echo -e "To stop services: ./redactify.sh stop"
    else
      print_error "Failed to start Redactify services."
    fi
    ;;
    
  stop)
    print_header "Stopping Redactify services"
    check_docker_compose
    docker-compose -f docker/docker-compose.yml down
    if [ $? -eq 0 ]; then
      echo -e "\n${GREEN}Redactify services have been stopped.${NC}"
    else
      print_error "Failed to stop Redactify services."
    fi
    ;;
    
  clean)
    print_header "Cleaning up Redactify Docker resources"
    
    # Stop all containers first
    check_docker_compose
    docker-compose -f docker/docker-compose.yml down
    
    # Remove Redactify images
    echo "Removing Redactify Docker images..."
    docker rmi $(docker images | grep redactify | awk '{print $3}') 2>/dev/null
    
    # Remove any related networks and volumes
    echo "Removing Docker networks and volumes..."
    docker network rm docker_redactify_net 2>/dev/null
    docker volume rm docker_redis_data 2>/dev/null
    
    echo -e "\n${GREEN}Cleanup completed.${NC}"
    ;;
    
  status)
    print_header "Redactify Services Status"
    check_docker_compose
    docker-compose -f docker/docker-compose.yml ps
    ;;
    
  logs)
    if [ -z "$service_name" ]; then
      print_header "Viewing logs for all Redactify services"
      docker-compose -f docker/docker-compose.yml logs --tail=100 -f
    else
      print_header "Viewing logs for $service_name service"
      docker-compose -f docker/docker-compose.yml logs --tail=100 -f $service_name
    fi
    ;;
    
  help|*)
    show_help
    ;;
esac
