#!/bin/bash
# Clean up all Docker resources related to Redactify

echo "Stopping all running containers..."
docker stop $(docker ps -a -q) 2>/dev/null || true

echo "Removing all containers..."
docker rm -f $(docker ps -a -q) 2>/dev/null || true

echo "Removing all Redactify images..."
docker rmi -f $(docker images | grep redactify | awk '{print $3}') 2>/dev/null || true

echo "Removing Redactify networks..."
docker network rm docker_redactify_net 2>/dev/null || true

echo "Removing Redactify volumes..."
docker volume rm docker_redis_data 2>/dev/null || true

echo "Performing additional system cleanup..."
docker system prune -f

echo "Docker cleanup completed successfully!"
