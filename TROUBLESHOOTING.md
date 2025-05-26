# Redactify Troubleshooting Guide

This guide provides solutions for common issues that might arise when running the Redactify application with Docker.

## Common Issues and Solutions

### 1. Docker Image Build Failures

#### Issue: Package installation errors during build

```
ERROR: Could not find a version that satisfies the requirement paddlepaddle>=2.5.0
```

**Solution:**

- Verify your internet connection
- Make sure the Dockerfile is using the correct Python version (3.11)
- Update the requirements.txt file to include compatible versions:

  ```
  paddlepaddle>=2.5.0
  tensorflow==2.19.0  # For Python 3.11 compatibility
  ```

- Try building with the `--no-cache` flag:

  ```bash
  docker build --no-cache -f Redactify/Dockerfile -t redactify-cpu .
  ```

#### Issue: System library dependencies missing

```
ImportError: libzbar.so.0: cannot open shared object file: No such file or directory
```

**Solution:**

- Add the missing system libraries to the Dockerfile:

  ```dockerfile
  RUN apt-get update && \
      apt-get install -y --no-install-recommends \
      libzbar0 \
      # other dependencies
  ```

- Rebuild the Docker image

### 2. Container Startup Issues

#### Issue: Redis connection errors

```
Error 111 connecting to redis:6379. Connection refused.
```

**Solution:**

- Make sure Redis container is running:

  ```bash
  docker-compose -f docker/docker-compose.yml ps
  ```

- Check Redis container logs:

  ```bash
  docker-compose -f docker/docker-compose.yml logs redis
  ```

- Verify Redis URL is correctly set to `redis://redis:6379/0` in the application
- Check if port 6379 is already in use on the host:

  ```bash
  sudo netstat -tulpn | grep 6379
  ```

- If Redis is running elsewhere on the host, either:
  - Stop the existing Redis server
  - Change the Redis port mapping in docker-compose.yml
  - Configure the application to use the existing Redis server

#### Issue: Web application doesn't start

```
ModuleNotFoundError: No module named 'paddle'
```

**Solution:**

- Make sure all dependencies are installed:

  ```bash
  docker exec -it redactify_web_1 pip list | grep paddle
  ```

- Verify that paddlepaddle is in requirements.txt
- Check Python version compatibility
- Rebuild the Docker image with latest requirements

### 3. Application Runtime Issues

#### Issue: File upload errors

```
413 Request Entity Too Large
```

**Solution:**

- Increase the client_max_body_size in Nginx:

  ```nginx
  http {
      client_max_body_size 100M;
  }
  ```

- Update the max_file_size_mb in config.yaml
- Restart the Nginx container:

  ```bash
  docker-compose -f docker/docker-compose.yml restart nginx
  ```

#### Issue: Slow PDF processing

```
Task processing is very slow or timing out
```

**Solution:**

- Increase Celery task time limits in config.yaml:

  ```yaml
  celery_task_soft_time_limit: 600
  celery_task_hard_time_limit: 720
  ```

- Increase worker resources in docker-compose.yml:

  ```yaml
  deploy:
    resources:
      limits:
        cpus: '4'
        memory: '8g'
  ```

- Scale the number of Celery workers:

  ```bash
  docker-compose -f docker/docker-compose.yml up -d --scale celery_worker=4
  ```

- For large PDFs, consider enabling GPU support if available

### 4. GPU-Related Issues

#### Issue: GPU not detected

```
I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized with oneAPI...
```

**Solution:**

- Verify GPU is properly installed and recognized by the system:

  ```bash
  nvidia-smi
  ```

- Ensure Docker has GPU support enabled:

  ```bash
  docker info | grep -i nvidia
  ```

- Use the GPU-specific Dockerfile:

  ```bash
  docker build -f Redactify/Dockerfile.gpu -t redactify-gpu .
  ```

- Configure docker-compose.override.yml for GPU support:

  ```yaml
  services:
    web:
      image: redactify-web-gpu
      deploy:
        resources:
          reservations:
            devices:
              - driver: nvidia
                count: 1
                capabilities: [gpu]
  ```

#### Issue: GPU memory errors

```
RuntimeError: CUDA out of memory
```

**Solution:**

- Adjust GPU memory allocation in config.yaml:

  ```yaml
  gpu_memory_fraction_tf_general: 0.2
  gpu_memory_fraction_tf_nlp: 0.3
  ```

- Close other GPU-using applications
- Monitor GPU memory usage:

  ```bash
  nvidia-smi -l 1
  ```

### 5. Redis-Related Issues

#### Issue: Redis memory errors

```
OOM command not allowed when used memory > 'maxmemory'
```

**Solution:**

- Configure Redis memory limits in redis.conf:

  ```
  maxmemory 2gb
  maxmemory-policy allkeys-lru
  ```

- Add configuration to docker-compose.yml:

  ```yaml
  redis:
    command: redis-server --maxmemory 2gb --maxmemory-policy allkeys-lru
  ```

- Monitor Redis memory usage:

  ```bash
  docker exec -it redactify_redis_1 redis-cli info memory
  ```

#### Issue: Redis persistence issues

```
Background save error
```

**Solution:**

- Check disk space:

  ```bash
  df -h
  ```

- Verify Redis volume permissions:

  ```bash
  docker exec -it redactify_redis_1 ls -la /data
  ```

- Configure Redis persistence settings:

  ```
  save 900 1
  save 300 10
  appendonly yes
  ```

### 6. Celery-Related Issues

#### Issue: Tasks are not being processed

```
Task received but not processed
```

**Solution:**

- Verify Celery worker is running:

  ```bash
  docker-compose -f docker/docker-compose.yml logs celery_worker
  ```

- Check that worker is connected to Redis:

  ```bash
  docker exec -it redactify_celery_worker_1 celery -A Redactify.services.celery_service.celery inspect ping
  ```

- Make sure worker is listening to the correct queue:

  ```bash
  docker exec -it redactify_celery_worker_1 celery -A Redactify.services.celery_service.celery inspect active_queues
  ```

- Check for task routing issues:

  ```bash
  docker exec -it redactify_celery_worker_1 celery -A Redactify.services.celery_service.celery inspect reserved
  ```

#### Issue: Task failures

```
Task XXX raised exception: YYY
```

**Solution:**

- Analyze error traceback from Celery logs
- Monitor task state:

  ```bash
  docker exec -it redactify_celery_worker_1 celery -A Redactify.services.celery_service.celery events
  ```

- Try running with higher log level:

  ```yaml
  command: celery -A Redactify.services.celery_service.celery worker --loglevel=debug -Q redaction
  ```

### 7. System Resource Issues

#### Issue: Container OOM (Out Of Memory) errors

```
Container killed due to memory usage
```

**Solution:**

- Increase container memory limits:

  ```yaml
  deploy:
    resources:
      limits:
        memory: '16g'
  ```

- Add swap space to host system:

  ```bash
  sudo fallocate -l 8G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  ```

- Optimize application memory usage:
  - Configure smaller batch sizes
  - Process fewer items in parallel

#### Issue: High CPU usage

```
Container using excessive CPU
```

**Solution:**

- Limit container CPU usage:

  ```yaml
  deploy:
    resources:
      limits:
        cpus: '4'
  ```

- Monitor process CPU usage:

  ```bash
  docker stats
  ```

- Identify and optimize CPU-intensive operations

### 8. Network-Related Issues

#### Issue: Connection timeouts

```
ConnectionTimeoutError: Connection timed out
```

**Solution:**

- Check network connectivity between containers:

  ```bash
  docker exec -it redactify_web_1 ping redis
  ```

- Verify service names match in docker-compose.yml
- Check for firewall rules blocking internal Docker traffic:

  ```bash
  sudo iptables -L | grep DOCKER
  ```

- Increase timeout settings:

  ```python
  # In app configuration
  REDIS_SOCKET_TIMEOUT = 60
  ```

#### Issue: Port conflicts

```
Error starting userland proxy: listen tcp 0.0.0.0:6379: bind: address already in use
```

**Solution:**

- Find which process is using the port:

  ```bash
  sudo netstat -tulpn | grep 6379
  ```

- Either:
  - Stop the conflicting service
  - Change the port mapping in docker-compose.yml
  - Use a different port for your application

## Recovering from Failures

### Full System Restore

If you need to completely reset the environment:

```bash
# Stop all containers
docker-compose -f docker/docker-compose.yml down

# Remove all related images
docker rmi $(docker images | grep redactify | awk '{print $3}')

# Remove volumes (warning: deletes all data!)
docker volume rm docker_redis_data

# Clean Docker system
docker system prune

# Rebuild and restart
docker build -f Redactify/Dockerfile -t redactify-cpu .
docker-compose -f docker/docker-compose.yml up -d
```

### Data Recovery

If Redis data is corrupted:

```bash
# Stop Redis container
docker-compose -f docker/docker-compose.yml stop redis

# Remove Redis data volume
docker volume rm docker_redis_data

# Create fresh volume
docker volume create docker_redis_data

# Start Redis again
docker-compose -f docker/docker-compose.yml up -d redis
```

## Monitoring and Maintenance

### Resource Usage Monitoring

```bash
# Monitor container resource usage
docker stats

# Check disk space
df -h

# Check logs for errors
docker-compose -f docker/docker-compose.yml logs --tail=100 | grep -i error
```

### Regular Maintenance Tasks

```bash
# Clean up unused Docker resources
docker system prune -f

# Backup Redis data
docker exec -it redactify_redis_1 redis-cli SAVE

# Update image with latest code
docker build -f Redactify/Dockerfile -t redactify-cpu .
docker-compose -f docker/docker-compose.yml up -d --no-deps web celery_worker
```

## Getting Help

If you continue experiencing issues after trying the solutions in this guide:

1. Check the logs for more detailed error information:

   ```bash
   docker-compose -f docker/docker-compose.yml logs --tail=200
   ```

2. Inspect the application state:

   ```bash
   # Check Redis status
   docker exec -it redactify_redis_1 redis-cli info

   # Check Celery tasks
   docker exec -it redactify_celery_worker_1 celery -A Redactify.services.celery_service.celery inspect active
   ```

3. Review container configuration:

   ```bash
   docker inspect redactify_web_1
   ```

4. Contact support with:
   - Docker and Docker Compose versions
   - Host system details (OS, RAM, CPU, GPU)
   - Error logs and debug information
   - Steps to reproduce the issue
