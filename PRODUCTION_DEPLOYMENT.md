# Redactify Production Deployment Guide

This guide provides detailed instructions for deploying the Redactify application in a production environment using Docker.

## Production System Requirements

- **CPU**: 4+ cores recommended
- **RAM**: 16GB minimum, 32GB recommended
- **Disk Space**: 50GB minimum for application, database, and document storage
- **Operating System**: Linux (Ubuntu 20.04 LTS or newer recommended)
- **Docker**: Version 20.10 or newer
- **Docker Compose**: Version 2.0 or newer
- **GPU (Optional)**: NVIDIA GPU with CUDA support for faster processing

## Security Considerations

### 1. Network Security

- **Firewall Configuration**: Only expose necessary ports (e.g., 443 for HTTPS)
- **HTTPS**: Secure all communications using SSL/TLS
- **Reverse Proxy**: Use Nginx or similar as a frontend for SSL termination and security headers

### 2. Data Security

- **Document Storage**: Use encrypted volumes for document storage
- **Redis Security**: Password-protect Redis and don't expose it externally
- **Secrets Management**: Use Docker secrets or environment variables for sensitive data

## Production Docker Compose Configuration

Create a `docker-compose.prod.yml` file for production deployments:

```yaml
version: '3.8'

services:
  redis:
    image: "redis:7-alpine"
    command: redis-server --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    restart: always
    networks:
      - redactify_net
    # Don't expose Redis port in production
    
  web:
    image: redactify-web-cpu  # Or redactify-web-gpu for GPU support
    restart: always
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 8G
    volumes:
      - redactify_uploads:/app/upload_files
      - redactify_temp:/app/temp_files
      - ${CONFIG_PATH:-./config.yaml}:/app/config.yaml:ro
    environment:
      - REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
      - FLASK_ENV=production
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    networks:
      - redactify_net
    depends_on:
      - redis
    # Don't expose web port directly in production
      
  celery_worker:
    image: redactify-celery-cpu  # Or redactify-celery-gpu for GPU support
    command: celery -A Redactify.services.celery_service.celery worker --loglevel=info -Q redaction --concurrency=${CELERY_CONCURRENCY:-4} -E
    restart: always
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 12G
      replicas: ${CELERY_REPLICAS:-2}
    volumes:
      - redactify_uploads:/app/upload_files
      - redactify_temp:/app/temp_files
      - ${CONFIG_PATH:-./config.yaml}:/app/config.yaml:ro
    environment:
      - REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
      - FLASK_ENV=production
    healthcheck:
      test: ["CMD", "celery", "-A", "Redactify.services.celery_service.celery", "inspect", "ping"]
      interval: 60s
      timeout: 10s
      retries: 3
      start_period: 40s
    networks:
      - redactify_net
    depends_on:
      - redis
  
  celery_beat:
    image: redactify-celery-cpu
    command: celery -A Redactify.services.celery_service.celery beat --loglevel=info
    restart: always
    volumes:
      - redactify_temp:/app/temp_files
      - ${CONFIG_PATH:-./config.yaml}:/app/config.yaml:ro
    environment:
      - REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
      - FLASK_ENV=production
    networks:
      - redactify_net
    depends_on:
      - redis
      
  nginx:
    image: nginx:stable-alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/conf:/etc/nginx/conf.d:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
      - nginx_cache:/var/cache/nginx
    restart: always
    depends_on:
      - web
    networks:
      - redactify_net

networks:
  redactify_net:
    driver: bridge
    
volumes:
  redis_data:
  redactify_uploads:
  redactify_temp:
  nginx_cache:
```

### Environment Variables File

Create a `.env` file for production settings (do not commit to version control):

```
# Redis settings
REDIS_PASSWORD=your_strong_password_here

# Celery worker settings
CELERY_CONCURRENCY=4
CELERY_REPLICAS=2

# Configuration path
CONFIG_PATH=/path/to/your/config.yaml
```

## Nginx Configuration for HTTPS

Create a basic Nginx configuration in `nginx/conf/redactify.conf`:

```nginx
# Rate limiting zone
limit_req_zone $binary_remote_addr zone=redactify_limit:10m rate=10r/s;

server {
    listen 80;
    server_name your-redactify-domain.com;
    
    # Redirect HTTP to HTTPS
    location / {
        return 301 https://$host$request_uri;
    }
}

server {
    listen 443 ssl http2;
    server_name your-redactify-domain.com;
    
    # SSL configuration
    ssl_certificate     /etc/nginx/ssl/redactify.crt;
    ssl_certificate_key /etc/nginx/ssl/redactify.key;
    ssl_protocols       TLSv1.2 TLSv1.3;
    ssl_ciphers         HIGH:!aNULL:!MD5;
    
    # Security headers
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header X-XSS-Protection "1; mode=block";
    add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:;";
    
    # Proxy to web service
    location / {
        limit_req zone=redactify_limit burst=20 nodelay;
        
        proxy_pass http://web:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeout settings for large file uploads
        proxy_connect_timeout 300;
        proxy_send_timeout 300;
        proxy_read_timeout 300;
        send_timeout 300;
        
        client_max_body_size 100M;
    }
    
    # Serve static files with caching
    location /static/ {
        proxy_pass http://web:5000/static/;
        proxy_cache nginx_cache;
        proxy_cache_valid 200 1h;
        expires 1h;
        add_header Cache-Control "public";
    }
    
    # Health check endpoint
    location /health {
        proxy_pass http://web:5000/health;
        access_log off;
        proxy_cache_bypass 1;
        proxy_no_cache 1;
    }
}
```

## Backup Strategy

### 1. Redis Backup

Configure Redis to create regular RDB snapshots:

```
# In redis.conf
dir /data
dbfilename dump.rdb
save 900 1
save 300 10
save 60 10000
```

Create a backup script for Redis:

```bash
#!/bin/bash
# Redis backup script

BACKUP_DIR="/path/to/backups/redis"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="$BACKUP_DIR/redis_backup_$TIMESTAMP.rdb"

# Create backup directory if not exists
mkdir -p $BACKUP_DIR

# Copy Redis dump file
docker cp redactify_redis_1:/data/dump.rdb $BACKUP_FILE

# Compress backup
gzip $BACKUP_FILE

# Keep only last 7 backups
find $BACKUP_DIR -name "redis_backup_*.rdb.gz" -type f -mtime +7 -delete
```

### 2. Document Backup

Create a backup script for uploaded documents:

```bash
#!/bin/bash
# Document backup script

BACKUP_DIR="/path/to/backups/documents"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="$BACKUP_DIR/documents_backup_$TIMESTAMP.tar.gz"

# Create backup directory if not exists
mkdir -p $BACKUP_DIR

# Create tar archive of uploads volume
docker run --rm -v redactify_uploads:/data -v $BACKUP_DIR:/backup \
  alpine tar czf /backup/documents_backup_$TIMESTAMP.tar.gz /data

# Keep only last 30 days of backups
find $BACKUP_DIR -name "documents_backup_*.tar.gz" -type f -mtime +30 -delete
```

## Monitoring Strategy

### 1. Container Monitoring

Use Docker's built-in health checks (already configured in docker-compose.prod.yml).

### 2. System Monitoring

Set up Prometheus and Grafana for monitoring:

```yaml
# Add to docker-compose.prod.yml
services:
  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus:/etc/prometheus
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
    ports:
      - "9090:9090"
    restart: always
    networks:
      - redactify_net
      
  node-exporter:
    image: prom/node-exporter
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.ignored-mount-points=^/(sys|proc|dev|host|etc)($$|/)'
    restart: always
    networks:
      - redactify_net
      
  cadvisor:
    image: gcr.io/cadvisor/cadvisor
    volumes:
      - /:/rootfs:ro
      - /var/run:/var/run:ro
      - /sys:/sys:ro
      - /var/lib/docker/:/var/lib/docker:ro
      - /dev/disk/:/dev/disk:ro
    restart: always
    networks:
      - redactify_net
      
  grafana:
    image: grafana/grafana
    volumes:
      - grafana_data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-admin}
    ports:
      - "3000:3000"
    restart: always
    networks:
      - redactify_net

volumes:
  prometheus_data:
  grafana_data:
```

### 3. Log Management

Use the ELK stack (Elasticsearch, Logstash, Kibana) or a simpler solution like Loki with Grafana:

```yaml
# Add to docker-compose.prod.yml
services:
  loki:
    image: grafana/loki:2.8.0
    volumes:
      - ./loki:/etc/loki
      - loki_data:/loki
    ports:
      - "3100:3100"
    command: -config.file=/etc/loki/loki-config.yaml
    restart: always
    networks:
      - redactify_net
      
  promtail:
    image: grafana/promtail:2.8.0
    volumes:
      - ./promtail:/etc/promtail
      - /var/lib/docker/containers:/var/lib/docker/containers:ro
      - /var/log:/var/log:ro
    command: -config.file=/etc/promtail/promtail-config.yaml
    restart: always
    networks:
      - redactify_net

volumes:
  loki_data:
```

## Deployment Procedure

### 1. Initial Setup

```bash
# Clone repository
git clone https://github.com/yourusername/Redactify.git
cd Redactify

# Create required directories
mkdir -p nginx/conf nginx/ssl loki promtail prometheus

# Create configuration files
# (copy the above configuration examples to their respective locations)

# Generate SSL certificates (replace with proper certificates for production)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout nginx/ssl/redactify.key -out nginx/ssl/redactify.crt
```

### 2. Build and Deploy

```bash
# Build Docker images
docker build -f Redactify/Dockerfile -t redactify-cpu .

# Create environment file
cat > .env << EOL
REDIS_PASSWORD=$(openssl rand -hex 16)
CELERY_CONCURRENCY=4
CELERY_REPLICAS=2
GRAFANA_PASSWORD=$(openssl rand -hex 8)
CONFIG_PATH=./config.yaml
EOL

# Create basic config.yaml
cat > config.yaml << EOL
redis_url: redis://:${REDIS_PASSWORD}@redis:6379/0
max_file_size_mb: 50
temp_file_max_age_seconds: 86400
log_level: INFO
EOL

# Start the stack
docker-compose -f docker-compose.prod.yml up -d
```

### 3. Verify Deployment

```bash
# Check container status
docker-compose -f docker-compose.prod.yml ps

# View logs
docker-compose -f docker-compose.prod.yml logs -f web

# Test the application
curl -k https://localhost/health
```

## Scaling Considerations

### 1. Horizontal Scaling

Increase the number of Celery workers:

```bash
# Scale to 4 worker replicas
docker-compose -f docker-compose.prod.yml up -d --scale celery_worker=4
```

### 2. Vertical Scaling

Adjust resource limits in docker-compose.prod.yml:

```yaml
deploy:
  resources:
    limits:
      cpus: '8'
      memory: 16G
```

### 3. Redis Scaling

For high-volume deployments, consider Redis Cluster or Redis Sentinel for high availability.

## Maintenance Procedures

### 1. Application Updates

```bash
# Pull latest code
git pull origin master

# Rebuild images
docker build -f Redactify/Dockerfile -t redactify-cpu .

# Restart services
docker-compose -f docker-compose.prod.yml up -d --no-deps --build web celery_worker celery_beat
```

### 2. Database Management

```bash
# Access Redis CLI
docker-compose -f docker-compose.prod.yml exec redis redis-cli -a $REDIS_PASSWORD

# Monitor Redis memory usage
docker-compose -f docker-compose.prod.yml exec redis redis-cli -a $REDIS_PASSWORD info memory
```

### 3. Log Rotation

Configure log rotation for container logs:

```
# /etc/logrotate.d/docker-containers
/var/lib/docker/containers/*/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    copytruncate
}
```

## Troubleshooting

### 1. Container Issues

```bash
# Check container logs
docker-compose -f docker-compose.prod.yml logs --tail=100 [service-name]

# Check container resource usage
docker stats

# Connect to a container
docker-compose -f docker-compose.prod.yml exec web bash
```

### 2. Application Issues

Check the application logs and health endpoint:

```bash
# Application logs
docker-compose -f docker-compose.prod.yml logs -f web

# Health check
curl -k https://your-domain.com/health
```

### 3. Redis Issues

```bash
# Check Redis info
docker-compose -f docker-compose.prod.yml exec redis redis-cli -a $REDIS_PASSWORD info

# Monitor Redis
docker-compose -f docker-compose.prod.yml exec redis redis-cli -a $REDIS_PASSWORD monitor
```

## Recovery Procedures

### 1. Redis Recovery

```bash
# Restore Redis backup
docker cp /path/to/backups/redis/redis_backup_20230101_120000.rdb redactify_redis_1:/data/dump.rdb
docker-compose -f docker-compose.prod.yml restart redis
```

### 2. Document Recovery

```bash
# Create a temporary container to restore documents
docker run --rm -v redactify_uploads:/data -v /path/to/backups/documents:/backup \
  alpine sh -c "rm -rf /data/* && tar xzf /backup/documents_backup_20230101_120000.tar.gz -C /data --strip-components=1"
```

## Security Hardening

### 1. Container Security

Enable Docker's security options:

```yaml
# Add to service definitions in docker-compose.prod.yml
security_opt:
  - no-new-privileges:true
  - seccomp:seccomp-profile.json
  
cap_drop:
  - ALL
cap_add:
  - NET_BIND_SERVICE
```

### 2. File Permissions

Ensure proper ownership of mounted volumes:

```yaml
# Create a custom entrypoint script for services
volumes:
  - ./docker-entrypoint.sh:/docker-entrypoint.sh
entrypoint: ["/docker-entrypoint.sh"]
```

### 3. Secrets Management

Use Docker secrets or a vault solution for sensitive information.

## Conclusion

This production deployment guide covers the essential aspects of deploying the Redactify application in a secure and scalable manner. Adjust the configurations to match your specific environment and requirements.
