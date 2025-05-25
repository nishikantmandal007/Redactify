#!/bin/bash
# Redactify Version and Health Check Script
# Shows version info and system health status

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
INSTALL_DIR="$HOME/redactify"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✅]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[⚠️]${NC} $1"
}

log_error() {
    echo -e "${RED}[❌]${NC} $1"
}

print_banner() {
    echo -e "${PURPLE}"
    cat << 'EOF'
    ____          __           __  _ ____      
   / __ \___  ___/ /___ ______/ /_(_) __/_  __
  / /_/ / _ \/ _  / __ `/ ___/ __/ / /_/ / / /
 / _, _/  __/ /_/ / /_/ / /__/ /_/ / __/ /_/ / 
/_/ |_|\___/\__,_/\__,_/\___/\__/_/_/  \__, /  
                                     /____/   
EOF
    echo -e "${NC}"
    echo -e "${BLUE}🔍 System Status & Version Information${NC}"
    echo "======================================================"
}

get_redactify_version() {
    if [[ -d "$INSTALL_DIR" ]] && [[ -f "$INSTALL_DIR/Redactify/main.py" ]]; then
        cd "$INSTALL_DIR"
        source venv/bin/activate 2>/dev/null || {
            log_error "Virtual environment not found or corrupted"
            return 1
        }
        
        # Try to get version from git
        if [[ -d ".git" ]]; then
            GIT_VERSION=$(git describe --tags --always 2>/dev/null || echo "unknown")
            GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
            GIT_BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")
            echo -e "${GREEN}📦 Redactify Version${NC}"
            echo "   Version: $GIT_VERSION"
            echo "   Commit:  $GIT_COMMIT"
            echo "   Branch:  $GIT_BRANCH"
        else
            echo -e "${YELLOW}📦 Redactify Version: Source installation (no git info)${NC}"
        fi
        
        # Check Python package versions
        echo ""
        echo -e "${BLUE}🐍 Python Environment${NC}"
        echo "   Python: $(python --version 2>&1 | cut -d' ' -f2)"
        echo "   Location: $INSTALL_DIR/venv"
        
        # Key package versions
        echo ""
        echo -e "${BLUE}📚 Key Dependencies${NC}"
        
        check_package_version() {
            local package=$1
            local version=$(pip show "$package" 2>/dev/null | grep Version | cut -d' ' -f2)
            if [[ -n "$version" ]]; then
                echo "   $package: $version"
            else
                echo "   $package: Not installed"
            fi
        }
        
        check_package_version "flask"
        check_package_version "celery"
        check_package_version "redis"
        check_package_version "presidio-analyzer"
        check_package_version "paddleocr"
        check_package_version "spacy"
        check_package_version "PyMuPDF"
        
    else
        log_error "Redactify installation not found at $INSTALL_DIR"
        return 1
    fi
}

check_system_status() {
    echo ""
    echo -e "${BLUE}🖥️  System Information${NC}"
    echo "   OS: $(uname -s) $(uname -r)"
    echo "   Architecture: $(uname -m)"
    echo "   Hostname: $(hostname)"
    echo "   User: $USER"
    echo "   Install Directory: $INSTALL_DIR"
    
    # Memory info
    if command -v free &> /dev/null; then
        MEMORY_INFO=$(free -h | awk '/^Mem:/ {print $2 " total, " $3 " used, " $7 " available"}')
        echo "   Memory: $MEMORY_INFO"
    fi
    
    # Disk space
    DISK_INFO=$(df -h "$INSTALL_DIR" 2>/dev/null | awk 'NR==2 {print $4 " free of " $2}' || echo "Unknown")
    echo "   Disk Space: $DISK_INFO"
}

check_services_status() {
    echo ""
    echo -e "${BLUE}🔧 Services Status${NC}"
    
    # Check Redis
    if command -v redis-cli &> /dev/null; then
        if redis-cli ping &> /dev/null; then
            REDIS_VERSION=$(redis-cli info server | grep redis_version | cut -d: -f2 | tr -d '\r')
            log_success "Redis: Running (v$REDIS_VERSION)"
        else
            log_warning "Redis: Installed but not running"
        fi
    else
        log_error "Redis: Not installed"
    fi
    
    # Check Celery workers
    WORKER_COUNT=$(pgrep -f "celery.*worker.*Redactify" | wc -l)
    if [[ $WORKER_COUNT -gt 0 ]]; then
        log_success "Celery Workers: $WORKER_COUNT running"
    else
        log_warning "Celery Workers: Not running"
    fi
    
    # Check Celery beat
    if pgrep -f "celery.*beat.*Redactify" &> /dev/null; then
        log_success "Celery Beat: Running"
    else
        log_warning "Celery Beat: Not running"
    fi
    
    # Check web server
    if pgrep -f "python.*Redactify.main" &> /dev/null; then
        log_success "Web Server: Running"
    else
        log_warning "Web Server: Not running"
    fi
    
    # Check systemd service
    if systemctl list-unit-files redactify.service &> /dev/null 2>&1; then
        if systemctl is-active redactify.service &> /dev/null; then
            log_success "Systemd Service: Active"
        else
            log_warning "Systemd Service: Inactive"
        fi
    else
        log_info "Systemd Service: Not configured"
    fi
}

check_gpu_status() {
    echo ""
    echo -e "${BLUE}⚡ GPU Information${NC}"
    
    if command -v nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
        GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        GPU_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        
        log_success "GPU: $GPU_NAME"
        echo "   Memory: ${GPU_USED}MB / ${GPU_MEMORY}MB used"
        echo "   Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)"
    else
        log_info "GPU: No NVIDIA GPU detected (CPU-only mode)"
    fi
}

check_network_endpoints() {
    echo ""
    echo -e "${BLUE}🌐 Network Endpoints${NC}"
    
    # Check if ports are in use
    check_port() {
        local port=$1
        local service=$2
        if netstat -ln 2>/dev/null | grep -q ":$port "; then
            log_success "$service: http://localhost:$port"
        else
            log_warning "$service: Port $port not listening"
        fi
    }
    
    check_port 5000 "Web Interface"
    check_port 5555 "Flower Monitor"
    check_port 6379 "Redis"
}

check_queue_status() {
    echo ""
    echo -e "${BLUE}📋 Queue Status${NC}"
    
    if command -v redis-cli &> /dev/null && redis-cli ping &> /dev/null; then
        cd "$INSTALL_DIR" 2>/dev/null && source venv/bin/activate 2>/dev/null || true
        
        python3 -c "
import redis
try:
    r = redis.Redis()
    redaction_queue = r.llen('redaction') or 0
    maintenance_queue = r.llen('maintenance') or 0
    celery_queue = r.llen('celery') or 0
    
    print(f'   Redaction tasks: {redaction_queue}')
    print(f'   Maintenance tasks: {maintenance_queue}')
    print(f'   General tasks: {celery_queue}')
    
    # Check for any failed tasks
    failed_keys = r.keys('celery-task-meta-*')
    failed_count = 0
    for key in failed_keys:
        task_data = r.get(key)
        if task_data and b'FAILURE' in task_data:
            failed_count += 1
    
    if failed_count > 0:
        print(f'   Failed tasks: {failed_count}')
    else:
        print('   Failed tasks: 0')
        
except Exception as e:
    print(f'   Error checking queues: {e}')
" 2>/dev/null || echo "   Unable to check queue status"
    else
        log_warning "Cannot check queues - Redis not accessible"
    fi
}

provide_help_info() {
    echo ""
    echo -e "${BLUE}🆘 Need Help?${NC}"
    echo ""
    echo -e "${YELLOW}Common Commands:${NC}"
    echo "   Start services:    cd ~/redactify && ./start-redactify.sh"
    echo "   Stop services:     cd ~/redactify && ./stop-redactify.sh"
    echo "   Monitor system:    cd ~/redactify && ./monitor-redactify.sh"
    echo "   View logs:         journalctl -f -u redactify"
    echo ""
    echo -e "${YELLOW}Troubleshooting:${NC}"
    echo "   Check requirements: curl -fsSL https://raw.githubusercontent.com/yourusername/Redactify/main/scripts/check-requirements.sh | bash"
    echo "   Reinstall:         curl -fsSL https://raw.githubusercontent.com/yourusername/Redactify/main/scripts/quick-install.sh | bash"
    echo "   Uninstall:         curl -fsSL https://raw.githubusercontent.com/yourusername/Redactify/main/scripts/uninstall.sh | bash"
    echo ""
    echo -e "${YELLOW}Documentation:${NC}"
    echo "   Main docs:         ~/redactify/docs/"
    echo "   Configuration:     ~/redactify/docs/configuration.md"
    echo "   Troubleshooting:   ~/redactify/docs/troubleshooting.md"
}

# Main execution
main() {
    print_banner
    get_redactify_version
    check_system_status
    check_services_status
    check_gpu_status
    check_network_endpoints
    check_queue_status
    provide_help_info
    
    echo ""
    echo "======================================================"
    echo -e "${GREEN}🎉 Status check complete!${NC}"
    echo ""
}

# Run main function
main "$@"
