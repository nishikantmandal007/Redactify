#!/bin/bash
# Redactify System Requirements Checker
# Verifies that your system meets the minimum requirements for Redactify

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Requirements
MIN_PYTHON_VERSION="3.10"
MIN_MEMORY_GB=4
MIN_DISK_GB=5
RECOMMENDED_MEMORY_GB=8

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
    echo -e "${BLUE}"
    echo "======================================================"
    echo "🔍 Redactify System Requirements Check"
    echo "======================================================"
    echo -e "${NC}"
}

check_python() {
    log_info "Checking Python version..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
        
        if [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -ge 10 ]]; then
            log_success "Python $PYTHON_VERSION (meets requirement: >= $MIN_PYTHON_VERSION)"
            return 0
        else
            log_error "Python $PYTHON_VERSION found, but >= $MIN_PYTHON_VERSION required"
            return 1
        fi
    else
        log_error "Python 3 not found. Please install Python >= $MIN_PYTHON_VERSION"
        return 1
    fi
}

check_memory() {
    log_info "Checking system memory..."
    
    if command -v free &> /dev/null; then
        MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
        
        if [[ $MEMORY_GB -ge $RECOMMENDED_MEMORY_GB ]]; then
            log_success "Memory: ${MEMORY_GB}GB (recommended: ${RECOMMENDED_MEMORY_GB}GB+)"
        elif [[ $MEMORY_GB -ge $MIN_MEMORY_GB ]]; then
            log_warning "Memory: ${MEMORY_GB}GB (minimum met, but ${RECOMMENDED_MEMORY_GB}GB+ recommended)"
        else
            log_error "Memory: ${MEMORY_GB}GB (minimum ${MIN_MEMORY_GB}GB required)"
            return 1
        fi
    else
        log_warning "Cannot check memory on this system"
    fi
    
    return 0
}

check_disk_space() {
    log_info "Checking disk space..."
    
    DISK_GB=$(df -BG . | awk 'NR==2{print $4}' | sed 's/G//')
    
    if [[ $DISK_GB -ge $MIN_DISK_GB ]]; then
        log_success "Disk space: ${DISK_GB}GB available (minimum: ${MIN_DISK_GB}GB)"
    else
        log_error "Disk space: ${DISK_GB}GB available (minimum ${MIN_DISK_GB}GB required)"
        return 1
    fi
    
    return 0
}

check_git() {
    log_info "Checking Git..."
    
    if command -v git &> /dev/null; then
        GIT_VERSION=$(git --version | cut -d' ' -f3)
        log_success "Git $GIT_VERSION installed"
    else
        log_error "Git not found. Please install Git"
        return 1
    fi
    
    return 0
}

check_curl() {
    log_info "Checking curl..."
    
    if command -v curl &> /dev/null; then
        log_success "curl is available"
    else
        log_error "curl not found. Please install curl"
        return 1
    fi
    
    return 0
}

check_package_manager() {
    log_info "Checking package manager..."
    
    if command -v apt &> /dev/null; then
        log_success "Package manager: apt (Debian/Ubuntu)"
        PKG_MANAGER="apt"
    elif command -v yum &> /dev/null; then
        log_success "Package manager: yum (RHEL/CentOS)"
        PKG_MANAGER="yum"
    elif command -v dnf &> /dev/null; then
        log_success "Package manager: dnf (Fedora)"
        PKG_MANAGER="dnf"
    elif command -v pacman &> /dev/null; then
        log_success "Package manager: pacman (Arch)"
        PKG_MANAGER="pacman"
    elif command -v brew &> /dev/null; then
        log_success "Package manager: brew (macOS)"
        PKG_MANAGER="brew"
    else
        log_warning "No recognized package manager found"
        PKG_MANAGER="unknown"
    fi
    
    return 0
}

check_redis() {
    log_info "Checking Redis..."
    
    if command -v redis-server &> /dev/null; then
        REDIS_VERSION=$(redis-server --version | grep -o 'v=[0-9.]*' | cut -d'=' -f2)
        log_success "Redis $REDIS_VERSION installed"
        
        # Check if Redis is running
        if redis-cli ping &> /dev/null; then
            log_success "Redis server is running"
        else
            log_warning "Redis is installed but not running"
        fi
    else
        log_warning "Redis not found (will be installed by quick-install script)"
    fi
    
    return 0
}

check_gpu() {
    log_info "Checking GPU support..."
    
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
        log_success "NVIDIA GPU detected: $GPU_INFO"
        log_info "GPU acceleration will be available"
    else
        log_info "No NVIDIA GPU detected - CPU-only mode will be used"
    fi
    
    return 0
}

check_internet() {
    log_info "Checking internet connectivity..."
    
    if curl -s --head https://google.com | head -n 1 | grep -q "200 OK"; then
        log_success "Internet connection available"
    else
        log_error "No internet connection. Internet is required for downloading dependencies"
        return 1
    fi
    
    return 0
}

provide_recommendations() {
    echo ""
    echo -e "${BLUE}📋 Recommendations:${NC}"
    echo ""
    
    if [[ $MEMORY_GB -lt $RECOMMENDED_MEMORY_GB ]]; then
        echo -e "${YELLOW}💾 Memory:${NC} Consider adding more RAM for better performance"
    fi
    
    if ! command -v redis-server &> /dev/null; then
        echo -e "${YELLOW}🔧 Redis:${NC} Will be automatically installed by the quick-install script"
    fi
    
    if ! command -v nvidia-smi &> /dev/null; then
        echo -e "${BLUE}⚡ GPU:${NC} For faster processing, consider using a system with NVIDIA GPU"
    fi
    
    echo -e "${BLUE}🚀 Ready to install?${NC} Run:"
    echo "   curl -fsSL https://raw.githubusercontent.com/nishikantmandal007/Redactify/main/scripts/quick-install.sh | bash"
}

# Main execution
main() {
    print_banner
    
    local overall_status=0
    
    check_python || overall_status=1
    check_memory || overall_status=1
    check_disk_space || overall_status=1
    check_git || overall_status=1
    check_curl || overall_status=1
    check_package_manager
    check_redis
    check_gpu
    check_internet || overall_status=1
    
    echo ""
    echo "======================================================"
    
    if [[ $overall_status -eq 0 ]]; then
        log_success "System requirements check PASSED"
        log_info "Your system is ready for Redactify installation!"
    else
        log_error "System requirements check FAILED"
        log_info "Please fix the issues above before installing Redactify"
    fi
    
    provide_recommendations
    
    exit $overall_status
}

# Run main function
main "$@"
