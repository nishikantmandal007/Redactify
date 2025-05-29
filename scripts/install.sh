#!/bin/bash
# Redactify Installation Script
# Automated installer for Redactify PDF PII Redaction Tool
#
# This script automatically:
# - Validates GitHub authentication for private repository access
# - Detects OS and installs system dependencies
# - Creates Python virtual environment
# - Installs all required dependencies
# - Sets up Redis
# - Downloads NLP models
# - Configures the application
# - Creates startup scripts

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
REPO_URL="https://github.com/nishikantmandal007/Redactify.git"
INSTALL_DIR="$HOME/redactify"
PYTHON_VERSION="3.10"
REDIS_VERSION="7"
REQUIRED_MEMORY_GB=4
REQUIRED_DISK_GB=5

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
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
             REDACTIFY INSTALLER              
EOF
    echo -e "${NC}"
    echo -e "${BLUE}🔒 Intelligent PII Redaction System${NC}"
    echo -e "${BLUE}📄 Automated PDF Privacy Protection${NC}"
    echo "=================================================="
}

check_github_access() {
    log_info "Validating GitHub repository access..."
    
    # Check if GITHUB_TOKEN is provided
    if [ -z "$GITHUB_TOKEN" ]; then
        log_error "GitHub Personal Access Token required for private repository access"
        echo ""
        echo "To install Redactify, you need a GitHub Personal Access Token:"
        echo "1. Go to: https://github.com/settings/tokens"
        echo "2. Click 'Generate new token (classic)'"
        echo "3. Select 'repo' scope for private repository access"
        echo "4. Copy the generated token"
        echo ""
        echo "Then run:"
        echo "export GITHUB_TOKEN='your_token_here'"
        echo "curl -H \"Authorization: token \$GITHUB_TOKEN\" -fsSL https://raw.githubusercontent.com/nishikantmandal007/Redactify/main/install.sh | bash"
        exit 1
    fi
    
    # Test GitHub API access
    if curl -s -H "Authorization: token $GITHUB_TOKEN" \
        "https://api.github.com/repos/nishikantmandal007/Redactify" > /dev/null 2>&1; then
        log_success "GitHub authentication successful"
    else
        log_error "Failed to access private repository. Please check your token permissions."
        exit 1
    fi
}

detect_os() {
    log_info "Detecting operating system..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v apt &> /dev/null; then
            OS="ubuntu"
            PKG_MANAGER="apt"
        elif command -v yum &> /dev/null; then
            OS="rhel"
            PKG_MANAGER="yum"
        elif command -v dnf &> /dev/null; then
            OS="fedora"
            PKG_MANAGER="dnf"
        elif command -v pacman &> /dev/null; then
            OS="arch"
            PKG_MANAGER="pacman"
        else
            log_error "Unsupported Linux distribution"
            exit 1
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        PKG_MANAGER="brew"
    else
        log_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
    
    log_success "Detected OS: $OS"
}

check_system_requirements() {
    log_info "Checking system requirements..."
    
    # Check memory
    if command -v free &> /dev/null; then
        MEMORY_GB=$(free -g | awk 'NR==2{printf "%.0f", $2}')
        if [[ $MEMORY_GB -ge $REQUIRED_MEMORY_GB ]]; then
            log_success "Memory: ${MEMORY_GB}GB (minimum: ${REQUIRED_MEMORY_GB}GB)"
        else
            log_warning "Low memory: ${MEMORY_GB}GB (recommended: ${REQUIRED_MEMORY_GB}GB+)"
        fi
    fi
    
    # Check disk space
    DISK_GB=$(df -BG . | awk 'NR==2{print $4}' | sed 's/G//')
    if [[ $DISK_GB -ge $REQUIRED_DISK_GB ]]; then
        log_success "Disk space: ${DISK_GB}GB available (minimum: ${REQUIRED_DISK_GB}GB)"
    else
        log_error "Insufficient disk space: ${DISK_GB}GB (minimum: ${REQUIRED_DISK_GB}GB required)"
        exit 1
    fi
    
    # Check Python version
    if command -v python3 &> /dev/null; then
        PYTHON_VER=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1-2)
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
            log_success "Python: $PYTHON_VER (compatible)"
        else
            log_error "Python 3.10+ required. Found: $PYTHON_VER"
            exit 1
        fi
    else
        log_warning "Python3 not found, will install"
    fi
}

install_system_dependencies() {
    log_info "Installing system dependencies..."
    
    case $OS in
        "ubuntu")
            sudo apt update
            sudo apt install -y \
                python3 python3-pip python3-venv python3-dev \
                redis-server \
                git curl wget \
                build-essential \
                libpoppler-cpp-dev \
                libgl1-mesa-glx \
                libglib2.0-0 \
                libsm6 \
                libxext6 \
                libxrender-dev \
                libgomp1 \
                libgcc-s1
            ;;
        "rhel"|"fedora")
            if [[ $PKG_MANAGER == "dnf" ]]; then
                sudo dnf update -y
                sudo dnf install -y \
                    python3 python3-pip python3-devel \
                    redis \
                    git curl wget \
                    gcc gcc-c++ make \
                    poppler-cpp-devel \
                    mesa-libGL \
                    glib2 \
                    libSM \
                    libXext \
                    libXrender \
                    libgomp \
                    libgcc
            else
                sudo yum update -y
                sudo yum install -y \
                    python3 python3-pip python3-devel \
                    redis \
                    git curl wget \
                    gcc gcc-c++ make \
                    poppler-cpp-devel
            fi
            ;;
        "arch")
            sudo pacman -Syu --noconfirm
            sudo pacman -S --noconfirm \
                python python-pip \
                redis \
                git curl wget \
                base-devel \
                poppler \
                mesa \
                glib2
            ;;
        "macos")
            if ! command -v brew &> /dev/null; then
                log_info "Installing Homebrew..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            fi
            brew update
            brew install python@3.11 redis git curl wget poppler
            ;;
    esac
    
    log_success "System dependencies installed"
}

setup_redis() {
    log_info "Setting up Redis..."
    
    case $OS in
        "ubuntu"|"rhel"|"fedora"|"arch")
            sudo systemctl enable redis-server || sudo systemctl enable redis
            sudo systemctl start redis-server || sudo systemctl start redis
            ;;
        "macos")
            brew services start redis
            ;;
    esac
    
    # Test Redis connection
    if redis-cli ping > /dev/null 2>&1; then
        log_success "Redis is running"
    else
        log_error "Failed to start Redis"
        exit 1
    fi
}

clone_repository() {
    log_info "Cloning Redactify repository..."
    
    # Remove existing installation
    if [[ -d "$INSTALL_DIR" ]]; then
        log_warning "Existing installation found, backing up..."
        mv "$INSTALL_DIR" "${INSTALL_DIR}.backup.$(date +%s)"
    fi
    
    # Clone repository with authentication
    git clone https://${GITHUB_TOKEN}@github.com/nishikantmandal007/Redactify.git "$INSTALL_DIR"
    cd "$INSTALL_DIR"
    
    log_success "Repository cloned to $INSTALL_DIR"
}

setup_python_environment() {
    log_info "Setting up Python virtual environment..."
    
    cd "$INSTALL_DIR"
    
    # Create virtual environment
    python3 -m venv venv
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    log_success "Python virtual environment created"
}

install_python_dependencies() {
    log_info "Installing Python dependencies..."
    
    cd "$INSTALL_DIR"
    source venv/bin/activate
    
    # Install main requirements
    pip install -r Redactify/requirements.txt
    
    # Check for GPU support and install GPU requirements if available
    if command -v nvidia-smi &> /dev/null && [[ -f "Redactify/requirements_gpu.txt" ]]; then
        log_info "GPU detected, installing GPU-accelerated packages..."
        pip install -r Redactify/requirements_gpu.txt
        log_success "GPU packages installed"
    else
        log_info "No GPU detected or GPU requirements not found, using CPU-only mode"
    fi
    
    log_success "Python dependencies installed"
}

download_nlp_models() {
    log_info "Downloading required NLP models..."
    
    cd "$INSTALL_DIR"
    source venv/bin/activate
    
    # Download spaCy model
    python -m spacy download en_core_web_sm
    
    log_success "NLP models downloaded"
}

setup_configuration() {
    log_info "Setting up configuration..."
    
    cd "$INSTALL_DIR"
    
    # Create configuration file
    cat > config.yaml << EOF
# Redactify Configuration
app:
  name: "Redactify"
  version: "1.0.0"
  debug: false
  host: "0.0.0.0"
  port: 5000

redis:
  host: "localhost"
  port: 6379
  db: 0
  password: null

upload:
  max_file_size_mb: 50
  allowed_extensions: [".pdf", ".png", ".jpg", ".jpeg"]
  temp_file_max_age_seconds: 86400

processing:
  max_concurrent_jobs: 4
  timeout_seconds: 300
  cleanup_interval_seconds: 3600

security:
  secret_key: "$(openssl rand -hex 32)"
  upload_path: "./upload_files"
  temp_path: "./temp_files"

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
EOF
    
    # Create necessary directories
    mkdir -p upload_files temp_files
    
    log_success "Configuration completed"
}

create_startup_scripts() {
    log_info "Creating startup scripts..."
    
    cd "$INSTALL_DIR"
    
    # Create start script
    cat > start-redactify.sh << 'EOF'
#!/bin/bash
# Redactify Startup Script

cd "$(dirname "$0")"
source venv/bin/activate

echo "🔒 Starting Redactify Services..."
echo "=================================="

# Check Redis
if ! redis-cli ping > /dev/null 2>&1; then
    echo "❌ Redis is not running. Please start Redis first."
    exit 1
fi
echo "✅ Redis is running"

# Start Celery worker in background
echo "🔄 Starting Celery worker..."
celery -A Redactify.services.celery_service.celery worker \
    --loglevel=info \
    --concurrency=4 \
    --queues=redaction,maintenance \
    --detach \
    --pidfile=celery_worker.pid \
    --logfile=logs/celery_worker.log

echo "🔄 Starting Celery beat scheduler..."
celery -A Redactify.services.celery_service.celery beat \
    --loglevel=info \
    --detach \
    --pidfile=celery_beat.pid \
    --logfile=logs/celery_beat.log

sleep 2

echo "🌐 Starting web application..."
echo "Access Redactify at: http://localhost:5000"
echo "Press Ctrl+C to stop"
echo ""

python Redactify/main.py
EOF

    # Create stop script
    cat > stop-redactify.sh << 'EOF'
#!/bin/bash
# Redactify Stop Script

cd "$(dirname "$0")"

echo "🛑 Stopping Redactify Services..."

# Stop Celery processes
if [[ -f celery_worker.pid ]]; then
    PID=$(cat celery_worker.pid)
    if kill -0 $PID 2>/dev/null; then
        kill $PID
        echo "✅ Celery worker stopped"
    fi
    rm -f celery_worker.pid
fi

if [[ -f celery_beat.pid ]]; then
    PID=$(cat celery_beat.pid)
    if kill -0 $PID 2>/dev/null; then
        kill $PID
        echo "✅ Celery beat stopped"
    fi
    rm -f celery_beat.pid
fi

# Kill any remaining processes
pkill -f "celery.*Redactify" 2>/dev/null || true
pkill -f "python.*Redactify" 2>/dev/null || true

echo "✅ All services stopped"
EOF

    # Create status script
    cat > status-redactify.sh << 'EOF'
#!/bin/bash
# Redactify Status Check

cd "$(dirname "$0")"

echo "🔍 Redactify Status Check"
echo "========================"

# Check Redis
if redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis: Running"
else
    echo "❌ Redis: Not running"
fi

# Check Celery worker
if [[ -f celery_worker.pid ]] && kill -0 $(cat celery_worker.pid) 2>/dev/null; then
    echo "✅ Celery Worker: Running (PID: $(cat celery_worker.pid))"
else
    echo "❌ Celery Worker: Not running"
fi

# Check Celery beat
if [[ -f celery_beat.pid ]] && kill -0 $(cat celery_beat.pid) 2>/dev/null; then
    echo "✅ Celery Beat: Running (PID: $(cat celery_beat.pid))"
else
    echo "❌ Celery Beat: Not running"
fi

# Check web application
if curl -s http://localhost:5000/api/health > /dev/null 2>&1; then
    echo "✅ Web Application: Running (http://localhost:5000)"
else
    echo "❌ Web Application: Not running"
fi

echo ""
echo "📊 Queue Status:"
source venv/bin/activate
python3 -c "
import redis
try:
    r = redis.Redis(host='localhost', port=6379, db=0)
    redaction_queue = r.llen('redaction')
    maintenance_queue = r.llen('maintenance')
    print(f'   Redaction queue: {redaction_queue} jobs')
    print(f'   Maintenance queue: {maintenance_queue} jobs')
except Exception as e:
    print(f'   Error checking queues: {e}')
" 2>/dev/null
EOF

    # Make scripts executable
    chmod +x start-redactify.sh stop-redactify.sh status-redactify.sh
    
    # Create logs directory
    mkdir -p logs
    
    log_success "Startup scripts created"
}

verify_installation() {
    log_info "Verifying installation..."
    
    cd "$INSTALL_DIR"
    source venv/bin/activate
    
    # Test imports
    python3 -c "
import sys
sys.path.append('Redactify')
try:
    from Redactify.app import create_app
    from Redactify.services.celery_service import celery
    import presidio_analyzer
    import presidio_anonymizer
    print('✅ Core dependencies verified')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"
    
    # Test Redis connection
    python3 -c "
import redis
try:
    r = redis.Redis(host='localhost', port=6379, db=0)
    r.ping()
    print('✅ Redis connection verified')
except Exception as e:
    print(f'❌ Redis error: {e}')
    sys.exit(1)
"
    
    # Test GPU availability (optional)
    python3 -c "
try:
    import tensorflow as tf
    if tf.config.list_physical_devices('GPU'):
        print('✅ GPU support available')
    else:
        print('ℹ️  GPU libraries installed but no GPU detected (CPU mode)')
except ImportError:
    print('ℹ️  GPU libraries not installed (CPU mode)')
" 2>/dev/null || true
    
    log_success "Installation verification completed"
}

print_completion_message() {
    echo ""
    echo -e "${GREEN}🎉 Redactify Installation Completed Successfully!${NC}"
    echo "================================================="
    echo ""
    echo -e "${BLUE}📍 Installation Location:${NC} $INSTALL_DIR"
    echo -e "${BLUE}🔧 Available Commands:${NC}"
    echo "   • Start services:    ./start-redactify.sh"
    echo "   • Stop services:     ./stop-redactify.sh"
    echo "   • Check status:      ./status-redactify.sh"
    echo ""
    echo -e "${BLUE}🌐 Access Information:${NC}"
    echo "   • Web Interface:     http://localhost:5000"
    echo "   • API Documentation: http://localhost:5000/docs"
    echo ""
    echo -e "${BLUE}📚 Next Steps:${NC}"
    echo "   1. cd $INSTALL_DIR"
    echo "   2. ./start-redactify.sh"
    echo "   3. Open http://localhost:5000 in your browser"
    echo ""
    echo -e "${YELLOW}💡 Tips:${NC}"
    echo "   • Use './status-redactify.sh' to monitor services"
    echo "   • Check logs in the 'logs/' directory for troubleshooting"
    echo "   • Configuration can be modified in 'config.yaml'"
    echo ""
}

cleanup_temp_files() {
    # Remove any temporary installation files
    rm -rf /tmp/redactify-install-* 2>/dev/null || true
}

# Main installation process
main() {
    print_banner
    
    # Pre-installation checks
    check_github_access
    detect_os
    check_system_requirements
    
    # System setup
    install_system_dependencies
    setup_redis
    
    # Application setup
    clone_repository
    setup_python_environment
    install_python_dependencies
    download_nlp_models
    setup_configuration
    create_startup_scripts
    
    # Verification and cleanup
    verify_installation
    cleanup_temp_files
    
    # Completion
    print_completion_message
}

# Error handling
trap 'log_error "Installation failed. Check the error messages above."; exit 1' ERR

# Run main installation
main "$@"
