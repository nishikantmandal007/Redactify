#!/bin/bash
# Redactify Quick Installation Script
# One-line installer for Redactify PDF PII Redaction Tool
#
# Usage: curl -fsSL https://raw.githubusercontent.com/nishikantmandal007/Redactify/main/scripts/quick-install.sh | bash
#
# This script automatically:
# - Detects OS and installs system dependencies
# - Creates a Python virtual environment
# - Installs Python dependencies
# - Installs and configures Redis
# - Downloads required NLP models
# - Sets up basic configuration
# - Starts services

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REPO_URL="https://github.com/nishikantmandal007/Redactify.git"
INSTALL_DIR="$HOME/redactify"
PYTHON_VERSION="3.10"
REDIS_VERSION="7"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_banner() {
    echo -e "${BLUE}"
    cat << 'EOF'
    ____          __           __  _ ____      
   / __ \___  ___/ /___ ______/ /_(_) __/_  __
  / /_/ / _ \/ _  / __ `/ ___/ __/ / /_/ / / /
 / _, _/  __/ /_/ / /_/ / /__/ /_/ / __/ /_/ / 
/_/ |_|\___/\__,_/\__,_/\___/\__/_/_/  \__, /  
                                     /____/   
        
🔒 Intelligent PII Redaction System
📄 Automated PDF Privacy Protection
EOF
    echo -e "${NC}"
    echo "======================================================"
    log_info "Starting Redactify Quick Installation..."
    echo "======================================================"
}

check_requirements() {
    log_info "Checking system requirements..."
    
    # Check if running as root
    if [[ $EUID -eq 0 ]]; then
        log_error "This script should not be run as root for security reasons"
        log_info "Please run as a regular user with sudo privileges"
        exit 1
    fi
    
    # Check available memory
    MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
    if [[ $MEMORY_GB -lt 4 ]]; then
        log_warning "System has less than 4GB RAM. Redactify may run slowly."
        log_info "Recommended: 8GB+ RAM for optimal performance"
    fi
    
    # Check available disk space
    DISK_GB=$(df -BG . | awk 'NR==2{print $4}' | sed 's/G//')
    if [[ $DISK_GB -lt 5 ]]; then
        log_error "Insufficient disk space. Need at least 5GB free space."
        exit 1
    fi
    
    log_success "System requirements check passed"
}

detect_os() {
    log_info "Detecting operating system..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ -f /etc/debian_version ]; then
            OS="debian"
            PKG_MANAGER="apt"
            log_info "Detected: Debian/Ubuntu Linux"
        elif [ -f /etc/redhat-release ]; then
            OS="redhat"
            PKG_MANAGER="yum"
            log_info "Detected: Red Hat/CentOS/Fedora Linux"
        elif [ -f /etc/arch-release ]; then
            OS="arch"
            PKG_MANAGER="pacman"
            log_info "Detected: Arch Linux"
        else
            OS="linux"
            log_warning "Unknown Linux distribution. Manual installation may be required."
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        PKG_MANAGER="brew"
        log_info "Detected: macOS"
    else
        log_error "Unsupported operating system: $OSTYPE"
        log_info "Supported: Linux (Debian/Ubuntu/RHEL/Arch), macOS"
        exit 1
    fi
}

install_system_deps() {
    log_info "Installing system dependencies..."
    
    case $OS in
        "debian")
            log_info "Updating package lists..."
            sudo apt update
            
            log_info "Installing system packages..."
            sudo apt install -y \
                python${PYTHON_VERSION} \
                python${PYTHON_VERSION}-venv \
                python${PYTHON_VERSION}-dev \
                python3-pip \
                redis-server \
                git \
                curl \
                wget \
                build-essential \
                pkg-config \
                poppler-utils \
                tesseract-ocr \
                libtesseract-dev \
                libgl1-mesa-glx \
                libglib2.0-0 \
                libsm6 \
                libxext6 \
                libxrender-dev \
                libgomp1 \
                nodejs \
                npm
                
            # Start and enable Redis
            sudo systemctl start redis-server
            sudo systemctl enable redis-server
            ;;
            
        "redhat")
            log_info "Installing EPEL repository..."
            sudo yum install -y epel-release
            
            log_info "Installing system packages..."
            sudo yum install -y \
                python${PYTHON_VERSION} \
                python${PYTHON_VERSION}-devel \
                python3-pip \
                redis \
                git \
                curl \
                wget \
                gcc \
                gcc-c++ \
                make \
                poppler-utils \
                tesseract \
                tesseract-devel \
                mesa-libGL \
                glib2 \
                libSM \
                libXext \
                libXrender \
                libgomp \
                nodejs \
                npm
                
            # Start and enable Redis
            sudo systemctl start redis
            sudo systemctl enable redis
            ;;
            
        "arch")
            log_info "Updating package database..."
            sudo pacman -Sy
            
            log_info "Installing system packages..."
            sudo pacman -S --noconfirm \
                python \
                python-pip \
                redis \
                git \
                curl \
                wget \
                base-devel \
                poppler \
                tesseract \
                tesseract-data-eng \
                mesa \
                glib2 \
                libsm \
                libxext \
                libxrender \
                nodejs \
                npm
                
            # Start and enable Redis
            sudo systemctl start redis
            sudo systemctl enable redis
            ;;
            
        "macos")
            # Check if Homebrew is installed
            if ! command -v brew &> /dev/null; then
                log_info "Installing Homebrew..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            fi
            
            log_info "Installing system packages..."
            brew update
            brew install \
                python@${PYTHON_VERSION} \
                redis \
                git \
                poppler \
                tesseract \
                node
                
            # Start Redis
            brew services start redis
            ;;
    esac
    
    log_success "System dependencies installed"
}

setup_python_env() {
    log_info "Setting up Python environment..."
    
    # Create installation directory
    if [ -d "$INSTALL_DIR" ]; then
        log_warning "Installation directory already exists: $INSTALL_DIR"
        read -p "Remove existing installation? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$INSTALL_DIR"
        else
            log_error "Installation cancelled"
            exit 1
        fi
    fi
    
    mkdir -p "$INSTALL_DIR"
    cd "$INSTALL_DIR"
    
    # Clone repository
    log_info "Cloning Redactify repository..."
    git clone "$REPO_URL" .
    
    # Create virtual environment
    log_info "Creating Python virtual environment..."
    python${PYTHON_VERSION} -m venv venv
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    log_info "Upgrading pip..."
    pip install --upgrade pip setuptools wheel
    
    log_success "Python environment created"
}

install_python_deps() {
    log_info "Installing Python dependencies..."
    
    cd "$INSTALL_DIR"
    source venv/bin/activate
    
    # Install main requirements
    log_info "Installing core dependencies..."
    pip install -r Redactify/requirements.txt
    
    # Check for GPU support
    if command -v nvidia-smi &> /dev/null; then
        log_info "NVIDIA GPU detected. Installing GPU dependencies..."
        pip install -r Redactify/requirements_gpu.txt
        GPU_AVAILABLE=true
    else
        log_info "No NVIDIA GPU detected. Using CPU-only mode."
        GPU_AVAILABLE=false
    fi
    
    log_success "Python dependencies installed"
}

download_nlp_models() {
    log_info "Downloading NLP models..."
    
    cd "$INSTALL_DIR"
    source venv/bin/activate
    
    # Download spaCy models
    log_info "Downloading spaCy English model..."
    python -m spacy download en_core_web_lg
    
    # Try to download transformer model if GPU is available
    if [ "$GPU_AVAILABLE" = true ]; then
        log_info "Downloading transformer model for better accuracy..."
        python -m spacy download en_core_web_trf || {
            log_warning "Failed to download transformer model. Using large model instead."
        }
    fi
    
    log_success "NLP models downloaded"
}

setup_configuration() {
    log_info "Setting up configuration..."
    
    cd "$INSTALL_DIR"
    
    # Create basic configuration if it doesn't exist
    if [ ! -f "config.yaml" ]; then
        log_info "Creating default configuration..."
        cp config.yaml config.yaml
    fi
    
    # Create necessary directories
    mkdir -p upload_files temp_files
    touch upload_files/.gitkeep temp_files/.gitkeep
    
    # Set proper permissions
    chmod 755 upload_files temp_files
    
    log_success "Configuration setup complete"
}

verify_installation() {
    log_info "Verifying installation..."
    
    cd "$INSTALL_DIR"
    source venv/bin/activate
    
    # Test Python environment
    python --version || {
        log_error "Python verification failed"
        exit 1
    }
    
    # Test Redis connectivity
    python -c "import redis; r = redis.Redis(); r.ping(); print('✅ Redis connected')" || {
        log_error "Redis verification failed"
        exit 1
    }
    
    # Test spaCy model
    python -c "import spacy; nlp = spacy.load('en_core_web_lg'); print('✅ spaCy model loaded')" || {
        log_error "spaCy model verification failed"
        exit 1
    }
    
    # Test core imports
    python -c "
import sys
sys.path.append('Redactify')
try:
    import presidio_analyzer
    import paddleocr
    import celery
    import flask
    print('✅ Core dependencies working')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
" || {
        log_error "Core dependencies verification failed"
        exit 1
    }
    
    # Test GPU availability if applicable
    if [ "$GPU_AVAILABLE" = true ]; then
        python -c "
import tensorflow as tf
if tf.config.list_physical_devices('GPU'):
    print('✅ GPU available')
else:
    print('ℹ️  GPU libraries installed but no GPU detected')
" || {
        log_warning "GPU verification failed, but installation can continue"
    }
    fi
    
    log_success "Installation verification complete"
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

# Check if Redis is running
if ! redis-cli ping > /dev/null 2>&1; then
    echo "❌ Redis is not running. Please start Redis first:"
    echo "   sudo systemctl start redis-server  # Linux"
    echo "   brew services start redis          # macOS"
    exit 1
fi

echo "✅ Redis is running"

# Start Celery worker in background
echo "🔄 Starting Celery worker..."
celery -A Redactify.services.celery_service.celery worker \
    --loglevel=info --concurrency=4 -Q redaction \
    --hostname=redaction@%h --detach

# Start maintenance worker in background
echo "🧹 Starting maintenance worker..."
celery -A Redactify.services.celery_service.celery worker \
    --loglevel=info --concurrency=1 -Q maintenance \
    --hostname=maintenance@%h --detach

# Wait a moment for workers to start
sleep 2

echo "🌐 Starting web server..."
echo "📱 Redactify will be available at: http://localhost:5000"
echo "🔄 Celery monitoring at: http://localhost:5555 (if Flower is installed)"
echo ""
echo "Press Ctrl+C to stop the web server"
echo "Note: Celery workers will continue running in background"

# Start the web application
python -m Redactify.main --host 0.0.0.0 --port 5000
EOF

    # Create stop script
    cat > stop-redactify.sh << 'EOF'
#!/bin/bash
# Redactify Stop Script

echo "🛑 Stopping Redactify Services..."

# Stop Celery workers
echo "Stopping Celery workers..."
pkill -f "celery.*worker" || echo "No Celery workers running"

# Stop Celery beat if running
pkill -f "celery.*beat" || echo "No Celery beat running"

echo "✅ Redactify services stopped"
EOF

    # Create monitoring script
    cat > monitor-redactify.sh << 'EOF'
#!/bin/bash
# Redactify Monitoring Script

cd "$(dirname "$0")"
source venv/bin/activate

echo "📊 Redactify System Status"
echo "========================="

# Check Redis
if redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis: Running"
else
    echo "❌ Redis: Not running"
fi

# Check Celery workers
WORKER_COUNT=$(pgrep -f "celery.*worker" | wc -l)
echo "🔄 Celery Workers: $WORKER_COUNT running"

# Check web server
if pgrep -f "python.*Redactify.main" > /dev/null; then
    echo "🌐 Web Server: Running"
else
    echo "❌ Web Server: Not running"
fi

# Show queue status
echo ""
echo "📋 Queue Status:"
python -c "
import redis
try:
    r = redis.Redis()
    redaction_queue = r.llen('redaction')
    maintenance_queue = r.llen('maintenance')
    print(f'   Redaction queue: {redaction_queue} tasks')
    print(f'   Maintenance queue: {maintenance_queue} tasks')
except:
    print('   Unable to connect to Redis')
"

# Show system resources
echo ""
echo "💻 System Resources:"
echo "   Memory: $(free -h | awk '/^Mem:/ {print $3 "/" $2}')"
echo "   CPU Load: $(uptime | awk -F'load average:' '{print $2}')"
EOF

    # Make scripts executable
    chmod +x start-redactify.sh stop-redactify.sh monitor-redactify.sh
    
    log_success "Startup scripts created"
}

setup_systemd_service() {
    if [[ "$OS" == "linux" ]] && command -v systemctl &> /dev/null; then
        log_info "Setting up systemd service (optional)..."
        
        read -p "Create systemd service for auto-start? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            sudo tee /etc/systemd/system/redactify.service > /dev/null << EOF
[Unit]
Description=Redactify PII Redaction Service
After=network.target redis.service
Wants=redis.service

[Service]
Type=forking
User=$USER
WorkingDirectory=$INSTALL_DIR
Environment=PATH=$INSTALL_DIR/venv/bin
ExecStart=$INSTALL_DIR/start-redactify.sh
ExecStop=$INSTALL_DIR/stop-redactify.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
            
            sudo systemctl daemon-reload
            sudo systemctl enable redactify.service
            
            log_success "Systemd service created and enabled"
            log_info "Start with: sudo systemctl start redactify"
        fi
    fi
}

print_completion_message() {
    echo ""
    echo -e "${GREEN}🎉 Redactify Installation Complete! 🎉${NC}"
    echo "=========================================="
    echo ""
    echo -e "${BLUE}📍 Installation Location:${NC} $INSTALL_DIR"
    echo ""
    echo -e "${BLUE}🚀 Quick Start:${NC}"
    echo "   cd $INSTALL_DIR"
    echo "   ./start-redactify.sh"
    echo ""
    echo -e "${BLUE}🌐 Access Points:${NC}"
    echo "   • Web Interface: http://localhost:5000"
    echo "   • Monitoring: http://localhost:5555 (install flower: pip install flower)"
    echo ""
    echo -e "${BLUE}📖 Useful Commands:${NC}"
    echo "   • Start:    ./start-redactify.sh"
    echo "   • Stop:     ./stop-redactify.sh"
    echo "   • Monitor:  ./monitor-redactify.sh"
    echo "   • Logs:     tail -f /var/log/redis/redis-server.log"
    echo ""
    echo -e "${BLUE}🔧 Configuration:${NC}"
    echo "   • Config file: $INSTALL_DIR/config.yaml"
    echo "   • Upload dir:  $INSTALL_DIR/upload_files"
    echo "   • Temp dir:    $INSTALL_DIR/temp_files"
    echo ""
    if [ "$GPU_AVAILABLE" = true ]; then
        echo -e "${GREEN}⚡ GPU acceleration: ENABLED${NC}"
    else
        echo -e "${YELLOW}💻 GPU acceleration: DISABLED (CPU only)${NC}"
    fi
    echo ""
    echo -e "${BLUE}📚 Documentation:${NC}"
    echo "   • Full docs: $INSTALL_DIR/docs/"
    echo "   • Config guide: $INSTALL_DIR/docs/configuration.md"
    echo "   • Troubleshooting: $INSTALL_DIR/docs/troubleshooting.md"
    echo ""
    echo -e "${YELLOW}⚠️  Remember to:${NC}"
    echo "   • Keep Redis running for background tasks"
    echo "   • Review config.yaml for your specific needs"
    echo "   • Check firewall settings if accessing remotely"
    echo ""
    echo -e "${GREEN}Happy redacting! 🔒📄${NC}"
}

# Main execution
main() {
    print_banner
    check_requirements
    detect_os
    install_system_deps
    setup_python_env
    install_python_deps
    download_nlp_models
    setup_configuration
    verify_installation
    create_startup_scripts
    setup_systemd_service
    print_completion_message
}

# Run main function
main "$@"
