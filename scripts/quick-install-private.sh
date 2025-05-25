#!/bin/bash
# Redactify Quick Installation Script (Private Repository)
# One-line installer for Redactify PDF PII Redaction Tool
#
# Usage with Personal Access Token:
# export GITHUB_TOKEN="your_personal_access_token"
# curl -H "Authorization: token $GITHUB_TOKEN" \
#   -fsSL https://raw.githubusercontent.com/nishikantmandal007/Redactify/main/scripts/quick-install-private.sh | bash
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
             REDACTIFY INSTALLER
EOF
    echo -e "${NC}"
}

check_github_auth() {
    log_info "Checking GitHub authentication..."
    
    if [ -z "$GITHUB_TOKEN" ]; then
        log_error "GITHUB_TOKEN environment variable is required for private repository access"
        log_info "Please set your GitHub Personal Access Token:"
        log_info "export GITHUB_TOKEN=\"your_personal_access_token\""
        log_info ""
        log_info "To create a Personal Access Token:"
        log_info "1. Go to https://github.com/settings/tokens"
        log_info "2. Click 'Generate new token (classic)'"
        log_info "3. Select 'repo' scope for private repository access"
        log_info "4. Copy the generated token"
        exit 1
    fi
    
    # Test GitHub API access
    if curl -s -H "Authorization: token $GITHUB_TOKEN" \
         https://api.github.com/repos/nishikantmandal007/Redactify > /dev/null 2>&1; then
        log_success "✅ GitHub authentication successful"
    else
        log_error "❌ GitHub authentication failed"
        log_info "Please check your GITHUB_TOKEN and repository access"
        exit 1
    fi
}

detect_os() {
    log_info "Detecting operating system..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ -f /etc/debian_version ]; then
            OS="debian"
            log_success "✅ Detected Debian/Ubuntu system"
        elif [ -f /etc/redhat-release ]; then
            OS="rhel"
            log_success "✅ Detected RHEL/CentOS/Fedora system"
        else
            OS="linux"
            log_warning "⚠️  Generic Linux detected, using generic package manager"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        log_success "✅ Detected macOS system"
    else
        log_error "❌ Unsupported operating system: $OSTYPE"
        exit 1
    fi
}

install_system_deps() {
    log_info "Installing system dependencies..."
    
    case $OS in
        debian)
            sudo apt-get update
            sudo apt-get install -y \
                python3 python3-pip python3-venv \
                redis-server \
                git \
                poppler-utils \
                tesseract-ocr \
                build-essential \
                libssl-dev \
                libffi-dev \
                python3-dev \
                pkg-config \
                libhdf5-dev
            ;;
        rhel)
            sudo yum update -y
            sudo yum install -y \
                python3 python3-pip \
                redis \
                git \
                poppler-utils \
                tesseract \
                gcc gcc-c++ make \
                openssl-devel \
                libffi-devel \
                python3-devel \
                pkgconfig \
                hdf5-devel
            ;;
        macos)
            if ! command -v brew &> /dev/null; then
                log_error "❌ Homebrew not found. Please install Homebrew first:"
                log_info "https://brew.sh"
                exit 1
            fi
            brew update
            brew install \
                python@3.10 \
                redis \
                git \
                poppler \
                tesseract \
                pkg-config \
                hdf5
            ;;
        *)
            log_error "❌ Package installation not supported for this OS"
            exit 1
            ;;
    esac
    
    log_success "✅ System dependencies installed"
}

setup_python_env() {
    log_info "Setting up Python environment..."
    
    # Create installation directory
    mkdir -p "$INSTALL_DIR"
    cd "$INSTALL_DIR"
    
    # Clone repository with authentication
    if [ -d "Redactify" ]; then
        log_info "Repository already exists, updating..."
        cd Redactify
        git pull https://$GITHUB_TOKEN@github.com/nishikantmandal007/Redactify.git
    else
        log_info "Cloning repository..."
        git clone https://$GITHUB_TOKEN@github.com/nishikantmandal007/Redactify.git
        cd Redactify
    fi
    
    # Create virtual environment
    python3 -m venv venv
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install Python dependencies
    pip install -r Redactify/requirements.txt
    
    log_success "✅ Python environment setup complete"
}

download_nlp_models() {
    log_info "Downloading NLP models..."
    
    source "$INSTALL_DIR/Redactify/venv/bin/activate"
    
    # Download spaCy models
    python -m spacy download en_core_web_sm
    python -m spacy download en_core_web_lg
    
    log_success "✅ NLP models downloaded"
}

setup_configuration() {
    log_info "Setting up configuration..."
    
    cd "$INSTALL_DIR/Redactify"
    
    # Copy example configuration if it doesn't exist
    if [ ! -f "config.yaml" ]; then
        if [ -f "config.example.yaml" ]; then
            cp config.example.yaml config.yaml
            log_info "Created config.yaml from example"
        else
            # Create basic configuration
            cat > config.yaml << EOF
# Redactify Configuration
app:
  host: "127.0.0.1"
  port: 5000
  debug: false

redis:
  host: "localhost"
  port: 6379
  db: 0

upload:
  max_file_size: 50MB
  allowed_extensions: [".pdf"]

security:
  secret_key: "$(openssl rand -hex 32)"
EOF
            log_info "Created basic config.yaml"
        fi
    fi
    
    log_success "✅ Configuration setup complete"
}

create_startup_scripts() {
    log_info "Creating startup scripts..."
    
    # Create startup script
    cat > "$INSTALL_DIR/start-redactify.sh" << EOF
#!/bin/bash
cd "$INSTALL_DIR/Redactify"
source venv/bin/activate

# Start Redis if not running
if ! pgrep redis-server > /dev/null; then
    redis-server --daemonize yes
fi

# Start Celery worker in background
celery -A Redactify.app.celery worker --loglevel=info &

# Start Flask application
python Redactify/app.py
EOF
    chmod +x "$INSTALL_DIR/start-redactify.sh"
    
    # Create stop script
    cat > "$INSTALL_DIR/stop-redactify.sh" << EOF
#!/bin/bash
pkill -f "celery.*Redactify"
pkill -f "python.*app.py"
EOF
    chmod +x "$INSTALL_DIR/stop-redactify.sh"
    
    log_success "✅ Startup scripts created"
}

verify_installation() {
    log_info "Verifying installation..."
    
    cd "$INSTALL_DIR/Redactify"
    source venv/bin/activate
    
    # Test imports
    if python -c "import Redactify; print('Redactify imported successfully')"; then
        log_success "✅ Redactify installation verified"
    else
        log_error "❌ Redactify installation verification failed"
        return 1
    fi
    
    # Check Redis
    if command -v redis-cli &> /dev/null && redis-cli ping > /dev/null 2>&1; then
        log_success "✅ Redis is accessible"
    else
        log_warning "⚠️  Redis may not be running"
    fi
}

main() {
    print_banner
    
    log_info "Starting Redactify installation..."
    log_info "Installation directory: $INSTALL_DIR"
    echo
    
    check_github_auth
    detect_os
    install_system_deps
    setup_python_env
    download_nlp_models
    setup_configuration
    create_startup_scripts
    verify_installation
    
    echo
    log_success "🎉 Redactify installation complete!"
    echo
    log_info "To start Redactify:"
    log_info "  $INSTALL_DIR/start-redactify.sh"
    echo
    log_info "To stop Redactify:"
    log_info "  $INSTALL_DIR/stop-redactify.sh"
    echo
    log_info "Access the web interface at: http://localhost:5000"
}

main "$@"
