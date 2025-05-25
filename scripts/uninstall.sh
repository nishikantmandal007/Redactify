#!/bin/bash
# Redactify Uninstaller Script
# Safely removes Redactify and all associated components

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
INSTALL_DIR="$HOME/redactify"

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
    echo -e "${RED}"
    echo "======================================================"
    echo "🗑️  Redactify Uninstaller"
    echo "======================================================"
    echo -e "${NC}"
    log_warning "This will remove Redactify and all associated files"
}

confirm_uninstall() {
    echo ""
    log_warning "The following will be removed:"
    echo "  • Redactify installation directory: $INSTALL_DIR"
    echo "  • Python virtual environment and all packages"
    echo "  • Configuration files and uploaded files"
    echo "  • Temporary processing files"
    echo "  • Startup scripts"
    
    if systemctl list-unit-files redactify.service &> /dev/null; then
        echo "  • Systemd service (redactify.service)"
    fi
    
    echo ""
    echo -e "${RED}⚠️  This action cannot be undone!${NC}"
    echo ""
    
    read -p "Are you sure you want to uninstall Redactify? (type 'yes' to confirm): " confirmation
    
    if [[ "$confirmation" != "yes" ]]; then
        log_info "Uninstallation cancelled"
        exit 0
    fi
}

stop_services() {
    log_info "Stopping Redactify services..."
    
    # Stop systemd service if it exists
    if systemctl list-unit-files redactify.service &> /dev/null 2>&1; then
        log_info "Stopping systemd service..."
        sudo systemctl stop redactify.service || true
        sudo systemctl disable redactify.service || true
    fi
    
    # Stop any running processes
    log_info "Stopping Celery workers..."
    pkill -f "celery.*worker.*Redactify" || true
    
    log_info "Stopping Celery beat..."
    pkill -f "celery.*beat.*Redactify" || true
    
    log_info "Stopping web server..."
    pkill -f "python.*Redactify.main" || true
    
    # Wait a moment for processes to stop
    sleep 2
    
    log_success "Services stopped"
}

remove_systemd_service() {
    if systemctl list-unit-files redactify.service &> /dev/null 2>&1; then
        log_info "Removing systemd service..."
        
        sudo systemctl stop redactify.service || true
        sudo systemctl disable redactify.service || true
        sudo rm -f /etc/systemd/system/redactify.service
        sudo systemctl daemon-reload
        
        log_success "Systemd service removed"
    fi
}

cleanup_redis_data() {
    log_info "Cleaning up Redis data..."
    
    # Only clean Redactify-specific data, not all Redis data
    if command -v redis-cli &> /dev/null && redis-cli ping &> /dev/null; then
        log_info "Clearing Redactify queues and results..."
        redis-cli del redaction maintenance celery-task-meta-* || true
        redis-cli eval "
        for i, name in ipairs(redis.call('KEYS', 'redactify:*')) do
            redis.call('DEL', name)
        end
        return #redis.call('KEYS', 'redactify:*')
        " 0 || true
        
        log_success "Redis data cleaned"
    else
        log_warning "Redis not accessible - skipping Redis cleanup"
    fi
}

remove_installation() {
    if [[ -d "$INSTALL_DIR" ]]; then
        log_info "Removing installation directory: $INSTALL_DIR"
        
        # Make sure we're not accidentally removing something else
        if [[ -f "$INSTALL_DIR/Redactify/main.py" ]] || [[ -f "$INSTALL_DIR/start-redactify.sh" ]]; then
            rm -rf "$INSTALL_DIR"
            log_success "Installation directory removed"
        else
            log_error "Directory $INSTALL_DIR doesn't appear to be a Redactify installation"
            log_error "Manual cleanup may be required"
        fi
    else
        log_warning "Installation directory not found: $INSTALL_DIR"
    fi
}

remove_temp_files() {
    log_info "Cleaning up temporary files..."
    
    # Clean up common temp locations
    rm -rf /tmp/redactify-* || true
    rm -rf /var/tmp/redactify-* || true
    
    log_success "Temporary files cleaned"
}

offer_redis_removal() {
    echo ""
    read -p "Do you want to remove Redis as well? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Removing Redis..."
        
        # Stop Redis
        sudo systemctl stop redis-server || sudo systemctl stop redis || true
        
        # Remove Redis packages based on OS
        if command -v apt &> /dev/null; then
            sudo apt remove -y redis-server redis-tools || true
            sudo apt autoremove -y || true
        elif command -v yum &> /dev/null; then
            sudo yum remove -y redis || true
        elif command -v dnf &> /dev/null; then
            sudo dnf remove -y redis || true
        elif command -v pacman &> /dev/null; then
            sudo pacman -R redis --noconfirm || true
        elif command -v brew &> /dev/null; then
            brew uninstall redis || true
        fi
        
        log_success "Redis removed"
    else
        log_info "Keeping Redis installed"
    fi
}

print_completion_message() {
    echo ""
    echo -e "${GREEN}✅ Redactify Uninstallation Complete${NC}"
    echo "======================================================"
    echo ""
    echo -e "${BLUE}📋 What was removed:${NC}"
    echo "  • Redactify application and all dependencies"
    echo "  • Python virtual environment"
    echo "  • Configuration and data files"
    echo "  • Startup scripts"
    echo "  • System services (if any)"
    echo ""
    echo -e "${BLUE}📋 What was preserved:${NC}"
    echo "  • System Python installation"
    echo "  • Other Python packages"
    echo "  • Redis (unless you chose to remove it)"
    echo ""
    echo -e "${GREEN}Thank you for using Redactify! 👋${NC}"
}

# Main execution
main() {
    print_banner
    confirm_uninstall
    stop_services
    remove_systemd_service
    cleanup_redis_data
    remove_installation
    remove_temp_files
    offer_redis_removal
    print_completion_message
}

# Run main function
main "$@"
