#!/bin/bash
#
# Redactify Installation Validation Script
# Validates that the quick-install.sh script is properly configured
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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
                 VALIDATION SCRIPT              
EOF
    echo -e "${NC}"
}

validate_script_structure() {
    local script_path="./quick-install.sh"
    
    if [ ! -f "$script_path" ]; then
        log_error "quick-install.sh not found in current directory"
        return 1
    fi
    
    log_info "Validating installation script structure..."
    
    # Check if script has proper repository URL
    if grep -q "nishikantmandal007/Redactify" "$script_path"; then
        log_success "✅ Correct repository URL found"
    else
        log_error "❌ Repository URL not found or incorrect"
        log_info "Looking for: nishikantmandal007/Redactify"
        log_info "Found URLs:"
        grep -n "github.com" "$script_path" || echo "No GitHub URLs found"
        return 1
    fi
    
    # Check for required functions
    local required_functions=(
        "detect_os"
        "install_system_deps"
        "setup_python_env"
        "download_nlp_models"
        "setup_configuration"
        "create_startup_scripts"
        "verify_installation"
    )
    
    for func in "${required_functions[@]}"; do
        if grep -q "^${func}()" "$script_path"; then
            log_success "✅ Function '$func' found"
        else
            log_error "❌ Function '$func' not found"
            return 1
        fi
    done
    
    # Check for required dependencies
    local required_deps=(
        "python"
        "redis"
        "git"
        "poppler-utils"
        "tesseract"
        "build-essential"
    )
    
    for dep in "${required_deps[@]}"; do
        if grep -q "$dep" "$script_path"; then
            log_success "✅ Dependency '$dep' installation found"
        else
            log_warning "⚠️  Dependency '$dep' may not be explicitly installed"
        fi
    done
    
    log_success "Script structure validation complete"
}

validate_documentation() {
    log_info "Validating documentation..."
    
    # Check README.md
    if [ -f "../README.md" ]; then
        if grep -q "curl.*quick-install.sh.*bash" "../README.md"; then
            log_success "✅ One-line installation found in README.md"
        else
            log_error "❌ One-line installation not found in README.md"
        fi
        
        if grep -q "nishikantmandal007/Redactify" "../README.md"; then
            log_success "✅ Correct repository URL in README.md"
        else
            log_error "❌ Repository URL incorrect in README.md"
        fi
    else
        log_warning "⚠️  README.md not found"
    fi
    
    # Check installation.md
    if [ -f "../installation.md" ]; then
        if grep -q "curl.*quick-install.sh.*bash" "../installation.md"; then
            log_success "✅ One-line installation found in installation.md"
        else
            log_error "❌ One-line installation not found in installation.md"
        fi
    else
        log_warning "⚠️  installation.md not found"
    fi
}

validate_requirements() {
    log_info "Validating requirements.txt..."
    
    if [ -f "../Redactify/requirements.txt" ]; then
        local req_file="../Redactify/requirements.txt"
        
        # Check for critical packages
        local critical_packages=(
            "flask"
            "celery"
            "redis"
            "presidio-analyzer"
            "presidio-anonymizer"
            "paddleocr"
            "spacy"
        )
        
        for package in "${critical_packages[@]}"; do
            if grep -qi "$package" "$req_file"; then
                log_success "✅ Package '$package' found in requirements.txt"
            else
                log_warning "⚠️  Package '$package' not found in requirements.txt"
            fi
        done
    else
        log_error "❌ requirements.txt not found"
        return 1
    fi
}

validate_scripts() {
    log_info "Validating other scripts..."
    
    local scripts=("check-requirements.sh" "uninstall.sh" "status.sh")
    
    for script in "${scripts[@]}"; do
        if [ -f "./$script" ]; then
            if [ -x "./$script" ]; then
                log_success "✅ Script '$script' exists and is executable"
            else
                log_warning "⚠️  Script '$script' exists but is not executable"
            fi
        else
            log_error "❌ Script '$script' not found"
        fi
    done
}

check_git_status() {
    log_info "Checking Git status..."
    
    cd "$(dirname "$0")/.."
    
    if git remote -v | grep -q "nishikantmandal007/Redactify"; then
        log_success "✅ Git remote points to correct repository"
    else
        log_warning "⚠️  Git remote may not point to the expected repository"
    fi
    
    # Check if there are uncommitted changes to critical files
    local critical_files=(
        "README.md"
        "installation.md"
        "scripts/quick-install.sh"
        "scripts/README.md"
    )
    
    for file in "${critical_files[@]}"; do
        if git diff --quiet "$file" 2>/dev/null; then
            log_success "✅ No uncommitted changes in $file"
        else
            log_warning "⚠️  Uncommitted changes in $file"
        fi
    done
}

main() {
    print_banner
    
    log_info "Starting Redactify installation validation..."
    echo
    
    cd "$(dirname "$0")"
    
    validate_script_structure
    echo
    
    validate_documentation
    echo
    
    validate_requirements
    echo
    
    validate_scripts
    echo
    
    check_git_status
    echo
    
    log_success "🎉 Validation complete!"
    echo
    log_info "If all checks passed, your installation script is ready!"
    log_info "Test it with: curl -fsSL https://raw.githubusercontent.com/nishikantmandal007/Redactify/main/scripts/quick-install.sh | bash"
}

main "$@"
