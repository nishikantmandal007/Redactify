#!/bin/bash
#
# Redactify Private Repository Installation Helper
# This script helps users install Redactify from a private GitHub repository
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
          PRIVATE REPOSITORY INSTALLER          
EOF
    echo -e "${NC}"
}

setup_github_token() {
    echo
    log_info "Setting up GitHub Personal Access Token..."
    echo
    
    if [ ! -z "$GITHUB_TOKEN" ]; then
        log_success "✅ GITHUB_TOKEN is already set"
        return 0
    fi
    
    echo -e "${YELLOW}To install from a private repository, you need a GitHub Personal Access Token.${NC}"
    echo
    echo "Steps to create a Personal Access Token:"
    echo "1. Go to https://github.com/settings/tokens"
    echo "2. Click 'Generate new token (classic)'"
    echo "3. Give it a descriptive name (e.g., 'Redactify Installation')"
    echo "4. Select the 'repo' scope for private repository access"
    echo "5. Click 'Generate token'"
    echo "6. Copy the generated token (you won't see it again!)"
    echo
    
    read -p "Do you already have a Personal Access Token? (y/n): " has_token
    
    if [[ "$has_token" =~ ^[Yy]$ ]]; then
        echo
        read -sp "Please enter your GitHub Personal Access Token: " token
        echo
        export GITHUB_TOKEN="$token"
        
        # Test the token
        log_info "Testing GitHub authentication..."
        if curl -s -H "Authorization: token $GITHUB_TOKEN" \
             https://api.github.com/repos/nishikantmandal007/Redactify > /dev/null 2>&1; then
            log_success "✅ GitHub authentication successful"
        else
            log_error "❌ GitHub authentication failed"
            log_info "Please check your token and repository access"
            exit 1
        fi
    else
        echo
        log_info "Please create a Personal Access Token first:"
        log_info "https://github.com/settings/tokens"
        echo
        log_info "Then run this script again."
        exit 1
    fi
}

run_installation() {
    log_info "Starting Redactify installation..."
    
    # Download and run the private installation script
    curl -H "Authorization: token $GITHUB_TOKEN" \
         -fsSL https://raw.githubusercontent.com/nishikantmandal007/Redactify/main/scripts/quick-install-private.sh | \
         GITHUB_TOKEN="$GITHUB_TOKEN" bash
}

save_token_instructions() {
    echo
    log_info "💡 To avoid entering the token again, you can save it to your shell profile:"
    echo
    echo "For bash users, add this to ~/.bashrc:"
    echo "export GITHUB_TOKEN=\"your_token_here\""
    echo
    echo "For zsh users, add this to ~/.zshrc:"
    echo "export GITHUB_TOKEN=\"your_token_here\""
    echo
    echo "Then restart your terminal or run: source ~/.bashrc (or ~/.zshrc)"
}

main() {
    print_banner
    
    log_info "Welcome to the Redactify Private Repository Installer"
    echo
    
    setup_github_token
    run_installation
    save_token_instructions
    
    echo
    log_success "🎉 Installation helper complete!"
}

main "$@"
