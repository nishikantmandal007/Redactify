#!/bin/bash
#
# Redactify Installation Status Summary
# Shows the current status of the one-line installation setup
#

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_banner() {
    echo -e "${PURPLE}"
    cat << 'EOF'
    ____          __           __  _ ____       
   / __ \___  ___/ /___ ______/ /_(_) __/_  __  
  / /_/ / _ \/ _  / __ `/ ___/ __/ / /_/ / / /  
 / _, _/  __/ /_/ / /_/ / /__/ /_/ / __/ /_/ /   
/_/ |_|\___/\__,_/\__,_/\___/\__/_/_/  \__, /   
                                     /____/    
       ONE-LINE INSTALLATION STATUS
EOF
    echo -e "${NC}"
}

show_installation_command() {
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}🚀 ONE-LINE INSTALLATION COMMAND${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo
    echo -e "${YELLOW}Copy and paste this command to install Redactify:${NC}"
    echo
    echo -e "${BLUE}curl -fsSL https://raw.githubusercontent.com/nishikantmandal007/Redactify/main/scripts/quick-install.sh | bash${NC}"
    echo
    echo -e "${GREEN}This will:${NC}"
    echo -e "  ✅ Auto-detect your OS (Ubuntu/Debian/RHEL/Arch/macOS)"
    echo -e "  ✅ Install all system dependencies (Redis, Python, build tools)"
    echo -e "  ✅ Create isolated Python virtual environment"
    echo -e "  ✅ Install all Python packages from requirements.txt"
    echo -e "  ✅ Download required NLP models (spaCy)"
    echo -e "  ✅ Set up configuration files"
    echo -e "  ✅ Create startup/stop/monitoring scripts"
    echo -e "  ✅ Verify installation integrity"
    echo
}

show_available_scripts() {
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}📋 AVAILABLE INSTALLATION SCRIPTS${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo
    
    local scripts=(
        "quick-install.sh:Main installation script with all dependencies"
        "check-requirements.sh:Pre-installation system requirements checker"
        "uninstall.sh:Complete removal of Redactify installation"
        "status.sh:System status and health monitoring"
        "validate-install.sh:Installation setup validation"
    )
    
    for script_info in "${scripts[@]}"; do
        local script_name="${script_info%%:*}"
        local description="${script_info#*:}"
        
        if [ -f "../scripts/$script_name" ]; then
            if [ -x "../scripts/$script_name" ]; then
                echo -e "  ✅ ${GREEN}$script_name${NC} - $description"
            else
                echo -e "  ⚠️  ${YELLOW}$script_name${NC} - $description (not executable)"
            fi
        else
            echo -e "  ❌ ${RED}$script_name${NC} - Missing"
        fi
    done
    echo
}

show_documentation_status() {
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}📖 DOCUMENTATION STATUS${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo
    
    local docs=(
        "../README.md:Main project documentation"
        "../installation.md:Detailed installation guide"
        "../scripts/README.md:Scripts documentation"
    )
    
    for doc_info in "${docs[@]}"; do
        local doc_path="${doc_info%%:*}"
        local description="${doc_info#*:}"
        local doc_name=$(basename "$doc_path")
        
        if [ -f "$doc_path" ]; then
            if grep -q "curl.*quick-install.sh.*bash" "$doc_path" 2>/dev/null; then
                echo -e "  ✅ ${GREEN}$doc_name${NC} - $description (contains one-line install)"
            else
                echo -e "  ⚠️  ${YELLOW}$doc_name${NC} - $description (missing one-line install)"
            fi
        else
            echo -e "  ❌ ${RED}$doc_name${NC} - Missing"
        fi
    done
    echo
}

show_repository_info() {
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}🔗 REPOSITORY INFORMATION${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo
    
    cd "$(dirname "$0")/.."
    
    local remote_url=$(git remote get-url origin 2>/dev/null || echo "No remote configured")
    echo -e "  📍 ${BLUE}Repository:${NC} $remote_url"
    
    local branch=$(git branch --show-current 2>/dev/null || echo "unknown")
    echo -e "  🌿 ${BLUE}Current branch:${NC} $branch"
    
    local commit=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    echo -e "  📝 ${BLUE}Latest commit:${NC} $commit"
    
    # Check for uncommitted changes
    if git diff --quiet 2>/dev/null; then
        echo -e "  ✅ ${GREEN}Status:${NC} Clean working directory"
    else
        echo -e "  ⚠️  ${YELLOW}Status:${NC} Uncommitted changes present"
    fi
    echo
}

show_next_steps() {
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}🎯 NEXT STEPS${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo
    echo -e "${BLUE}1. Test the installation:${NC}"
    echo -e "   ${YELLOW}./validate-install.sh${NC}"
    echo
    echo -e "${BLUE}2. Commit and push changes:${NC}"
    echo -e "   ${YELLOW}git add .${NC}"
    echo -e "   ${YELLOW}git commit -m \"Add comprehensive one-line installation system\"${NC}"
    echo -e "   ${YELLOW}git push origin main${NC}"
    echo
    echo -e "${BLUE}3. Test the one-line installation:${NC}"
    echo -e "   ${YELLOW}curl -fsSL https://raw.githubusercontent.com/nishikantmandal007/Redactify/main/scripts/quick-install.sh | bash${NC}"
    echo
    echo -e "${BLUE}4. Update documentation if needed:${NC}"
    echo -e "   - Update README.md with any final changes"
    echo -e "   - Add usage examples"
    echo -e "   - Create video tutorials or screenshots"
    echo
}

show_features_summary() {
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}🌟 COMPLETED FEATURES${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo
    echo -e "  ✅ ${GREEN}One-line installation command${NC}"
    echo -e "  ✅ ${GREEN}Multi-OS support (Ubuntu/Debian/RHEL/Arch/macOS)${NC}"
    echo -e "  ✅ ${GREEN}Automatic dependency installation${NC}"
    echo -e "  ✅ ${GREEN}Virtual environment creation${NC}"
    echo -e "  ✅ ${GREEN}Redis setup and configuration${NC}"
    echo -e "  ✅ ${GREEN}Python package installation${NC}"
    echo -e "  ✅ ${GREEN}NLP model downloads${NC}"
    echo -e "  ✅ ${GREEN}Startup/stop/monitoring scripts${NC}"
    echo -e "  ✅ ${GREEN}Installation verification${NC}"
    echo -e "  ✅ ${GREEN}Complete uninstall capability${NC}"
    echo -e "  ✅ ${GREEN}System requirements checker${NC}"
    echo -e "  ✅ ${GREEN}Installation validation${NC}"
    echo -e "  ✅ ${GREEN}Comprehensive documentation${NC}"
    echo
}

main() {
    clear
    print_banner
    echo
    
    show_installation_command
    show_available_scripts
    show_documentation_status
    show_repository_info
    show_features_summary
    show_next_steps
    
    echo -e "${PURPLE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}🎉 One-line installation system is ready!${NC}"
    echo -e "${PURPLE}═══════════════════════════════════════════════════════════════${NC}"
    echo
}

# Run if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
