# 🚀 Redactify Installation Scripts

This directory contains automated scripts for installing, managing, and maintaining Redactify.

## 📋 Available Scripts

### 🔧 Installation Scripts

#### `quick-install.sh` - One-Line Installation

**The recommended way to install Redactify**

```bash
curl -fsSL htt- ✅ Provides next steps for deployment

---

## 🎉 Final Words

**Happy Redacting! 🔒📄**

Your documents are now secure with Redactify - Secure • Automated • Intelligent

</div>githubusercontent.com/nishikantmandal007/Redactify/main/scripts/quick-install.sh | bash
```

**What it does:**

- ✅ Detects your operating system automatically
- ✅ Installs all system dependencies (Redis, Python, build tools)
- ✅ Creates isolated Python virtual environment in `~/redactify/venv`
- ✅ Installs all Python packages (Flask, Celery, Presidio, PaddleOCR, etc.)
- ✅ Downloads required NLP models (spaCy)
- ✅ Sets up configuration files
- ✅ Creates startup/stop scripts
- ✅ Verifies installation integrity
- ✅ Optionally sets up systemd service

**Supported Systems:**

- Ubuntu/Debian Linux
- RHEL/CentOS/Fedora Linux
- Arch Linux
- macOS

**Installation Location:** `~/redactify`

---

#### `check-requirements.sh` - System Requirements Checker

**Check if your system is ready for Redactify**

```bash
curl -fsSL https://raw.githubusercontent.com/nishikantmandal007/Redactify/main/scripts/check-requirements.sh | bash
```

**What it checks:**

- Python 3.10+ availability
- System memory (4GB minimum, 8GB recommended)
- Available disk space (5GB minimum)
- Git and curl availability
- Package manager detection
- Redis status
- GPU support (NVIDIA)
- Internet connectivity

**Use this before installation to avoid issues**

---

### 🗑️ Maintenance Scripts

#### `uninstall.sh` - Complete Removal

**Safely remove Redactify and all components**

```bash
curl -fsSL https://raw.githubusercontent.com/nishikantmandal007/Redactify/main/scripts/uninstall.sh | bash
```

**What it removes:**

- ✅ Complete Redactify installation
- ✅ Python virtual environment and all packages
- ✅ Configuration and data files
- ✅ Startup scripts
- ✅ Systemd services
- ✅ Redactify-specific Redis data
- ✅ Temporary files
- ✅ Optionally removes Redis server

**Safe:** Only removes Redactify-specific components

---

## 🏃‍♂️ Quick Start Workflow

### 1. Check Requirements (Optional)

```bash
curl -fsSL https://raw.githubusercontent.com/nishikantmandal007/Redactify/main/scripts/check-requirements.sh | bash
```

### 2. Install Redactify

```bash
curl -fsSL https://raw.githubusercontent.com/nishikantmandal007/Redactify/main/scripts/quick-install.sh | bash
```

### 3. Start Using Redactify

```bash
cd ~/redactify
./start-redactify.sh
```

Open **<http://localhost:5000>** in your browser

### 4. Managing Redactify

```bash
# Start all services
./start-redactify.sh

# Stop all services  
./stop-redactify.sh

# Check system status
./monitor-redactify.sh
```

---

## 🔧 Advanced Usage

### Custom Installation Directory

To install in a different location:

```bash
export REDACTIFY_INSTALL_DIR="/path/to/custom/location"
curl -fsSL https://raw.githubusercontent.com/nishikantmandal007/Redactify/main/scripts/quick-install.sh | bash
```

### GPU-Only Installation

To install with GPU support only:

```bash
export REDACTIFY_FORCE_GPU=true
curl -fsSL https://raw.githubusercontent.com/nishikantmandal007/Redactify/main/scripts/quick-install.sh | bash
```

### Offline Installation

For systems without internet access:

1. Download the repository and scripts on a connected machine
2. Transfer to the target machine
3. Run the local script:

```bash
./scripts/quick-install.sh --offline
```

---

## 🐛 Troubleshooting

### Installation Fails

1. **Check requirements first:**

   ```bash
   ./scripts/check-requirements.sh
   ```

2. **Run with debug output:**

   ```bash
   bash -x ./scripts/quick-install.sh
   ```

3. **Check logs:**

   ```bash
   # Installation logs
   tail -f /tmp/redactify-install.log
   
   # Runtime logs
   tail -f ~/redactify/logs/redactify.log
   ```

### Permission Issues

```bash
# Fix ownership
sudo chown -R $USER:$USER ~/redactify

# Fix permissions
chmod +x ~/redactify/*.sh
```

### Redis Connection Issues

```bash
# Check Redis status
redis-cli ping

# Restart Redis
sudo systemctl restart redis-server  # Linux
brew services restart redis          # macOS
```

### Virtual Environment Issues

```bash
# Recreate virtual environment
cd ~/redactify
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r Redactify/requirements.txt
```

---

## 🔒 Security Considerations

### Script Verification

Always verify scripts before running:

```bash
# Download and inspect first
curl -fsSL https://raw.githubusercontent.com/nishikantmandal007/Redactify/main/scripts/quick-install.sh > install.sh
less install.sh  # Review the script
bash install.sh  # Run after review
```

### Virtual Environment Isolation

All Python packages are installed in an isolated virtual environment at `~/redactify/venv`, ensuring no conflicts with system Python packages.

### Service Security

- Services run as your user (not root)
- Redis is configured for local access only
- Temporary files are automatically cleaned up
- Sensitive data is handled securely

---

## 📚 Additional Resources

- **Main Documentation:** [../README.md](../README.md)
- **Installation Guide:** [../installation.md](../installation.md)
- **Configuration:** [../docs/configuration.md](../docs/configuration.md)
- **Troubleshooting:** [../docs/troubleshooting.md](../docs/troubleshooting.md)
- **Developer Guide:** [../docs/developer-guide.md](../docs/developer-guide.md)

---

## 🤝 Contributing

Found an issue with the installation scripts? Please:

1. Check existing issues: [GitHub Issues](https://github.com/nishikantmandal007/Redactify/issues)
2. Create a new issue with:
   - Your operating system and version
   - Python version
   - Full error output
   - Steps to reproduce

---

<div align="center">

## 🔍 validate-install.sh

**Validates the installation setup**

```bash
./validate-install.sh
```

**What it does:**

- ✅ Validates installation script structure and configuration
- ✅ Checks all required functions and dependencies are present
- ✅ Verifies documentation contains correct URLs
- ✅ Validates requirements.txt has critical packages
- ✅ Checks script permissions and Git status
- ✅ Provides comprehensive installation readiness report

---

## 📊 installation-status.sh

**Shows the status of the one-line installation setup**

```bash
./installation-status.sh
```

**What it does:**

- ✅ Displays one-line installation command
- ✅ Lists all available installation scripts
- ✅ Checks documentation status
- ✅ Shows repository information
- ✅ Summarizes completed features
- ✅ Provides next steps for deployment

---

**Happy Redacting! 🔒📄**

**Secure • Automated • Intelligent**

</div>
