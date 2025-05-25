# 🔒 Private Repository Installation Guide

This guide explains how to install Redactify from a private GitHub repository.

## 🚀 Quick Start

### Option 1: Interactive Installation (Recommended)

The easiest way to install from a private repository:

```bash
curl -fsSL https://raw.githubusercontent.com/nishikantmandal007/Redactify/main/scripts/install-helper.sh | bash
```

This script will:

- Guide you through creating a GitHub Personal Access Token
- Test your authentication
- Automatically install Redactify
- Provide setup instructions

### Option 2: Manual Installation

If you already have a GitHub Personal Access Token:

```bash
export GITHUB_TOKEN='your_personal_access_token'
curl -H "Authorization: token $GITHUB_TOKEN" \
  -fsSL https://raw.githubusercontent.com/nishikantmandal007/Redactify/main/scripts/quick-install-private.sh | bash
```

## 🔑 GitHub Personal Access Token Setup

### Creating a Token

1. Go to [GitHub Settings → Personal Access Tokens](https://github.com/settings/tokens)
2. Click **"Generate new token (classic)"**
3. Give it a descriptive name: `Redactify Installation`
4. Set expiration as needed (30 days, 90 days, or no expiration)
5. Select the following scopes:
   - ✅ **`repo`** - Full control of private repositories
6. Click **"Generate token"**
7. **Copy the token immediately** (you won't see it again!)

### Saving the Token

To avoid entering the token repeatedly, save it to your shell profile:

**For Bash users:**

```bash
echo 'export GITHUB_TOKEN="your_token_here"' >> ~/.bashrc
source ~/.bashrc
```

**For Zsh users:**

```bash
echo 'export GITHUB_TOKEN="your_token_here"' >> ~/.zshrc
source ~/.zshrc
```

## 🛡️ Security Best Practices

### Token Security

- Never share your Personal Access Token
- Use tokens with minimal required permissions
- Set reasonable expiration dates
- Revoke tokens when no longer needed

### Repository Access

- Only grant repository access to trusted users
- Use branch protection rules
- Enable two-factor authentication on your GitHub account

## 🔧 Troubleshooting

### Authentication Failed

```bash
# Test your token manually
curl -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/nishikantmandal007/Redactify
```

If this fails:

- Check that your token is correct
- Verify the repository name and owner
- Ensure the token has `repo` scope
- Check if the token has expired

### Token Not Set

```bash
# Check if token is set
echo $GITHUB_TOKEN
```

If empty, set it:

```bash
export GITHUB_TOKEN="your_token_here"
```

### Permission Denied

- Verify you have access to the private repository
- Check that the repository owner has granted you access
- Ensure the token has the correct permissions

## 📋 What Gets Installed

The private installation script installs the same components as the public version:

- ✅ System dependencies (Python, Redis, build tools)
- ✅ Python virtual environment
- ✅ All Python packages from requirements.txt
- ✅ NLP models (spaCy)
- ✅ Configuration files
- ✅ Startup/stop scripts
- ✅ Service verification

**Installation Location:** `~/redactify`

## 🔗 Related Documentation

- [Main Installation Guide](../installation.md)
- [Scripts Overview](README.md)
- [Configuration Guide](../docs/configuration.md)
- [Troubleshooting](../docs/troubleshooting.md)

## 🤝 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review the main [troubleshooting guide](../docs/troubleshooting.md)
3. Contact the repository maintainer
4. Create an issue (if you have repository access)

---

**Happy Secure Redacting! 🔒📄**
