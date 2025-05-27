# CI/CD Pipeline Setup for Redactify

This document explains the GitHub Actions workflows for building and deploying Redactify.

## Workflows

### 1. Build and Push Docker Images (`.github/workflows/docker-ci.yml`)

**Triggers:**

- Push to `main` branch
- Pull requests to `main` branch

**What it does:**

- Builds Docker images for both web and celery services
- Pushes images to GitHub Container Registry (GHCR)
- Uses build caching for faster builds
- Only pushes on `main` branch (not on PRs)

**Image locations:**

- Web: `ghcr.io/nishikantmandal007/redactify/redactify-web:latest`
- Celery: `ghcr.io/nishikantmandal007/redactify/redactify-celery:latest`

### 2. Deploy to Azure (`.github/workflows/azure-deploy.yml`)

**Triggers:**

- After successful completion of "Build and Push Docker Images" workflow
- Manual trigger via GitHub Actions UI

**What it does:**

- Creates Azure resources (Resource Group, Container Apps Environment, Redis)
- Deploys web and celery worker containers to Azure Container Apps
- Sets up environment variables and networking

## Setup Instructions

### 1. GitHub Container Registry Permissions

The workflows automatically use `GITHUB_TOKEN` with package write permissions. No additional setup needed.

### 2. Azure Setup (for Azure deployment)

1. **Create Azure Service Principal:**

   ```bash
   az ad sp create-for-rbac --name "redactify-github-actions" \
     --role contributor \
     --scopes /subscriptions/{subscription-id} \
     --sdk-auth
   ```

2. **Add GitHub Secret:**
   - Go to GitHub repository → Settings → Secrets and variables → Actions
   - Add secret named `AZURE_CREDENTIALS` with the JSON output from step 1

3. **Install Azure CLI Extensions:**

   ```bash
   az extension add --name containerapp
   ```

### 3. Manual Deployment

You can also trigger deployments manually:

- Go to GitHub repository → Actions → "Deploy to Azure" → "Run workflow"

## Cost Estimation for Azure Student Credits

**Estimated monthly costs:**

- Container Apps (2 apps): ~$25-35/month
- Azure Cache for Redis (Basic C0): ~$15/month
- **Total: ~$40-50/month** (well within $100 student credit)

## Monitoring and Troubleshooting

### Check deployment status

```bash
# List all container apps
az containerapp list --resource-group redactify-rg -o table

# Check app logs
az containerapp logs show --name redactify-web --resource-group redactify-rg

# Check Redis status
az redis show --name redactify-redis --resource-group redactify-rg
```

### Manual Docker commands for testing

```bash
# Pull and run locally
docker pull ghcr.io/nishikantmandal007/redactify/redactify-web:latest
docker run -p 5000:5000 ghcr.io/nishikantmandal007/redactify/redactify-web:latest
```

## Environment Variables

Both workflows set these environment variables:

- `REDACTIFY_REDIS_URL`: Redis connection string
- `FLASK_ENV`: Set to `production`

## Security Notes

- Docker images are stored in GitHub Container Registry (private by default)
- Azure resources use managed identities where possible
- Redis connection uses SSL and authentication keys
- Container Apps ingress is configured for HTTPS only

## Next Steps

1. Push your code to GitHub to trigger the first build
2. Set up Azure credentials for deployment
3. Monitor your Azure spending in the Azure portal
4. Consider adding health checks and monitoring
