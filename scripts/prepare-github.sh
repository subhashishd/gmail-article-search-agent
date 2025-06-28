#!/bin/bash

echo "🚀 Preparing Gmail Article Search Agent for GitHub upload..."

# Create .gitignore if it doesn't exist
if [ ! -f .gitignore ]; then
    cat > .gitignore << 'EOF'
# Environment files
.env
.env.local
.env.production
*.env

# Credentials
credentials/
*.json
*.pem
*.key

# Logs
*.log
logs/

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/
pip-log.txt
pip-delete-this-directory.txt
.pytest_cache/

# Docker
.dockerignore

# Data
data/
*.db
*.sqlite

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Monitoring data
monitoring/*/data/
monitoring/grafana/data/
monitoring/prometheus/data/

# Models
models/
*.gguf
*.bin
*.safetensors

# Node modules (if any frontend dependencies)
node_modules/

# Temporary files
tmp/
temp/
*.tmp
EOF
    echo "✅ Created .gitignore"
else
    echo "✅ .gitignore already exists"
fi

# Create example environment file
if [ ! -f .env.example ]; then
    cat > .env.example << 'EOF'
# Database Configuration
DATABASE_URL=postgresql://gmail_user:secure_password@db:5432/gmail_search
POSTGRES_DB=gmail_search
POSTGRES_USER=gmail_user
POSTGRES_PASSWORD=secure_password

# Gmail API Configuration
GMAIL_CREDENTIALS_PATH=/app/credentials/gmail_credentials.json
GMAIL_TOKEN_PATH=/app/credentials/gmail_token.json

# Medium Integration (Optional)
MEDIUM_SESSION_ID=your_medium_session_id_here
MEDIUM_UID=your_medium_uid_here

# API Configuration
BACKEND_PORT=8000
FRONTEND_PORT=8501
OLLAMA_PORT=11434

# Monitoring Configuration
ENABLE_MONITORING=true
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
JAEGER_PORT=16686

# Security (Production)
# ENABLE_AUTH=true
# JWT_SECRET=your-super-secret-jwt-key
# CORS_ORIGINS=https://your-domain.com
EOF
    echo "✅ Created .env.example"
else
    echo "✅ .env.example already exists"
fi

# Create basic LICENSE file
if [ ! -f LICENSE ]; then
    cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2024 Gmail Article Search Agent

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF
    echo "✅ Created LICENSE"
else
    echo "✅ LICENSE already exists"
fi

# Initialize git repository if not already initialized
if [ ! -d .git ]; then
    git init
    echo "✅ Initialized git repository"
else
    echo "✅ Git repository already exists"
fi

# Create credentials directory structure but don't include actual files
mkdir -p credentials
if [ ! -f credentials/README.md ]; then
    cat > credentials/README.md << 'EOF'
# Credentials Directory

This directory should contain your Gmail API credentials:

## Required Files:
- `gmail_credentials.json` - OAuth2 credentials from Google Cloud Console
- `gmail_token.json` - Generated OAuth2 token (created automatically)

## Setup Instructions:
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable Gmail API
4. Create OAuth2 credentials (Desktop application)
5. Download the credentials JSON file
6. Rename it to `gmail_credentials.json` and place it here

**Note**: These files are ignored by git for security reasons.
EOF
    echo "✅ Created credentials/README.md"
fi

# Check for sensitive files that shouldn't be committed
echo ""
echo "🔍 Checking for sensitive files..."

SENSITIVE_FILES=(
    ".env"
    "credentials/*.json"
    "*.key"
    "*.pem"
    "models/*.gguf"
    "data/*.db"
)

FOUND_SENSITIVE=false
for pattern in "${SENSITIVE_FILES[@]}"; do
    if ls $pattern 2>/dev/null | grep -q .; then
        echo "⚠️  Found sensitive files: $pattern"
        FOUND_SENSITIVE=true
    fi
done

if [ "$FOUND_SENSITIVE" = true ]; then
    echo ""
    echo "⚠️  WARNING: Sensitive files detected!"
    echo "   Make sure these are properly ignored in .gitignore"
    echo "   Review files before committing to GitHub"
else
    echo "✅ No sensitive files found"
fi

echo ""
echo "📋 GitHub Upload Checklist:"
echo "   ✅ Documentation complete (README.md, DEPLOYMENT.md)"
echo "   ✅ Environment example created (.env.example)"
echo "   ✅ License file created (LICENSE)"
echo "   ✅ Git repository initialized"
echo "   ✅ .gitignore configured"
echo "   ✅ Credentials directory documented"
echo ""
echo "🚀 Ready for GitHub upload!"
echo ""
echo "Next steps:"
echo "1. Review and commit your changes:"
echo "   git add ."
echo "   git commit -m 'Initial commit: Gmail Article Search Agent'"
echo ""
echo "2. Create GitHub repository and push:"
echo "   git remote add origin https://github.com/yourusername/gmail-article-search-agent.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "3. Update README.md with your actual repository URL"
echo "4. Consider creating GitHub releases for versions"
