# Gmail API Setup Instructions

This guide will help you set up Gmail API credentials for the Gmail Article Search Agent.

## Prerequisites

1. A Google account with Gmail access
2. Medium Daily Digest emails in your Gmail inbox

## Step 1: Create a Google Cloud Project

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Click "Select a project" and then "New Project"
3. Enter a project name (e.g., "Gmail Article Search")
4. Click "Create"

## Step 2: Enable Gmail API

1. In the Google Cloud Console, ensure your project is selected
2. Go to "APIs & Services" > "Library"
3. Search for "Gmail API"
4. Click on "Gmail API" and then "Enable"

## Step 3: Create Credentials

### Option A: Service Account (Recommended for local development)

1. Go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "Service Account"
3. Enter a service account name (e.g., "gmail-search-agent")
4. Click "Create and Continue"
5. Skip the optional steps and click "Done"
6. Click on the created service account
7. Go to the "Keys" tab
8. Click "Add Key" > "Create new key"
9. Select "JSON" format and click "Create"
10. Save the downloaded JSON file as `credentials.json`

### Option B: OAuth 2.0 (Alternative)

1. Go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "OAuth client ID"
3. If prompted, configure the OAuth consent screen first
4. Select "Desktop application" as the application type
5. Enter a name (e.g., "Gmail Article Search")
6. Click "Create"
7. Download the JSON file and save as `credentials.json`

## Step 4: Configure Domain-Wide Delegation (For Service Account Only)

**Note**: This step requires Google Workspace admin access. If you don't have admin access, use OAuth 2.0 instead.

1. In the service account details, copy the "Client ID"
2. Go to your Google Workspace Admin Console
3. Navigate to "Security" > "API Controls" > "Domain-wide Delegation"
4. Click "Add new" and enter the Client ID
5. Add the scope: `https://www.googleapis.com/auth/gmail.readonly`
6. Click "Authorize"

## Step 5: Place Credentials File

1. Create a `credentials` directory in your project root:
   ```bash
   mkdir credentials
   ```

2. Move your `credentials.json` file to this directory:
   ```bash
   mv /path/to/downloaded/credentials.json ./credentials/credentials.json
   ```

## Step 6: Update Environment Variables

The `.env` file should already be configured correctly:

```env
GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/credentials.json
```

## Troubleshooting

### Common Issues

1. **"Access denied" errors**: 
   - Ensure the Gmail API is enabled
   - Check that your service account has the correct permissions
   - For domain-wide delegation, ensure it's properly configured

2. **"File not found" errors**:
   - Verify the credentials file is in the correct location
   - Check file permissions

3. **"Invalid credentials" errors**:
   - Re-download the credentials file
   - Ensure you're using the correct type (Service Account vs OAuth)

### Testing Your Setup

You can test your Gmail API setup by:

1. Starting the application with Docker Compose
2. Using the "Test Gmail Connection" button in the frontend
3. Checking the backend logs for authentication success/failure

## Security Notes

- Never commit your `credentials.json` file to version control
- The `credentials` directory is already added to `.gitignore`
- Store credentials securely in production environments
- Consider using Google Cloud Secret Manager for production deployments

## Next Steps

After setting up credentials:

1. Start the application with `docker-compose up`
2. Access the frontend at `http://localhost:8501`
3. Test the Gmail connection
4. Fetch and index your Medium articles
5. Start searching!
