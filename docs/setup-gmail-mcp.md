# Gmail MCP Server Integration Setup

This document explains how the Gmail Article Search Agent uses the Gmail MCP (Model Context Protocol) server for improved Gmail integration.

## What is Gmail MCP Server?

The Gmail MCP server is a standardized interface that provides secure and efficient access to Gmail operations through the Model Context Protocol. This approach offers several advantages:

- **Standardized Interface**: Uses the MCP protocol for consistent Gmail operations
- **Better Security**: Handles authentication and permissions more securely
- **Simplified Integration**: Reduces complexity compared to direct Gmail API integration
- **Improved Error Handling**: Better error management and retry logic

## Architecture Changes

### Before (Direct Gmail API)
```
FastAPI Backend → Google Gmail API → Gmail Account
```

### After (MCP Integration)
```
FastAPI Backend → Gmail MCP Server → Google Gmail API → Gmail Account
```

## Prerequisites

1. **Node.js and npm**: The Gmail MCP server is a Node.js application
2. **Gmail API Credentials**: Same as before - you still need `credentials.json`
3. **Docker**: The setup includes Node.js in the backend container

## How It Works

### 1. MCP Server Startup
When the backend service starts, it automatically:
- Spawns a Gmail MCP server process
- Configures it with your Gmail credentials
- Establishes a JSON-RPC communication channel

### 2. Gmail Operations
All Gmail operations now go through MCP:
- **Email Search**: Uses MCP `search_emails` tool
- **Message Retrieval**: Uses MCP `get_message` tool
- **Authentication**: Handled by the MCP server

### 3. Communication Protocol
The backend communicates with the MCP server using JSON-RPC 2.0:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "search_emails",
    "arguments": {
      "query": "Medium Daily Digest after:2024/01/01",
      "max_results": 50
    }
  }
}
```

## Implementation Details

### GmailMCPService Class
The new `GmailMCPService` class provides:
- **Async Operations**: All Gmail operations are now async
- **MCP Communication**: JSON-RPC communication with the MCP server
- **Error Handling**: Improved error management
- **Resource Cleanup**: Proper MCP server lifecycle management

### Backward Compatibility
A wrapper `GmailService` class maintains compatibility:
- Same public interface as before
- Synchronous methods that wrap async operations
- No changes needed to existing code

## Configuration

### Environment Variables
```env
# MCP-specific settings
MCP_SERVER_TIMEOUT=30
MCP_SERVER_RETRIES=3

# Gmail credentials (same as before)
GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/credentials.json
```

### Docker Setup
The backend Dockerfile now includes:
- Node.js and npm installation
- Gmail MCP server package installation
- All dependencies for MCP communication

## Benefits of MCP Integration

1. **Better Abstraction**: Cleaner separation between Gmail logic and application logic
2. **Improved Reliability**: MCP server handles connection management and retries
3. **Standardized Protocol**: Uses industry-standard MCP for tool interactions
4. **Enhanced Security**: Better credential management through MCP
5. **Future-Proof**: Easy to extend with other MCP tools

## Troubleshooting

### MCP Server Issues
```bash
# Check if Node.js is available
docker exec -it gmail-search-backend node --version

# Check MCP server installation
docker exec -it gmail-search-backend npm list -g @modelcontextprotocol/server-gmail

# View backend logs for MCP errors
docker-compose logs -f backend
```

### Common Problems

1. **MCP Server Startup Failed**
   - Ensure Node.js is properly installed
   - Check that Gmail credentials exist
   - Verify MCP server package installation

2. **Authentication Errors**
   - Same troubleshooting as regular Gmail API
   - Check credentials.json format and permissions
   - Verify Gmail API is enabled

3. **Communication Errors**
   - Check backend logs for JSON-RPC errors
   - Ensure MCP server process is running
   - Verify timeout and retry settings

## Testing MCP Integration

### 1. Test Gmail Connection
Use the `/test-gmail` endpoint to verify MCP integration:
```bash
curl -X POST http://localhost:8000/test-gmail
```

### 2. Check Backend Logs
Monitor logs for MCP-specific messages:
```bash
docker-compose logs -f backend | grep -i mcp
```

### 3. Verify Email Fetching
Test the complete flow:
```bash
curl -X POST http://localhost:8000/fetch-articles
curl http://localhost:8000/fetch-status
```

## Migration Notes

If you're upgrading from the direct Gmail API integration:

1. **No Changes Needed**: The public interface remains the same
2. **New Dependencies**: Node.js and MCP server are automatically installed
3. **Same Credentials**: Use the same `credentials.json` file
4. **Improved Performance**: Async operations may be faster
5. **Better Error Messages**: More detailed error reporting

## Next Steps

After setting up MCP integration:

1. Test the Gmail connection using the frontend
2. Fetch some articles to verify end-to-end functionality
3. Monitor logs for any MCP-related issues
4. Enjoy improved Gmail integration reliability!

## Advanced Configuration

For advanced users, you can customize MCP behavior by:

1. **Adjusting Timeouts**: Modify `MCP_SERVER_TIMEOUT` environment variable
2. **Retry Logic**: Configure `MCP_SERVER_RETRIES` for better reliability
3. **Custom MCP Tools**: Extend the MCP server with additional Gmail tools
4. **Protocol Debugging**: Enable detailed JSON-RPC logging for troubleshooting
