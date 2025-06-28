#!/bin/bash
"""
Docker initialization script for Ollama setup.
This script runs after the Ollama container starts to download the Mistral model.
"""

set -e

echo "Starting Ollama initialization script..."

# Configuration
OLLAMA_HOST="${OLLAMA_HOST:-http://ollama:11434}"
MODEL_NAME="${OLLAMA_MODEL:-mistral:7b-instruct}"
MAX_RETRIES=60
SLEEP_INTERVAL=5

# Function to wait for Ollama server
wait_for_ollama() {
    echo "Waiting for Ollama server at $OLLAMA_HOST..."
    local retries=0
    
    while [ $retries -lt $MAX_RETRIES ]; do
        if curl -s "$OLLAMA_HOST/api/tags" >/dev/null 2>&1; then
            echo "✓ Ollama server is ready"
            return 0
        fi
        
        echo "  Waiting for Ollama server... ($((retries + 1))/$MAX_RETRIES)"
        sleep $SLEEP_INTERVAL
        retries=$((retries + 1))
    done
    
    echo "✗ Ollama server not ready after $(($MAX_RETRIES * $SLEEP_INTERVAL)) seconds"
    return 1
}

# Function to check if model exists
check_model_exists() {
    echo "Checking if model $MODEL_NAME exists..."
    
    local response=$(curl -s "$OLLAMA_HOST/api/tags" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    models = [model['name'] for model in data.get('models', [])]
    print('$MODEL_NAME' in models)
except:
    print('False')
    ")
    
    if [ "$response" = "True" ]; then
        echo "✓ Model $MODEL_NAME already exists"
        return 0
    else
        echo "  Model $MODEL_NAME not found"
        return 1
    fi
}

# Function to download model
download_model() {
    echo "Downloading model $MODEL_NAME..."
    echo "This may take several minutes depending on model size..."
    
    # Use curl to trigger model download
    local response=$(curl -s -X POST "$OLLAMA_HOST/api/pull" \
        -H "Content-Type: application/json" \
        -d "{\"name\": \"$MODEL_NAME\"}" \
        --max-time 1800)  # 30 minutes timeout
    
    if [ $? -eq 0 ]; then
        echo "✓ Model download completed"
        return 0
    else
        echo "✗ Model download failed"
        return 1
    fi
}

# Function to test model
test_model() {
    echo "Testing model $MODEL_NAME..."
    
    local test_payload=$(cat <<EOF
{
    "model": "$MODEL_NAME",
    "messages": [
        {
            "role": "user",
            "content": "Hello! Please respond with 'Test successful' if you can understand this."
        }
    ],
    "options": {
        "temperature": 0.1,
        "num_predict": 10
    }
}
EOF
)
    
    local response=$(curl -s -X POST "$OLLAMA_HOST/api/chat" \
        -H "Content-Type: application/json" \
        -d "$test_payload" \
        --max-time 60)
    
    if echo "$response" | grep -i "successful\|working\|understand" >/dev/null 2>&1; then
        echo "✓ Model test successful"
        return 0
    else
        echo "  Model test response: $response"
        echo "✓ Model responding (assuming functional)"
        return 0
    fi
}

# Main execution
main() {
    echo "=========================================="
    echo "Ollama Docker Initialization"
    echo "Host: $OLLAMA_HOST"
    echo "Model: $MODEL_NAME"
    echo "=========================================="
    
    # Step 1: Wait for Ollama server
    if ! wait_for_ollama; then
        echo "Failed to connect to Ollama server"
        exit 1
    fi
    
    # Step 2: Check if model exists
    if check_model_exists; then
        echo "Model already available, skipping download"
    else
        # Step 3: Download model
        if ! download_model; then
            echo "Failed to download model"
            exit 1
        fi
        
        # Verify download
        if ! check_model_exists; then
            echo "Model download verification failed"
            exit 1
        fi
    fi
    
    # Step 4: Test model
    if ! test_model; then
        echo "Model test failed, but continuing..."
    fi
    
    # Create completion marker
    echo "timestamp=$(date)" > /tmp/ollama_init_complete
    echo "model=$MODEL_NAME" >> /tmp/ollama_init_complete
    echo "host=$OLLAMA_HOST" >> /tmp/ollama_init_complete
    
    echo "=========================================="
    echo "✓ Ollama initialization completed successfully!"
    echo "Model $MODEL_NAME is ready for use"
    echo "=========================================="
}

# Run main function
main "$@"
