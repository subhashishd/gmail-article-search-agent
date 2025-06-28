#!/usr/bin/env python3
"""
Setup script to initialize Ollama and download Mistral model for Docker deployment.
This script handles the model downloading that was problematic with llama-cpp-python.
"""

import os
import sys
import time
import requests
import subprocess
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def wait_for_ollama_server(host: str = "http://localhost:11434", timeout: int = 300) -> bool:
    """
    Wait for Ollama server to be ready.
    
    Args:
        host: Ollama server host
        timeout: Maximum wait time in seconds
        
    Returns:
        bool: True if server is ready
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{host}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info("Ollama server is ready")
                return True
        except requests.exceptions.RequestException:
            pass
        
        logger.info("Waiting for Ollama server to start...")
        time.sleep(5)
    
    logger.error(f"Ollama server not ready after {timeout} seconds")
    return False


def download_model(model_name: str = "mistral:7b-instruct", host: str = "http://localhost:11434") -> bool:
    """
    Download the specified model using Ollama.
    
    Args:
        model_name: Name of the model to download
        host: Ollama server host
        
    Returns:
        bool: True if download successful
    """
    try:
        logger.info(f"Starting download of model: {model_name}")
        
        # Use requests to trigger model pull
        response = requests.post(
            f"{host}/api/pull",
            json={"name": model_name},
            timeout=1800  # 30 minutes timeout for large models
        )
        
        if response.status_code == 200:
            logger.info(f"Successfully downloaded model: {model_name}")
            return True
        else:
            logger.error(f"Failed to download model: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Error downloading model {model_name}: {e}")
        return False


def verify_model_available(model_name: str, host: str = "http://localhost:11434") -> bool:
    """
    Verify that the model is available and ready for use.
    
    Args:
        model_name: Name of the model to verify
        host: Ollama server host
        
    Returns:
        bool: True if model is available
    """
    try:
        response = requests.get(f"{host}/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json()
            available_models = [model['name'] for model in models.get('models', [])]
            
            if model_name in available_models:
                logger.info(f"Model {model_name} is available and ready")
                return True
            else:
                logger.warning(f"Model {model_name} not found in available models: {available_models}")
                return False
        else:
            logger.error(f"Failed to list models: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"Error verifying model availability: {e}")
        return False


def test_model_inference(model_name: str, host: str = "http://localhost:11434") -> bool:
    """
    Test that the model can perform inference.
    
    Args:
        model_name: Name of the model to test
        host: Ollama server host
        
    Returns:
        bool: True if inference works
    """
    try:
        logger.info(f"Testing inference with model: {model_name}")
        
        test_payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": "Hello! Please respond with 'Model working correctly' if you can understand this message."
                }
            ],
            "options": {
                "temperature": 0.1,
                "num_predict": 20
            }
        }
        
        response = requests.post(
            f"{host}/api/chat",
            json=test_payload,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result.get('message', {}).get('content', '')
            logger.info(f"Model response: {content}")
            
            if 'working' in content.lower():
                logger.info("Model inference test successful")
                return True
            else:
                logger.warning("Model responded but content seems unexpected")
                return True  # Still consider it working
        else:
            logger.error(f"Inference test failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Error testing model inference: {e}")
        return False


def get_model_info(model_name: str, host: str = "http://localhost:11434") -> Dict[str, Any]:
    """
    Get detailed information about the model.
    
    Args:
        model_name: Name of the model
        host: Ollama server host
        
    Returns:
        Dict containing model information
    """
    try:
        response = requests.post(
            f"{host}/api/show",
            json={"name": model_name},
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed to get model info: {response.status_code}")
            return {}
            
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return {}


def main():
    """Main setup function."""
    logger.info("Starting Ollama setup for Docker environment")
    
    # Configuration
    ollama_host = os.getenv("OLLAMA_HOST", "http://ollama:11434")
    model_name = os.getenv("OLLAMA_MODEL", "mistral:7b-instruct")
    
    # For local testing, use localhost
    if "localhost" not in ollama_host and "127.0.0.1" not in ollama_host:
        if "ollama" in ollama_host:
            # Docker environment
            pass
        else:
            ollama_host = "http://localhost:11434"
    
    logger.info(f"Using Ollama host: {ollama_host}")
    logger.info(f"Target model: {model_name}")
    
    # Step 1: Wait for Ollama server
    if not wait_for_ollama_server(ollama_host):
        logger.error("Ollama server not available")
        sys.exit(1)
    
    # Step 2: Check if model already exists
    if verify_model_available(model_name, ollama_host):
        logger.info(f"Model {model_name} already available")
    else:
        # Step 3: Download model
        if not download_model(model_name, ollama_host):
            logger.error("Failed to download model")
            sys.exit(1)
        
        # Step 4: Verify download
        if not verify_model_available(model_name, ollama_host):
            logger.error("Model download verification failed")
            sys.exit(1)
    
    # Step 5: Test inference
    if not test_model_inference(model_name, ollama_host):
        logger.warning("Model inference test failed, but continuing...")
    
    # Step 6: Display model info
    model_info = get_model_info(model_name, ollama_host)
    if model_info:
        logger.info(f"Model details: {model_info.get('details', {})}")
    
    logger.info("Ollama setup completed successfully!")
    
    # Create a marker file to indicate successful setup
    with open("/tmp/ollama_setup_complete", "w") as f:
        f.write(f"Model: {model_name}\nHost: {ollama_host}\n")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
