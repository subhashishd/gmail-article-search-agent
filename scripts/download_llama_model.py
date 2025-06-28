#!/usr/bin/env python3
"""
Download script for Llama-2-7B-Chat quantized model.

This script downloads the quantized GGUF model optimized for 8GB RAM systems.
The model is downloaded from Hugging Face and saved locally.
"""

import os
import requests
from pathlib import Path
import hashlib

def download_file(url: str, local_path: str, expected_size: int = None) -> bool:
    """Download a file with progress tracking."""
    try:
        print(f"Downloading {url}")
        print(f"Saving to: {local_path}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Download with streaming
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\rProgress: {progress:.1f}% ({downloaded}/{total_size} bytes)", end='')
        
        print(f"\n✓ Download complete: {local_path}")
        
        # Verify file size if provided
        if expected_size:
            actual_size = os.path.getsize(local_path)
            if actual_size != expected_size:
                print(f"⚠️  Warning: File size mismatch. Expected: {expected_size}, Got: {actual_size}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False

def main():
    """Download Llama-2-7B-Chat quantized model."""
    
    print("🦙 Llama-2-7B-Chat Model Downloader")
    print("=" * 50)
    print("This will download a quantized 4-bit model (~3.9GB)")
    print("Optimized for MacBook Air 8GB RAM")
    print()
    
    # Model configuration
    model_info = {
        "name": "Llama-2-7B-Chat-GGUF",
        "file": "llama-2-7b-chat.q4_0.gguf",
        "url": "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.q4_0.gguf",
        "size": 4081004832,  # ~3.9GB
        "description": "4-bit quantized model for efficient inference"
    }
    
    # Local paths
    models_dir = Path("models")
    model_path = models_dir / model_info["file"]
    
    print(f"Model: {model_info['name']}")
    print(f"File: {model_info['file']}")
    print(f"Size: {model_info['size'] / (1024**3):.1f} GB")
    print(f"Description: {model_info['description']}")
    print()
    
    # Check if model already exists
    if model_path.exists():
        existing_size = model_path.stat().st_size
        if existing_size == model_info["size"]:
            print("✓ Model already exists and appears complete")
            print(f"Path: {model_path.absolute()}")
            return
        else:
            print(f"⚠️  Existing model file found but size mismatch")
            print(f"Expected: {model_info['size']}, Found: {existing_size}")
            response = input("Re-download? (y/n): ").lower().strip()
            if response != 'y':
                return
    
    # Confirm download
    print(f"This will download {model_info['size'] / (1024**3):.1f} GB to:")
    print(f"  {model_path.absolute()}")
    print()
    
    response = input("Continue? (y/n): ").lower().strip()
    if response != 'y':
        print("Download cancelled")
        return
    
    # Download the model
    print("\nStarting download...")
    success = download_file(
        url=model_info["url"],
        local_path=str(model_path),
        expected_size=model_info["size"]
    )
    
    if success:
        print("\n🎉 Model download completed successfully!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install llama-cpp-python")
        print("2. The model is ready to use with the local LLM service")
        print(f"3. Model location: {model_path.absolute()}")
        
        # Verify the model file
        if model_path.exists():
            actual_size = model_path.stat().st_size
            print(f"\nFile verification:")
            print(f"  Size: {actual_size / (1024**3):.2f} GB")
            print(f"  Expected: {model_info['size'] / (1024**3):.2f} GB")
            
            if actual_size == model_info["size"]:
                print("  ✓ File size matches expected")
            else:
                print("  ⚠️  File size mismatch - download may be incomplete")
    else:
        print("\n❌ Model download failed")
        print("Please check your internet connection and try again")

if __name__ == "__main__":
    main()
