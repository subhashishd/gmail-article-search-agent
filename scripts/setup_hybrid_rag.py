#!/usr/bin/env python3
"""
Setup script for Hybrid RAG system.

This script helps set up the hybrid RAG implementation by:
1. Installing required dependencies
2. Downloading the local LLM model
3. Testing the system configuration
4. Providing status information
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def run_command(command, description=""):
    """Run a shell command and return success status."""
    try:
        print(f"\n🔄 {description}" if description else f"\n🔄 Running: {command}")
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Success")
            if result.stdout.strip():
                print(f"Output: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ Failed")
            if result.stderr.strip():
                print(f"Error: {result.stderr.strip()}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible (need 3.8+)")
        return False

def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing Dependencies")
    print("=" * 40)
    
    # Install llama-cpp-python with specific configuration for macOS
    success = True
    
    print("Installing llama-cpp-python (optimized for macOS)...")
    # For macOS, we can try to use Metal acceleration if available
    cmd = "pip install llama-cpp-python"
    
    if not run_command(cmd, "Installing llama-cpp-python"):
        print("⚠️  Standard installation failed, trying with CPU-only build...")
        cmd_fallback = "CMAKE_ARGS='-DLLAMA_METAL=off' pip install llama-cpp-python --force-reinstall"
        if not run_command(cmd_fallback, "Installing llama-cpp-python (CPU-only)"):
            success = False
    
    # Verify other dependencies
    deps_to_check = [
        "sentence-transformers",
        "transformers",
        "psycopg2-binary",
        "fastapi",
        "streamlit"
    ]
    
    for dep in deps_to_check:
        try:
            __import__(dep.replace('-', '_'))
            print(f"✅ {dep} is available")
        except ImportError:
            print(f"⚠️  {dep} not found, installing...")
            if not run_command(f"pip install {dep}", f"Installing {dep}"):
                success = False
    
    return success

def download_model():
    """Download the Llama model."""
    print("\n🦙 Downloading Llama-2-7B-Chat Model")
    print("=" * 40)
    
    # Check if model already exists
    model_path = Path("models/llama-2-7b-chat.q4_0.gguf")
    
    if model_path.exists():
        size_gb = model_path.stat().st_size / (1024**3)
        print(f"✅ Model already exists ({size_gb:.1f} GB)")
        return True
    
    print("Model not found. Running download script...")
    return run_command("python download_llama_model.py", "Downloading Llama model")

def test_services():
    """Test that all services can be imported and initialized."""
    print("\n🧪 Testing Services")
    print("=" * 40)
    
    try:
        # Test imports
        print("Testing imports...")
        
        from backend.services.local_llm_service import local_llm_service
        print("✅ Local LLM service imported")
        
        from backend.services.hybrid_rag_service import hybrid_rag_service
        print("✅ Hybrid RAG service imported")
        
        from backend.services.local_search_service import local_search_service
        print("✅ Local search service imported")
        
        # Test LLM availability
        print("\nTesting LLM availability...")
        llm_available = local_llm_service.is_available()
        if llm_available:
            print("✅ LLM service is available and ready")
        else:
            print("⚠️  LLM service is not available (model may not be downloaded)")
        
        # Test vector service
        print("\nTesting vector service...")
        try:
            stats = local_search_service.get_article_stats()
            print(f"✅ Vector service connected (Total articles: {stats.get('total_articles', 0)})")
        except Exception as e:
            print(f"⚠️  Vector service error: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Service test failed: {e}")
        return False

def display_system_info():
    """Display system information and next steps."""
    print("\n📊 System Information")
    print("=" * 40)
    
    # Check available RAM
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
        print(f"💾 Total RAM: {ram_gb:.1f} GB")
        
        if ram_gb >= 8:
            print("✅ RAM is sufficient for Llama-2-7B")
        else:
            print("⚠️  Low RAM detected. Consider using TinyLlama instead.")
            
    except ImportError:
        print("📊 Install 'psutil' to see RAM information")
    
    # Model information
    model_path = Path("models/llama-2-7b-chat.q4_0.gguf")
    if model_path.exists():
        size_gb = model_path.stat().st_size / (1024**3)
        print(f"🦙 Model: Llama-2-7B-Chat ({size_gb:.1f} GB)")
        print(f"📁 Location: {model_path.absolute()}")
    else:
        print("🦙 Model: Not downloaded")
    
    # Architecture summary
    print(f"\n🏗️  Architecture: Hybrid RAG")
    print(f"   • Vector Search: Pre-filtering (~200 candidates)")
    print(f"   • LLM Analysis: Deep relevance scoring")
    print(f"   • Combined Scoring: 40% vector + 60% LLM")

def main():
    """Main setup routine."""
    print("🚀 Hybrid RAG Setup")
    print("=" * 50)
    print("Setting up hybrid RAG system with local LLM")
    print("This will install dependencies and download the Llama model")
    print()
    
    # Confirm setup
    response = input("Continue with setup? (y/n): ").lower().strip()
    if response != 'y':
        print("Setup cancelled")
        return
    
    success = True
    
    # Step 1: Check Python version
    if not check_python_version():
        success = False
    
    # Step 2: Install dependencies
    if success:
        if not install_dependencies():
            success = False
    
    # Step 3: Download model (optional if fails)
    if success:
        print("\nThe next step will download ~3.9GB. Continue?")
        response = input("Download Llama model? (y/n): ").lower().strip()
        if response == 'y':
            download_model()  # Don't fail setup if model download fails
    
    # Step 4: Test services
    if success:
        test_services()
    
    # Step 5: Display info and next steps
    print("\n" + "=" * 50)
    display_system_info()
    
    print("\n🎉 Setup Complete!")
    print("\nNext steps:")
    print("1. Start the backend: python -m backend.main")
    print("2. Start the frontend: streamlit run frontend/streamlit_app.py")
    print("3. Test search functionality with the new hybrid RAG")
    
    if not Path("models/llama-2-7b-chat.q4_0.gguf").exists():
        print("\n⚠️  Note: LLM model not downloaded. The system will use:")
        print("   • Vector search for candidate retrieval")
        print("   • DistilGPT-2 fallback for basic analysis")
        print("   • To get full hybrid RAG, run: python download_llama_model.py")

if __name__ == "__main__":
    main()
