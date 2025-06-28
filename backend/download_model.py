
from sentence_transformers import SentenceTransformer
import os
import time
import sys

def download_with_retry(model_name, max_retries=3, timeout=180):
    """Download model with retry logic and timeout handling."""
    for attempt in range(max_retries):
        try:
            print(f"Downloading model: {model_name} (attempt {attempt + 1}/{max_retries})")
            
            # Set shorter timeout to avoid hanging
            os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = str(timeout)
            os.environ['TRANSFORMERS_CACHE'] = './models'
            
            model = SentenceTransformer(model_name)
            model.save("./models")
            print(f"Model {model_name} downloaded and saved successfully")
            return True
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download model after {max_retries} attempts")
                print("Model will be downloaded on first use instead")
                return False

# Download and cache the embedding model
model_name = "all-MiniLM-L6-v2"
success = download_with_retry(model_name)

if not success:
    print("Warning: Could not pre-download embedding model")
    print("Application will download it on first use")
    # Don't fail the build - just continue
    sys.exit(0)
