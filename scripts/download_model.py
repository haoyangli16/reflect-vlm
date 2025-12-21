#!/usr/bin/env python3
"""
Download model from HuggingFace with retry logic and mirror support.
This pre-downloads the model to avoid timeout issues during evaluation.
"""
import os
import sys
import time
from huggingface_hub import snapshot_download

def download_model_with_retry(model_id, max_retries=3, cache_dir=None):
    """Download model with retry logic."""
    
    # Set up HuggingFace mirror endpoint for China
    if 'HF_ENDPOINT' not in os.environ:
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    print(f"Using HuggingFace endpoint: {os.environ.get('HF_ENDPOINT', 'default')}")
    print(f"Downloading model: {model_id}")
    
    for attempt in range(max_retries):
        try:
            print(f"\nAttempt {attempt + 1}/{max_retries}")
            local_path = snapshot_download(
                repo_id=model_id,
                cache_dir=cache_dir,
                resume_download=True,
                local_files_only=False,
            )
            print(f"\n✓ Successfully downloaded model to: {local_path}")
            return local_path
        except Exception as e:
            print(f"✗ Download failed: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("\n✗ All retry attempts failed!")
                raise

if __name__ == "__main__":
    model_id = "yunhaif/ReflectVLM-llava-v1.5-13b-base"
    
    # Use custom cache dir if specified
    cache_dir = os.environ.get('HF_HOME', None)
    
    try:
        download_model_with_retry(model_id, max_retries=5, cache_dir=cache_dir)
        print("\n" + "="*80)
        print("Model download completed successfully!")
        print("You can now run the evaluation script.")
        print("="*80)
    except Exception as e:
        print("\n" + "="*80)
        print(f"FAILED to download model: {e}")
        print("\nTroubleshooting:")
        print("1. Check your network connection")
        print("2. Try using the proxy:")
        print("   export https_proxy=104.250.52.76:2080")
        print("   export http_proxy=104.250.52.76:2080")
        print("3. Or try the mirror:")
        print("   export HF_ENDPOINT=https://hf-mirror.com")
        print("="*80)
        sys.exit(1)
