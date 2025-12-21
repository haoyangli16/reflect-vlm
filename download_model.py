#!/usr/bin/env python
"""
Download ReflectVLM model with retry logic and mirror support.
Works better in China with HuggingFace mirror.
"""
import os
import sys
import time
from huggingface_hub import snapshot_download

# Use HuggingFace mirror for China
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

model_name = "yunhaif/ReflectVLM-llava-v1.5-13b-base"
cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

print(f"Downloading {model_name}...")
print(f"Using mirror: {os.environ.get('HF_ENDPOINT', 'default')}")
print(f"Cache directory: {cache_dir}")
print("\nThis may take a while (the model is ~13GB)...\n")

max_retries = 3
for attempt in range(max_retries):
    try:
        model_path = snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            resume_download=True,
            max_workers=4,
        )
        print(f"\n✓ Model downloaded successfully to: {model_path}")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Download attempt {attempt + 1}/{max_retries} failed: {e}")
        if attempt < max_retries - 1:
            wait_time = (attempt + 1) * 10
            print(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        else:
            print("\n✗ All download attempts failed.")
            print("You can try:")
            print("1. Use the proxy: export https_proxy=104.250.52.76:2080")
            print("2. Download manually from: https://hf-mirror.com/yunhaif/ReflectVLM-llava-v1.5-13b-base")
            sys.exit(1)
