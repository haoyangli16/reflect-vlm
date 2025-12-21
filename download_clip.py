#!/usr/bin/env python
"""Download CLIP vision tower model"""
import fix_triton_import  # Import this first to fix LLVM issues
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import CLIPVisionModel, CLIPImageProcessor

print('Downloading CLIP vision model: openai/clip-vit-large-patch14-336')
print('Using HuggingFace mirror: https://hf-mirror.com')

model = CLIPVisionModel.from_pretrained('openai/clip-vit-large-patch14-336')
processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-large-patch14-336')

print('Download complete!')
print(f'Model cached at: {model.config._name_or_path}')
