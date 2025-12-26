import torch
import sys
import os
import argparse
from PIL import Image
import numpy as np
import transformers
import tokenizers


# Add the reflect-vlm directory to path so we can import roboworld and llava
# Adjusted to be relative to this script's location
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def test_compatibility(model_path):
    print(f"--- Environment Check ---")
    print(f"Transformers version: {transformers.__version__}")
    print(f"Tokenizers version:   {tokenizers.__version__}")
    print(f"PyTorch version:      {torch.__version__}")
    print(f"CUDA available:       {torch.cuda.is_available()}")
    print("-" * 25)

    try:
        from roboworld.agent.llava import LlavaAgent
        print("✅ Successfully imported LlavaAgent from roboworld")
    except ImportError as e:
        print(f"❌ Failed to import LlavaAgent: {e}")
        return

    print(f"\nAttempting to load model from: {model_path}")
    if not os.path.exists(model_path):
        print(f"❌ Path does not exist: {model_path}")
        return

    try:
        # We use 4bit loading to save memory during the test
        agent = LlavaAgent(model_path=model_path, load_4bit=True)
        print("✅ Model and Tokenizer loaded successfully!")
        
        # Test 1: Image Encoding
        print("\nTesting Image Encoding...")
        dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        features = agent.encode_image(dummy_img)
        print(f"✅ Image encoding successful. Feature shape: {features.shape}")

        # Test 2: Dummy Inference
        print("\nTesting Inference (Generation)...")
        dummy_goal = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        prompt = "What action should I take next?"
        
        # Using act() method which triggers the full pipeline
        action = agent.act(image=dummy_img, goal_image=dummy_goal, inp=prompt)
        print(f"✅ Inference successful! Response: '{action}'")
        
        print("\n" + "="*30)
        print("🎉 COMPATIBILITY VERIFIED")
        print("The newer transformers/tokenizers versions are compatible with your LLaVA model.")
        print("="*30)

    except Exception as e:
        print(f"\n❌ COMPATIBILITY ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\nSuggestion: If you see 'NoneType object has no attribute...', it often means "
              "the model's config format changed between transformers versions.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/share/project/lhy/thirdparty/reflect-vlm/ReflectVLM-llava-v1.5-13b-base")
    args = parser.parse_args()
    
    test_compatibility(args.model_path)
