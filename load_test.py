import os
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Environment setup
os.environ["ACCELERATE_DISABLE_RICH"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

MODEL_ID = "google/gemma-3-270m-it"

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def cleanup():
    gc.collect()
    device = get_device()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()

def test_load():
    device = get_device()
    print(f"--- Hardware-Aware Load Test ---")
    print(f"Detected Device: {device}")
    
    print("Pre-load cleanup...")
    cleanup()
    
    print("1. Initializing Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    print(f"2. Loading weights to CPU first...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="cpu",
        low_cpu_mem_usage=False,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    
    print("3. Finalizing weight tying...")
    model.tie_weights()
    
    print(f"4. Moving model to {device}...")
    model = model.to(device)
    
    print("5. Verifying with a dummy generation...")
    inputs = tokenizer("Hello, world!", return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10)
    
    print(f"SUCCESS: Model loaded on {device} and generated output!")
    print(f"Output: {tokenizer.decode(outputs[0])}")

if __name__ == "__main__":
    try:
        test_load()
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
