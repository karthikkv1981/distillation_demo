import torch
import gc
import sys
import os
import json

# Set MPS fallback and disable accelerate progress bars before any torch imports
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["ACCELERATE_DISABLE_RICH"] = "1"

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

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

STUDENT_ID = "google/gemma-3-270m-it"
ADAPTER_PATH = "distilled_student_lora"

def generate_response(model, tokenizer, prompt_text):
    device = get_device()
    messages = [{"role": "user", "content": prompt_text}]
    inputs = tokenizer.apply_chat_template(
        messages, 
        tokenize=True, 
        add_generation_prompt=True, 
        return_tensors="pt",
        return_dict=True
    ).to(device)
    
    outputs = model.generate(
        **inputs, 
        max_new_tokens=250, 
        do_sample=False
    )
    
    prompt_length = inputs["input_ids"].shape[1]
    response_tokens = outputs[0][prompt_length:]
    return tokenizer.decode(response_tokens, skip_special_tokens=True).strip()


def main():
    device = get_device()
    print("Loading baseline evaluation results...")
    try:
        with open("baseline_eval.json", "r") as f:
            baseline_results = json.load(f)
    except FileNotFoundError:
        print("Error: baseline_eval.json not found. Run evaluate_before.py first.")
        return
        
    print(f"\nPreparing Distilled Student Model {STUDENT_ID} on {device}...", flush=True)
    
    # Pre-load cleanup
    cleanup()
    
    # Load base model on CPU first
    print("Loading base student weights strictly to CPU...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        STUDENT_ID, 
        torch_dtype=torch.float32, 
        low_cpu_mem_usage=False,
        device_map="cpu"
    )
    print(f"Finalizing weights and moving base student to {device}...", flush=True)
    base_model.tie_weights()
    base_model = base_model.to(device)
    print("Base student loaded successfully!", flush=True)
    
    # Load PEFT adapter
    try:
        print("Loading LoRA adapters...")
        student_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH).to(device)
        student_tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
        print("Distilled student ready for evaluation!")
    except Exception as e:
        print(f"Failed to load LoRA adapter. Did train_student.py finish? Error: {e}")
        return
        
    final_results = []
    
    print("\nGenerating STUDENT AFTER responses...")
    for i, item in enumerate(baseline_results):
        prompt = item["prompt"]
        print(f"Generating [{i+1}/{len(baseline_results)}]...")
        
        after_resp = generate_response(student_model, student_tokenizer, prompt)
        
        final_results.append({
            "prompt": prompt,
            "teacher": item["teacher"],
            "student_before": item["student_before"],
            "student_after": after_resp
        })
        
    print("\nFinal Comparison:")
    for item in final_results:
        print("====================================")
        print(f"PROMPT:\n{item['prompt']}")
        print("------------------------------------")
        print(f"TEACHER:\n{item['teacher']}")
        print("------------------------------------")
        print(f"STUDENT BEFORE:\n{item['student_before']}")
        print("------------------------------------")
        print(f"STUDENT AFTER:\n{item['student_after']}")
        print("====================================\n")
        
    with open("final_eval.json", "w") as f:
        json.dump(final_results, f, indent=2)
    print("Saved final results to final_eval.json")

if __name__ == "__main__":
    main()
