import torch
import gc
import sys
import os
import json

# Set MPS fallback and disable accelerate progress bars before any torch imports
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["ACCELERATE_DISABLE_RICH"] = "1"

from transformers import AutoModelForCausalLM, AutoTokenizer

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

TEACHER_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
STUDENT_ID = "google/gemma-3-270m-it"

TEST_PROMPTS = [
    "Design a production RAG system for 10M documents",
    "Design an audit-compliant GenAI system for banking",
    "Implement LLM evaluation framework for hallucination detection",
    "Design multi-agent orchestration architecture",
    "Add governance and explainability layer to GenAI platform"
]

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
    results = []
    
    # 1. GENERATE TEACHER RESPONSES
    print(f"Preparing Teacher Model {TEACHER_ID} on {device}...", flush=True)
    
    # Pre-load cleanup
    cleanup()
    
    teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_ID)
    print("Loading teacher weights strictly to CPU...", flush=True)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        TEACHER_ID, 
        torch_dtype=torch.float32, 
        low_cpu_mem_usage=False,
        device_map="cpu"
    )
    print(f"Finalizing weights and moving teacher to {device}...", flush=True)
    teacher_model.tie_weights()
    teacher_model = teacher_model.to(device)
    print("Teacher loaded successfully!", flush=True)
    
    teacher_responses = []
    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"Teacher generating [{i+1}/{len(TEST_PROMPTS)}]...")
        resp = generate_response(teacher_model, teacher_tokenizer, prompt)
        teacher_responses.append(resp)
        
    print("Teacher generation complete. Freeing memory...")
    del teacher_model
    del teacher_tokenizer
    cleanup()

    # 2. GENERATE STUDENT RESPONSES
    print(f"\nPreparing Untrained Student Model {STUDENT_ID} on {device}...", flush=True)
    
    # Pre-load cleanup
    cleanup()
    
    student_tokenizer = AutoTokenizer.from_pretrained(STUDENT_ID)
    print("Loading student weights strictly to CPU...", flush=True)
    student_model = AutoModelForCausalLM.from_pretrained(
        STUDENT_ID, 
        torch_dtype=torch.float32, 
        low_cpu_mem_usage=False,
        device_map="cpu"
    )
    print(f"Finalizing weights and moving student to {device}...", flush=True)
    student_model.tie_weights()
    student_model = student_model.to(device)
    print("Student loaded successfully!", flush=True)
    
    student_responses = []
    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"Student generating [{i+1}/{len(TEST_PROMPTS)}]...")
        resp = generate_response(student_model, student_tokenizer, prompt)
        student_responses.append(resp)
        
    # Free memory again
    del student_model
    del student_tokenizer
    cleanup()
        
    # 3. PRINT & SAVE RESULTS
    for i, prompt in enumerate(TEST_PROMPTS):
        print("\n====================================")
        print(f"PROMPT:\n{prompt}")
        print("------------------------------------")
        print(f"TEACHER:\n{teacher_responses[i]}")
        print("------------------------------------")
        print(f"STUDENT BEFORE:\n{student_responses[i]}")
        print("====================================")
        
        results.append({
            "prompt": prompt,
            "teacher": teacher_responses[i],
            "student_before": student_responses[i]
        })
        
    with open("baseline_eval.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved baseline results to baseline_eval.json")

if __name__ == "__main__":
    main()
