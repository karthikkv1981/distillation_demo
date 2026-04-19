import json
import os
import gc

# Set MPS fallback and disable accelerate progress bars before any torch imports
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["ACCELERATE_DISABLE_RICH"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import argparse

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

def get_prompts(num_prompts):
    base_prompts = [
        "Design a production RAG system for {scale} documents in the {industry} industry.",
        "Design an audit-compliant GenAI system for {industry} scaling to {requests} requests.",
        "Implement LLM evaluation framework for hallucination detection focusing on {metric} for {use_case}.",
        "Design multi-agent orchestration architecture for {use_case} at {scale} scale.",
        "Add governance and explainability layer to GenAI platform for {industry} handling {data_type} data.",
        "Scale inference to {requests} requests per day for a {use_case} application.",
        "Implement vector DB lifecycle management for {data_type} data in {industry}.",
        "Design secure prompt management system for {industry} applications handling {scale} queries.",
        "Build a fine-tuning pipeline for a {use_case} model focusing on {metric}.",
        "Architect a hybrid search RAG pipeline for {scale} {data_type} records in {industry}."
    ]
    
    scales = ["1M", "5M", "10M", "50M", "100M"]
    industries = ["banking", "healthcare", "retail", "manufacturing", "insurance", "legal", "e-commerce", "government"]
    metrics = ["context relevance", "factual accuracy", "toxicity", "groundedness", "answer semantic similarity"]
    use_cases = ["customer support", "code generation", "document summarization", "data analysis", "financial forecasting"]
    requests = ["100K", "500K", "1M", "10M"]
    data_types = ["text", "multimodal", "image", "code snippets", "financial reports"]

    prompts = []
    # Generate the requested number of prompts by sampling
    random.seed(42)
    while len(prompts) < num_prompts:
        base = random.choice(base_prompts)
        prompt = base.format(
            scale=random.choice(scales),
            industry=random.choice(industries),
            metric=random.choice(metrics),
            use_case=random.choice(use_cases),
            requests=random.choice(requests),
            data_type=random.choice(data_types)
        )
        if prompt not in prompts:
            prompts.append(prompt)
    return prompts

def main(num_prompts):
    device = get_device()
    # We use SmolLM2-360M-Instruct as the teacher.
    model_id = "HuggingFaceTB/SmolLM2-360M-Instruct"
    print(f"Preparing teacher model {model_id} on {device}...", flush=True)
    
    # Pre-load cleanup
    cleanup()
    
    # Using float16 for standard Mac inference speed
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    print("Loading teacher weights strictly to CPU...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cpu",
        low_cpu_mem_usage=False,
        torch_dtype=torch.float32,
    )
    
    print(f"Finalizing weights and moving teacher to {device}...", flush=True)
    model.tie_weights()
    model = model.to(device)
    print("Teacher loaded successfully!", flush=True)
    
    print("Preparing prompts...", flush=True)
    prompts = get_prompts(num_prompts)
    
    os.makedirs("dataset", exist_ok=True)
    output_file = "dataset/train.jsonl"
    
    # If the file exists and has enough lines, skip generation
    if os.path.exists(output_file):
        print(f"Checking existing dataset at {output_file}...", flush=True)
        with open(output_file, 'r') as f:
            lines = f.readlines()
        if len(lines) >= num_prompts:
            print(f"Dataset already exists with at least {num_prompts} examples. Skipping generation.", flush=True)
            return

    print("--- Starting Generation Phase ---", flush=True)
    print(f"NOTE: The first generation may take 30-60s for {device} kernel compilation...", flush=True)
    print(f"Generating {num_prompts} synthetic examples. This will take some time...", flush=True)
    
    with open(output_file, "w") as f:
        for i, prompt_text in enumerate(prompts):
            print(f"[{i+1}/{num_prompts}] Preparing prompt: {prompt_text}", flush=True)
            
            messages = [
                {"role": "user", "content": prompt_text}
            ]
            
            print("Tokenizing input...", flush=True)
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True
            ).to(device)
            
            print(f"Invoking model.generate on {device}...", flush=True)
            outputs = model.generate(
                **inputs,
                max_new_tokens=250,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            print("Generation complete!", flush=True)
            
            prompt_length = inputs["input_ids"].shape[1]
            response_tokens = outputs[0][prompt_length:] # exclude prompt
            response_text = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
            
            data_dict = {
                "instruction": prompt_text,
                "output": response_text
            }
            
            f.write(json.dumps(data_dict) + "\n")
            f.flush()

    print(f"Dataset generation complete. Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=200, help="Number of synthetic examples to generate")
    args = parser.parse_args()
    main(args.num_samples)
