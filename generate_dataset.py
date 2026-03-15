import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import random

def get_prompts():
    base_prompts = [
        "Design a production RAG system for {scale} documents in the {industry} industry.",
        "Design an audit-compliant GenAI system for {industry}.",
        "Implement LLM evaluation framework for hallucination detection focusing on {metric}.",
        "Design multi-agent orchestration architecture for {use_case}.",
        "Add governance and explainability layer to GenAI platform for {industry}.",
        "Scale inference to {requests} requests per day for a {use_case} application.",
        "Implement vector DB lifecycle management for {data_type} data.",
        "Design secure prompt management system for {industry} applications."
    ]
    
    scales = ["1M", "5M", "10M", "50M", "100M"]
    industries = ["banking", "healthcare", "retail", "manufacturing", "insurance", "legal", "e-commerce", "government"]
    metrics = ["context relevance", "factual accuracy", "toxicity", "groundedness", "answer semantic similarity"]
    use_cases = ["customer support", "code generation", "document summarization", "data analysis", "financial forecasting"]
    requests = ["100K", "500K", "1M", "10M"]
    data_types = ["text", "multimodal", "image", "code snippets", "financial reports"]

    prompts = []
    # Generate exactly 200 prompts by sampling
    random.seed(42)
    while len(prompts) < 200:
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

def main():
    # We use SmolLM2-360M-Instruct as the teacher.
    model_id = "HuggingFaceTB/SmolLM2-360M-Instruct"
    print(f"Loading teacher model {model_id} on MPS...")
    
    # Using float16 for standard Mac inference speed
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="mps",
        torch_dtype=torch.float16,
    )
    
    prompts = get_prompts()
    
    os.makedirs("dataset", exist_ok=True)
    output_file = "dataset/train.jsonl"
    
    # If the file exists and has 200 lines, skip generation
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            lines = f.readlines()
        if len(lines) == 200:
            print(f"Dataset already exists with 200 examples at {output_file}. Skipping generation.")
            return

    print("Generating 200 synthetic examples. This will take some time...")
    
    with open(output_file, "w") as f:
        for i, prompt_text in enumerate(prompts):
            print(f"[{i+1}/200] Generating response for: {prompt_text}")
            
            messages = [
                {"role": "user", "content": prompt_text}
            ]
            
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to("mps")
            
            outputs = model.generate(
                inputs,
                max_new_tokens=250,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            
            response_tokens = outputs[0][inputs.shape[1]:] # exclude prompt
            response_text = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
            
            data_dict = {
                "instruction": prompt_text,
                "output": response_text
            }
            
            f.write(json.dumps(data_dict) + "\n")
            f.flush()

    print(f"Dataset generation complete. Saved to {output_file}")

if __name__ == "__main__":
    main()
