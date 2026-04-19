import torch
import sys
import gc
import os
import argparse
from datasets import load_dataset

# Set MPS fallback and disable accelerate progress bars before any torch imports
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["ACCELERATE_DISABLE_RICH"] = "1"

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

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

def formatting_prompts_func(example):
    return f"User: {example['instruction']}\n\nAssistant: {example['output']}"

def main(num_samples=None):
    device = get_device()
    print(f"Preparing to train {STUDENT_ID} on {device}...", flush=True)
    
    # Pre-load cleanup
    cleanup()
    
    # Load dataset
    if not os.path.exists("dataset/train.jsonl"):
        print("Error: dataset/train.jsonl not found. Run generate_dataset.py first.")
        return
        
    dataset = load_dataset("json", data_files="dataset/train.jsonl", split="train")
    
    if num_samples is not None and len(dataset) > num_samples:
        dataset = dataset.select(range(num_samples))
        print(f"Subsampled dataset to {num_samples} examples for faster training.", flush=True)
        
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Standard LoRA on MPS (Avoid bitsandbytes/device_map="auto" hangs)
    print("Loading model weights strictly to CPU (float32)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        STUDENT_ID,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False,
        device_map="cpu"
    )
    
    print(f"Finalizing weights and moving model to {device}...", flush=True)
    model.tie_weights()
    model = model.to(device)
    print("Successfully loaded model for training!", flush=True)
        
    # Standard LoRA config (Used by SFTTrainer to wrap the model automatically)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    sft_config = SFTConfig(
        output_dir="distilled_student_lora",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        optim="adamw_torch",
        fp16=False,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        max_length=1024,
        report_to="none"
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
        args=sft_config,
        formatting_func=formatting_prompts_func
    )
    
    print("Starting distillation training...")
    trainer.train()
    
    print("Saving LoRA adapters...")
    trainer.model.save_pretrained("distilled_student_lora")
    tokenizer.save_pretrained("distilled_student_lora")
    print("Training complete! Adapters saved to 'distilled_student_lora'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=None, help="Number of synthetic examples to use")
    args = parser.parse_args()
    main(args.num_samples)
