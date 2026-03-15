import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

STUDENT_ID = "google/gemma-3-270m-it"

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        text = f"User: {example['instruction'][i]}\n\nAssistant: {example['output'][i]}"
        output_texts.append(text)
    return output_texts

def main():
    print(f"Preparing to train {STUDENT_ID}...")
    
    # Load dataset
    if not os.path.exists("dataset/train.jsonl"):
        print("Error: dataset/train.jsonl not found. Run generate_dataset.py first.")
        return
        
    dataset = load_dataset("json", data_files="dataset/train.jsonl", split="train")
    
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Attempt QLoRA 4-bit setup
    # Note: 4-bit bitsandbytes has limited support on Mac MPS. We gracefully fallback to float16.
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        print("Attempting to load model in 4-bit precision (QLoRA)...")
        model = AutoModelForCausalLM.from_pretrained(
            STUDENT_ID,
            quantization_config=bnb_config,
            device_map="auto" 
        )
        model = prepare_model_for_kbit_training(model)
        print("Successfully loaded in 4-bit!")
    except Exception as e:
        print(f"\n4-bit quantization fallback activated due to: {e}")
        print("Falling back to standard float16 LoRA for Apple Silicon MPS compatibility...")
        model = AutoModelForCausalLM.from_pretrained(
            STUDENT_ID,
            torch_dtype=torch.float16,
            device_map="mps"
        )
        
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Do not modify base weights -- PEFT automatically freezes base weights
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    training_args = TrainingArguments(
        output_dir="distilled_student_lora",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        optim="adamw_torch",
        use_mps_device=(model.device.type == "mps"),
        fp16=False, # MPS mixed precision is partial, explicit float16 model weights handles it
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant"
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_args,
        formatting_func=formatting_prompts_func
    )
    
    print("Starting distillation training...")
    trainer.train()
    
    print("Saving LoRA adapters...")
    trainer.model.save_pretrained("distilled_student_lora")
    tokenizer.save_pretrained("distilled_student_lora")
    print("Training complete! Adapters saved to 'distilled_student_lora'.")

if __name__ == "__main__":
    main()
