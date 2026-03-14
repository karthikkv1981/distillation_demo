import json
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

STUDENT_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
ADAPTER_PATH = "distilled_student_lora"

def generate_response(model, tokenizer, prompt_text):
    messages = [{"role": "user", "content": prompt_text}]
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("mps")
    outputs = model.generate(
        inputs, 
        max_new_tokens=250, 
        do_sample=True, 
        temperature=0.7, 
        top_p=0.9
    )
    response_tokens = outputs[0][inputs.shape[1]:]
    return tokenizer.decode(response_tokens, skip_special_tokens=True).strip()

def main():
    print("Loading baseline evaluation results...")
    try:
        with open("baseline_eval.json", "r") as f:
            baseline_results = json.load(f)
    except FileNotFoundError:
        print("Error: baseline_eval.json not found. Run evaluate_before.py first.")
        return
        
    print(f"\nLoading Distilled Student Model {STUDENT_ID} from {ADAPTER_PATH} on MPS...")
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        STUDENT_ID, 
        torch_dtype=torch.float16, 
        device_map="mps"
    )
    
    # Load PEFT adapter
    try:
        student_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        student_tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
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
