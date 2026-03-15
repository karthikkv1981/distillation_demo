import json
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    results = []
    
    # 1. GENERATE TEACHER RESPONSES
    print(f"Loading Teacher Model {TEACHER_ID} on MPS...")
    teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_ID)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        TEACHER_ID, 
        torch_dtype=torch.float16, 
        device_map="mps"
    )
    
    teacher_responses = []
    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"Teacher generating [{i+1}/{len(TEST_PROMPTS)}]...")
        resp = generate_response(teacher_model, teacher_tokenizer, prompt)
        teacher_responses.append(resp)
        
    print("Teacher generation complete. Freeing memory...")
    del teacher_model
    del teacher_tokenizer
    gc.collect()
    torch.mps.empty_cache()

    # 2. GENERATE STUDENT RESPONSES
    print(f"\nLoading Untrained Student Model {STUDENT_ID} on MPS...")
    student_tokenizer = AutoTokenizer.from_pretrained(STUDENT_ID)
    student_model = AutoModelForCausalLM.from_pretrained(
        STUDENT_ID, 
        torch_dtype=torch.float16, 
        device_map="mps"
    )
    
    student_responses = []
    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"Student generating [{i+1}/{len(TEST_PROMPTS)}]...")
        resp = generate_response(student_model, student_tokenizer, prompt)
        student_responses.append(resp)
        
    # Free memory again
    del student_model
    del student_tokenizer
    gc.collect()
    torch.mps.empty_cache()
        
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
