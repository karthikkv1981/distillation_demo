import json
import re

def compute_metrics(text):
    # 1. Response length (words)
    words = text.split()
    length = len(words)
    
    # 2. Arch components
    keywords = [
        "rag", "vector db", "vector database", "evaluation", 
        "monitoring", "governance", "scaling", "security", 
        "latency", "cost optimization", "cost"
    ]
    text_lower = text.lower()
    keyword_count = sum(1 for k in keywords if k in text_lower)
    
    # 3. Structured sections (headings '#' or numbered lists '1.')
    structure_count = 0
    for line in text.split('\n'):
        line = line.strip()
        if line.startswith('#') or re.match(r'^\d+\.', line):
            structure_count += 1
            
    return {
        "length": length,
        "keywords": keyword_count,
        "structures": structure_count
    }

def main():
    try:
        with open("final_eval.json", "r") as f:
            final_results = json.load(f)
    except FileNotFoundError:
        print("Error: final_eval.json not found. Run evaluate_after.py first.")
        return
        
    print("\nKnowledge Distillation Metrics")
    print(f"{'Metric':<18} | {'Teacher Avg':<15} | {'Student Before':<15} | {'Student After':<15}")
    print("-" * 75)
    
    metrics = {
        "teacher": {"length": 0, "keywords": 0, "structures": 0},
        "student_before": {"length": 0, "keywords": 0, "structures": 0},
        "student_after": {"length": 0, "keywords": 0, "structures": 0}
    }
    
    num_samples = len(final_results)
    if num_samples == 0:
        print("No results found.")
        return
        
    for item in final_results:
        t_m = compute_metrics(item["teacher"])
        sb_m = compute_metrics(item["student_before"])
        sa_m = compute_metrics(item["student_after"])
        
        for k in ["length", "keywords", "structures"]:
            metrics["teacher"][k] += t_m[k]
            metrics["student_before"][k] += sb_m[k]
            metrics["student_after"][k] += sa_m[k]
            
    # Calculate averages
    for model in ["teacher", "student_before", "student_after"]:
        for k in ["length", "keywords", "structures"]:
            metrics[model][k] /= num_samples
            
    keys = [
        ("Length (words)", "length"),
        ("Arch Keywords", "keywords"),
        ("Struct count", "structures")
    ]
    
    for display_name, k in keys:
        t_val = f"{metrics['teacher'][k]:.1f}"
        sb_val = f"{metrics['student_before'][k]:.1f}"
        sa_val = f"{metrics['student_after'][k]:.1f}"
        print(f"{display_name:<18} | {t_val:<15} | {sb_val:<15} | {sa_val:<15}")
    print("-" * 75)

if __name__ == "__main__":
    main()
