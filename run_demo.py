import subprocess
import time
import psutil
import os
import sys

def run_script(script_name):
    print(f"\n{'='*60}")
    print(f"RUNNING: {script_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # We want to use the virtual environment python if it exists
    venv_python = "venv/bin/python" if os.path.exists("venv/bin/python") else sys.executable
    
    process = subprocess.Popen(
        [venv_python, script_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    max_memory = 0
    p = psutil.Process(process.pid)
    
    while True:
        try:
            # Poll memory usage
            if p.is_running() and p.status() != psutil.STATUS_ZOMBIE:
                mem = p.memory_info().rss / (1024 * 1024) # MB
                if mem > max_memory:
                    max_memory = mem
                    
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
            
    rc = process.poll()
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n[Completed {script_name}] Duration: {duration:.1f}s | Max Memory: {max_memory:.1f} MB")
    
    if rc != 0:
        print(f"Error: {script_name} failed with return code {rc}")
        print("Stopping pipeline.")
        exit(rc)
        
    return duration, max_memory

def main():
    print("Starting Knowledge Distillation Demo Pipeline")
    print("WARNING: This demo requires the Hugging Face `google/gemma-3-270m-it` model.")
    print("Make sure you have logged in via `huggingface-cli login` and accepted the Gemma terms.")
    print("For full execution, 8GB of RAM + MPS capability on Mac is required.\n")
    
    total_start = time.time()
    
    stats = {}
    
    scripts = [
        "generate_dataset.py",
        "evaluate_before.py",
        "train_student.py",
        "evaluate_after.py",
        "metrics.py"
    ]
    
    for script in scripts:
        dur, mem = run_script(script)
        stats[script] = {"duration": dur, "memory": mem}
        
    total_duration = time.time() - total_start
    
    print(f"\n{'='*60}")
    print(f"{'PIPELINE SUMMARY':^60}")
    print(f"{'='*60}")
    print(f"{'Phase':<22} | {'Duration (s)':<15} | {'Max Memory (MB)':<15}")
    print("-" * 60)
    
    for script, data in stats.items():
        print(f"{script:<22} | {data['duration']:<15.1f} | {data['memory']:<15.1f}")
        
    print("-" * 60)
    print(f"{'TOTAL':<22} | {total_duration:<15.1f} | {'-':<15}")
    print(f"{'='*60}\nKnowledge Distillation Demo complete!")

if __name__ == "__main__":
    main()
