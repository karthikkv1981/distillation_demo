import subprocess
import time
import psutil
import os
import sys
import argparse

def run_script(script_name, script_args=[]):
    print(f"\n{'='*60}")
    print(f"RUNNING: {script_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # We want to use the virtual environment python if it exists
    venv_python = "venv/bin/python" if os.path.exists("venv/bin/python") else sys.executable
    
    # Use -u for unbuffered output to ensure we see prints immediately
    process = subprocess.Popen(
        [venv_python, "-u", script_name] + script_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1 # Line buffered
    )
    
    # Set the pipe to non-blocking mode
    os.set_blocking(process.stdout.fileno(), False)
    
    max_memory = 0
    p = psutil.Process(process.pid)
    
    while True:
        try:
            # Poll memory usage
            if p.is_running() and p.status() != psutil.STATUS_ZOMBIE:
                mem = p.memory_info().rss / (1024 * 1024) # MB
                if mem > max_memory:
                    max_memory = mem
            
            # Non-blocking read
            try:
                output_chunk = process.stdout.read(4096)
                if output_chunk:
                    sys.stdout.write(output_chunk)
                    sys.stdout.flush()
                elif process.poll() is not None:
                    # Check one last time for data
                    final_chunk = process.stdout.read()
                    if final_chunk:
                        sys.stdout.write(final_chunk)
                        sys.stdout.flush()
                    break
            except (IOError, TypeError):
                # No data available right now
                if process.poll() is not None:
                    break
            
            time.sleep(0.05) # Responsive but not cpu-heavy
            
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            if process.poll() is not None:
                break
            
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=None, help="Number of synthetic examples to generate for training (e.g. 5 for a quick test)")
    args = parser.parse_args()

    if args.num_samples is None:
        print("\n--- Model Distillation Configuration ---")
        try:
            user_input = input("How many samples would you like to use for this run? [Press Enter for Default: 200]: ").strip()
            num_samples = int(user_input) if user_input else 200
        except ValueError:
            print("Invalid input. Defaulting to 200 samples.")
            num_samples = 200
    else:
        num_samples = args.num_samples

    print(f"\nPipeline configured for {num_samples} samples.")
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
        script_args = ["--num_samples", str(num_samples)] if script in ["generate_dataset.py", "train_student.py"] else []
        dur, mem = run_script(script, script_args)
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
