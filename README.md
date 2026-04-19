# Knowledge Distillation Demo

This project demonstrates a portable Knowledge Distillation pipeline that runs on Apple Silicon (MPS), NVIDIA GPUs (CUDA), or CPU fallback. It uses a "Teacher" model (**SmolLM2-360M-Instruct**) to generate synthetic data and fine-tune a "Student" model (**Gemma 3 270M IT**) using LoRA/QLoRA.

## Prerequisites

- **Hardware**: 
  - **Mac**: Apple Silicon (M1/M2/M3/M4) recommended.
  - **Windows/Linux**: NVIDIA GPU (CUDA) or standard CPU fallback.
  - **Minimum Memory**: **8GB of RAM** (16GB+ recommended).
- **Software**: Python 3.10 or higher.
- **Hugging Face Account**: Required to access the Gemma model.

## Setup Instructions

### 1. Environment Setup

Create a virtual environment and install the required dependencies:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Hugging Face Authentication

To run the demo, you must authenticate with Hugging Face and have access to the Gemma model.

1.  **Get a Token**: Create an "Access Token" (Read type) at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
2.  **Accept Model Terms**: Visit the [google/gemma-3-270m-it](https://huggingface.co/google/gemma-3-270m-it) page and accept the license terms.
3.  **Login**: Use the provided script to log in:
    ```bash
    python3 hf_login.py --token <YOUR_HF_TOKEN>
    ```
    Alternatively, you can use:
    ```bash
    huggingface-cli login
    ```

## Running the Demo

```bash
python3 run_demo.py
```

> [!TIP]
> **Don't want to wait for 200 samples to generate?** 
> When you run `python3 run_demo.py`, you will now be interactively asked how many samples you want to use for the run! Just type `5` and press enter for a lightning-fast demonstration. 
> 
> Alternatively, you can bypass the prompt entirely by passing the `--num_samples` flag:
> ```bash
> python3 run_demo.py --num_samples 5
> ```

### What is Knowledge Distillation? (The Layman's Explanation)

Think of Knowledge Distillation like a **Master and an Apprentice**. 
- The **Teacher Model (Master)** is a very smart, capable AI that requires a more computing power than student model. In this demo, the Master is **`SmolLM2-360M-Instruct`**.
- The **Student Model (Apprentice)** is a smaller, faster AI that isn't quite as smart out of the box, but can run easily on an everyday laptop. In this demo, the Apprentice is **`Gemma 3 270M IT`**.

Instead of making the Student learn everything from scratch, we have the Teacher generate "cheat sheets" or textbooks full of high-quality answers. We then use this perfectly curated material to train the Student. The end result is a small, fast AI that mimics the intelligence of the large, heavy AI!

> [!NOTE]
> **Understanding the Terminology:**
> - **Knowledge Distillation** is the overall *strategy* (The Master writes a textbook, the Apprentice studies it).
> - **Fine-Tuning** is the actual *action* of studying (The Apprentice reading the textbook and updating their brain).
> - **LoRA** is the *efficient technique* used to study (The Apprentice writing on a notepad instead of rewiring their whole brain). 
> 
> *In standard fine-tuning, humans have to manually write thousands of high-quality answers. What makes "Knowledge Distillation" special is that we use a massive, super-smart AI to automatically generate all that training data for us!*

### Step-by-Step Breakdown

When you run `python3 run_demo.py`, you are kicking off this exact Master/Apprentice training session locally on your machine. The pipeline automatically detects your hardware (CUDA, MPS, or CPU) and optimizes accordingly. Here is exactly what happens:

1. **Step 1: Making the Textbook (`generate_dataset.py`)** 
   *We wake up the Teacher Model.* We give it 200 tricky prompts (like "Design an AI system") and ask it to write out detailed, perfect answers. We save these answers into a file. This is the new "textbook" the Student will study.
   > **Example Textbook Entry:**
   > * **Prompt:** "Design an audit-compliant GenAI system for banking."
   > * **Teacher's Answer:** "To build a compliant banking system, you must implement localized vector databases, strict end-to-end encryption, and a robust PII-redaction pipeline before any prompt reaches the LLM..."

2. **Step 2: The Pre-Test (`evaluate_before.py`)** 
   *We test the Student before it studies.* We ask the raw, untrained Student Model to answer a few specific questions. It will probably give mediocre or generic answers. We save these to compare later.
   > **Example Pre-Test Answer:**
   > * **Prompt:** "Design an audit-compliant GenAI system for banking."
   > * **Untrained Student:** "Banking is very important. You should use computers and passwords. I can help you write code if you tell me what to do!" *(Notice how it completely misses the technical architectural nuances).*

3. **Step 3: Studying time (`train_student.py`)** 
   *The Student goes to school.* We feed the Teacher's 200 textbook answers to the Student Model. The Student adjusts its internal "brain" (weights) to try and sound exactly like the Teacher. 
   
   **Why do we use LoRA here?** 
   Normally, teaching an AI means opening up its entire "brain" and updating billions of connections at once. That requires massive, expensive supercomputers. **LoRA (Low-Rank Adaptation)** is like giving the Student a separate "notepad" to write its new learnings on, rather than performing brain surgery to rewire everything. We freeze the original brain and only train the small notepad. This allows the heavy training process to happen incredibly fast directly on your MacBook!
   
   > **What actually happens here:** The Student reads the prompt, attempts to guess the answer, and then checks its answer against the Teacher's actual textbook answer. It calculates the difference (the "loss"), and mathematically tweaks its tiny LoRA "notepad" so its next guess will be slightly closer to the Teacher's.

4. **Step 4: The Final Exam (`evaluate_after.py`)** 
   *We test the Student after it graduates.* We ask the tuned Student Model the exact same questions we asked in Step 2. We save these new, hopefully vastly improved, answers.
   > **Example Final Exam Answer:**
   > * **Prompt:** "Design an audit-compliant GenAI system for banking."
   > * **Graduated Student:** "An audit-compliant banking GenAI requires secure isolated environments. First, deploy an an on-premise vector store. Ensure all queries pass through a PII-scrubber layer..." *(It learned how to sound like an architect!)*

5. **Step 5: Grading (`metrics.py`)** 
   *We print the report card.* This script puts the Teacher's answer, the Student's Pre-Test answer, and the Student's Final Exam answer side-by-side so you can see exactly how much the Apprentice learned from the Master.

## Project Structure

- `run_demo.py`: Main entry point and pipeline orchestrator.
- `hf_login.py`: Helper script for Hugging Face authentication.
- `generate_dataset.py`: Synthetic data generation script.
- `train_student.py`: LoRA training script for the student model.
- `evaluate_before.py` / `evaluate_after.py`: Evaluation scripts.
- `metrics.py`: Summarization and comparison logic.
- `requirements.txt`: Python package dependencies.
- `dataset/`: Directory where the synthetic training data is stored.
- `distilled_student_lora/`: Directory where the trained LoRA adapters are saved.

## Troubleshooting

- **Memory Issues**: If the process crashes, ensure you have closed other memory-intensive applications. 8GB is the minimum required.
- **Quantization Errors**: The script attempts 4-bit quantization (QLoRA). If your environment doesn't support it, it will automatically fall back to standard float16.
