# Knowledge Distillation Demo

This project demonstrates a Knowledge Distillation pipeline on Apple Silicon (MPS). It uses a "Teacher" model (**SmolLM2-360M-Instruct**) to generate synthetic data and fine-tune a "Student" model (**Gemma 3 270M IT**) using LoRA/QLoRA.

## Prerequisites

- **Hardware**: Mac with Apple Silicon (M1/M2/M3/M4) and at least **8GB of RAM**.
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

Once authenticated, you can run the entire pipeline with a single command:

```bash
python3 run_demo.py
```

### What happens during the demo?

The `run_demo.py` script orchestrates the following steps:

1.  **`generate_dataset.py`**: Uses the teacher model to generate 200 synthetic instruction-response pairs.
2.  **`evaluate_before.py`**: Measures the base performance of the student model on a small test set.
3.  **`train_student.py`**: Fine-tunes the student model on the synthetic dataset using LoRA/QLoRA.
4.  **`evaluate_after.py`**: Measures the performance of the student model after training.
5.  **`metrics.py`**: Compares the results and generates a summary report.

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
