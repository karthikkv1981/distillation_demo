# Knowledge Distillation: Student Exercises

This document contains 10 exercises designed to deepen your understanding of the knowledge distillation pipeline, model optimization, and deployment.

---

### 1. Upgrade the Teacher Model
**Task:** The current pipeline uses `SmolLM2-360M-Instruct` as the teacher. Replace it with a significantly larger model (2B+ parameters).
- **Options:** `google/gemma-2-2b-it`, `meta-llama/Llama-3.2-3B-Instruct`, or `Qwen/Qwen2.5-1.5B-Instruct`.
- **Files to Modify:** `generate_dataset.py`, `evaluate_before.py`.
- **Goal:** Observe how a more capable teacher improves the "textbook" quality and the student's final exam performance.

### 2. Fix Alignment Issues
**Task:** Identify why the student model isn't perfectly mimicking the teacher's style (e.g., it might be too brief or miss technical nuances).
- **Approach:** 
    - Adjust the `temperature` and `top_p` in `generate_dataset.py`.
    - Increase the `num_train_epochs` or `learning_rate` in `train_student.py`.
    - Modify the formatting function to include a specific system prompt like "You are a senior systems architect."
- **Goal:** Achieve a "Struct count" and "Arch Keywords" metric in `metrics.py` that closely matches the teacher's.

### 3. Cloud Deployment (AWS/GCP)
**Task:** Run the entire distillation pipeline (generation, training, evaluation) in a cloud environment like AWS (EC2/SageMaker) or Google Cloud (Vertex AI/Compute Engine).
- **AWS:** Use an EC2 instance (e.g., `g4dn.xlarge`) or a SageMaker Notebook.
- **GCP:** Use a Compute Engine instance with an L4 or T4 GPU.
- **Goal:** Successfully run `run_demo.py` in a headless Linux environment and manage Hugging Face authentication via environment variables.

### 4. Efficient Training with QLoRA
**Task:** Modify `train_student.py` to use 4-bit quantization (QLoRA) instead of standard weights.
- **Requirement:** This requires a CUDA-enabled GPU (bitsandbytes support).
- **Implementation:** Update the `BitsAndBytesConfig` and use `prepare_model_for_kbit_training`.
- **Goal:** Measure the reduction in peak VRAM usage compared to standard LoRA.

### 5. Expanding the Technical Domains
**Task:** The current dataset generator focuses on RAG, Banking, and Healthcare. Add 5 new technical domains to `get_prompts` in `generate_dataset.py`.
- **New Domains:** Cybersecurity, Bioinformatics, Game Development, IoT, and Cloud Infrastructure.
- **Goal:** Train the student to become a specialist in a broader range of engineering fields.

### 6. Hyperparameter Exploration
**Task:** Conduct an experiment by changing the LoRA rank (`r`) and alpha (`lora_alpha`).
- **Test cases:** 
    - `r=8, alpha=16`
    - `r=32, alpha=64`
    - `r=64, alpha=128`
- **Goal:** Document how increasing the rank affects the training time and the student's ability to learn complex patterns.

### 7. Semantic Evaluation Metrics
**Task:** Update `metrics.py` to go beyond word counts and keyword matching.
- **Implementation:** Use the `sentence-transformers` library to calculate the **Cosine Similarity** between the teacher's response and the student's response.
- **Goal:** Provide a mathematical score (0.0 to 1.0) of how semantically close the student's answer is to the "ground truth" teacher answer.

### 8. Interactive Distillation UI
**Task:** Create a visual dashboard for the demo.
- **Tool:** Use [Gradio](https://gradio.app/) or [Streamlit](https://streamlit.io/).
- **Features:** A text box for a user prompt, and three output columns showing:
    1. Teacher's Answer
    2. Student's Answer (Before Training)
    3. Student's Answer (After Training)
- **Goal:** Make the distillation results accessible to non-technical stakeholders.

### 9. Data Quality Filtering
**Task:** Implement a "Quality Gate" in `generate_dataset.py`.
- **Logic:** After the teacher generates a response, check if it meets certain criteria (e.g., word count > 100, contains at least 2 technical keywords, no repetitive loops). If it fails, discard it and regenerate.
- **Goal:** Ensure the student is only learning from the highest quality "textbook" entries.

### 10. Multi-Student Comparison
**Task:** Train two different student architectures on the same teacher-generated dataset.
- **Comparison:** Compare `google/gemma-3-270m-it` against `Qwen/Qwen2.5-0.5B-Instruct`.
- **Goal:** Determine which small-parameter model architecture is more efficient at absorbing knowledge from a larger teacher.
