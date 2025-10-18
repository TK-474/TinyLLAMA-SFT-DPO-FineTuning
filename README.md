# 🦙 Fine-Tuning TinyLLaMA (SFT + DPO)

**🔗 Explore the models on Hugging Face:**  
👉 [TK47 on Hugging Face](https://huggingface.co/TK47)

---

## 📘 Project Overview

This repository contains the work for **Introduction to Text Analytics – Assignment 5**, focusing on **fine-tuning the TinyLLaMA-1.1B-Chat-v1.0 model** through two major stages:
1. **Supervised Fine-Tuning (SFT)** using the **Databricks Dolly-15k** dataset.  
2. **Preference Fine-Tuning (DPO)** using the **UltraFeedback-binarized-preferences** dataset.

The goal was to analyze the effects of different **LoRA (Low-Rank Adaptation)** configurations on model performance and understand how preference optimization can align model outputs closer to human judgment.

---

## 👥 Group Members

| Name | ID |
|------|----|
| Laiba Zubair | 24472 |
| Saad Lakhani | 24471 |
| **Talal Khan** | **25253** |

---

## 🧠 Base Model & Environment

- **Base Model:** [`TinyLlama/TinyLlama-1.1B-Chat-v1.0`](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- **Platform:** Kaggle (Dual T4 GPUs)  
- **Frameworks Used:**  
  - `transformers`  
  - `peft` (for LoRA)  
  - `trl` (for DPO)  
  - `datasets`  
  - `evaluate`

> ⚙️ Google Colab could not handle trials with higher epochs, so all experiments were conducted on **Kaggle** with direct upload of checkpoints to Hugging Face.

---

## 🧩 Part 1 – Supervised Instruction Fine-Tuning (SFT)

### 🧱 Dataset
**Databricks Dolly-15k** – 15,000 instruction–response pairs across categories like brainstorming, classification, QA, summarization, etc.

- Tokenization: Max length = 256  
- Data Split: 90% Train / 10% Test

### ⚙️ LoRA Configuration Trials

| Trial | Rank (r) | Alpha (α) | Dropout | Target Modules | LR | Batch | Epochs |
|:------|:----------|:----------|:---------|:----------------|:--|:-------|:--------|
| 1 | 8 | 32 | 0.05 | q_proj, v_proj | 2e-4 | 8 | 2 |
| 2 | 16 | 64 | 0.1 | q_proj, v_proj, k_proj, o_proj | 1e-4 | 8 | 2 |
| 3 | 32 | 128 | 0.2 | q_proj, v_proj | 5e-5 | 4 | 4 |
| 4 | 64 | 256 | 0.1 | q_proj, v_proj, k_proj, o_proj, gate_proj | 3e-4 | 8 | 3 |
| 5 | 8 | 64 | 0.05 | gate_proj, up_proj, down_proj | 2e-5 | 4 | 5 |

### 📊 Evaluation Metrics

| Metric | Base | Trial 1 | Trial 2 | **Trial 3** | Trial 4 | Trial 5 |
|:--------|:------|:--------|:--------|:-------------|:--------|:--------|
| BLEU | 3.05 | 2.58 | 2.39 | **2.37** | 2.57 | 2.57 |
| Perplexity ↓ | 11.90 | 8.30 | 8.31 | **8.24** | 10.53 | 8.25 |

> Despite minor BLEU drops, **Trial 3** exhibited the **lowest perplexity**, making it the best-performing SFT model.

---

## 🧭 Part 2 – Preference Fine-Tuning (DPO)

### 🧱 Dataset
**UltraFeedback-binarized-preferences** – ~63k examples of prompt–chosen–rejected triplets derived from human-aligned ratings.

Training splits:
- Trials 1–2: 500 train / 200 eval  
- Trials 3–5: 1000 train / 500 eval

### ⚙️ DPO Configuration Trials

| Trial | LoRA Config | DPO Config Summary |
|:------|:-------------|:-------------------|
| 1 | r=16, α=64, q_proj, v_proj, dropout=0.15 | β=0.1, lr=5e-4, batch=4, epochs=3 |
| 2 | r=8, α=32, q_proj, v_proj, k_proj | β=0.2, lr=5e-4, batch=3, epochs=2 |
| 3 | r=16, α=64, q_proj, dropout=0.2 | β=0.05, lr=2e-4, batch=4, epochs=3 |
| 4 | r=16, α=32, q_proj, k_proj, v_proj, o_proj | β=0.15, lr=4e-5, batch=2, epochs=1 |
| 5 | r=12, α=128, q_proj, k_proj, dropout=0.2 | β=0.2, lr=5e-6, batch=6, epochs=2 |

> Gradient accumulation and checkpointing were enabled to handle limited GPU memory.

### 📊 Evaluation Metrics

| Metric | Base | Trial 1 | Trial 2 | Trial 3 | Trial 4 | Trial 5 |
|:--------|:------|:--------|:--------|:--------|:--------|:--------|
| BLEU | 5.85 | 7.02 | 5.85 | **7.02** | 6.53 | **7.02** |
| Perplexity ↓ | 9.38 | 5.01 | 9.38 | **5.01** | 9.52 | **5.01** |

> Trials **1, 3, and 5** showed the best performance in both BLEU and perplexity, indicating successful preference alignment.

---

## 🧾 Key Learnings

- **LoRA Target Selection Matters:** Attention matrices (q, k, v) yielded the best improvements.  
- **Learning Rate & Batch Size are Critical:** Small adjustments led to large differences in model behavior.  
- **Adding More Matrices ≠ Better Results:** Overparameterization led to worse BLEU and higher perplexity.  
- **Resource Constraints:** Each SFT trial ≈ 4.5 hrs, DPO ≈ 5.5 hrs (on dual T4 GPUs).  
- **Ethical Response Handling:** Limited data (500–1000 rows) affected alignment on “refuse-to-answer” ethical questions.

---

## 📦 Repository Contents

├── data/ # Dataset loading scripts
├── notebooks/ # Kaggle notebooks for SFT and DPO
├── models/ # Checkpoint directories
├── results/ # Evaluation results and tables
├── utils/ # Helper functions
├── README.md # Project documentation (this file)
└── requirements.txt # Dependencies


---

## 📈 Results Summary

- **Best SFT model:** Trial 3  
  - BLEU: 2.37 | Perplexity: 8.24  
  - Hugging Face: [tinyllama-sft-t3](https://huggingface.co/TK47/tinyllama-sft-t3)

- **Best DPO model:** Trial 5  
  - BLEU: 7.02 | Perplexity: 5.01  
  - Hugging Face: [tinyllama-sft-dpo-trials/dpo_trial_1](https://huggingface.co/TK47/tinyllama-sft-dpo-trials/tree/main/dpo_trial_1)

---

## 🧩 References

- Peiyuan Zhang et al. (2024). *TinyLLaMA: An Open-Source Small Language Model*. arXiv.  
- Mike Conover et al. (2023). *Databricks Dolly-15k: Open Instruction-Tuned LLM Dataset*. Databricks.  
- Ganqu Cui et al. (2023). *UltraFeedback: Boosting LMs with High-Quality Feedback*. arXiv.  
- Long Ouyang et al. (2022). *Training Language Models to Follow Instructions with Human Feedback*. arXiv.  

---

## 💬 Acknowledgements

Special thanks to the **Introduction to Text Analytics** faculty for enabling hands-on exploration of fine-tuning methods and to **Kaggle** for GPU resources.


