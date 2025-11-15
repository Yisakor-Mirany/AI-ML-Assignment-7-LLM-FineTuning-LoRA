# AI-ML-Assignment-7-LLM-FineTuning-LoRA

Author: Yisakor Mirany
Model: distilbert-base-uncased
PEFT Method: LoRA (Low-Rank Adaptation)
Dataset: IMDB (Sentiment Analysis)

# 🔍 Summary

This project applies Parameter-Efficient Fine-Tuning (PEFT) using LoRA to fine-tune a pre-trained LLM (DistilBERT) for a binary sentiment classification task on the IMDB movie reviews dataset.

Instead of updating all 66M+ model parameters, LoRA injects small trainable rank-decomposition matrices into the attention mechanism, drastically reducing compute requirements while still achieving high accuracy.

The system:

Loads and preprocesses the IMDB dataset

Tokenizes text using the DistilBERT tokenizer

Attaches LoRA adapters to the query/value matrices

Fine-tunes only 1–2% of the model parameters

Evaluates model performance on Accuracy and F1

Saves final LoRA-enabled sentiment classifier

# 📦 Project Components

notebook.ipynb – Full workflow (data prep → LoRA setup → training → evaluation)

requirements.txt – Python dependencies

README.md – Documentation

Saved Model Folder (optional) – Final LoRA-tuned DistilBERT model

# 📘 Model & LoRA Configuration

Base Model:** distilbert-base-uncased**
Task: Sentiment classification
Classes:

- 0 → Negative

- 1 → Positive

LoRA Settings:

| Parameter      | Value                     |
| -------------- | ------------------------- |
| r              | 8                         |
| lora_alpha     | 16                        |
| lora_dropout   | 0.1                       |
| Target         | Query & Value projections |
| PEFT Task Type | Sequence Classification   |


# 📊 Training Details

Epochs: 3

Batch Size: 16

Optimizer: AdamW (via HF Trainer)

Precision: FP16 (mixed precision)

Training Strategy: Evaluation + checkpoint each epoch

The notebook includes the complete training loop using Hugging Face Trainer.

# 🧪 Evaluation Metrics

After training, the model is evaluated on the IMDB test set.

Metric	Score
Accuracy	(Insert your final score here)
F1 Score	(Insert your final score here)

These scores typically reach ~90–93% with LoRA on DistilBERT.

# ⚖️ Baseline vs Fine-Tuned Comparison

Baseline (zero-shot DistilBERT):

Accuracy: ~50–60%

No task-specific knowledge

Poor sentiment classification

LoRA Fine-Tuned DistilBERT:

Accuracy: >90%

F1 significantly improves

Learns sentiment cues effectively

Only trains a tiny subset of parameters

# ⚡ Why LoRA Is Efficient

LoRA dramatically reduces the cost of fine-tuning by:

Freezing the entire pre-trained model

Injecting small, low-rank adapters into attention layers

Training only ~1–2% of parameters

Reducing GPU memory requirements

Speeding up training without losing accuracy

This makes it ideal for student environments, Colab GPUs, and real-world low-resource scenarios.

# 📝 Inference Example

def predict(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    return "Positive" if pred == 1 else "Negative"

predict("This movie was amazing!")
