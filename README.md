#  FLAN-T5 Multitask Fine-Tuning with LoRA

This project explores multitask fine-tuning of [`google/flan-t5-base`](https://huggingface.co/google/flan-t5-base) using **Low-Rank Adaptation (LoRA)** via the 🤗 [PEFT](https://github.com/huggingface/peft) library. The dataset combines two instruction-like NLP tasks:

- ** Summarization** from news articles (CNN/DailyMail)
- ** MCQ Generation** from reading comprehension passages (RACE)

---

##  Why This Project?

The aim is to train a **unified FLAN-T5 model** that can dynamically handle both summarization and question generation using task-specific tokens. This is achieved by:

- Prefixing inputs with special tokens (`<summarize>`, `<mcq>`)
- Training on a single dataset with two tasks
- Using **LoRA** for parameter-efficient fine-tuning

---

##  What This Project Does

-  Loads and formats the **RACE** and **CNN/DailyMail** datasets  
-  Injects **task-specific prefixes** to guide generation  
-  Fine-tunes the model using 🤗 Transformers, Accelerate, and PEFT  
- **Resumes training** from the most recent checkpoint (if available)  
-  Produces a multi-task capable model with minimal compute

---

## ⚙ Key Features

| Feature                   | Description                                         |
|--------------------------|-----------------------------------------------------|
|  LoRA Fine-Tuning       | Efficiently updates only adapter weights            |
|  Resume Support         | Automatically loads the latest checkpoint           |
|  Multi-task Dataset     | Two distinct tasks, one model                       |
|  Difficulty-Aware MCQs  | Easy, Medium, Hard labels from RACE                 |
|  Special Prompt Tokens  | `<summarize>` and `<mcq>` guide generation behavior |

---

##  Special Tokens in Training

Special tokens are used to signal the type of task to the model:

| Token        | Purpose                          |
|--------------|----------------------------------|
| `<summarize>`| Prefix for summarization tasks   |
| `<mcq>`      | Prefix for MCQ generation tasks  |

These tokens are added before training:

```python
SPECIAL_TOKENS = ["<summarize>", "<mcq>"]
tokenizer.add_tokens(SPECIAL_TOKENS)
model.resize_token_embeddings(len(tokenizer))
```

---

##  Tokenization Strategy

**Summarization
Input:  <summarize> Summarize the following article: [article]
Target: Summary: [highlight]

**MCQ Generation
Input:  <mcq> Generate MCQs (Easy: 2, Medium: 1, Hard: 1): [passage]
Target: MCQs:
1. Question: ...
   Options:
   A. ...
   B. ...
   ...
   Correct: B
   Difficulty: Medium

---

##  Custom Multi-Task Dataset
The dataset merges:

10,000 MCQ samples from the RACE dataset

10,000 summarization samples from the cnn_dailymail dataset (v3.0.0)

**Format
{
  "input_text": "<mcq> Generate MCQs (Easy: 1, Medium: 1, Hard: 1): The cell is the basic unit of life...",
  "target_text": "MCQs:\n1. Question: What is the basic unit of life?\n   Options:\n   A. Atom B. Cell C. Organ D. Tissue\n   Correct: B\n   Difficulty: Easy\n..."
}

## Dataset Summary
| Task          | Samples  | Notes                               |
| ------------- | -------- | ----------------------------------- |
| MCQ (RACE)    | \~10,000 | Difficulty-balanced, labeled        |
| Summarization | \~10,000 | Extracted from CNN/DailyMail v3.0.0 |
| **Total**     | \~20,000 | Unified with task prefixing         |

---

##  Training
Training is performed using flan-t5-base with LoRA adapters on selected attention layers (q, v)

Uses Hugging Face Trainer and Seq2SeqTrainingArguments

Can resume from checkpoint-* inside flan-t5-lora-checkpoints/

---

##  Why This Matters
By conditioning generation on task-specific prefixes, the same model can:

Generalize across tasks

Avoid interference between summarization and QA generation

Achieve efficiency via LoRA without modifying the full model weights
