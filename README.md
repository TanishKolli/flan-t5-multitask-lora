# 📚 FLAN-T5 Multitask Fine-Tuning with LoRA

This project explores multitask fine-tuning of [`google/flan-t5-base`](https://huggingface.co/google/flan-t5-base) using **Low-Rank Adaptation (LoRA)** via the 🤗 [PEFT](https://github.com/huggingface/peft) library. The dataset combines two instruction-like NLP tasks:

- **📝 Summarization** from news articles (CNN/DailyMail)
- **❓ MCQ Generation** from reading comprehension passages (RACE)

---

## 🧠 Why This Project?

The aim is to train a **unified FLAN-T5 model** that can dynamically handle both summarization and question generation using task-specific tokens. This is achieved by:

- Prefixing inputs with special tokens (`<summarize>`, `<mcq>`)
- Training on a single dataset with two tasks
- Using **LoRA** for parameter-efficient fine-tuning

---

## 🛠️ What This Project Does

- ✅ Loads and formats the **RACE** and **CNN/DailyMail** datasets  
- ✅ Injects **task-specific prefixes** to guide generation  
- ✅ Fine-tunes the model using 🤗 Transformers, Accelerate, and PEFT  
- ✅ **Resumes training** from the most recent checkpoint (if available)  
- ✅ Produces a multi-task capable model with minimal compute

---

## ⚙️ Key Features

| Feature                   | Description                                         |
|--------------------------|-----------------------------------------------------|
| 🔧 LoRA Fine-Tuning       | Efficiently updates only adapter weights            |
| 🔁 Resume Support         | Automatically loads the latest checkpoint           |
| 🧩 Multi-task Dataset     | Two distinct tasks, one model                       |
| 🎯 Difficulty-Aware MCQs  | Easy, Medium, Hard labels from RACE                 |
| 🧬 Special Prompt Tokens  | `<summarize>` and `<mcq>` guide generation behavior |

---

## 🧬 Special Tokens in Training

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
