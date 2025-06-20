# --- Install Dependencies ---
!pip install -q datasets transformers accelerate peft bitsandbytes
!pip install --upgrade transformers peft accelerate datasets

# --- Mount Google Drive ---
from google.colab import drive
drive.mount('/content/drive')

# --- Imports ---
import os
import json
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

# --- Constants and Config ---
MODEL_NAME = "google/flan-t5-base"
SPECIAL_TOKENS = ["<summarize>", "<mcq>"]
MAX_INPUT_LENGTH = 512
MAX_OUTPUT_LENGTH = 256

# --- Paths ---
DRIVE_BASE = "/content/drive/MyDrive"
CHECKPOINT_DIR = os.path.join(DRIVE_BASE, "flan-t5-lora-checkpoints")
DATA_PATH = os.path.join(DRIVE_BASE, "combined_mcq_summarization_finetune.jsonl")

# --- Load Dataset ---
try:
    df = pd.read_json(DATA_PATH, lines=True)
    dataset = Dataset.from_pandas(df)
    print(f" Dataset loaded: {len(dataset)} samples")
except Exception as e:
    raise RuntimeError(f" Could not load dataset: {e}")

# --- Initialize Tokenizer and Model ---
print("\n🔄 Initializing tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Add special tokens
tokenizer.add_tokens(SPECIAL_TOKENS)
model.resize_token_embeddings(len(tokenizer))

# Apply LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# --- Tokenization Function ---
def tokenize(example):
    input_text = example["input_text"].strip()
    output_text = example["target_text"].strip()

    if input_text.lower().startswith("summarize"):
        prefix = "<summarize> "
    elif input_text.lower().startswith("generate mcqs"):
        prefix = "<mcq> "
    else:
        prefix = ""

    model_inputs = tokenizer(
        prefix + input_text,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
    )

    labels = tokenizer(
        output_text,
        max_length=MAX_OUTPUT_LENGTH,
        truncation=True,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# --- Tokenize Dataset ---
print("\n Tokenizing...")
columns_to_remove = [col for col in dataset.column_names if col not in ['input_text', 'target_text']]
tokenized_dataset = dataset.map(tokenize, remove_columns=columns_to_remove)
print(f"Tokenized samples: {len(tokenized_dataset)}")

# --- Data Collator ---
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# --- Training Args ---
training_args = Seq2SeqTrainingArguments(
    output_dir=CHECKPOINT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    num_train_epochs=3,
    save_steps=1000,
    save_total_limit=2,
    logging_steps=20,
    logging_strategy="steps",
    fp16=False,
    do_train=True,
    report_to="tensorboard"
)

# --- Initialize Trainer ---
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# --- Function to Get Latest Checkpoint ---
def get_latest_checkpoint(path):
    if not os.path.exists(path):
        return None
    checkpoints = [
        os.path.join(path, d) for d in os.listdir(path)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(path, d))
    ]
    if not checkpoints:
        return None
    return sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1]

# --- Resume or Start Training ---
print(f"\n Starting Training...")

latest_ckpt = get_latest_checkpoint(CHECKPOINT_DIR)

if latest_ckpt:
    print(f"🔁 Resuming from latest checkpoint: {latest_ckpt}")
    trainer.train(resume_from_checkpoint=latest_ckpt)
else:
    print(f"▶️ No checkpoint found. Starting from scratch.")
    trainer.train()

print("\n✅ Training Completed.")
