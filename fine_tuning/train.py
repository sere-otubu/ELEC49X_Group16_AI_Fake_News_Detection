from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import numpy as np
import evaluate
import os
from huggingface_hub import login
from getpass import getpass

# --- 1. Configuration ---
MODEL_NAME = "roberta-base"
OUTPUT_DIR = "my_finetuned_model"

# !!! IMPORTANT: Change this to your actual Hugging Face model name !!!
HUB_MODEL_NAME = "sereotubu/fake-news-detector-isot" 
# -----------------------------------------------------------------

def login_to_huggingface():
    """Logs into Hugging Face Hub"""
    token = os.environ.get('HF_TOKEN')
    if token:
        print("Logging in with HF_TOKEN environment variable...")
        login(token=token)
    else:
        print("Please enter your Hugging Face token:")
        token = getpass()
        login(token=token)

def train_model():
    # --- 2. Load Data ---
    if not os.path.exists('train.csv') or not os.path.exists('validation.csv'):
        print("Error: 'train.csv' or 'validation.csv' not found.")
        print("Please run 'python split_dataset.py' first.")
        return

    dataset = load_dataset('csv', data_files={'train': 'train.csv', 'validation': 'validation.csv'})

    # --- 3. Preprocessing ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # --- 4. Load Model ---
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=2,
        id2label={0: "false", 1: "true"},
        label2id={"false": 0, "true": 1}
    )

    # --- 5. Evaluation Metric ---
    metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        # Use pos_label=1 to calculate F1 for the "true" class
        return metric.compute(predictions=predictions, references=labels, pos_label=1)

    # --- 6. Set Training Arguments ---
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",        # Use the older argument name
        save_strategy="epoch",        # Use the older argument name
        num_train_epochs=3,               
        per_device_train_batch_size=8,    
        per_device_eval_batch_size=8,
        learning_rate=5e-6,               
        load_best_model_at_end=True,      
        metric_for_best_model="f1",       
        logging_dir='./logs',
        push_to_hub=True,                 
        hub_model_id=HUB_MODEL_NAME,
    )

    # --- 7. Create Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer, # This is fine, the warning is just a notice
    )

    # --- 8. Train! ---
    print("Starting fine-tuning...")
    trainer.train()

    # --- 9. Save and Upload ---
    print("Training complete. Saving and uploading model...")
    trainer.save_model(OUTPUT_DIR)
    trainer.push_to_hub()

    print("="*50)
    print(f"✅ All done! Your fine-tuned model is saved locally in '{OUTPUT_DIR}'")
    print(f"and has been pushed to your Hugging Face Hub at: https://huggingface.co/{HUB_MODEL_NAME}")

if __name__ == "__main__":
    try:
        login_to_huggingface()
        train_model()
    except Exception as e:
        print(f"\nAn error occurred: {e}")