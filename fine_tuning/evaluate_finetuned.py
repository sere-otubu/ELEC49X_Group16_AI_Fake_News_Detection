import torch
from transformers import pipeline
from datasets import load_dataset
from sklearn.metrics import classification_report
from tqdm import tqdm
import os

# !!! IMPORTANT: Change this to your actual Hugging Face model name !!!
HUB_MODEL_NAME = "sereotubu/fake-news-detector-isot"
# -----------------------------------------------------------------

def evaluate_finetuned():
    print(f"Loading fine-tuned model from Hub: {HUB_MODEL_NAME}...")
    device = 0 if torch.cuda.is_available() else -1

    try:
        fine_tuned_classifier = pipeline(
            "text-classification",
            model=HUB_MODEL_NAME,
            device=device
        )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure your HUB_MODEL_NAME is correct and the model is public (or you are logged in).")
        return

    TEST_FILE = 'test.csv'
    if not os.path.exists(TEST_FILE):
        print(f"Error: {TEST_FILE} not found. Run 'python split_dataset.py' first.")
        return

    try:
        dataset = load_dataset('csv', data_files={'test': TEST_FILE})['test']
        dataset = dataset.map(lambda e: {"label_str": "true" if e['label'] == 1 else "false"})
        print(f"Loaded {len(dataset)} test samples from {TEST_FILE}.")
    except Exception as e:
        print(f"Error loading {TEST_FILE}: {e}")
        return

    texts = dataset['text']
    true_labels = dataset['label_str']
    
    print("\nStarting fine-tuned evaluation...")
    predicted_labels = []
    # Use batch_size for faster inference
    for result in tqdm(fine_tuned_classifier(texts, batch_size=8), total=len(texts)):
        predicted_labels.append(result['label'])

    print("\n" + "="*50)
    print(f"  Fine-Tuned Model Performance Report ({HUB_MODEL_NAME})")
    print("="*50)
    if true_labels:
        print(classification_report(true_labels, predicted_labels, labels=["true", "false"]))
    else:
        print("No predictions were made.")
    print("="*50)

if __name__ == "__main__":
    evaluate_finetuned()