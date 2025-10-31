import torch
from transformers import pipeline
from datasets import load_dataset
from sklearn.metrics import classification_report
from tqdm import tqdm
import os

def evaluate_baseline():
    print("Loading RoBERTa model (zero-shot)...")
    device = 0 if torch.cuda.is_available() else -1
    
    try:
        classifier = pipeline(
            "zero-shot-classification",
            model="roberta-large-mnli",
            device=device
        )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
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

    y_true = []
    y_pred = []

    print("\nStarting baseline evaluation...")
    for item in tqdm(dataset):
        text = item['text']
        true_label = item['label_str']

        if not text or not isinstance(text, str) or len(text.strip()) == 0:
            continue
        
        try:
            result = classifier(
                text,
                candidate_labels=["truthful news", "fake news"],
                hypothesis_template="This text is {}."
            )
            
            labels = result['labels']
            scores = result['scores']
            truth_probability = scores[labels.index("truthful news")]
            predicted_label = "true" if truth_probability > 0.5 else "false"

            y_true.append(true_label)
            y_pred.append(predicted_label)
        except Exception as e:
            print(f"\n[Warning] Error during prediction: {e}. Skipping item.")
            continue

    print("\n" + "="*50)
    print("  Baseline Model Performance Report (roberta-large-mnli)")
    print("="*50)
    if y_true:
        print(classification_report(y_true, y_pred, labels=["true", "false"]))
    else:
        print("No predictions were made.")
    print("="*50)

if __name__ == "__main__":
    evaluate_baseline()