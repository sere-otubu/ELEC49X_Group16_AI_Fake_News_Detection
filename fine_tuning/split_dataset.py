import pandas as pd
from sklearn.model_selection import train_test_split
import os

# --- 1. Define File Names ---
true_file = 'True.csv'
fake_file = 'Fake.csv'
train_file = 'train.csv'
validation_file = 'validation.csv'
test_file = 'test.csv'

def split_data():
    # --- 2. Check if files exist ---
    if not os.path.exists(true_file) or not os.path.exists(fake_file):
        print(f"Error: Make sure '{true_file}' and '{fake_file}' are in the 'fine_tuning' folder.")
        return

    print("Files found. Starting processing...")
    
    # --- 3. Load and Label Data ---
    print(f"Loading {true_file}...")
    df_true = pd.read_csv(true_file)
    df_true['label'] = 1  # 1 for "truthful news"

    print(f"Loading {fake_file}...")
    df_fake = pd.read_csv(fake_file)
    df_fake['label'] = 0  # 0 for "fake news"

    # --- 4. Combine and Clean Data ---
    print("Combining datasets...")
    df_combined = pd.concat([df_true, df_fake], ignore_index=True)

    print(f"Original article count: {len(df_combined)}")
    # Drop any articles that have the exact same text
    df_combined.drop_duplicates(subset=['text'], inplace=True, keep='first')
    print(f"De-duplicated article count: {len(df_combined)}")

    # We only need the 'text' and 'label' columns
    df_final = df_combined[['text', 'label']].dropna()

    # --- 5. Shuffle the Data ---
    print("Shuffling data...")
    df_shuffled = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

    # --- 6. Split the Data (80% Train, 10% Validation, 10% Test) ---
    print("Splitting data into train, validation, and test sets...")
    
    df_train, df_temp = train_test_split(
        df_shuffled,
        test_size=0.2,
        random_state=42,
        stratify=df_shuffled['label']
    )

    df_validation, df_test = train_test_split(
        df_temp,
        test_size=0.5,
        random_state=42,
        stratify=df_temp['label']
    )

    # --- 7. Save the New Files ---
    print(f"Saving {train_file}...")
    df_train.to_csv(train_file, index=False)
    
    print(f"Saving {validation_file}...")
    df_validation.to_csv(validation_file, index=False)
    
    print(f"Saving {test_file}...")
    df_test.to_csv(test_file, index=False)

    print("\n" + "="*50)
    print("✅ All done!")
    print(f"Total articles: {len(df_shuffled)}")
    print(f"Training articles: {len(df_train)}")
    print(f"Validation articles: {len(df_validation)}")
    print(f"Test articles: {len(df_test)}")
    print("\nYour 'train.csv', 'validation.csv', and 'test.csv' files are ready.")

if __name__ == "__main__":
    split_data()