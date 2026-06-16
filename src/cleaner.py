import os
import pandas as pd

def clean_dataset(file_path):
    df = pd.read_csv(file_path)
    
    # 1. Auto-drop structural exact duplicates
    df = df.drop_duplicates()
    
    # 2. Smart fill gaps (Fills numerical with median, text with 'UNKNOWN')
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna('UNKNOWN')
            
    # Save the polished dataset
    os.makedirs('cleaned_data', exist_ok=True)
    clean_filename = os.path.join('cleaned_data', 'clean_' + os.path.basename(file_path))
    df.to_csv(clean_filename, index=False)
    print(f"Saved pristine data to: {clean_filename}")

if __name__ == "__main__":
    data_dir = "data"
    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            clean_dataset(os.path.join(data_dir, file))
