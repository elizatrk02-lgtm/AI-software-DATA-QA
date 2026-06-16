import os
import pandas as pd
from pyod.models.iforest import IsolationForest

def analyze_data(file_path):
    print(f"Analyzing: {file_path}")
    df = pd.read_csv(file_path)
    
    # 1. Check for Gaps
    gaps = df.isnull().sum()
    print("\n[Gaps Found]:\n", gaps[gaps > 0])
    
    # 2. Check for Repetitions
    dup_count = df.duplicated().sum()
    print(f"\n[Duplicate Rows]: {dup_count}")
    
    # 3. AI Outlier Detection (for numerical columns)
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(num_cols) > 0:
        # Fill NaNs temporarily for the AI model
        df_clean = df[num_cols].fillna(0)
        
        # Train Isolation Forest
        clf = IsolationForest(contamination=0.05) # Flags top 5% anomalies
        clf.fit(df_clean)
        
        # Predict: -1 indicates an anomaly/error
        df['anomaly'] = clf.labels_
        anomalies = df[df['anomaly'] == -1]
        print(f"\n[AI Flagged Errors/Outliers]: Found {len(anomalies)} suspicious rows.")
        print(anomalies[num_cols].head())

# Run on all CSVs in the data folder
if __name__ == "__main__":
    data_dir = "data"
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith(".csv"):
                analyze_data(os.path.join(data_dir, file))

