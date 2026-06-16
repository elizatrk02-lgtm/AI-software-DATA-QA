import os
import pandas as pd
from pyod.models.iforest import IsolationForest

def clean_data(df):
    # 1. Drop exact duplicates
    df_clean = df.drop_duplicates()
    
    # 2. Smart fill gaps (median for numbers, 'UNKNOWN' for text)
    for col in df_clean.columns:
        if df_clean[col].dtype in ['float64', 'int64']:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        else:
            df_clean[col] = df_clean[col].fillna('UNKNOWN')
    return df_clean

def analyze_data(file_path):
    df = pd.read_csv(file_path)
    
    # 1. Check for Gaps
    gaps = df.isnull().sum()
    gap_html = gaps[gaps > 0].to_frame(name='Missing Count').to_html(classes='table') if gaps.sum() > 0 else "<p>No gaps found!</p>"
    
    # 2. Check for Repetitions
    dup_count = df.duplicated().sum()
    dup_html = df[df.duplicated(keep=False)].to_html(classes='table') if dup_count > 0 else "<p>No duplicate rows found!</p>"
    
    # 3. AI Outlier Detection
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    ai_html = "<p>No numerical columns for AI analysis.</p>"
    anomaly_count = 0
    
    if len(num_cols) > 0:
        df_clean_num = df[num_cols].fillna(0)
        clf = IsolationForest(contamination=0.05)
        clf.fit(df_clean_num)
        df['anomaly'] = clf.labels_
        anomalies = df[df['anomaly'] == -1]
        anomaly_count = len(anomalies)
        ai_html = anomalies.to_html(classes='table') if anomaly_count > 0 else "<p>No AI anomalies detected.</p>"

    # 4. Save a pristine, auto-cleaned dataset
    df_fixed = clean_data(df.drop(columns=['anomaly'], errors='ignore'))
    os.makedirs('public/cleaned_data', exist_ok=True)
    df_fixed.to_csv(f"public/cleaned_data/clean_{os.path.basename(file_path)}", index=False)

    # 5. Construct HTML Dashboard String
    html_content = f"""
    <html>
    <head>
        <title>AI Data QA Report</title>
        <link rel="stylesheet" href="https://jsdelivr.net">
    </head>
    <body class="container my-5">
        <h1 class="mb-4">AI Data QA Report: {os.path.basename(file_path)}</h1>
        <div class="alert alert-info">Cleaned dataset available for download in repository.</div>
        <div class="card my-3"><div class="card-header"><h3>Data Gaps (Missing Values)</h3></div><div class="card-body">{gap_html}</div></div>
        <div class="card my-3"><div class="card-header"><h3>Repetitions (Duplicate Rows)</h3></div><div class="card-body">{dup_html}</div></div>
        <div class="card my-3"><div class="card-header"><h3>AI Flagged Anomalies</h3></div><div class="card-body">{ai_html}</div></div>
    </body>
    </html>
    """
    
    os.makedirs('public', exist_ok=True)
    with open('public/index.html', 'w') as f:
        f.write(html_content)

    # 6. Safety Rule: Fail the code run if more than 2 critical AI errors exist
    if anomaly_count > 2:
        print(f"\nCRITICAL ERROR: Found {anomaly_count} AI anomalies. Rejecting file!")
        exit(1)

if __name__ == "__main__":
    data_dir = "data"
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith(".csv"):
                analyze_data(os.path.join(data_dir, file))
                break
