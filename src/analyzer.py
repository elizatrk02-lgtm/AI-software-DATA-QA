import os
import pandas as pd
from pyod.models.iforest import IsolationForest

def clean_data(df):
    df_clean = df.drop_duplicates()
    for col in df_clean.columns:
        if df_clean[col].dtype in ['float64', 'int64']:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        else:
            df_clean[col] = df_clean[col].fillna('UNKNOWN')
    return df_clean

def analyze_data(file_path):
    # Detect file type and load correctly
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path, engine='openpyxl')
    else:
        df = pd.read_csv(file_path)
    
    gaps = df.isnull().sum()
    gap_html = gaps[gaps > 0].to_frame(name='Missing Count').to_html(classes='table') if gaps.sum() > 0 else "<p>No gaps found!</p>"
    
    dup_count = df.duplicated().sum()
    dup_html = df[df.duplicated(keep=False)].to_html(classes='table') if dup_count > 0 else "<p>No duplicate rows found!</p>"
    
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

    # Save cleaned version back out as the same format
    df_fixed = clean_data(df.drop(columns=['anomaly'], errors='ignore'))
    os.makedirs('public/cleaned_data', exist_ok=True)
    
    base_name = os.path.basename(file_path)
    if file_path.endswith('.xlsx'):
        df_fixed.to_excel(f"public/cleaned_data/clean_{base_name}", index=False, engine='openpyxl')
    else:
        df_fixed.to_csv(f"public/cleaned_data/clean_{base_name}", index=False)

    html_content = f"""
    <html>
    <head>
        <title>AI Data QA Report</title>
        <link rel="stylesheet" href="https://jsdelivr.net">
    </head>
    <body class="container my-5">
        <h1 class="mb-4">AI Data QA Report: {base_name}</h1>
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

    if anomaly_count > 2:
        print(f"\nCRITICAL ERROR: Found {anomaly_count} AI anomalies. Rejecting file!")
        exit(1)

if __name__ == "__main__":
    data_dir = "data"
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith(".csv") or file.endswith(".xlsx"):
                analyze_data(os.path.join(data_dir, file))
                break
