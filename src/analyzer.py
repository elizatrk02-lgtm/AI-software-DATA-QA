import os
import pandas as pd
from pyod.models.iforest import IsolationForest

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
    if len(num_cols) > 0:
        df_clean = df[num_cols].fillna(0)
        clf = IsolationForest(contamination=0.05)
        clf.fit(df_clean)
        df['anomaly'] = clf.labels_
        anomalies = df[df['anomaly'] == -1]
        ai_html = anomalies.to_html(classes='table') if len(anomalies) > 0 else "<p>No AI anomalies detected.</p>"

    # 4. Construct HTML Dashboard String
    html_content = f"""
    <html>
    <head>
        <title>AI Data QA Report</title>
        <link rel="stylesheet" href="https://jsdelivr.net">
    </head>
    <body class="container my-5">
        <h1 class="mb-4">AI Data QA Report: {os.path.basename(file_path)}</h1>
        <div class="card my-3"><div class="card-header"><h3>Data Gaps (Missing Values)</h3></div><div class="card-body">{gap_html}</div></div>
        <div class="card my-3"><div class="card-header"><h3>Repetitions (Duplicate Rows)</h3></div><div class="card-body">{dup_html}</div></div>
        <div class="card my-3"><div class="card-header"><h3>AI Flagged Anomalies</h3></div><div class="card-body">{ai_html}</div></div>
    </body>
    </html>
    """
    
    # Ensure deployment directory exists
    os.makedirs('public', exist_ok=True)
    with open('public/index.html', 'w') as f:
        f.write(html_content)

if __name__ == "__main__":
    data_dir = "data"
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith(".csv"):
                analyze_data(os.path.join(data_dir, file))
                break # Generates report for the first found CSV


