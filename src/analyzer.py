import os
import re
import pandas as pd
from pyod.models.iforest import IsolationForest

def clean_data(df):
    df_clean = df.drop_duplicates().copy()
    
    for col in df_clean.columns:
        # Fix numbers and missing entries
        if df_clean[col].dtype in ['float64', 'int64']:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        else:
            df_clean[col] = df_clean[col].fillna('UNKNOWN')
            # Text Data Normalization: Remove trailing/leading white spaces
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].astype(str).str.strip()
    return df_clean

def check_pattern_errors(df):
    pattern_logs = []
    
    # Simple regex rules for validation
    email_regex = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    phone_regex = r'^\+?[\d\s-]{7,15}$'
    
    for col in df.columns:
        col_lower = col.lower()
        
        # 1. Email Format Verification
        if 'email' in col_lower:
            bad_emails = df[~df[col].astype(str).str.match(email_regex, na=True) & df[col].notna()]
            if not bad_emails.empty:
                pattern_logs.append(f"<b>{col}</b>: Found {len(bad_emails)} malformed email strings.")
                
        # 2. Phone Number Format Verification
        elif 'phone' in col_lower or 'tel' in col_lower:
            bad_phones = df[~df[col].astype(str).str.match(phone_regex, na=True) & df[col].notna()]
            if not bad_phones.empty:
                pattern_logs.append(f"<b>{col}</b>: Found {len(bad_phones)} irregular phone strings.")
                
    if not pattern_logs:
        return "<p>All structured text fields match expected format templates!</p>"
    return "<ul>" + "".join([f"<li>{log}</li>" for log in pattern_logs]) + "</ul>"

def analyze_data(file_path):
    # Detect file type and load correctly
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path, engine='openpyxl')
    else:
        df = pd.read_csv(file_path)
    
    # 1. Check for Gaps
    gaps = df.isnull().sum()
    gap_html = gaps[gaps > 0].to_frame(name='Missing Count').to_html(classes='table table-striped') if gaps.sum() > 0 else "<p>No gaps found!</p>"
    
    # 2. Check for Repetitions
    dup_count = df.duplicated().sum()
    dup_html = df[df.duplicated(keep=False)].to_html(classes='table table-striped') if dup_count > 0 else "<p>No duplicate rows found!</p>"
    
    # 3. Pattern / Template Verification Rules
    pattern_html = check_pattern_errors(df)
    
    # 4. AI Outlier Detection
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
        ai_html = anomalies.to_html(classes='table table-striped') if anomaly_count > 0 else "<p>No AI anomalies detected.</p>"

    # 5. Save cleaned version out to deployment folder
    df_fixed = clean_data(df.drop(columns=['anomaly'], errors='ignore'))
    os.makedirs('public/cleaned_data', exist_ok=True)
    
    base_name = os.path.basename(file_path)
    if file_path.endswith('.xlsx'):
        df_fixed.to_excel(f"public/cleaned_data/clean_{base_name}", index=False, engine='openpyxl')
    else:
        df_fixed.to_csv(f"public/cleaned_data/clean_{base_name}", index=False)

    # 6. Construct Comprehensive HTML Dashboard String
    html_content = f"""
    <html>
    <head>
        <title>AI Data QA Report</title>
        <link rel="stylesheet" href="https://jsdelivr.net">
    </head>
    <body class="container my-5">
        <h1 class="mb-4">📋 AI Data QA Report: {base_name}</h1>
        <div class="alert alert-success">✨ A pristine, reformatted version is ready for download in your repo directory.</div>
        
        <div class="card my-3"><div class="card-header bg-warning text-dark"><h3>⚠️ Data Gaps (Missing Values)</h3></div><div class="card-body">{gap_html}</div></div>
        <div class="card my-3"><div class="card-header bg-danger text-white"><h3>🔄 Repetitions (Duplicate Rows)</h3></div><div class="card-body">{dup_html}</div></div>
        <div class="card my-3"><div class="card-header bg-info text-white"><h3>📝 Text & Template Rule Violations</h3></div><div class="card-body">{pattern_html}</div></div>
        <div class="card my-3"><div class="card-header bg-primary text-white"><h3>🧠 AI Flagged Anomalies</h3></div><div class="card-body">{ai_html}</div></div>
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
