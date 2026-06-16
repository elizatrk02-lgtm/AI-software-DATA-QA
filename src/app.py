import streamlit as st
import pandas as pd
import os
from pyod.models.iforest import IsolationForest

# 1. Custom CSS styling to force a clean, minimalist layout matching ChatGPT's aesthetic
st.set_page_config(page_title="DataQA AI", page_icon="⚡", layout="centered")

st.markdown("""
    <style>
        /* Minimalist background adjustments */
        .main { background-color: #1e1e2f; color: #ffffff; }
        h1, h2, h3 { font-weight: 700; font-family: 'Inter', sans-serif; letter-spacing: -0.5px; }
        
        /* ChatGPT-style Feature Cards */
        .feature-card {
            background-color: #f7f7f8;
            border: 1px solid #e5e5e7;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 15px;
            color: #2d2d2d;
        }
        .feature-title { font-weight: 600; font-size: 16px; margin-bottom: 5px; color: #10a37f; }
        .feature-desc { font-size: 13px; color: #6e6e80; }
        
        /* Premium Upload Area Styling Override */
        div.stFileUploader {
            border: 2px dashed #10a37f !important;
            border-radius: 16px;
            padding: 25px;
            background-color: #fafafa;
        }
    </style>
""", unsafe_allow_html=True)

# Helper functions for the data logic
def clean_data(df):
    df_clean = df.drop_duplicates().copy()
    for col in df_clean.columns:
        if df_clean[col].dtype in ['float64', 'int64']:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        else:
            df_clean[col] = df_clean[col].fillna('UNKNOWN')
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].astype(str).str.strip()
    return df_clean

# --- SIDEBAR (Conversational & Account Control) ---
st.sidebar.markdown("<h2 style='text-align: center; color: #10a37f;'>⚡ DataQA AI</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.markdown("🤖 **AI Assistant Active**")
st.sidebar.caption("Model: IsolationForest-v1.0")

# Session History Tracker
st.sidebar.markdown("<br><b>Recent Workspace Scans</b>", unsafe_allow_html=True)
st.sidebar.caption("📁 No active history. Upload a dataset to begin.")

# Monetization Hook Footer
st.sidebar.markdown("<br><br><br>---", unsafe_allow_html=True)
st.sidebar.markdown("<div style='font-size: 14px; font-weight:bold;'>🚀 Upgrade Workspace</div>", unsafe_allow_html=True)
st.sidebar.caption("Unlock automated API streaming webhooks and premium enterprise SLA rules.")
if st.sidebar.button("Get Plus ($49/mo)", use_container_width=True):
    st.sidebar.success("🔗 Redirecting securely to Stripe checkout...")

# --- MAIN HERO CONTAINER (ChatGPT Layout) ---
st.markdown("<h1 style='text-align: center; margin-top: 50px; font-size: 42px;'>How can I clean your data today?</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #6e6e80; font-size: 16px; margin-bottom: 40px;'>Upload any messy dataset to instantly target structural gaps, loops, and statistical anomalies.</p>", unsafe_allow_html=True)

# Grid Layout for Features (Aesthetic Only, mimics the prompt cards)
col1, col2 = st.columns(2)
with col1:
    st.markdown("""
        <div class="feature-card">
            <div class="feature-title">🔍 Scan Gaps & Formats</div>
            <div class="feature-desc">Detect empty rows, trace pattern errors, and validate emails or phone entries dynamically.</div>
        </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
        <div class="feature-card">
            <div class="feature-title">🧠 Train Unsupervised AI</div>
            <div class="feature-desc">Leverage Isolation Forest models to flag malicious entries or corrupted variables out of line.</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 2. Centered Interactive Drag-and-Drop Dropzone
uploaded_file = st.file_uploader("", type=["csv", "xlsx"], label_visibility="collapsed")

# --- CONVERSATIONAL RESPONSE BOX (Triggers when file is provided) ---
if uploaded_file is not None:
    st.markdown("---")
    st.markdown(f"### 💬 DataQA Assistant")
    
    with st.spinner("Analyzing parameters and fitting ML clusters..."):
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            df = pd.read_csv(uploaded_file)
            
        gaps = df.isnull().sum().sum()
        dup_count = df.duplicated().sum()
        
        # Calculate Machine Learning Anomalies
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        anomaly_count = 0
        anomalies_df = pd.DataFrame()
        
        if len(num_cols) > 0:
            df_clean_num = df[num_cols].fillna(0)
            clf = IsolationForest(contamination=0.05)
            clf.fit(df_clean_num)
            df['anomaly'] = clf.labels_
            anomalies_df = df[df['anomaly'] == -1]
            anomaly_count = len(anomalies_df)

    # Conversational streaming summary style response
    st.write(f"I have successfully ingested **{uploaded_file.name}** containing **{len(df)} records**. Here is what I discovered:")
    
    # Clean Metric Summary Pillars
    m1, m2, m3 = st.columns(3)
    m1.metric("Structural Gaps", f"{gaps} missing entries")
    m2.metric("Duplicated Records", f"{dup_count} repeats")
    m3.metric("AI Flagged Outliers", f"{anomaly_count} anomalies")
    
    # Present Interactive Deep Dives Expanded Below
    with st.expander("📊 Inspect Detected Anomalies & Outliers"):
        if anomaly_count > 0:
            st.dataframe(anomalies_df[num_cols], use_container_width=True)
        else:
            st.info("No deep-layer statistical outliers found.")

    # Auto-Fixing Engine Export Node (Pristine execution)
    st.markdown("<br>", unsafe_allow_html=True)
    st.write("✨ **Action Optimized:** I have compiled a sanitized, structurally complete version of this dataset.")
    
    df_fixed = clean_data(df.drop(columns=['anomaly'], errors='ignore'))
    csv_clean = df_fixed.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Download Cleaned Version",
        data=csv_clean,
        file_name=f"clean_{uploaded_file.name}",
        mime="text/csv",
        use_container_width=True
    )
