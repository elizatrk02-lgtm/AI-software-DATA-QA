import streamlit as st
import pandas as pd
import os
from analyzer import clean_data, check_pattern_errors
from pyod.models.iforest import IsolationForest

# Page Setup
st.set_page_config(page_title="AI Data QA Startup", page_icon="🤖", layout="wide")
st.title("🤖 Enterprise AI Data QA Platform")
st.write("Upload your data files to instantly detect gaps, repetitions, and machine-learning anomalies.")

# 1. File Uploader UI Component
uploaded_file = st.file_uploader("Drag and drop your CSV or XLSX file here", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Load Data safely
    if uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file, engine='openpyxl')
    else:
        df = pd.read_csv(uploaded_file)
        
    st.success(f"Successfully loaded {uploaded_file.name} ({len(df)} rows)")
    
    # Create Layout Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["⚠️ Gaps & Structural Errors", "🔄 Repetitions", "📝 Pattern Rules", "🧠 AI Outliers"])
    
    # Tab 1: Data Gaps
    with tab1:
        st.subheader("Missing Values Analyzer")
        gaps = df.isnull().sum()
        if gaps.sum() > 0:
            st.dataframe(gaps[gaps > 0].to_frame(name='Missing Count'), use_container_width=True)
        else:
            st.info("No data gaps found!")

    # Tab 2: Repetitions
    with tab2:
        st.subheader("Duplicate Record Analysis")
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            st.warning(f"Found {dup_count} duplicated rows.")
            st.dataframe(df[df.duplicated(keep=False)], use_container_width=True)
        else:
            st.info("No duplicates found!")

    # Tab 3: Pattern Rules
    with tab3:
        st.subheader("Format & String Validations")
        # Reuse your logic directly inside the UI app frame
        email_regex = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        for col in df.columns:
            if 'email' in col.lower():
                bad_emails = df[~df[col].astype(str).str.match(email_regex, na=True) & df[col].notna()]
                if not bad_emails.empty:
                    st.error(f"Column '{col}' has {len(bad_emails)} malformed email formats.")
                    st.dataframe(bad_emails)

    # Tab 4: AI Engine
    with tab4:
        st.subheader("Isolation Forest Anomaly Scoring")
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if len(num_cols) > 0:
            df_clean_num = df[num_cols].fillna(0)
            clf = IsolationForest(contamination=0.05)
            clf.fit(df_clean_num)
            df['anomaly'] = clf.labels_
            anomalies = df[df['anomaly'] == -1]
            
            if not anomalies.empty:
                st.error(f"AI Engine flagged {len(anomalies)} statistical anomalies!")
                st.dataframe(anomalies[num_cols])
            else:
                st.info("No multidimensional anomalies flagged by AI.")
        else:
            st.info("No numeric fields available for ML processing.")

    # Auto-Fixing Download Feature (Your Monetization Upsell Hook)
    st.markdown("---")
    st.subheader("✨ Auto-Cleaned Export Engine")
    df_fixed = clean_data(df.drop(columns=['anomaly'], errors='ignore'))
    
    csv_clean = df_fixed.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Cleaned Dataset Pristine File",
        data=csv_clean,
        file_name=f"clean_{uploaded_file.name}",
        mime="text/csv"
    )
# Insert into src/app.py beneath the download engine layout:
st.sidebar.markdown("---")
st.sidebar.subheader("🌟 Go Premium")
st.sidebar.write("Unlock unlimited automated schema checking, direct API connectivity, and custom Slack error webhooks.")
if st.sidebar.button("Upgrade to Premium ($49/mo)"):
    st.sidebar.info("🔗 [Click here to complete payment via Stripe Checkout](https://stripe.com)")
