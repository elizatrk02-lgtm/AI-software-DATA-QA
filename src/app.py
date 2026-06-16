import streamlit as st
import pandas as pd
import numpy as np
import os
from pyod.models.iforest import IsolationForest

# Visualization & Advanced Math Stack
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 1. Custom CSS styling to force a clean, minimalist layout matching ChatGPT's aesthetic
st.set_page_config(page_title="DataQA AI", page_icon="⚡", layout="centered")

st.markdown("""
    <style>
        .main { background-color: #1e1e2f; color: #ffffff; }
        h1, h2, h3 { font-weight: 700; font-family: 'Inter', sans-serif; letter-spacing: -0.5px; }
        
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
        
        div.stFileUploader {
            border: 2px dashed #10a37f !important;
            border-radius: 16px;
            padding: 25px;
            background-color: #fafafa;
        }
    </style>
""", unsafe_allow_html=True)

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

# --- SIDEBAR ---
st.sidebar.markdown("<h2 style='text-align: center; color: #10a37f;'>⚡ DataQA AI</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.markdown("🤖 **AI & Math Engine Active**")
st.sidebar.caption("Statistical Modules: SciPy / Seaborn / PyOD")

st.sidebar.markdown("<br><br><br>---", unsafe_allow_html=True)
st.sidebar.markdown("<div style='font-size: 14px; font-weight:bold;'>🚀 Upgrade Workspace</div>", unsafe_allow_html=True)
if st.sidebar.button("Get Plus ($49/mo)", use_container_width=True):
    st.sidebar.success("🔗 Secure routing active...")

# --- MAIN HERO CONTAINER ---
st.markdown("<h1 style='text-align: center; margin-top: 50px; font-size: 42px;'>How can I clean your data today?</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #6e6e80; font-size: 16px; margin-bottom: 40px;'>Upload any dataset to run instant ML anomaly detection and comprehensive statistical graphs.</p>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="feature-card"><div class="feature-title">📊 Statistical Mapping</div><div class="feature-desc">Compute skewness, kurtosis, distributions, and multi-variable correlation plots instantly.</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="feature-card"><div class="feature-title">🧠 Advanced AI Diagnostics</div><div class="feature-desc">Run Isolation Forest modeling gates to target non-linear outliers across columns.</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["csv", "xlsx"], label_visibility="collapsed")

if uploaded_file is not None:
    st.markdown("---")
    st.markdown(f"### 💬 DataQA Assistant")
    
    with st.spinner("Processing dimensions and mapping matrix coordinates..."):
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            df = pd.read_csv(uploaded_file)
            
        gaps = df.isnull().sum().sum()
        dup_count = df.duplicated().sum()
        
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

    st.write(f"Injected **{uploaded_file.name}** containing **{len(df)} records**. Analysis complete:")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Structural Gaps", f"{gaps} entries missing")
    m2.metric("Duplicated Records", f"{dup_count} loops")
    m3.metric("AI Flagged Outliers", f"{anomaly_count} variants")

    # =========================================================================
    # NEW STEP: COMPREHENSIVE STATISTICAL INTELLIGENCE GRID
    # =========================================================================
    with st.expander("📈 Advanced Statistical Analytics & Math Breakdown", expanded=True):
        if len(num_cols) > 0:
            st.markdown("#### 🔢 Descriptives & Higher-Order Moments")
            
            # Mathematical summary dataframe generation
            stats_summary = []
            for col in num_cols:
                col_data = df[col].dropna()
                if len(col_data) > 1:
                    skew = stats.skew(col_data)
                    kurt = stats.kurtosis(col_data)
                    stats_summary.append({
                        "Column Metric": col,
                        "Mean (μ)": f"{col_data.mean():.2f}",
                        "Median": f"{col_data.median():.2f}",
                        "Std Dev (σ)": f"{col_data.std():.2f}",
                        "Skewness": f"{skew:.2f}",
                        "Kurtosis": f"{kurt:.2f}"
                    })
            
            if stats_summary:
                st.dataframe(pd.DataFrame(stats_summary), use_container_width=True, hide_index=True)
            
            st.markdown("#### 📊 Automated Distribution Graphing")
            # Select target metric row to render graphs dynamically
            selected_col = st.selectbox("Select metric column to graph:", num_cols)
            
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            clean_col_data = df[selected_col].dropna()
            
            # Plot 1: Histogram & KDE (Continuous distribution curve)
            sns.histplot(clean_col_data, kde=True, ax=ax[0], color="#10a37f")
            ax[0].set_title(f"Distribution Shape for {selected_col}")
            ax[0].set_xlabel(selected_col)
            
            # Plot 2: Box & Whisker Plot (Statistical range dispersion mapping)
            sns.boxplot(x=clean_col_data, ax=ax[1], color="#1e1e2f")
            ax[1].set_title(f"Box Plot Range Interquartiles")
            ax[1].set_xlabel(selected_col)
            
            st.pyplot(fig)
            plt.close()

            # Plot 3: Multi-variable Matrix Correlation (if 2 or more columns exist)
            if len(num_cols) >= 2:
                st.markdown("#### 🔗 Relational Linear Correlation Matrix")
                fig_corr, ax_corr = plt.subplots(figsize=(6, 4))
                corr_matrix = df[num_cols].corr()
                sns.heatmap(corr_matrix, annot=True, cmap="viridis", fmt=".2f", ax=ax_corr, cbar=True)
                ax_corr.set_title("Pearson Correlation Coefficients (r)")
                st.pyplot(fig_corr)
                plt.close()
        else:
            st.info("No quantitative floating fields found to run multi-moment calculations.")

    # --- CORE WORKSPACE ACTIONS ---
    with st.expander("📊 Inspect Raw AI Anomalies"):
        if anomaly_count > 0:
            st.dataframe(anomalies_df[num_cols], use_container_width=True)
        else:
            st.info("No algorithmic outliers logged.")

    st.markdown("<br>", unsafe_allow_html=True)
    df_fixed = clean_data(df.drop(columns=['anomaly'], errors='ignore'))
    csv_clean = df_fixed.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Export Cleaned Operational File",
        data=csv_clean,
        file_name=f"clean_{uploaded_file.name}",
        mime="text/csv",
        use_container_width=True
    )
