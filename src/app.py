import streamlit as st
import pandas as pd
import numpy as np
import os
from pyod.models.iforest import IsolationForest

# Visualization & Comprehensive Math Stack
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression

# 1. Custom CSS styling matching ChatGPT UI
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
st.sidebar.markdown("🤖 **AI, Math, & Inference Active**")
st.sidebar.caption("Modules: SciPy Stats / Scikit-Learn Regression")

st.sidebar.markdown("<br><br><br>---", unsafe_allow_html=True)
st.sidebar.markdown("<div style='font-size: 14px; font-weight:bold;'>🚀 Upgrade Workspace</div>", unsafe_allow_html=True)
if st.sidebar.button("Get Plus ($49/mo)", use_container_width=True):
    st.sidebar.success("🔗 Secure routing active...")

# --- MAIN HERO CONTAINER ---
st.markdown("<h1 style='text-align: center; margin-top: 50px; font-size: 42px;'>How can I clean your data today?</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #6e6e80; font-size: 16px; margin-bottom: 40px;'>Upload any dataset to run advanced regression lines, distribution models, and inference tests.</p>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="feature-card"><div class="feature-title">📈 Trendlines & Regressions</div><div class="feature-desc">Isolate dependencies, establish correlation margins, and predict numeric targets.</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="feature-card"><div class="feature-title">🧪 Parametric Hypothesis Testing</div><div class="feature-desc">Compute comparative metrics between groupings automatically using ANOVA or T-tests.</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["csv", "xlsx"], label_visibility="collapsed")

if uploaded_file is not None:
    st.markdown("---")
    st.markdown(f"### 💬 DataQA Assistant")
    
    if uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file, engine='openpyxl')
    else:
        df = pd.read_csv(uploaded_file)
        
    gaps = df.isnull().sum().sum()
    dup_count = df.duplicated().sum()
    
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    anomaly_count = 0
    anomalies_df = pd.DataFrame()
    
    if len(num_cols) > 0:
        df_clean_num = df[num_cols].fillna(0)
        clf = IsolationForest(contamination=0.05)
        clf.fit(df_clean_num)
        df['anomaly'] = clf.labels_
        anomalies_df = df[df['anomaly'] == -1]
        anomaly_count = len(anomalies_df)

    st.write(f"Inverted **{uploaded_file.name}** containing **{len(df)} records**. Workspace metrics generated below:")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Structural Gaps", f"{gaps} entries")
    m2.metric("Duplicated Records", f"{dup_count} loops")
    m3.metric("AI Flagged Outliers", f"{anomaly_count} variants")

    # =========================================================================
    # CORE UPGRADE BLOCK: HIGHER LEVEL INFERENCE & REGRESSION INTERACTIVE MODULES
    # =========================================================================
    
    # Section A: Predictive Trendlines and Regression Modeling
    with st.expander("📈 Predictive Trendlines & Correlation Regression Models", expanded=True):
        if len(num_cols) >= 2:
            st.markdown("#### 📉 Dynamic Single-Variable Ordinary Least Squares (OLS)")
            x_var = st.selectbox("Select Independent Variable (X-Axis):", num_cols, index=0)
            y_var = st.selectbox("Select Target Dependent Variable (Y-Axis):", num_cols, index=min(1, len(num_cols)-1))
            
            # Drop empty coordinates to fit pure matrix boundaries
            reg_df = df[[x_var, y_var]].dropna()
            
            if len(reg_df) > 3:
                X = reg_df[[x_var]].values
                y = reg_df[y_var].values
                
                model = LinearRegression()
                model.fit(X, y)
                predictions = model.predict(X)
                r_squared = model.score(X, y)
                
                # Visual output rendering
                fig_reg, ax_reg = plt.subplots(figsize=(7, 3.5))
                sns.scatterplot(data=reg_df, x=x_var, y=y_var, ax=ax_reg, color="#1e1e2f", alpha=0.7, label="Data Records")
                ax_reg.plot(reg_df[x_var], predictions, color="#10a37f", linewidth=2.5, label=f"Trendline (R² = {r_squared:.2f})")
                ax_reg.set_title(f"Linear Relationship: {y_var} vs {x_var}")
                ax_reg.legend()
                st.pyplot(fig_reg)
                plt.close()
                
                st.caption(f"**Mathematical Model Interpretation:** For every single unit increment in *{x_var}*, *{y_var}* is expected to shift by **{model.coef_[0]:.4f}** units (Intercept: {model.intercept_:.2f}).")
            else:
                st.info("Insufficient valid non-null vector points to calculate linear alignment coefficients.")
        else:
            st.info("Predictive regressions demand a minimum layout configuration of 2 numerical column variables.")

    # Section B: Comparative Inference Hypothesis Testing (T-Tests / ANOVA)
    with st.expander("🧪 Comparative Inference Testing (t-Test / ANOVA Matrix Gates)", expanded=False):
        if len(num_cols) > 0 and len(cat_cols) > 0:
            st.markdown("#### 🔬 Evaluate Categorical Influence on Continuous Targets")
            selected_cat = st.selectbox("Select Categorical Group Vector:", cat_cols)
            selected_num = st.selectbox("Select Target Measurement Array:", num_cols)
            
            # Structure subsets inside variable array categories
            clean_test_df = df[[selected_cat, selected_num]].dropna()
            groups = clean_test_df[selected_cat].unique()
            
            group_arrays = [clean_test_df[clean_test_df[selected_cat] == g][selected_num].values for g in groups]
            group_counts = [len(arr) for arr in group_arrays]
            
            # Enforce validation boundary: Ensure each group matrix features at least two rows
            if len(groups) >= 2 and min(group_counts) >= 2:
                if len(groups) == 2:
                    st.markdown(f"##### 📊 Running Independent Two-Sample Student's t-Test")
                    t_stat, p_val = stats.ttest_ind(group_arrays[0], group_arrays[1], equal_var=False)
                    st.write(f"Comparing Category **{groups[0]}** vs **{groups[1]}** across **{selected_num}**:")
                    st.metric("Calculated p-Value Probability", f"{p_val:.5f}")
                else:
                    st.markdown(f"##### 📊 Running One-Way Analysis of Variance (ANOVA)")
                    f_stat, p_val = stats.f_oneway(*group_arrays)
                    st.write(f"Evaluating across **{len(groups)} discrete categories** within variable array *{selected_cat}*:")
                    st.metric("Calculated ANOVA p-Value", f"{p_val:.5f}")
                
                # Decipher hypothesis benchmarks for operators
                if p_val < 0.05:
                    st.success(f"🎉 **Statistically Significant Relationship (p < 0.05):** The structural groupings within *{selected_cat}* exert a highly meaningful impact on the distribution variances of *{selected_num}*.")
                else:
                    st.warning(f"⚖️ **No Statistical Divergence (p ≥ 0.05):** The baseline variances observed between classifications appear mathematically matching. Variation is likely down to random distribution noise.")
            else:
                st.info("The selected grouping does not feature enough data variations or unique classification paths to run parametric tests.")
        else:
            st.info("Hypothesis matrix gates require at least 1 Categorical field and 1 Quantitative numerical field to initiate split profiling columns.")

    # =========================================================================
    # CORE PIPELINE SECTIONS
    # =========================================================================
    with st.expander("📊 Inspect Raw AI Anomalies"):
        if anomaly_count > 0:
            st.dataframe(anomalies_df[num_cols], use_container_width=True)
        else:
            st.info("No algorithmic outliers logged.")
