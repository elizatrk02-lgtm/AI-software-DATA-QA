import streamlit as st
import pandas as pd
import numpy as np
import os
import io
from pyod.models.iforest import IsolationForest

# Visualization & Comprehensive Math Stack
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression

# PDF Generation Engine Modules
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

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

# --- PDF GENERATION UTILITY FUNCTION ---
def generate_pdf_report(filename, total_rows, gaps, duplicates, anomalies):
    """Generates an enterprise-ready PDF byte stream report completely in-memory."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    
    # Custom Brand Palette Styles
    title_style = ParagraphStyle(
        'PDFTitle', parent=styles['Heading1'], fontSize=24, leading=28, textColor=colors.HexColor('#10a37f'), spaceAfter=15
    )
    meta_style = ParagraphStyle(
        'PDFMeta', parent=styles['Normal'], fontSize=10, textColor=colors.HexColor('#6e6e80'), spaceAfter=20
    )
    section_style = ParagraphStyle(
        'PDFSection', parent=styles['Heading2'], fontSize=16, leading=20, textColor=colors.HexColor('#1e1e2f'), spaceBefore=15, spaceAfter=10
    )
    body_style = ParagraphStyle(
        'PDFBody', parent=styles['Normal'], fontSize=11, leading=15, spaceAfter=8
    )

    elements = []
    
    # Header Banner
    elements.append(Paragraph("⚡ Enterprise DataQA AI Verification Audit", title_style))
    elements.append(Paragraph(f"<b>Target Dataset:</b> {filename}<br><b>Generated:</b> 2026 Audit Pipeline", meta_style))
    elements.append(Spacer(1, 10))
    
    # Summary Section
    elements.append(Paragraph("1. Executive Quality Summary", section_style))
    elements.append(Paragraph(f"This document verifies the operational layout configuration and statistical validity vectors of the database rows processed by the Unsupervised DataQA AI engine.", body_style))
    
    # Tabular Performance Metrics Grid
    data = [
        [Paragraph('<b>Metric Inspected</b>', body_style), Paragraph('<b>Value Logged</b>', body_style), Paragraph('<b>System Risk Status</b>', body_style)],
        ['Total Processed Records', str(total_rows), 'Ingestion Complete'],
        ['Structural Data Gaps', str(gaps), 'Needs Resolution' if gaps > 0 else 'Passed'],
        ['Duplicated Record Loops', str(duplicates), 'Needs Cleansing' if duplicates > 0 else 'Passed'],
        ['AI Core Outliers Flagged', str(anomalies), 'Attention Required' if anomalies > 0 else 'Passed']
    ]
    
    t = Table(data, colWidths=[200, 150, 150])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#fafafa')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('BOTTOMPADDING', (0,0), (-1,0), 8),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e5e5e7')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f7f7f8')])
    ]))
    elements.append(t)
    elements.append(Spacer(1, 20))
    
    # AI Logic and Disclaimer Sections
    elements.append(Paragraph("2. Algorithmic Diagnostics Blueprint", section_style))
    elements.append(Paragraph("Statistical variances were cross-examined using an unsupervised Isolation Forest model configured at an expected contamination threshold metric of 5%. Flagged anomalies reflect multivariable data coordinate deviations out of normal operational bounds.", body_style))
    
    # Build Document Profile to Buffer
    doc.build(elements)
    buffer.seek(0)
    return buffer

# --- SIDEBAR ---
st.sidebar.markdown("<h2 style='text-align: center; color: #10a37f;'>⚡ DataQA AI</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.markdown("🤖 **AI, Math, & Inference Active**")
st.sidebar.caption("Modules: SciPy Stats / ReportLab PDF Generator")

st.sidebar.markdown("<br><br><br>---", unsafe_allow_html=True)
st.sidebar.markdown("<div style='font-size: 14px; font-weight:bold;'>🚀 Upgrade Workspace</div>", unsafe_allow_html=True)
if st.sidebar.button("Get Plus ($49/mo)", use_container_width=True):
    st.sidebar.success("🔗 Secure routing active...")

# --- MAIN HERO CONTAINER ---
st.markdown("<h1 style='text-align: center; margin-top: 50px; font-size: 42px;'>How can I clean your data today?</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #6e6e80; font-size: 16px; margin-bottom: 40px;'>Upload any dataset to run advanced regression lines, distribution models, and export PDF summaries.</p>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="feature-card"><div class="feature-title">📈 Trendlines & Regressions</div><div class="feature-desc">Isolate dependencies, establish correlation margins, and predict numeric targets.</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="feature-card"><div class="feature-title">📄 Audit Report Engine</div><div class="feature-desc">Compile complex machine learning outliers into verified, download-ready PDF briefs.</div></div>', unsafe_allow_html=True)

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

    # Regressions & Distributions Sections
    with st.expander("📈 Predictive Trendlines & Correlation Regression Models", expanded=True):
        if len(num_cols) >= 2:
            st.markdown("#### 📉 Dynamic Single-Variable Ordinary Least Squares (OLS)")
            x_var = st.selectbox("Select Independent Variable (X-Axis):", num_cols, index=0)
            y_var = st.selectbox("Select Target Dependent Variable (Y-Axis):", num_cols, index=min(1, len(num_cols)-1))
            
            reg_df = df[[x_var, y_var]].dropna()
            if len(reg_df) > 3:
                X = reg_df[[x_var]].values
                y = reg_df[y_var].values
                model = LinearRegression().fit(X, y)
                predictions = model.predict(X)
                r_squared = model.score(X, y)
                
                fig_reg, ax_reg = plt.subplots(figsize=(7, 3.5))
                sns.scatterplot(data=reg_df, x=x_var, y=y_var, ax=ax_reg, color="#1e1e2f", alpha=0.7, label="Data Records")
                ax_reg.plot(reg_df[x_var], predictions, color="#10a37f", linewidth=2.5, label=f"Trendline (R² = {r_squared:.2f})")
                ax_reg.set_title(f"Linear Relationship: {y_var} vs {x_var}")
                ax_reg.legend()
                st.pyplot(fig_reg)
                plt.close()
            else:
                st.info("Insufficient valid vector coordinates.")
        else:
            st.info("Predictive regressions demand a minimum of 2 numerical column fields.")

    # Core Action Controls & PDF Generation Download Interface
    st.markdown("<br>", unsafe_allow_html=True)
    st.write("### ✨ Document Export Channels")
    
    # Layout Action Buttons side-by-side matching a modern workflow platform
    btn_col1, btn_col2 = st.columns(2)
    
    with btn_col1:
        # Action 1: Export Data Cleansing file (.csv)
        df_fixed = clean_data(df.drop(columns=['anomaly'], errors='ignore'))
        # Action 1: Export Data Cleansing file (.csv)
        df_fixed = clean_data(df.drop(columns=['anomaly'], errors='ignore'))
        csv_clean = df_fixed.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 Export Cleaned Dataset (.CSV)",
            data=csv_clean,
            file_name=f"clean_{uploaded_file.name}",
            mime="text/csv",
            use_container_width=True
        )
        
    with btn_col2:
        # Action 2: Generate Corporate Executive PDF Document Audit
        pdf_stream = generate_pdf_report(uploaded_file.name, len(df), gaps, dup_count, anomaly_count)
        
        st.download_button(
            label="📄 Download Executive Audit Summary (.PDF)",
            data=pdf_stream,
            file_name=f"DataQA_Audit_Report_{os.path.splitext(uploaded_file.name)[0]}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
