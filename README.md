# 🤖 AI Data QA Software

An automated, machine learning-powered data quality assurance tool built natively for GitHub Actions. This software automatically scans datasets for missing gaps, structural duplicates, and statistical anomalies.

## 🚀 Features
* **Data Gap Detection:** Automatically flags columns containing missing (`NaN`) values.
* **Repetition Scan:** Finds and logs exact duplicate rows.
* **AI Anomaly Detection:** Uses an Unsupervised **Isolation Forest** model to find complex numerical outliers.
* **Auto-Fixing:** Cleans datasets by removing duplicates and replacing missing data with statistical medians.
* **Automated Web Dashboard:** Generates an interactive HTML dashboard hosted instantly via GitHub Pages.
* **CI/CD Quality Gate:** Automatically rejects pull requests if data corruption exceeds acceptable thresholds.

## 📁 Repository Structure
* `data/` — Drop your raw `.csv` or `.xlsx` files here to trigger the AI scanner.
* `src/analyzer.py` — The core Python script containing the data cleaning and AI model logic.
* `.github/workflows/data_qa.yml` — The automation script that spins up the cloud environment.

## 🛠️ How to Use
1. Commit or upload any data file into the `data/` directory.
2. Navigate to the **Actions** tab in this repository to watch the AI evaluate your data in real-time.
3. Access your live visual reports via the **GitHub Pages** URL generated in your repository settings.
