# analysis_engine.py

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency, zscore
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for Flask
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
import os
import warnings
from dotenv import load_dotenv
import json
import uuid # For unique plot filenames
from io import BytesIO # If returning plot data directly
import base64 # For embedding plots in HTML

# --- Configuration ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Google Gemini (Same as before)
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        print("Gemini AI configured successfully.")
        USE_GEMINI = True
    except Exception as e:
        print(f"Error configuring Gemini AI: {e}. Explanations will be basic.")
        USE_GEMINI = False
else:
    print("Gemini API Key not found. Explanations will be basic.")
    USE_GEMINI = False

# Constants (same as before)
ALPHA = 0.05
IF_CONTAMINATION = 'auto'
Z_SCORE_THRESHOLD = 3
IQR_MULTIPLIER = 1.5

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- Helper Functions (Keep load_data, identify_column_types, handle_missing_values, get_gemini_explanation) ---
# (Copy these functions from the previous script here)
def load_data(historic_path, new_path):
    """Loads historic and new datasets from specified paths."""
    try:
        # Basic validation for CSV
        if not historic_path.lower().endswith('.csv') or not new_path.lower().endswith('.csv'):
             raise ValueError("Invalid file type. Please upload CSV files.")

        df_historic = pd.read_csv(historic_path)
        df_new = pd.read_csv(new_path)
        print(f"Loaded historic data: {df_historic.shape}")
        print(f"Loaded new data: {df_new.shape}")

        # Ensure consistent columns (preserve order from original CSVs)
        common_cols = [col for col in df_historic.columns if col in df_new.columns]
        extra_hist_cols = [col for col in df_historic.columns if col not in df_new.columns]
        extra_new_cols = [col for col in df_new.columns if col not in df_historic.columns]
        if extra_hist_cols:
            print(f"Warning: Columns in historic but not new: {extra_hist_cols}")
        if extra_new_cols:
            print(f"Warning: Columns in new but not historic: {extra_new_cols}. These cannot be directly compared for drift.")

        # Ensure consistent data types (same logic as before)
        for col in common_cols:
             if df_historic[col].dtype != df_new[col].dtype:
                 try:
                     if pd.api.types.is_numeric_dtype(df_historic[col]) or pd.api.types.is_numeric_dtype(df_new[col]):
                         df_historic[col] = pd.to_numeric(df_historic[col], errors='coerce')
                         df_new[col] = pd.to_numeric(df_new[col], errors='coerce')
                     else:
                         df_historic[col] = df_historic[col].astype(str)
                         df_new[col] = df_new[col].astype(str)
                     print(f"Attempted type harmonization for column '{col}'.")
                 except Exception as e:
                     print(f"Warning: Could not harmonize dtype for column '{col}': {e}. Skipping drift analysis for this column.")
                     if col in common_cols: common_cols.remove(col) # Ensure removal if error occurs

        # Return only common columns for comparison, preserving order
        return df_historic[common_cols], df_new[common_cols]

    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        raise ValueError(f"File not found: {e}") # Re-raise for Flask
    except pd.errors.EmptyDataError:
        print("Error: One or both CSV files are empty.")
        raise ValueError("One or both CSV files are empty.")
    except ValueError as e: # Catch our CSV check or pd.read_csv errors
         print(f"Error reading CSV: {e}")
         raise ValueError(f"Error processing CSV: {e}") # Re-raise for Flask
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        raise RuntimeError(f"An unexpected error occurred during data loading: {e}") # Re-raise

def identify_column_types(df):
    # --- (Same code as before) ---
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    text_cols = []
    potential_text = categorical_cols[:]
    for col in potential_text:
        if df[col].nunique() > 0.5 * len(df[col]) or \
           (df[col].str.len().mean() > 50 if pd.api.types.is_string_dtype(df[col]) else False):
             if col in categorical_cols:
                 is_likely_text = (df[col].nunique() > 0.1 * len(df[col]) and \
                                   (df[col].dropna().astype(str).str.len().mean() > 30 if not df[col].dropna().empty else False))
                 if is_likely_text:
                     text_cols.append(col)
                     categorical_cols.remove(col)
    print(f"Numerical columns: {numerical_cols}")
    print(f"Categorical columns: {categorical_cols}")
    print(f"Text columns: {text_cols}")
    return numerical_cols, categorical_cols, text_cols

def handle_missing_values(df, strategy='auto'):
    # --- (Same code as before) ---
    df_filled = df.copy()
    for col in df_filled.columns:
        if df_filled[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df_filled[col]):
                fill_value = df_filled[col].median() if strategy == 'median' else df_filled[col].mean()
                if strategy == 'auto': fill_value = df_filled[col].median()
                print(f"Filled missing values in numerical column '{col}' with {fill_value:.2f} ({'median' if strategy != 'mean' else 'mean'})")
            elif pd.api.types.is_categorical_dtype(df_filled[col]) or pd.api.types.is_object_dtype(df_filled[col]):
                fill_value = df_filled[col].mode()[0] if not df_filled[col].mode().empty else 'Missing'
                print(f"Filled missing values in categorical/text column '{col}' with '{fill_value}' (mode or 'Missing')")
    return df_filled


def get_gemini_explanation(prompt):
    # --- (Same code as before) ---
    if not USE_GEMINI:
        return "Gemini AI not available. Basic explanation provided."
    try:
        response = gemini_model.generate_content(prompt)
        explanation = response.text.strip().replace('*', '').replace('`', '')
        return explanation
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return f"Error fetching explanation from Gemini: {e}"

def get_preview_tables(df_historic, df_new):
    """Return HTML tables for the first 10 rows of both datasets, preserving column order."""
    historic_html = df_historic.head(10).to_html(classes='preview-table', index=False, border=0, columns=list(df_historic.columns))
    new_html = df_new.head(10).to_html(classes='preview-table', index=False, border=0, columns=list(df_new.columns))
    return historic_html, new_html

# --- Drift Detection Functions (No LLM call here) ---
def detect_numerical_drift(series_historic, series_new, column_name):
    series_historic_na = series_historic.dropna()
    series_new_na = series_new.dropna()
    if len(series_historic_na) < 2 or len(series_new_na) < 2:
        print(f"Skipping KS test for '{column_name}': insufficient non-NaN data.")
        return None
    ks_stat, p_value = ks_2samp(series_historic_na, series_new_na)
    drift_score = 1 - p_value
    drift_detected = p_value < ALPHA
    result = {
        "column": column_name, "type": "Numerical", "test": "Kolmogorov-Smirnov",
        "statistic": ks_stat, "p_value": p_value, "drift_score": drift_score,
        "drift_detected": drift_detected
    }
    return result

def detect_categorical_drift(series_historic, series_new, column_name):
    all_categories = sorted(list(set(series_historic.unique()) | set(series_new.unique())))
    hist_counts = series_historic.value_counts().reindex(all_categories, fill_value=0)
    new_counts = series_new.value_counts().reindex(all_categories, fill_value=0)
    contingency_table = pd.DataFrame({'historic': hist_counts, 'new': new_counts})
    if contingency_table.empty or contingency_table.shape[0] < 1 or contingency_table.shape[1] < 2 or contingency_table.sum().sum() == 0:
        return None
    contingency_table = contingency_table.loc[contingency_table.sum(axis=1) > 0, contingency_table.sum(axis=0) > 0]
    if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
        return None
    try:
        chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
        drift_score = 1 - p_value
        drift_detected = p_value < ALPHA
    except ValueError:
        return None
    result = {
        "column": column_name, "type": "Categorical", "test": "Chi-squared",
        "statistic": chi2_stat, "p_value": p_value, "drift_score": drift_score,
        "drift_detected": drift_detected
    }
    return result

# --- Anomaly Detection Functions (No LLM call here) ---
def detect_anomalies_isolation_forest(df_historic, df_new, numerical_cols):
    """Detect anomalies in new data using Isolation Forest trained on historic data."""
    if not numerical_cols:
        return None
    df_hist = df_historic[numerical_cols].copy()
    df_new_num = df_new[numerical_cols].copy()
    df_hist = handle_missing_values(df_hist, strategy='median')
    df_new_num = handle_missing_values(df_new_num, strategy='median')
    scaler = StandardScaler()
    df_hist_scaled = scaler.fit_transform(df_hist)
    df_new_scaled = scaler.transform(df_new_num)
    model = IsolationForest(contamination=IF_CONTAMINATION, random_state=42, n_estimators=100)
    model.fit(df_hist_scaled)
    anomaly_scores = model.decision_function(df_new_scaled)
    predictions = model.predict(df_new_scaled)
    anomaly_df = pd.DataFrame({
        'index_in_new': df_new.index,
        'anomaly_score': anomaly_scores,
        'is_anomaly': [pred == -1 for pred in predictions]
    })
    return anomaly_df[anomaly_df['is_anomaly']]

def detect_statistical_anomalies(df_new, df_historic, numerical_cols):
    """Detect univariate anomalies in new data using Z-score and IQR based on historic data."""
    anomalies = {}
    for col in numerical_cols:
        hist_col = df_historic[col].dropna()
        if hist_col.empty:
            continue
        hist_mean = hist_col.mean()
        hist_std = hist_col.std()
        hist_q1 = hist_col.quantile(0.25)
        hist_q3 = hist_col.quantile(0.75)
        hist_iqr = hist_q3 - hist_q1
        lower_bound = hist_q1 - IQR_MULTIPLIER * hist_iqr
        upper_bound = hist_q3 + IQR_MULTIPLIER * hist_iqr
        if hist_std > 0:
            zscores = (df_new[col] - hist_mean) / hist_std
        else:
            zscores = pd.Series(np.inf, index=df_new.index)
            zscores[df_new[col] == hist_mean] = 0
            zscores[df_new[col].isnull()] = np.nan
        z_anomalies = df_new.index[abs(zscores) > Z_SCORE_THRESHOLD]
        iqr_anomalies = df_new.index[(df_new[col] < lower_bound) | (df_new[col] > upper_bound)]
        null_anomalies = df_new.index[df_new[col].isnull()]
        col_anomalies = {}
        for idx in null_anomalies:
            col_anomalies[idx] = {'method': 'Unexpected Null', 'value': None}
        for idx in z_anomalies:
            if idx not in col_anomalies:
                col_anomalies[idx] = {'method': 'Z-score', 'value': df_new.loc[idx, col]}
        for idx in iqr_anomalies:
            if idx not in col_anomalies:
                col_anomalies[idx] = {'method': 'IQR', 'value': df_new.loc[idx, col]}
        if col_anomalies:
            anomalies[col] = col_anomalies
    return anomalies

# --- Prioritization and Reporting Functions (No LLM call here) ---
def prioritize_issues(drift_results, if_anomalies, stat_anomalies):
    issues = []
    for result in drift_results:
        if result and result['drift_detected']:
            priority_score = result['drift_score']
            issues.append({
                'type': 'Data Drift', 'column': result['column'], 'severity_score': priority_score,
                'description': f"Significant {result['type']} drift detected.",
                'details': f"Test: {result['test']}, p-value: {result['p_value']:.4e}, Drift Score: {result['drift_score']:.4f}",
            })
    # ... add anomalies similarly ...
    issues.sort(key=lambda x: x['severity_score'], reverse=True)
    prioritization_reasoning = "Issues ranked by severity score. Addressing high-severity issues first is recommended."
    return issues, prioritization_reasoning

# --- Generate tabular report ---
def generate_report_html(drift_results, if_anomalies, stat_anomalies, prioritized_issues, plot_paths):
    report = """
    <h2>Data Drift & Anomaly Detection Report</h2>
    <hr>
    <h3> Columns</h3>
    <table class='result-table'><thead><tr><th>Column</th><th>Type</th><th>Test</th><th>Statistic</th><th>P-value</th><th>Drift Score</th></tr></thead><tbody>
    """
    for result in drift_results:
        if result and result['drift_detected']:
            report += f"<tr class='highlight'><td>{result['column']}</td><td>{result['type']}</td><td>{result['test']}</td><td>{result['statistic']:.4f}</td><td>{result['p_value']:.4e}</td><td>{result['drift_score']:.4f}</td></tr>"
    report += "</tbody></table>"
    # ... add anomalies table ...
    report += "<hr><h3>Prioritized Issues</h3><table class='result-table'><thead><tr><th>Type</th><th>Column</th><th>Severity</th><th>Description</th><th>Details</th></tr></thead><tbody>"
    for issue in prioritized_issues[:15]:
        report += f"<tr class='highlight'><td>{issue['type']}</td><td>{issue.get('column','')}</td><td>{issue['severity_score']:.3f}</td><td>{issue['description']}</td><td>{issue['details']}</td></tr>"
    report += "</tbody></table>"
    report += "<hr><p><em>End of Report</em></p>"
    return report

# --- Main Analysis Function ---
def run_analysis(historic_filepath, new_filepath):
    all_plots_base64 = {}
    try:
        df_historic, df_new = load_data(historic_filepath, new_filepath)
        if df_historic.empty or df_new.empty:
            return "Error: One or both datasets are empty after loading common columns.", {}, '', ''
        num_cols_hist, cat_cols_hist, text_cols_hist = identify_column_types(df_historic)
        num_cols_new, cat_cols_new, text_cols_new = identify_column_types(df_new)
        common_cols = df_historic.columns.tolist()
        common_num_cols = [col for col in common_cols if col in num_cols_hist and col in num_cols_new]
        common_cat_cols = [col for col in common_cols if col in cat_cols_hist and col in cat_cols_new]
        drift_results = []
        for col in common_num_cols:
            result = detect_numerical_drift(df_historic[col], df_new[col], col)
            if result:
                drift_results.append(result)
        for col in common_cat_cols:
            result = detect_categorical_drift(df_historic[col], df_new[col], col)
            if result:
                drift_results.append(result)
        if_anomalies = detect_anomalies_isolation_forest(df_historic, df_new, common_num_cols)
        stat_anomalies = detect_statistical_anomalies(df_new, df_historic, common_num_cols)
        prioritized_issues, prioritization_reasoning = prioritize_issues(drift_results, if_anomalies, stat_anomalies)
        report_html = generate_report_html(drift_results, if_anomalies, stat_anomalies, prioritized_issues, all_plots_base64)
        historic_preview, new_preview = get_preview_tables(df_historic, df_new)
        return report_html, all_plots_base64, historic_preview, new_preview
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        print(f"Analysis Error: {e}")
        error_html = f"<h2>Analysis Failed</h2><hr><p><strong>Error:</strong> {e}</p>"
        return error_html, {}, '', ''
    except Exception as e:
        print(f"Unexpected Analysis Error: {e}")
        error_html = f"<h2>Analysis Failed</h2><hr><p><strong>An unexpected error occurred:</strong> {e}</p>"
        return error_html, {}, '', ''