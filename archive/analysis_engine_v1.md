# analysis_engine.py

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency, zscore
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import google.generativeai as genai
import os
import warnings
from dotenv import load_dotenv
import json
import re
from collections import defaultdict

# --- Configuration ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Google Gemini
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # Using 1.5 Flash as it's generally available and capable for this task
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        print("Gemini AI configured successfully (gemini-1.5-flash).")
        USE_GEMINI = True
    except Exception as e:
        print(f"Error configuring Gemini AI: {e}. Explanations will be basic.")
        USE_GEMINI = False
else:
    print("Gemini API Key not found. Explanations will be basic.")
    USE_GEMINI = False

# Constants
ALPHA = 0.05  # Significance level for statistical tests
IF_CONTAMINATION = 'auto' # Contamination for Isolation Forest
Z_SCORE_THRESHOLD = 3    # Z-score threshold for outliers
IQR_MULTIPLIER = 1.5     # IQR multiplier for outliers
MAX_ANOMALIES_TO_LLM = 15 # Max anomalous rows details to send to LLM to avoid excessive prompt length
MAX_DRIFT_COLS_TO_LLM = 10 # Max drifted columns details to send to LLM

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# --- Helper Functions ---

def load_data(historic_path, new_path):
    """Loads historic and new datasets, finds common columns, and attempts type harmonization."""
    try:
        if not historic_path.lower().endswith('.csv') or not new_path.lower().endswith('.csv'):
             raise ValueError("Invalid file type. Please upload CSV files.")

        df_historic = pd.read_csv(historic_path)
        df_new = pd.read_csv(new_path)
        print(f"Loaded historic data: {df_historic.shape}")
        print(f"Loaded new data: {df_new.shape}")

        if df_historic.empty or df_new.empty:
            raise ValueError("One or both loaded dataframes are empty.")

        # Use original columns for reporting potential mismatches
        original_hist_cols = df_historic.columns.tolist()
        original_new_cols = df_new.columns.tolist()

        # Identify common columns based on name
        common_cols = [col for col in original_hist_cols if col in original_new_cols]
        extra_hist_cols = [col for col in original_hist_cols if col not in common_cols]
        extra_new_cols = [col for col in original_new_cols if col not in common_cols]

        # Warnings about mismatched columns
        if extra_hist_cols:
            print(f"Warning: Columns in historic but not new (will be ignored for comparison): {extra_hist_cols}")
        if extra_new_cols:
            print(f"Warning: Columns in new but not historic (will be ignored for comparison): {extra_new_cols}")

        if not common_cols:
             raise ValueError("No common columns found between the historic and new datasets.")

        # Select only common columns for comparison
        df_historic_common = df_historic[common_cols].copy()
        df_new_common = df_new[common_cols].copy()

        # Attempt to harmonize data types for common columns
        harmonized_common_cols = []
        for col in common_cols:
             dtype_hist = df_historic_common[col].dtype
             dtype_new = df_new_common[col].dtype

             if dtype_hist != dtype_new:
                 print(f"Attempting type harmonization for column '{col}' (Historic: {dtype_hist}, New: {dtype_new}).")
                 try:
                     # Prioritize numeric conversion if either is numeric
                     is_hist_numeric_like = pd.api.types.is_numeric_dtype(dtype_hist) or df_historic_common[col].apply(lambda x: isinstance(x, (int, float))).all()
                     is_new_numeric_like = pd.api.types.is_numeric_dtype(dtype_new) or df_new_common[col].apply(lambda x: isinstance(x, (int, float))).all()

                     if is_hist_numeric_like or is_new_numeric_like:
                         df_historic_common[col] = pd.to_numeric(df_historic_common[col], errors='coerce')
                         df_new_common[col] = pd.to_numeric(df_new_common[col], errors='coerce')
                         if df_historic_common[col].isnull().all() or df_new_common[col].isnull().all():
                              print(f"Warning: Column '{col}' became all NaN after numeric conversion. Check data.")
                         print(f"  Harmonized '{col}' to numeric.")
                         harmonized_common_cols.append(col)
                     else: # Fallback to string
                         df_historic_common[col] = df_historic_common[col].astype(str)
                         df_new_common[col] = df_new_common[col].astype(str)
                         print(f"  Harmonized '{col}' to string.")
                         harmonized_common_cols.append(col)
                 except Exception as e:
                     print(f"Warning: Could not harmonize dtype for column '{col}': {e}. Skipping this column for analysis.")
             else:
                harmonized_common_cols.append(col) # Types match, keep it

        if not harmonized_common_cols:
            raise ValueError("No common columns remaining after type harmonization.")

        # Return dataframes with only harmonized common columns
        return df_historic_common[harmonized_common_cols], df_new_common[harmonized_common_cols]

    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        raise ValueError(f"File not found: {e.filename}")
    except pd.errors.EmptyDataError as e:
        print(f"Error: CSV file is empty: {e}")
        # Identify which file is empty if possible (less straightforward with pd.read_csv)
        raise ValueError(f"CSV file seems empty. Please check the input files.")
    except ValueError as e: # Catch our specific errors or pd.read_csv value errors
         print(f"Error reading or processing CSV: {e}")
         raise ValueError(f"Error processing CSV: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        raise RuntimeError(f"An unexpected error occurred: {e}")

def identify_column_types(df):
    """Identifies numerical, categorical, and potential text columns."""
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    # Basic text detection (can be refined)
    text_cols = []
    potential_text = list(categorical_cols) # Operate on a copy
    for col in potential_text:
        # Heuristic: High cardinality or long average string length suggests text
        unique_ratio = df[col].nunique() / len(df[col]) if len(df[col]) > 0 else 0
        avg_len = df[col].astype(str).str.len().mean() if pd.api.types.is_string_dtype(df[col]) else 0
        # Adjust threshold: if > 20% unique AND avg length > 25 chars, or just > 50% unique
        if (unique_ratio > 0.2 and avg_len > 25) or unique_ratio > 0.5:
             text_cols.append(col)
             if col in categorical_cols: # Avoid error if already removed
                 categorical_cols.remove(col)

    print(f"Identified Column Types:")
    print(f"  Numerical: {numerical_cols}")
    print(f"  Categorical: {categorical_cols}")
    print(f"  Text (heuristic): {text_cols}")
    return numerical_cols, categorical_cols, text_cols

def handle_missing_values(df, numerical_strategy='median', categorical_strategy='mode'):
    """Handles missing values based on column type."""
    df_filled = df.copy()
    missing_info = {}
    for col in df_filled.columns:
        if df_filled[col].isnull().any():
            missing_count = df_filled[col].isnull().sum()
            missing_pct = (missing_count / len(df_filled)) * 100
            fill_value = None
            strategy_used = None

            if pd.api.types.is_numeric_dtype(df_filled[col]):
                if numerical_strategy == 'median':
                    fill_value = df_filled[col].median()
                    strategy_used = 'median'
                elif numerical_strategy == 'mean':
                    fill_value = df_filled[col].mean()
                    strategy_used = 'mean'
                else: # Default to median
                    fill_value = df_filled[col].median()
                    strategy_used = 'median'
                df_filled[col].fillna(fill_value, inplace=True)
                print(f"Filled {missing_count} missing values in numerical column '{col}' with {strategy_used} ({fill_value:.2f})")

            elif pd.api.types.is_object_dtype(df_filled[col]) or pd.api.types.is_categorical_dtype(df_filled[col]):
                if categorical_strategy == 'mode':
                    mode_val = df_filled[col].mode()
                    if not mode_val.empty:
                        fill_value = mode_val[0]
                        strategy_used = 'mode'
                    else: # Handle cases with no mode (e.g., all unique NaNs)
                        fill_value = 'Missing'
                        strategy_used = 'constant ("Missing")'
                else: # Default to constant 'Missing'
                     fill_value = 'Missing'
                     strategy_used = 'constant ("Missing")'
                df_filled[col].fillna(fill_value, inplace=True)
                print(f"Filled {missing_count} missing values in categorical/text column '{col}' with {strategy_used} ('{fill_value}')")
            else:
                 print(f"Warning: Column '{col}' has an unsupported type for missing value handling.")

            missing_info[col] = {'count': missing_count, 'percentage': missing_pct, 'fill_value': fill_value, 'strategy': strategy_used}

    return df_filled, missing_info


def get_preview_tables(df_historic, df_new):
    """Return HTML tables for the first 10 rows of both datasets."""
    # Ensure index is shown for reference
    historic_html = df_historic.head(10).to_html(classes='preview-table', index=True, border=0)
    new_html = df_new.head(10).to_html(classes='preview-table', index=True, border=0)
    return historic_html, new_html

# --- Drift Detection Functions ---
def detect_numerical_drift(series_historic, series_new, column_name, alpha=ALPHA):
    """Performs KS test for numerical drift."""
    series_historic_na = series_historic.dropna()
    series_new_na = series_new.dropna()
    if len(series_historic_na) < 2 or len(series_new_na) < 2:
        print(f"Skipping KS test for '{column_name}': insufficient non-NaN data.")
        return None
    try:
        ks_stat, p_value = ks_2samp(series_historic_na, series_new_na)
        # Clamp p-value slightly above zero for stability if it's exactly zero
        p_value = max(p_value, 1e-100)
        drift_score = 1 - p_value # Higher score means more drift (lower p-value)
        drift_detected = p_value < alpha
        confidence = 1 - p_value # Confidence that drift exists
        return {
            "column": column_name, "type": "Numerical", "test": "Kolmogorov-Smirnov",
            "statistic": ks_stat, "p_value": p_value, "drift_score": drift_score,
            "drift_detected": drift_detected, "confidence": confidence
        }
    except Exception as e:
        print(f"Error during KS test for '{column_name}': {e}")
        return None

def detect_categorical_drift(series_historic, series_new, column_name, alpha=ALPHA):
    """Performs Chi-squared test for categorical drift."""
    series_historic = series_historic.astype(str).fillna('NaN') # Ensure consistent type and handle NaN explicitly
    series_new = series_new.astype(str).fillna('NaN')

    all_categories = sorted(list(set(series_historic.unique()) | set(series_new.unique())))

    if len(all_categories) < 2:
         print(f"Skipping Chi2 test for '{column_name}': only one category observed across datasets.")
         return None

    hist_counts = series_historic.value_counts().reindex(all_categories, fill_value=0)
    new_counts = series_new.value_counts().reindex(all_categories, fill_value=0)

    contingency_table = pd.DataFrame({'historic': hist_counts, 'new': new_counts})

    # Filter out rows (categories) or columns (datasets) with zero counts
    contingency_table = contingency_table.loc[contingency_table.sum(axis=1) > 0, contingency_table.sum(axis=0) > 0]

    # Need at least 2x2 table for chi2
    if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
        print(f"Skipping Chi2 test for '{column_name}': contingency table too small after removing zeros ({contingency_table.shape}).")
        return None

    try:
        chi2_stat, p_value, _, expected_freq = chi2_contingency(contingency_table)
         # Check for low expected frequencies (common issue for Chi2 validity)
        if (expected_freq < 5).any().any():
            print(f"Warning: Low expected frequency (<5) in Chi2 test for '{column_name}'. P-value may be less reliable.")

        p_value = max(p_value, 1e-100) # Clamp p-value
        drift_score = 1 - p_value
        drift_detected = p_value < alpha
        confidence = 1 - p_value

        return {
            "column": column_name, "type": "Categorical", "test": "Chi-squared",
            "statistic": chi2_stat, "p_value": p_value, "drift_score": drift_score,
            "drift_detected": drift_detected, "confidence": confidence
        }
    except ValueError as e:
        # Often happens if sums are zero or dimensions mismatch unexpectedly
        print(f"Error during Chi2 test calculation for '{column_name}': {e}. Contingency table:\n{contingency_table}")
        return None
    except Exception as e:
         print(f"Unexpected error during Chi2 test for '{column_name}': {e}")
         return None

# --- Anomaly Detection Functions ---
def detect_anomalies_isolation_forest(df_historic, df_new, numerical_cols, contamination=IF_CONTAMINATION, random_state=42):
    """Detect anomalies in new data using Isolation Forest trained on historic numerical data."""
    if not numerical_cols:
        print("Skipping Isolation Forest: No numerical columns identified.")
        return pd.DataFrame(columns=['index_in_new', 'anomaly_score', 'is_anomaly'])

    # Prepare data: Select columns, handle missing, scale
    df_hist_num = df_historic[numerical_cols].copy()
    df_new_num = df_new[numerical_cols].copy()

    # Handle missing values using median imputation (consistent)
    df_hist_filled, _ = handle_missing_values(df_hist_num, numerical_strategy='median')
    df_new_filled, _ = handle_missing_values(df_new_num, numerical_strategy='median')

    # Scale data based on historic distribution
    scaler = StandardScaler()
    try:
        if df_hist_filled.empty:
             print("Warning: Historic numerical data is empty after handling missing values. Cannot train Isolation Forest.")
             return pd.DataFrame(columns=['index_in_new', 'anomaly_score', 'is_anomaly'])
        if df_hist_filled.var().sum() == 0: # Check if there's any variance
             print("Warning: Historic numerical data has zero variance after scaling. Isolation Forest may not be effective.")
             # Return empty results as IF might fail or give meaningless scores
             return pd.DataFrame(columns=['index_in_new', 'anomaly_score', 'is_anomaly'])

        df_hist_scaled = scaler.fit_transform(df_hist_filled)

        if df_new_filled.empty:
             print("Warning: New numerical data is empty after handling missing values. Cannot predict anomalies.")
             return pd.DataFrame(columns=['index_in_new', 'anomaly_score', 'is_anomaly'])

        df_new_scaled = scaler.transform(df_new_filled) # Use same scaler

    except ValueError as e:
         print(f"Error during scaling for Isolation Forest: {e}")
         return pd.DataFrame(columns=['index_in_new', 'anomaly_score', 'is_anomaly'])


    # Train and predict using Isolation Forest
    try:
        model = IsolationForest(contamination=contamination, random_state=random_state, n_estimators=100)
        model.fit(df_hist_scaled)

        # decision_function: lower score means more anomalous
        # predict: -1 for anomalies, 1 for inliers
        anomaly_scores = model.decision_function(df_new_scaled)
        predictions = model.predict(df_new_scaled)

        # Create result DataFrame
        anomaly_df = pd.DataFrame({
            'index_in_new': df_new.index,  # Use original index from df_new
            'anomaly_score': anomaly_scores, # Lower is more anomalous
            'is_anomaly': predictions == -1 # True if anomaly
        })
        print(f"Isolation Forest detected {anomaly_df['is_anomaly'].sum()} potential anomalies.")
        # Return all scores, filtering can happen later if needed
        return anomaly_df

    except Exception as e:
        print(f"Error during Isolation Forest execution: {e}")
        return pd.DataFrame(columns=['index_in_new', 'anomaly_score', 'is_anomaly'])


def detect_statistical_anomalies(df_new, df_historic, numerical_cols, z_threshold=Z_SCORE_THRESHOLD, iqr_multiplier=IQR_MULTIPLIER):
    """Detect univariate anomalies (Nulls, Z-score, IQR) in new data based on historic stats."""
    anomalies_by_row = defaultdict(list) # Map: index -> list of anomaly dicts for that row

    for col in numerical_cols:
        hist_col = df_historic[col].dropna()
        new_col = df_new[col] # Keep NaNs for null detection

        if hist_col.empty:
            print(f"Skipping statistical anomaly detection for '{col}': No historic data.")
            continue

        # Calculate historic stats
        hist_mean = hist_col.mean()
        hist_std = hist_col.std()
        hist_q1 = hist_col.quantile(0.25)
        hist_q3 = hist_col.quantile(0.75)
        hist_iqr = hist_q3 - hist_q1

        # Define bounds
        iqr_lower_bound = hist_q1 - iqr_multiplier * hist_iqr
        iqr_upper_bound = hist_q3 + iqr_multiplier * hist_iqr
        z_lower_bound = hist_mean - z_threshold * hist_std if hist_std > 0 else hist_mean
        z_upper_bound = hist_mean + z_threshold * hist_std if hist_std > 0 else hist_mean

        # Iterate through new data points for the column
        for idx, value in new_col.items():
            anomaly_found = False
            if pd.isnull(value):
                # Check if nulls were rare or non-existent historically
                historic_null_ratio = df_historic[col].isnull().mean()
                if historic_null_ratio < 0.01: # If nulls were <1% historically, flag new nulls
                    anomalies_by_row[idx].append({
                        'column': col, 'value': None, 'method': 'Unexpected Null',
                        'severity_score': 0.8, # Assign a relatively high severity score
                        'historic_context': f"Historic null rate: {historic_null_ratio:.2%}"
                    })
                    anomaly_found = True
            elif pd.api.types.is_numeric_dtype(value): # Ensure value is numeric before comparison
                # IQR check
                if value < iqr_lower_bound or value > iqr_upper_bound:
                    distance = min(abs(value - iqr_lower_bound), abs(value - iqr_upper_bound))
                    severity = 1.0 - np.exp(-0.1 * abs(distance / (hist_iqr + 1e-6))) # Higher distance -> higher severity (approaches 1)
                    anomalies_by_row[idx].append({
                        'column': col, 'value': value, 'method': 'IQR Outlier',
                        'severity_score': max(0.5, severity), # Ensure minimum severity for being flagged
                        'historic_context': f"IQR Bounds: ({iqr_lower_bound:.2f}, {iqr_upper_bound:.2f})"
                    })
                    anomaly_found = True

                # Z-score check (only if std dev is meaningful)
                if hist_std > 1e-6: # Avoid division by zero or near-zero std
                    z = (value - hist_mean) / hist_std
                    if abs(z) > z_threshold:
                        severity = 1.0 - np.exp(-0.1 * (abs(z) - z_threshold)) # Severity increases beyond threshold
                        # Avoid double-flagging if already caught by IQR, unless Z-score is much more extreme
                        if not anomaly_found or abs(z) > z_threshold * 1.5: # Add if not found or Z is very high
                             anomalies_by_row[idx].append({
                                 'column': col, 'value': value, 'method': 'Z-score Outlier',
                                 'z_score': z,
                                 'severity_score': max(0.5, severity) ,
                                 'historic_context': f"Mean: {hist_mean:.2f}, StdDev: {hist_std:.2f}"
                             })
                             anomaly_found = True

    print(f"Statistical methods identified potential anomalies in {len(anomalies_by_row)} rows.")
    return dict(anomalies_by_row)


# --- LLM Integration ---

def format_data_for_llm(df, indices, max_rows=10):
    """Formats specified rows of a DataFrame into a string for the LLM prompt."""
    if not indices or df.empty:
        return "No relevant data rows to display."
    
    indices_to_show = list(indices)[:max_rows]
    # Ensure indices exist in the dataframe
    valid_indices = [idx for idx in indices_to_show if idx in df.index]
    if not valid_indices:
        return "Specified indices not found in the dataframe."

    # Select rows and format as string (e.g., markdown table or similar)
    return df.loc[valid_indices].to_markdown(index=True)


def get_llm_analysis(drift_results, stat_anomalies_by_row, if_anomalies_df, df_new, df_historic, common_num_cols, common_cat_cols, alpha=ALPHA, max_drift=MAX_DRIFT_COLS_TO_LLM, max_anom=MAX_ANOMALIES_TO_LLM):
    """
    Generates a comprehensive prompt for Gemini, sends it, and parses the response.
    """
    if not USE_GEMINI:
        return {
            "summary": "LLM analysis skipped (Gemini not configured or unavailable).",
            "prioritized_issues": [],
            "drift_explanations": {},
            "anomaly_explanations": {}
        }

    # 1. Prepare Drift Summary for Prompt
    significant_drift = [r for r in drift_results if r.get('drift_detected', False)]
    significant_drift.sort(key=lambda x: x.get('drift_score', 0), reverse=True)
    drift_summary_prompt = "Detected Data Drift (Significant based on p < {}):\n".format(alpha)
    if not significant_drift:
        drift_summary_prompt += "No significant drift detected.\n"
    else:
        drift_summary_prompt += "Column | Type | Test | P-Value | Drift Score (1-P)\n"
        drift_summary_prompt += "-------|------|------|---------|-------------\n"
        for item in significant_drift[:max_drift]:
            drift_summary_prompt += f"{item['column']} | {item['type']} | {item['test']} | {item['p_value']:.3e} | {item['drift_score']:.3f}\n"
        if len(significant_drift) > max_drift:
             drift_summary_prompt += f"... (and {len(significant_drift) - max_drift} more drifted columns)\n"

    # 2. Prepare Anomaly Summary for Prompt
    anomaly_summary_prompt = "Detected Anomalies in New Dataset:\n"
    anomalous_rows_indices = set()

    # Combine statistical and IF anomalies, prioritizing by severity/score
    all_anomalies = []
    # Add statistical anomalies
    for idx, anomalies in stat_anomalies_by_row.items():
        for anom in anomalies:
             all_anomalies.append({'index': idx, 'column': anom['column'], 'value': anom['value'],
                                   'method': anom['method'], 'score': anom['severity_score'], 'type': 'Statistical'})
             anomalous_rows_indices.add(idx)

    # Add Isolation Forest anomalies (if any)
    if not if_anomalies_df.empty:
        if_anom_rows = if_anomalies_df[if_anomalies_df['is_anomaly']].sort_values('anomaly_score') # Lower score = more anomalous
        for idx, row in if_anom_rows.iterrows():
            # Use the actual index from df_new
            original_idx = row['index_in_new']
            # Map IF score to severity (e.g., lower score -> higher severity)
            # Simple linear scale: score ranges roughly -0.2 (anom) to 0.2 (normal) for default IF
            # Map score S to severity P: P = max(0, min(1, 0.5 - S * 2.5))
            severity = max(0, min(1, 0.5 - row['anomaly_score'] * 2.5))
            all_anomalies.append({'index': original_idx, 'column': 'Multiple (IF)', 'value': 'N/A',
                                  'method': 'Isolation Forest', 'score': severity, 'type': 'Multivariate'})
            anomalous_rows_indices.add(original_idx)

    # Sort all anomalies by severity score (descending)
    all_anomalies.sort(key=lambda x: x['score'], reverse=True)

    if not all_anomalies:
        anomaly_summary_prompt += "No significant anomalies detected by statistical or Isolation Forest methods.\n"
    else:
         anomaly_summary_prompt += f"Found potential anomalies in {len(anomalous_rows_indices)} unique rows. Top {max_anom} individual anomalies (by severity):\n"
         anomaly_summary_prompt += "Index | Column | Method | Value | Severity Score\n"
         anomaly_summary_prompt += "------|--------|--------|-------|--------------\n"
         for anom in all_anomalies[:max_anom]:
             # Format value for display
             val_str = f"{anom['value']:.2f}" if isinstance(anom['value'], (int, float)) else str(anom['value'])
             val_str = (val_str[:30] + '...') if len(val_str) > 30 else val_str # Truncate long values
             anomaly_summary_prompt += f"{anom['index']} | {anom['column']} | {anom['method']} | {val_str} | {anom['score']:.3f}\n"
         if len(all_anomalies) > max_anom:
             anomaly_summary_prompt += f"... (and {len(all_anomalies) - max_anom} more anomaly instances)\n"

    # 3. Prepare Data Snippets for Anomalous Rows
    anomalous_rows_data_prompt = "Data for Top Anomalous Rows (showing up to {} rows):\n".format(max_anom)
    # Get unique indices from the top anomalies to show data context
    top_anomalous_indices = sorted(list(set(a['index'] for a in all_anomalies[:max_anom])))
    anomalous_rows_data_prompt += format_data_for_llm(df_new, top_anomalous_indices, max_rows=max_anom)

    # 4. Construct the Final Prompt
    prompt = f"""
You are an expert AI data analyst. You have analyzed a 'new' dataset against a 'historic' baseline dataset.
Historic dataset shape: {df_historic.shape}
New dataset shape: {df_new.shape}
Numerical columns compared: {common_num_cols}
Categorical columns compared: {common_cat_cols}

Here is a summary of the detected data drift and anomalies:

--- Data Drift Summary ---
{drift_summary_prompt}
--- Anomaly Summary ---
{anomaly_summary_prompt}
--- Anomalous Row Data Snippet ---
{anomalous_rows_data_prompt}

--- Your Task ---
Based ONLY on the information provided above, please provide:

1.  **Overall Summary:** A brief (2-3 sentence) summary of the main data quality issues found (drift and anomalies).
2.  **Drift Analysis:**
    *   For the most significant drifted columns (up to 3):
        *   Explain the potential cause or implication of the drift (e.g., "Shift in distribution of 'Age' might indicate a change in user demographics"). Refer to the test results.
        *   Suggest a potential business impact if applicable.
3.  **Anomaly Analysis:**
    *   For the top anomalies (considering severity and row index):
        *   Explain *why* specific rows/values are flagged as anomalous based on the method and score/context provided (e.g., "Row 123 has 'Income' of 999999, which is an IQR outlier compared to the historic range..."). Refer to the data snippet.
        *   Highlight the specific column and value causing the anomaly within the row data provided.
        *   Suggest potential causes (e.g., data entry error, sensor malfunction, genuinely new pattern).
4.  **Prioritized List of Issues (Top 3-5):**
    *   List the most critical issues (can be drift or anomalies).
    *   For each issue, provide:
        *   **Issue:** Brief description (e.g., "Significant drift in 'TransactionAmount'", "High severity outliers in 'SensorReading' at indices [10, 25]").
        *   **Reasoning:** Why it's prioritized (e.g., "High drift score indicates major distribution change", "Multiple high-severity outliers suggest systemic issue", "Potential high impact on downstream models").
        *   **Recommendation:** A brief, actionable recommendation (e.g., "Investigate source of change in 'TransactionAmount'", "Validate outlier 'SensorReading' values", "Review data entry process for 'ProductCategory'").

**Output Format:** Please provide the response as a JSON object with the following keys: "summary", "drift_analysis" (string), "anomaly_analysis" (string), "prioritized_issues" (list of objects, each with "issue", "reasoning", "recommendation"). Ensure the analysis ONLY uses the provided data summaries and snippets. Do not invent external context.
"""

    # 5. Call Gemini API
    print("\n--- Calling Gemini API for analysis and prioritization ---")
    # print("Prompt snippet:\n", prompt[:1000] + "\n...") # Uncomment for debugging prompt
    llm_response_text = "LLM analysis failed." # Default message
    try:
        response = gemini_model.generate_content(prompt)
        # Clean the response: remove backticks, markdown code blocks etc.
        cleaned_response = response.text.strip()
        # Basic cleaning: remove markdown fences if Gemini uses them for JSON
        cleaned_response = re.sub(r'^```json\s*', '', cleaned_response, flags=re.IGNORECASE)
        cleaned_response = re.sub(r'\s*```$', '', cleaned_response)
        cleaned_response = cleaned_response.replace('\\n', '\n') # Ensure newlines are real
        print("--- Gemini Response Received ---")
        # print(cleaned_response) # Uncomment for debugging raw response

        # Attempt to parse the JSON response
        llm_result = json.loads(cleaned_response)

        # Validate structure (basic check)
        if not all(k in llm_result for k in ["summary", "drift_analysis", "anomaly_analysis", "prioritized_issues"]):
             raise ValueError("LLM response missing required keys.")
        if not isinstance(llm_result["prioritized_issues"], list):
             raise ValueError("'prioritized_issues' should be a list.")

        print("--- Gemini Analysis Parsed Successfully ---")
        return llm_result

    except json.JSONDecodeError as e:
        print(f"Error: Could not decode LLM response as JSON: {e}")
        print("Raw LLM response:", cleaned_response)
        llm_response_text = f"LLM analysis failed: Could not parse response. Raw response was:\n{cleaned_response}"
    except genai.types.generation_types.StopCandidateException as e:
         print(f"Error: Gemini generation stopped: {e}")
         llm_response_text = f"LLM analysis failed: Generation stopped prematurely ({e}). This might be due to safety settings or prompt issues."
    except Exception as e:
        print(f"Error calling or parsing Gemini API: {e}")
        llm_response_text = f"LLM analysis failed: {e}"

    # Fallback if parsing fails or Gemini wasn't used
    return {
        "summary": llm_response_text if not USE_GEMINI else "LLM analysis failed.",
        "drift_analysis": "Could not generate drift analysis.",
        "anomaly_analysis": "Could not generate anomaly analysis.",
        "prioritized_issues": [] # Return empty list for prioritization
    }


# --- Reporting Function ---

def generate_report_html(drift_results, stat_anomalies_by_row, if_anomalies_df, llm_analysis, df_new):
    """Generates an HTML report summarizing findings and LLM insights."""

    # --- LLM Insights Section ---
    report = "<h2>AI Analysis Summary</h2>"
    report += f"<p><strong>Overall Summary:</strong> {llm_analysis.get('summary', 'Not available.')}</p>"

    report += "<details><summary><strong>Detailed Drift Analysis (AI Insights)</strong></summary>"
    report += f"<pre>{llm_analysis.get('drift_analysis', 'Not available.')}</pre></details>"

    report += "<details><summary><strong>Detailed Anomaly Analysis (AI Insights)</strong></summary>"
    report += f"<pre>{llm_analysis.get('anomaly_analysis', 'Not available.')}</pre></details>"

    report += "<h3>Prioritized Issues (AI Recommended)</h3>"
    prioritized = llm_analysis.get('prioritized_issues', [])
    if prioritized:
        report += "<table class='result-table'><thead><tr><th>Issue</th><th>Reasoning</th><th>Recommendation</th></tr></thead><tbody>"
        for issue in prioritized:
            report += f"<tr><td>{issue.get('issue', 'N/A')}</td><td>{issue.get('reasoning', 'N/A')}</td><td>{issue.get('recommendation', 'N/A')}</td></tr>"
        report += "</tbody></table>"
    else:
        report += "<p>No prioritized issues provided by the AI or analysis failed.</p>"

    report += "<hr>"

    # --- Data Drift Details Section ---
    report += "<h2>Data Drift Details</h2>"
    significant_drift = [r for r in drift_results if r and r.get('drift_detected')]
    if significant_drift:
        report += f"<p>Found significant drift (p < {ALPHA}) in {len(significant_drift)} columns:</p>"
        report += "<table class='result-table'><thead><tr><th>Column</th><th>Type</th><th>Test</th><th>Statistic</th><th>P-value</th><th>Drift Score (1-P)</th><th>Confidence</th></tr></thead><tbody>"
        # Sort by drift score descending
        significant_drift.sort(key=lambda x: x.get('drift_score', 0), reverse=True)
        for result in significant_drift:
            report += f"<tr class='highlight-drift'><td>{result['column']}</td><td>{result['type']}</td><td>{result['test']}</td><td>{result['statistic']:.4f}</td><td>{result['p_value']:.4e}</td><td>{result['drift_score']:.4f}</td><td>{result['confidence']:.4f}</td></tr>"
        report += "</tbody></table>"
    else:
        report += "<p>No significant data drift detected between the historic and new datasets based on the tests performed.</p>"

    report += "<hr>"

    # --- Anomaly Details Section ---
    report += "<h2>Anomaly Detection Details</h2>"

    # Combine anomalies for reporting
    all_anomalies_report = []
    anomalous_row_indices_report = set()

    # Add statistical anomalies
    for idx, anomalies in stat_anomalies_by_row.items():
        for anom in anomalies:
             all_anomalies_report.append({'index': idx, 'column': anom['column'], 'value': anom['value'],
                                          'method': anom['method'], 'score': anom['severity_score'],
                                          'context': anom.get('historic_context', 'N/A'), 'type': 'Statistical'})
             anomalous_row_indices_report.add(idx)

    # Add Isolation Forest anomalies
    if not if_anomalies_df.empty:
        if_anom_rows = if_anomalies_df[if_anomalies_df['is_anomaly']].sort_values('anomaly_score')
        for _, row in if_anom_rows.iterrows():
            original_idx = row['index_in_new']
            severity = max(0, min(1, 0.5 - row['anomaly_score'] * 2.5)) # Remap score to severity
            all_anomalies_report.append({'index': original_idx, 'column': 'Multiple', 'value': 'N/A',
                                         'method': 'Isolation Forest', 'score': severity,
                                         'context': f"IF Score: {row['anomaly_score']:.3f}", 'type': 'Multivariate'})
            anomalous_row_indices_report.add(original_idx)

    # Sort by severity
    all_anomalies_report.sort(key=lambda x: x['score'], reverse=True)

    if all_anomalies_report:
        report += f"<p>Found {len(all_anomalies_report)} potential anomaly instances across {len(anomalous_row_indices_report)} unique rows.</p>"

        # Table of top anomalies
        report += "<h3>Top Anomaly Instances (by Severity Score)</h3>"
        report += "<table class='result-table'><thead><tr><th>Row Index</th><th>Column(s)</th><th>Value</th><th>Method</th><th>Severity Score</th><th>Context/Details</th></tr></thead><tbody>"
        for anom in all_anomalies_report[:20]: # Show top 20 instances
             val_str = f"{anom['value']:.2f}" if isinstance(anom['value'], (int, float)) else str(anom['value'])
             val_str = (val_str[:50] + '...') if len(val_str) > 50 else val_str # Truncate long values
             highlight_class = 'highlight-anomaly-high' if anom['score'] > 0.7 else 'highlight-anomaly-med'
             report += f"<tr class='{highlight_class}'><td>{anom['index']}</td><td>{anom['column']}</td><td>{val_str}</td><td>{anom['method']} ({anom['type']})</td><td>{anom['score']:.3f}</td><td>{anom['context']}</td></tr>"
        report += "</tbody></table>"
        if len(all_anomalies_report) > 20:
             report += f"<p>... and {len(all_anomalies_report) - 20} more anomaly instances detected.</p>"

        # Section for full anomalous rows (collapsible)
        report += f"<details><summary><strong>Show Data for Top {min(len(anomalous_row_indices_report), MAX_ANOMALIES_TO_LLM)} Anomalous Rows</strong></summary>"
        top_anomalous_indices_report = sorted(list(anomalous_row_indices_report))[:MAX_ANOMALIES_TO_LLM]

        if top_anomalous_indices_report:
            # Filter df_new safely for existing indices
            valid_indices = [idx for idx in top_anomalous_indices_report if idx in df_new.index]
            if valid_indices:
                 report += "<div class='preview-table-container' style='max-height: 400px;'>" # Style directly for simplicity here
                 report += df_new.loc[valid_indices].to_html(classes='preview-table', index=True, border=0)
                 report += "</div>"
            else:
                 report += "<p>Could not retrieve data for the specified anomalous row indices.</p>"
        else:
             report += "<p>No specific anomalous row indices to display data for.</p>"
        report += "</details>"

    else:
        report += "<p>No significant anomalies detected by the configured methods.</p>"

    report += "<hr><p><em>End of Report</em></p>"
    return report

# --- Main Analysis Function ---
def run_analysis(historic_filepath, new_filepath):
    """Orchestrates the data loading, analysis, LLM interaction, and reporting."""
    try:
        # 1. Load and Prepare Data
        df_historic, df_new = load_data(historic_filepath, new_filepath)
        if df_historic.empty or df_new.empty:
            # load_data should raise error, but double check
            raise ValueError("Data loading resulted in empty common dataframes.")

        print("\n--- Handling Missing Values ---")
        # Use simpler fill for analysis consistency, report what was done if needed
        df_historic_filled, hist_missing_info = handle_missing_values(df_historic.copy())
        df_new_filled, new_missing_info = handle_missing_values(df_new.copy()) # Keep original df_new for reporting

        print("\n--- Identifying Column Types ---")
        # Use original DFs for type identification before filling potentially changes things
        num_cols, cat_cols, text_cols = identify_column_types(df_new) # Assume new defines the types primarily

        # Ensure columns exist in both filled dataframes after potential harmonization/filling
        common_cols = df_historic_filled.columns.intersection(df_new_filled.columns).tolist()
        common_num_cols = [col for col in num_cols if col in common_cols]
        common_cat_cols = [col for col in cat_cols if col in common_cols]

        # 2. Perform Drift Detection
        print("\n--- Detecting Data Drift ---")
        drift_results = []
        for col in common_num_cols:
            result = detect_numerical_drift(df_historic_filled[col], df_new_filled[col], col)
            if result: drift_results.append(result)
        for col in common_cat_cols:
             # Use original series for Chi2 if possible to capture original categories before filling
             # But ensure fillna('') or similar if NaNs are present and important
             hist_series_cat = df_historic[col].astype(str).fillna('__NaN__')
             new_series_cat = df_new[col].astype(str).fillna('__NaN__')
             result = detect_categorical_drift(hist_series_cat, new_series_cat, col)
             if result: drift_results.append(result)

        # 3. Perform Anomaly Detection
        print("\n--- Detecting Anomalies ---")
        # Use filled data for IF training/prediction, but use original DFs for stats where appropriate
        if_anomalies_df = detect_anomalies_isolation_forest(df_historic_filled, df_new_filled, common_num_cols)
        stat_anomalies_by_row = detect_statistical_anomalies(df_new, df_historic, common_num_cols) # Use original DFs here

        # 4. Get LLM Analysis (if enabled)
        llm_analysis = get_llm_analysis(drift_results, stat_anomalies_by_row, if_anomalies_df,
                                        df_new, df_historic, common_num_cols, common_cat_cols)


        # 5. Generate Report
        print("\n--- Generating Report ---")
        report_html = generate_report_html(drift_results, stat_anomalies_by_row, if_anomalies_df, llm_analysis, df_new)

        # 6. Get Data Previews
        historic_preview, new_preview = get_preview_tables(df_historic, df_new) # Use original loaded data

        return report_html, historic_preview, new_preview # Removed plot data

    except (ValueError, RuntimeError, FileNotFoundError) as e:
        print(f"Analysis Error: {e}")
        error_html = f"<h2>Analysis Failed</h2><hr><p><strong>Error:</strong> {e}</p><p>Please check the input files and ensure they are valid CSVs with common columns.</p>"
        # Return empty previews on error
        return error_html, '', ''
    except Exception as e:
        import traceback
        print(f"Unexpected Analysis Error: {e}")
        print(traceback.format_exc()) # Print stack trace for debugging unexpected errors
        error_html = f"<h2>Analysis Failed</h2><hr><p><strong>An unexpected error occurred:</strong> {e}</p><p>Please check the logs or contact support.</p>"
        return error_html, '', ''