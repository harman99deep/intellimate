# analysis_engine.py

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency, iqr as scipy_iqr # Use scipy's iqr for robustness
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
ALPHA = 0.05
IF_CONTAMINATION = 'auto'
Z_SCORE_THRESHOLD = 3
IQR_MULTIPLIER = 1.5
MAX_ANOMALIES_TO_LLM = 15
MAX_DRIFT_COLS_TO_LLM = 10
ID_COLUMN_SUBSTRINGS = ['id', 'key', 'uuid', 'identifier'] # Case-insensitive check
ID_UNIQUENESS_THRESHOLD = 0.9 # If >90% unique, likely an ID

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# --- Helper Functions ---

# load_data, identify_column_types, handle_missing_values remain largely the same
# (assuming they are working as per previous versions)
# ... (Keep the existing load_data, identify_column_types, handle_missing_values functions here) ...
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

        original_hist_cols = df_historic.columns.tolist()
        original_new_cols = df_new.columns.tolist()
        common_cols = [col for col in original_hist_cols if col in original_new_cols]
        extra_hist_cols = [col for col in original_hist_cols if col not in common_cols]
        extra_new_cols = [col for col in original_new_cols if col not in common_cols]

        if extra_hist_cols:
            print(f"Warning: Columns in historic but not new (will be ignored for comparison): {extra_hist_cols}")
        if extra_new_cols:
            print(f"Warning: Columns in new but not historic (will be ignored for comparison): {extra_new_cols}")

        if not common_cols:
             raise ValueError("No common columns found between the historic and new datasets.")

        df_historic_common = df_historic[common_cols].copy()
        df_new_common = df_new[common_cols].copy()

        harmonized_common_cols = []
        for col in common_cols:
             dtype_hist = df_historic_common[col].dtype
             dtype_new = df_new_common[col].dtype
             if dtype_hist != dtype_new:
                 print(f"Attempting type harmonization for column '{col}' (Historic: {dtype_hist}, New: {dtype_new}).")
                 try:
                     is_hist_numeric_like = pd.api.types.is_numeric_dtype(dtype_hist) or df_historic_common[col].apply(lambda x: isinstance(x, (int, float))).all()
                     is_new_numeric_like = pd.api.types.is_numeric_dtype(dtype_new) or df_new_common[col].apply(lambda x: isinstance(x, (int, float))).all()
                     if is_hist_numeric_like or is_new_numeric_like:
                         df_historic_common[col] = pd.to_numeric(df_historic_common[col], errors='coerce')
                         df_new_common[col] = pd.to_numeric(df_new_common[col], errors='coerce')
                         if df_historic_common[col].isnull().all() or df_new_common[col].isnull().all():
                              print(f"Warning: Column '{col}' became all NaN after numeric conversion. Check data.")
                         print(f"  Harmonized '{col}' to numeric.")
                         harmonized_common_cols.append(col)
                     else:
                         df_historic_common[col] = df_historic_common[col].astype(str)
                         df_new_common[col] = df_new_common[col].astype(str)
                         print(f"  Harmonized '{col}' to string.")
                         harmonized_common_cols.append(col)
                 except Exception as e:
                     print(f"Warning: Could not harmonize dtype for column '{col}': {e}. Skipping this column for analysis.")
             else:
                harmonized_common_cols.append(col)

        if not harmonized_common_cols:
            raise ValueError("No common columns remaining after type harmonization.")

        return df_historic_common[harmonized_common_cols], df_new_common[harmonized_common_cols]

    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        raise ValueError(f"File not found: {e.filename}")
    except pd.errors.EmptyDataError as e:
        print(f"Error: CSV file is empty: {e}")
        raise ValueError(f"CSV file seems empty. Please check the input files.")
    except ValueError as e:
         print(f"Error reading or processing CSV: {e}")
         raise ValueError(f"Error processing CSV: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        raise RuntimeError(f"An unexpected error occurred: {e}")

def identify_column_types(df):
    """Identifies numerical, categorical, and potential text columns."""
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    text_cols = []
    potential_text = list(categorical_cols)
    for col in potential_text:
        try: # Add try-except for robustness on diverse data
            unique_ratio = df[col].nunique(dropna=False) / len(df[col]) if len(df[col]) > 0 else 0
            # Check if string methods are applicable before calling them
            if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == 'object':
                avg_len = df[col].astype(str).str.len().mean()
            else:
                 avg_len = 0 # Not string-like, don't calculate avg length

            # Refined heuristic for text detection
            is_likely_text = (unique_ratio > 0.5 and avg_len > 15) or \
                             (unique_ratio > 0.2 and avg_len > 30) or \
                             (avg_len > 100) # Very long strings are likely text

            if is_likely_text:
                text_cols.append(col)
                if col in categorical_cols:
                    categorical_cols.remove(col)
        except TypeError as e:
             print(f"Warning: Could not analyze column '{col}' for text type due to data type issues: {e}. Treating as standard categorical.")
        except Exception as e:
            print(f"Warning: Unexpected error analyzing column '{col}' for text type: {e}. Treating as standard categorical.")


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
            try:
                if pd.api.types.is_numeric_dtype(df_filled[col]):
                    if numerical_strategy == 'median':
                        fill_value = df_filled[col].median()
                        strategy_used = 'median'
                    elif numerical_strategy == 'mean':
                        fill_value = df_filled[col].mean()
                        strategy_used = 'mean'
                    else:
                        fill_value = df_filled[col].median()
                        strategy_used = 'median'
                    if pd.isnull(fill_value): # Handle case where median/mean is NaN (e.g., all NaNs)
                         fill_value = 0
                         strategy_used += ' (fallback to 0)'
                    df_filled[col].fillna(fill_value, inplace=True)
                    print(f"Filled {missing_count} missing values in numerical column '{col}' with {strategy_used} ({fill_value:.2f})")
                elif pd.api.types.is_object_dtype(df_filled[col]) or pd.api.types.is_categorical_dtype(df_filled[col]):
                    if categorical_strategy == 'mode':
                        mode_val = df_filled[col].mode()
                        if not mode_val.empty:
                            fill_value = mode_val[0]
                            strategy_used = 'mode'
                        else:
                            fill_value = 'Missing'
                            strategy_used = 'constant ("Missing")'
                    else:
                        fill_value = 'Missing'
                        strategy_used = 'constant ("Missing")'
                    df_filled[col].fillna(fill_value, inplace=True)
                    print(f"Filled {missing_count} missing values in categorical/text column '{col}' with {strategy_used} ('{fill_value}')")
                else:
                    print(f"Warning: Column '{col}' has an unsupported type for missing value handling.")
                    strategy_used = 'skipped'

                missing_info[col] = {'count': missing_count, 'percentage': missing_pct, 'fill_value': fill_value, 'strategy': strategy_used}
            except TypeError as e:
                 print(f"Warning: Could not fill missing values in column '{col}' due to data type issues: {e}")
                 missing_info[col] = {'count': missing_count, 'percentage': missing_pct, 'fill_value': None, 'strategy': 'error'}
            except Exception as e:
                 print(f"Warning: Unexpected error filling missing values in column '{col}': {e}")
                 missing_info[col] = {'count': missing_count, 'percentage': missing_pct, 'fill_value': None, 'strategy': 'error'}

    return df_filled, missing_info

def get_preview_tables(df_historic, df_new):
    """Return HTML tables for the first 10 rows of both datasets."""
    try:
        historic_html = df_historic.head(10).to_html(classes='preview-table', index=True, border=0)
        new_html = df_new.head(10).to_html(classes='preview-table', index=True, border=0)
        return historic_html, new_html
    except Exception as e:
        print(f"Error generating preview tables: {e}")
        return "<p>Error generating historic data preview.</p>", "<p>Error generating new data preview.</p>"

def is_potential_id_column(col_name, series):
    """Heuristic check if a column is likely an ID."""
    col_name_lower = col_name.lower()
    if not isinstance(series, pd.Series): # Basic type check
        return False

    # Check 1: Name contains ID-like substrings
    if any(sub in col_name_lower for sub in ID_COLUMN_SUBSTRINGS):
        return True

    # Check 2: High uniqueness (avoid calculating on huge non-string columns if possible)
    # Only apply uniqueness check if it's likely string/object or has many values
    if series.dtype == 'object' or pd.api.types.is_string_dtype(series.dtype) or len(series) > 1000:
        try:
            # dropna=False is important here, NaNs don't contribute to ID-ness
            unique_ratio = series.nunique(dropna=False) / len(series) if len(series) > 0 else 0
            if unique_ratio >= ID_UNIQUENESS_THRESHOLD:
                print(f"Column '{col_name}' flagged as potential ID based on high uniqueness ({unique_ratio:.2f})")
                return True
        except Exception as e:
             print(f"Warning: Could not calculate uniqueness for column '{col_name}': {e}")

    return False

# --- Numerical Summary ---
def calculate_numerical_summaries(df_historic, df_new, numerical_cols):
    """Calculates Min, Max, IQR for numerical columns in both dataframes."""
    summaries = {}
    for col in numerical_cols:
        hist_col = df_historic[col].dropna()
        new_col = df_new[col].dropna()
        summary = {}
        try:
            if not hist_col.empty:
                summary['hist_min'] = hist_col.min()
                summary['hist_max'] = hist_col.max()
                summary['hist_iqr'] = scipy_iqr(hist_col)
            else:
                 summary['hist_min'] = np.nan
                 summary['hist_max'] = np.nan
                 summary['hist_iqr'] = np.nan

            if not new_col.empty:
                summary['new_min'] = new_col.min()
                summary['new_max'] = new_col.max()
                summary['new_iqr'] = scipy_iqr(new_col)
            else:
                 summary['new_min'] = np.nan
                 summary['new_max'] = np.nan
                 summary['new_iqr'] = np.nan

            summaries[col] = summary
        except Exception as e:
            print(f"Warning: Could not calculate summary statistics for column '{col}': {e}")
            summaries[col] = {'hist_min': 'Error', 'hist_max': 'Error', 'hist_iqr': 'Error',
                              'new_min': 'Error', 'new_max': 'Error', 'new_iqr': 'Error'}
    return summaries


# --- Drift Detection Functions ---
# Keep the core logic but maybe add a simplified result field later if needed
def detect_numerical_drift(series_historic, series_new, column_name, alpha=ALPHA):
    """Performs KS test for numerical drift. Returns technical details."""
    series_historic_na = series_historic.dropna()
    series_new_na = series_new.dropna()
    if len(series_historic_na) < 2 or len(series_new_na) < 2:
        return None
    try:
        ks_stat, p_value = ks_2samp(series_historic_na, series_new_na)
        p_value = max(p_value, 1e-100)
        drift_score = 1 - p_value
        drift_detected = p_value < alpha
        confidence = 1 - p_value
        # Add simplified result interpretation
        result_interpretation = "Significant change detected" if drift_detected else "No significant change detected"
        return {
            "column": column_name, "type": "Numerical", "test": "Kolmogorov-Smirnov",
            "statistic": ks_stat, "p_value": p_value, "drift_score": drift_score,
            "drift_detected": drift_detected, "confidence": confidence,
            "simple_result": result_interpretation, "simple_test_name": "Distribution Shape Test"
        }
    except Exception as e:
        print(f"Error during KS test for '{column_name}': {e}")
        return None

def detect_categorical_drift(series_historic, series_new, column_name, alpha=ALPHA):
    """Performs Chi-squared test for categorical drift. Returns technical details."""
    series_historic = series_historic.astype(str).fillna('__NaN__')
    series_new = series_new.astype(str).fillna('__NaN__')
    all_categories = sorted(list(set(series_historic.unique()) | set(series_new.unique())))
    if len(all_categories) < 2: return None
    hist_counts = series_historic.value_counts().reindex(all_categories, fill_value=0)
    new_counts = series_new.value_counts().reindex(all_categories, fill_value=0)
    contingency_table = pd.DataFrame({'historic': hist_counts, 'new': new_counts})
    contingency_table = contingency_table.loc[contingency_table.sum(axis=1) > 0, contingency_table.sum(axis=0) > 0]
    if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2: return None
    try:
        chi2_stat, p_value, _, expected_freq = chi2_contingency(contingency_table)
        if (expected_freq < 5).any().any():
            print(f"Warning: Low expected frequency (<5) in Chi2 test for '{column_name}'. P-value may be less reliable.")
        p_value = max(p_value, 1e-100)
        drift_score = 1 - p_value
        drift_detected = p_value < alpha
        confidence = 1 - p_value
        result_interpretation = "Significant change detected" if drift_detected else "No significant change detected"
        return {
            "column": column_name, "type": "Categorical", "test": "Chi-squared",
            "statistic": chi2_stat, "p_value": p_value, "drift_score": drift_score,
            "drift_detected": drift_detected, "confidence": confidence,
             "simple_result": result_interpretation, "simple_test_name": "Category Distribution Test"
        }
    except ValueError as e:
        print(f"Error during Chi2 test calculation for '{column_name}': {e}. Contingency table:\n{contingency_table}")
        return None
    except Exception as e:
         print(f"Unexpected error during Chi2 test for '{column_name}': {e}")
         return None

# --- Anomaly Detection Functions ---
# Keep core logic, but add simplified explanations for methods
def detect_anomalies_isolation_forest(df_historic, df_new, numerical_cols, contamination=IF_CONTAMINATION, random_state=42):
    """Detect anomalies using Isolation Forest. Returns scores and flags."""
    if not numerical_cols:
        return pd.DataFrame(columns=['index_in_new', 'anomaly_score', 'is_anomaly', 'severity_score'])
    df_hist_num = df_historic[numerical_cols].copy()
    df_new_num = df_new[numerical_cols].copy()
    df_hist_filled, _ = handle_missing_values(df_hist_num, numerical_strategy='median')
    df_new_filled, _ = handle_missing_values(df_new_num, numerical_strategy='median')
    scaler = StandardScaler()
    try:
        if df_hist_filled.empty or df_hist_filled.var().sum() == 0:
             print("Warning: Historic numerical data empty or has zero variance. Skipping Isolation Forest.")
             return pd.DataFrame(columns=['index_in_new', 'anomaly_score', 'is_anomaly', 'severity_score'])
        df_hist_scaled = scaler.fit_transform(df_hist_filled)
        if df_new_filled.empty:
             print("Warning: New numerical data empty. Cannot predict anomalies with Isolation Forest.")
             return pd.DataFrame(columns=['index_in_new', 'anomaly_score', 'is_anomaly', 'severity_score'])
        df_new_scaled = scaler.transform(df_new_filled)
    except ValueError as e:
         print(f"Error during scaling for Isolation Forest: {e}")
         return pd.DataFrame(columns=['index_in_new', 'anomaly_score', 'is_anomaly', 'severity_score'])
    try:
        model = IsolationForest(contamination=contamination, random_state=random_state, n_estimators=100)
        model.fit(df_hist_scaled)
        anomaly_scores = model.decision_function(df_new_scaled) # Lower score -> more anomalous
        predictions = model.predict(df_new_scaled) # -1 for anomalies
        # Map score to severity (0 to 1, higher is more severe)
        # Adjusted scale based on typical IF score range (~ -0.2 to 0.2)
        severity_scores = np.clip(0.5 - anomaly_scores * 2.0, 0, 1) # Simple linear mapping

        anomaly_df = pd.DataFrame({
            'index_in_new': df_new.index,
            'anomaly_score': anomaly_scores,
            'is_anomaly': predictions == -1,
            'severity_score': severity_scores # Store severity
        })
        print(f"Isolation Forest flagged {anomaly_df['is_anomaly'].sum()} potential anomalies based on combined features.")
        return anomaly_df
    except Exception as e:
        print(f"Error during Isolation Forest execution: {e}")
        return pd.DataFrame(columns=['index_in_new', 'anomaly_score', 'is_anomaly', 'severity_score'])

def detect_statistical_anomalies(df_new, df_historic, numerical_cols, numerical_summaries, z_threshold=Z_SCORE_THRESHOLD, iqr_multiplier=IQR_MULTIPLIER):
    """Detect univariate anomalies (Nulls, Z-score, IQR) based on historic stats and summaries."""
    anomalies_by_row = defaultdict(list)

    for col in numerical_cols:
        # Get pre-calculated summaries for context
        summary = numerical_summaries.get(col, {})
        hist_mean = df_historic[col].mean() # Recalculate mean/std here if needed, or pass them in summaries
        hist_std = df_historic[col].std()
        hist_q1 = df_historic[col].quantile(0.25) # Use actual quantiles for bounds
        hist_q3 = df_historic[col].quantile(0.75)
        hist_iqr_val = hist_q3 - hist_q1 if pd.notnull(hist_q1) and pd.notnull(hist_q3) else 0

        # Define bounds based on actual historic quantiles/stats
        iqr_lower_bound = hist_q1 - iqr_multiplier * hist_iqr_val if pd.notnull(hist_q1) else -np.inf
        iqr_upper_bound = hist_q3 + iqr_multiplier * hist_iqr_val if pd.notnull(hist_q3) else np.inf
        z_lower_bound = hist_mean - z_threshold * hist_std if pd.notnull(hist_std) and hist_std > 1e-6 else hist_mean
        z_upper_bound = hist_mean + z_threshold * hist_std if pd.notnull(hist_std) and hist_std > 1e-6 else hist_mean

        # Iterate through new data points
        for idx, value in df_new[col].items():
            anomaly_found = False
            # 1. Unexpected Nulls
            if pd.isnull(value):
                historic_null_ratio = df_historic[col].isnull().mean()
                if historic_null_ratio < 0.01:
                    anomalies_by_row[idx].append({
                        'column': col, 'value': None, 'method': 'Unexpected Missing Value',
                        'severity_score': 0.8, # High severity for unexpected nulls
                        'historic_context': f"Historically <1% missing",
                        'simple_method': 'Unexpected Missing'
                    })
                    anomaly_found = True
            # 2. Numeric Outliers (IQR/Z-score)
            elif pd.api.types.is_numeric_dtype(value):
                is_iqr_outlier = False
                # IQR check
                if value < iqr_lower_bound or value > iqr_upper_bound:
                    distance = min(abs(value - iqr_lower_bound), abs(value - iqr_upper_bound))
                    severity = 1.0 - np.exp(-0.1 * abs(distance / (hist_iqr_val + 1e-6)))
                    anomalies_by_row[idx].append({
                        'column': col, 'value': value, 'method': 'IQR Outlier',
                        'severity_score': max(0.5, severity),
                        'historic_context': f"Typical Range (IQR based): ({iqr_lower_bound:.2f}, {iqr_upper_bound:.2f})",
                        'simple_method': 'Outside Typical Range'
                    })
                    anomaly_found = True
                    is_iqr_outlier = True

                # Z-score check (only if std dev is meaningful)
                if pd.notnull(hist_std) and hist_std > 1e-6:
                    z = (value - hist_mean) / hist_std
                    if abs(z) > z_threshold:
                        severity = 1.0 - np.exp(-0.1 * (abs(z) - z_threshold))
                        # Add if not already flagged by IQR, or if Z-score is much more extreme
                        if not is_iqr_outlier or abs(z) > z_threshold * 1.5:
                             anomalies_by_row[idx].append({
                                 'column': col, 'value': value, 'method': 'Z-score Outlier', 'z_score': z,
                                 'severity_score': max(0.5, severity) ,
                                 'historic_context': f"Far from Historic Avg ({hist_mean:.2f}, StdDev: {hist_std:.2f})",
                                 'simple_method': 'Far From Average'
                             })
                             anomaly_found = True

    print(f"Statistical methods identified potential anomalies in {len(anomalies_by_row)} rows.")
    return dict(anomalies_by_row)


# --- LLM Integration ---

def format_data_for_llm(df, indices, max_rows=10):
    """Formats specified rows of a DataFrame into a markdown string for the LLM prompt."""
    if not indices or df.empty:
        return "No relevant data rows to display."
    indices_to_show = sorted(list(indices))[:max_rows]
    valid_indices = [idx for idx in indices_to_show if idx in df.index]
    if not valid_indices:
        return "Specified indices not found in the dataframe."
    try:
        # Use tabulate for better markdown formatting if available
        return df.loc[valid_indices].to_markdown(index=True)
    except ImportError:
        # Fallback to basic string formatting if tabulate is not installed
        return df.loc[valid_indices].to_string()


def get_llm_analysis(drift_results, stat_anomalies_by_row, if_anomalies_df, df_new, df_historic, common_num_cols, common_cat_cols, numerical_summaries, alpha=ALPHA, max_drift=MAX_DRIFT_COLS_TO_LLM, max_anom=MAX_ANOMALIES_TO_LLM):
    """Generates a comprehensive prompt for Gemini using simplified language concepts, sends it, and parses the response."""
    if not USE_GEMINI:
        # Provide a basic prioritization based on scores if LLM is off
        basic_issues = []
        # Add drift issues
        drift_issues = [r for r in drift_results if r.get('drift_detected', False)]
        drift_issues.sort(key=lambda x: x.get('drift_score', 0), reverse=True)
        for item in drift_issues[:3]: # Top 3 drift
            basic_issues.append({
                'issue': f"Significant change in '{item['column']}' ({item['type']})",
                'reasoning': f"High change severity score ({item['drift_score']:.3f}) indicates distribution shift.",
                'recommendation': f"Investigate cause of change in '{item['column']}'."
            })
        # Add anomaly issues (simplified)
        all_anomalies_basic = []
        for idx, anomalies in stat_anomalies_by_row.items():
            for anom in anomalies: all_anomalies_basic.append({'index': idx, 'col': anom['column'], 'score': anom['severity_score'], 'type': anom.get('simple_method', 'Statistical')})
        if not if_anomalies_df.empty:
             if_anom_rows = if_anomalies_df[if_anomalies_df['is_anomaly']]
             for idx, row in if_anom_rows.iterrows(): all_anomalies_basic.append({'index': row['index_in_new'], 'col': 'Multiple', 'score': row['severity_score'], 'type': 'Combined Features'})
        all_anomalies_basic.sort(key=lambda x: x['score'], reverse=True)
        anomaly_rows_summary = defaultdict(float)
        for anom in all_anomalies_basic: anomaly_rows_summary[anom['index']] = max(anomaly_rows_summary[anom['index']], anom['score']) # Max severity per row
        sorted_anom_rows = sorted(anomaly_rows_summary.items(), key=lambda item: item[1], reverse=True)
        for idx, score in sorted_anom_rows[:3]: # Top 3 anomaly rows
             basic_issues.append({
                 'issue': f"Unusual data found in row {idx}",
                 'reasoning': f"High anomaly severity score ({score:.3f}). Check specific columns flagged for this row.",
                 'recommendation': f"Validate data for row {idx}, investigate root cause (e.g., entry error, system issue)."
             })
        # Sort combined basic issues approximately by score/severity
        basic_issues.sort(key=lambda x: float(re.search(r'\((\d\.\d+)\)', x['reasoning']).group(1)) if re.search(r'\((\d\.\d+)\)', x['reasoning']) else 0, reverse=True)

        return {
            "summary": "LLM analysis skipped (Gemini not configured or unavailable). Basic prioritization provided.",
            "prioritized_issues": basic_issues[:5], # Return top 5 basic issues
            "drift_analysis": "LLM analysis skipped.",
            "anomaly_analysis": "LLM analysis skipped."
        }

    # --- Prepare LLM Prompt ---
    # 1. Drift Summary (Simplified Terms)
    significant_drift = [r for r in drift_results if r.get('drift_detected', False)]
    significant_drift.sort(key=lambda x: x.get('drift_score', 0), reverse=True)
    drift_summary_prompt = f"Detected Data Distribution Changes (where change is statistically significant, based on alpha < {alpha}):\n"
    if not significant_drift:
        drift_summary_prompt += "No significant distribution changes detected.\n"
    else:
        drift_summary_prompt += "Column | Type | Change Result | Change Severity (0-1) | Confidence (0-1)\n"
        drift_summary_prompt += "-------|------|---------------|-----------------------|-----------------\n"
        for item in significant_drift[:max_drift]:
            drift_summary_prompt += f"{item['column']} | {item['type']} | {item['simple_result']} | {item['drift_score']:.3f} | {item['confidence']:.3f}\n"
        if len(significant_drift) > max_drift:
             drift_summary_prompt += f"... (and {len(significant_drift) - max_drift} more changed columns)\n"

    # 2. Numerical Summaries for Context
    summary_prompt = "Numerical Data Summary (Historic vs New):\n"
    summary_prompt += "Column | Historic Range (Min-Max) | New Range (Min-Max) | Historic Typical Spread (IQR) | New Typical Spread (IQR)\n"
    summary_prompt += "-------|--------------------------|---------------------|-------------------------------|------------------------\n"
    for col, stats in numerical_summaries.items():
        hist_range = f"{stats.get('hist_min', 'N/A'):.2f}-{stats.get('hist_max', 'N/A'):.2f}" if pd.notna(stats.get('hist_min')) else "N/A"
        new_range = f"{stats.get('new_min', 'N/A'):.2f}-{stats.get('new_max', 'N/A'):.2f}" if pd.notna(stats.get('new_min')) else "N/A"
        hist_iqr = f"{stats.get('hist_iqr', 'N/A'):.2f}" if pd.notna(stats.get('hist_iqr')) else "N/A"
        new_iqr = f"{stats.get('new_iqr', 'N/A'):.2f}" if pd.notna(stats.get('new_iqr')) else "N/A"
        summary_prompt += f"{col} | {hist_range} | {new_range} | {hist_iqr} | {new_iqr}\n"

    # 3. Anomaly Summary (Simplified Terms)
    anomaly_summary_prompt = "Detected Unusual Data Points (Anomalies) in New Dataset:\n"
    all_anomalies = []
    anomalous_rows_indices = set()
    # Combine statistical and IF anomalies
    for idx, anomalies in stat_anomalies_by_row.items():
        for anom in anomalies:
             all_anomalies.append({'index': idx, 'column': anom['column'], 'value': anom['value'],
                                   'method': anom.get('simple_method', 'Statistical'), 'score': anom['severity_score'], 'type': 'Single Column'})
             anomalous_rows_indices.add(idx)
    if not if_anomalies_df.empty:
        if_anom_rows = if_anomalies_df[if_anomalies_df['is_anomaly']].sort_values('severity_score', ascending=False) # Sort by severity
        for _, row in if_anom_rows.iterrows():
            original_idx = row['index_in_new']
            all_anomalies.append({'index': original_idx, 'column': 'Multiple', 'value': 'N/A',
                                  'method': 'Combined Features Check', 'score': row['severity_score'], 'type': 'Multiple Columns'})
            anomalous_rows_indices.add(original_idx)

    all_anomalies.sort(key=lambda x: x['score'], reverse=True)

    if not all_anomalies:
        anomaly_summary_prompt += "No significant unusual data points detected.\n"
    else:
         anomaly_summary_prompt += f"Found potential unusual data in {len(anomalous_rows_indices)} unique rows. Top {max_anom} instances (by severity):\n"
         anomaly_summary_prompt += "Row Index | Column(s) | Reason Flagged | Value | Anomaly Severity (0-1)\n"
         anomaly_summary_prompt += "----------|-----------|----------------|-------|----------------------\n"
         for anom in all_anomalies[:max_anom]:
             val_str = f"{anom['value']:.2f}" if isinstance(anom['value'], (int, float)) else str(anom['value'])
             val_str = (val_str[:30] + '...') if len(val_str) > 30 else val_str
             anomaly_summary_prompt += f"{anom['index']} | {anom['column']} | {anom['method']} | {val_str} | {anom['score']:.3f}\n"
         if len(all_anomalies) > max_anom:
             anomaly_summary_prompt += f"... (and {len(all_anomalies) - max_anom} more instances)\n"

    # 4. Data Snippets for Anomalous Rows
    anomalous_rows_data_prompt = "Data for Top Anomalous Rows (showing up to {} rows):\n".format(max_anom)
    top_anomalous_indices = sorted(list(set(a['index'] for a in all_anomalies[:max_anom])))
    anomalous_rows_data_prompt += format_data_for_llm(df_new, top_anomalous_indices, max_rows=max_anom)

    # 5. Construct the Final Prompt
    prompt = f"""
You are an expert AI data analyst communicating findings to a business user. You have analyzed a 'new' dataset against a 'historic' baseline. Focus on clear, simple language and actionable insights. Avoid technical jargon like p-value, KS test, Chi-squared, Z-score, IQR, Isolation Forest unless explaining the concept simply.

--- Analysis Context ---
Historic dataset shape: {df_historic.shape}
New dataset shape: {df_new.shape}
Columns compared (excluding potential IDs): Numerical: {common_num_cols}, Categorical: {common_cat_cols}

--- Numerical Data Summary ---
{summary_prompt}

--- Data Distribution Changes Summary ---
{drift_summary_prompt}

--- Unusual Data Points (Anomalies) Summary ---
{anomaly_summary_prompt}

--- Anomalous Row Data Snippet ---
{anomalous_rows_data_prompt}

--- Your Task ---
Based ONLY on the information provided above, please provide the following in **simple, business-friendly language**:

1.  **Overall Summary:** A brief (2-3 sentence) summary of the main data quality issues found (distribution changes and unusual data). Highlight the most impactful findings.
2.  **Distribution Change Analysis:**
    *   For the columns with the most significant changes (highest 'Change Severity', up to 3):
        *   Explain *what* changed in simple terms (e.g., "The average 'Age' seems higher now", "The variety of 'Product Types' has shifted", "The range of 'Transaction Amount' is wider/narrower than before"). Use the Numerical Data Summary for context if applicable.
        *   Suggest a potential business impact (e.g., "This might affect sales forecasts", "Customer demographics may be changing").
3.  **Unusual Data Analysis:**
    *   For the most severe unusual data points (top anomalies):
        *   Explain *why* the data is flagged as unusual in simple terms (e.g., "Row 123 has an 'Income' much higher than the typical range seen historically", "Row 456 has a combination of values that is rare compared to past data", "Row 789 is missing a value in 'Region' where it was almost always present before"). Refer to the 'Reason Flagged' and the data snippet.
        *   Suggest potential causes (e.g., data entry mistake, system error, genuinely new customer behavior, expected missing data).
4.  **Prioritized List of Issues (Top 3-5):**
    *   List the most critical issues (can be distribution changes or unusual data).
    *   For each issue:
        *   **Issue:** Brief description in simple terms (e.g., "Major shift in 'TransactionAmount' distribution", "Several rows with extremely high 'SensorReading' values", "Increase in missing 'Region' data").
        *   **Reasoning:** Why it's important (e.g., "Could impact financial reporting accuracy", "Suggests potential sensor problems needing checks", "May break downstream processes relying on 'Region'").
        *   **Recommendation:** A brief, actionable next step (e.g., "Investigate source system for 'TransactionAmount' changes", "Ask engineers to check sensors for rows [list indices]", "Review why 'Region' data might be missing more often").

**Output Format:** Provide the response as a JSON object with keys: "summary", "drift_analysis" (string), "anomaly_analysis" (string), "prioritized_issues" (list of objects, each with "issue", "reasoning", "recommendation"). Ensure all explanations use **simple, non-technical language**.
"""

    # 6. Call Gemini API
    print("\n--- Calling Gemini API for simplified analysis and prioritization ---")
    llm_response_text = "LLM analysis failed."
    try:
        response = gemini_model.generate_content(prompt)
        cleaned_response = response.text.strip()
        cleaned_response = re.sub(r'^```json\s*', '', cleaned_response, flags=re.IGNORECASE)
        cleaned_response = re.sub(r'\s*```$', '', cleaned_response)
        cleaned_response = cleaned_response.replace('\\n', '\n')
        print("--- Gemini Response Received ---")
        llm_result = json.loads(cleaned_response)
        if not all(k in llm_result for k in ["summary", "drift_analysis", "anomaly_analysis", "prioritized_issues"]):
             raise ValueError("LLM response missing required keys.")
        if not isinstance(llm_result["prioritized_issues"], list):
             raise ValueError("'prioritized_issues' should be a list.")
        print("--- Gemini Analysis Parsed Successfully ---")
        return llm_result
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode LLM response as JSON: {e}")
        llm_response_text = f"LLM analysis failed: Could not parse response. Raw response was:\n{cleaned_response}"
    except genai.types.generation_types.StopCandidateException as e:
         print(f"Error: Gemini generation stopped: {e}")
         llm_response_text = f"LLM analysis failed: Generation stopped ({e}). Check safety settings or prompt complexity."
    except Exception as e:
        print(f"Error calling or parsing Gemini API: {e}")
        llm_response_text = f"LLM analysis failed: {e}"

    # Fallback if parsing fails or Gemini wasn't used (reuse basic prioritization from start of function)
    fallback_issues = [] # Recalculate basic issues here if needed, or retrieve from scope if possible
    # (Simplified logic for fallback - copy/adapt from the !USE_GEMINI block if needed)
    return {
        "summary": llm_response_text,
        "drift_analysis": "Could not generate drift analysis.",
        "anomaly_analysis": "Could not generate anomaly analysis.",
        "prioritized_issues": [] # Return empty list or basic issues if calculated
    }


# --- Reporting Function ---

def generate_report_html(drift_results, stat_anomalies_by_row, if_anomalies_df, llm_analysis, df_new, numerical_summaries, excluded_id_cols):
    """Generates a user-friendly HTML report summarizing findings and LLM insights."""

    # --- LLM Insights Section ---
    report = "<h2>AI Analysis Summary (Powered by Gemini)</h2>"
    report += f"<p><strong>Overall Summary:</strong> {llm_analysis.get('summary', 'Not available.')}</p>"
    report += "<details><summary><strong>Detailed Distribution Change Analysis (AI Insights)</strong></summary>"
    report += f"<pre>{llm_analysis.get('drift_analysis', 'Not available.')}</pre></details>"
    report += "<details><summary><strong>Detailed Unusual Data Analysis (AI Insights)</strong></summary>"
    report += f"<pre>{llm_analysis.get('anomaly_analysis', 'Not available.')}</pre></details>"
    report += "<h3>Prioritized Issues (AI Recommended)</h3>"
    prioritized = llm_analysis.get('prioritized_issues', [])
    if prioritized:
        report += "<table class='result-table'><thead><tr><th>Issue</th><th>Why it Matters</th><th>Recommendation</th></tr></thead><tbody>"
        for issue in prioritized:
            report += f"<tr><td>{issue.get('issue', 'N/A')}</td><td>{issue.get('reasoning', 'N/A')}</td><td>{issue.get('recommendation', 'N/A')}</td></tr>"
        report += "</tbody></table>"
    else:
        report += "<p>No prioritized issues provided by the AI or analysis failed.</p>"
    report += "<hr>"

    # --- Numerical Summary Section ---
    report += "<h2>Numerical Data Summary</h2>"
    if numerical_summaries:
        report += "<p>Comparison of value ranges and typical spread (IQR) for numerical columns.</p>"
        report += "<table class='result-table'><thead><tr><th>Column</th><th>Historic Range (Min-Max)</th><th>New Range (Min-Max)</th><th>Historic Typical Spread (IQR)</th><th>New Typical Spread (IQR)</th></tr></thead><tbody>"
        for col, stats in numerical_summaries.items():
            hist_min = stats.get('hist_min', np.nan)
            hist_max = stats.get('hist_max', np.nan)
            new_min = stats.get('new_min', np.nan)
            new_max = stats.get('new_max', np.nan)
            hist_iqr = stats.get('hist_iqr', np.nan)
            new_iqr = stats.get('new_iqr', np.nan)

            hist_range_str = f"{hist_min:.2f} - {hist_max:.2f}" if pd.notna(hist_min) else "N/A"
            new_range_str = f"{new_min:.2f} - {new_max:.2f}" if pd.notna(new_min) else "N/A"
            hist_iqr_str = f"{hist_iqr:.2f}" if pd.notna(hist_iqr) else "N/A"
            new_iqr_str = f"{new_iqr:.2f}" if pd.notna(new_iqr) else "N/A"

            # Basic highlighting if range or IQR changed notably (e.g., by > 20%) - can be refined
            range_changed = False
            if pd.notna(hist_min) and pd.notna(new_min) and not np.isclose(hist_min, new_min, rtol=0.2) or \
               pd.notna(hist_max) and pd.notna(new_max) and not np.isclose(hist_max, new_max, rtol=0.2):
                range_changed = True
            iqr_changed = False
            if pd.notna(hist_iqr) and pd.notna(new_iqr) and hist_iqr > 1e-6 and not np.isclose(hist_iqr, new_iqr, rtol=0.2):
                 iqr_changed = True

            row_class = "highlight-drift" if range_changed or iqr_changed else ""

            report += f"<tr class='{row_class}'><td>{col}</td><td>{hist_range_str}</td><td>{new_range_str}</td><td>{hist_iqr_str}</td><td>{new_iqr_str}</td></tr>"
        report += "</tbody></table>"
    else:
        report += "<p>No numerical columns found or summaries could not be calculated.</p>"
    report += "<hr>"


    # --- Data Drift Details Section (Simplified) ---
    report += "<h2>Data Distribution Change Details</h2>"
    significant_drift = [r for r in drift_results if r and r.get('drift_detected')]
    if excluded_id_cols:
         report += f"<p><i>Note: Potential ID columns excluded from this analysis: {', '.join(excluded_id_cols)}.</i></p>"

    if significant_drift:
        report += f"<p>Found columns where the data distribution has likely changed significantly compared to the historic data:</p>"
        report += "<table class='result-table'><thead><tr><th>Column</th><th>Data Type</th><th>Change Check Method</th><th>Result</th><th>Change Severity (0-1)</th><th>Confidence (0-1)</th></tr></thead><tbody>"
        significant_drift.sort(key=lambda x: x.get('drift_score', 0), reverse=True)
        for result in significant_drift:
            report += f"<tr class='highlight-drift'><td>{result['column']}</td><td>{result['type']}</td><td>{result.get('simple_test_name','N/A')}</td><td>{result.get('simple_result','N/A')}</td><td>{result['drift_score']:.3f}</td><td>{result['confidence']:.3f}</td></tr>"
        report += "</tbody></table>"
    else:
        report += "<p>No significant data distribution changes detected based on the tests performed.</p>"
    report += "<hr>"

    # --- Anomaly Details Section (Simplified) ---
    report += "<h2>Unusual Data Point Details</h2>"
    all_anomalies_report = []
    anomalous_row_indices_report = set()
    # Combine and simplify anomalies
    for idx, anomalies in stat_anomalies_by_row.items():
        for anom in anomalies:
             all_anomalies_report.append({'index': idx, 'column': anom['column'], 'value': anom['value'],
                                          'method': anom.get('simple_method', 'Statistical'), 'score': anom['severity_score'],
                                          'context': anom.get('historic_context', 'N/A'), 'type': 'Single Column'})
             anomalous_row_indices_report.add(idx)
    if not if_anomalies_df.empty:
        if_anom_rows = if_anomalies_df[if_anomalies_df['is_anomaly']]
        for _, row in if_anom_rows.iterrows():
            original_idx = row['index_in_new']
            all_anomalies_report.append({'index': original_idx, 'column': 'Multiple', 'value': 'N/A',
                                         'method': 'Combined Features Check', 'score': row['severity_score'],
                                         'context': f"Score reflects unusual combination", 'type': 'Multiple Columns'})
            anomalous_row_indices_report.add(original_idx)

    all_anomalies_report.sort(key=lambda x: x['score'], reverse=True)

    if all_anomalies_report:
        report += f"<p>Found {len(all_anomalies_report)} potential unusual data instances across {len(anomalous_row_indices_report)} unique rows.</p>"
        report += "<h3>Top Unusual Data Instances (by Severity Score)</h3>"
        report += "<table class='result-table'><thead><tr><th>Row Index</th><th>Column(s)</th><th>Value</th><th>Reason Flagged</th><th>Severity (0-1)</th><th>Historic Context/Details</th></tr></thead><tbody>"
        for anom in all_anomalies_report[:20]:
             val_str = f"{anom['value']:.2f}" if isinstance(anom['value'], (int, float)) else str(anom['value'])
             val_str = (val_str[:50] + '...') if len(val_str) > 50 else val_str
             highlight_class = 'highlight-anomaly-high' if anom['score'] > 0.7 else 'highlight-anomaly-med'
             report += f"<tr class='{highlight_class}'><td>{anom['index']}</td><td>{anom['column']}</td><td>{val_str}</td><td>{anom['method']} ({anom['type']})</td><td>{anom['score']:.3f}</td><td>{anom['context']}</td></tr>"
        report += "</tbody></table>"
        if len(all_anomalies_report) > 20:
             report += f"<p>... and {len(all_anomalies_report) - 20} more instances detected.</p>"
        report += f"<details><summary><strong>Show Data for Top {min(len(anomalous_row_indices_report), MAX_ANOMALIES_TO_LLM)} Rows with Unusual Data</strong></summary>"
        top_anomalous_indices_report = sorted(list(anomalous_row_indices_report))[:MAX_ANOMALIES_TO_LLM]
        if top_anomalous_indices_report:
            valid_indices = [idx for idx in top_anomalous_indices_report if idx in df_new.index]
            if valid_indices:
                 report += "<div class='preview-table-container' style='max-height: 400px;'>"
                 # Ensure generated HTML is safe
                 try:
                     anomalous_data_html = df_new.loc[valid_indices].to_html(classes='preview-table', index=True, border=0)
                     report += anomalous_data_html
                 except Exception as table_error:
                      report += f"<p>Error generating table for anomalous rows: {table_error}</p>"
                 report += "</div>"
            else:
                 report += "<p>Could not retrieve data for the specified anomalous row indices.</p>"
        else:
             report += "<p>No specific anomalous row indices to display data for.</p>"
        report += "</details>"
    else:
        report += "<p>No significant unusual data points detected by the configured methods.</p>"
    report += "<hr><p><em>End of Report</em></p>"
    return report


# --- Main Analysis Function ---
def run_analysis(historic_filepath, new_filepath):
    """Orchestrates the data loading, analysis, LLM interaction, and simplified reporting."""
    excluded_id_cols = [] # Keep track of excluded ID columns
    try:
        # 1. Load and Prepare Data
        df_historic, df_new = load_data(historic_filepath, new_filepath)
        print("\n--- Handling Missing Values ---")
        df_historic_filled, _ = handle_missing_values(df_historic.copy())
        df_new_filled, _ = handle_missing_values(df_new.copy())
        print("\n--- Identifying Column Types ---")
        num_cols, cat_cols, text_cols = identify_column_types(df_new)
        common_cols = df_historic_filled.columns.intersection(df_new_filled.columns).tolist()
        common_num_cols = [col for col in num_cols if col in common_cols]
        common_cat_cols = [col for col in cat_cols if col in common_cols]

        # --- Exclude Potential ID Columns from Drift Analysis ---
        print("\n--- Checking for Potential ID Columns (for Drift Exclusion) ---")
        cols_to_check_drift = common_num_cols + common_cat_cols
        drift_num_cols = []
        drift_cat_cols = []
        for col in common_num_cols:
             if not is_potential_id_column(col, df_new[col]):
                 drift_num_cols.append(col)
             else:
                 excluded_id_cols.append(col)
                 print(f"Excluding potential ID column '{col}' from drift analysis.")
        for col in common_cat_cols:
            # Check original DF 'new' for uniqueness as filling might change it
            if not is_potential_id_column(col, df_new[col]):
                 drift_cat_cols.append(col)
            else:
                 if col not in excluded_id_cols: # Avoid duplicates
                     excluded_id_cols.append(col)
                     print(f"Excluding potential ID column '{col}' from drift analysis.")

        # 2. Calculate Numerical Summaries (Range, IQR)
        print("\n--- Calculating Numerical Summaries ---")
        # Use original data before filling for more accurate Min/Max/IQR
        numerical_summaries = calculate_numerical_summaries(df_historic, df_new, common_num_cols) # Use all common num cols for summary

        # 3. Perform Drift Detection (on non-ID columns)
        print("\n--- Detecting Data Drift (Excluding IDs) ---")
        drift_results = []
        for col in drift_num_cols: # Use filtered list
            result = detect_numerical_drift(df_historic_filled[col], df_new_filled[col], col)
            if result: drift_results.append(result)
        for col in drift_cat_cols: # Use filtered list
             hist_series_cat = df_historic[col].astype(str).fillna('__NaN__')
             new_series_cat = df_new[col].astype(str).fillna('__NaN__')
             result = detect_categorical_drift(hist_series_cat, new_series_cat, col)
             if result: drift_results.append(result)

        # 4. Perform Anomaly Detection (on all relevant columns)
        print("\n--- Detecting Anomalies ---")
        if_anomalies_df = detect_anomalies_isolation_forest(df_historic_filled, df_new_filled, common_num_cols) # Use all num cols
        # Pass summaries for context to statistical anomalies
        stat_anomalies_by_row = detect_statistical_anomalies(df_new, df_historic, common_num_cols, numerical_summaries)

        # 5. Get LLM Analysis (if enabled)
        llm_analysis = get_llm_analysis(drift_results, stat_anomalies_by_row, if_anomalies_df,
                                        df_new, df_historic, drift_num_cols, drift_cat_cols, # Pass drift cols context
                                        numerical_summaries, alpha=ALPHA) # Pass summaries

        # 6. Generate Simplified Report
        print("\n--- Generating Simplified Report ---")
        report_html = generate_report_html(drift_results, stat_anomalies_by_row, if_anomalies_df, llm_analysis, df_new, numerical_summaries, excluded_id_cols)

        # 7. Get Data Previews
        historic_preview, new_preview = get_preview_tables(df_historic, df_new)

        return report_html, historic_preview, new_preview

    except (ValueError, RuntimeError, FileNotFoundError) as e:
        print(f"Analysis Error: {e}")
        error_html = f"<h2>Analysis Failed</h2><hr><p><strong>Error:</strong> {e}</p><p>Please check the input files and ensure they are valid CSVs with common columns.</p>"
        return error_html, '', ''
    except Exception as e:
        import traceback
        print(f"Unexpected Analysis Error: {e}")
        print(traceback.format_exc())
        error_html = f"<h2>Analysis Failed</h2><hr><p><strong>An unexpected error occurred:</strong> {e}</p><p>Please check the logs or contact support.</p>"
        return error_html, '', ''