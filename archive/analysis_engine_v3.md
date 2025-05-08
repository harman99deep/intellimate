# analysis_engine.py

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency, iqr as scipy_iqr, ttest_1samp, chisquare
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import google.generativeai as genai
import os
import warnings
from dotenv import load_dotenv
import json
import re
from collections import defaultdict
import psycopg2 # For PostgreSQL

# --- Configuration ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# DB Config (from .env file)
DB_HOST = os.getenv("DB_HOST") # Removed defaults, should be set or handled
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
HISTORIC_TABLE_NAME = os.getenv("HISTORIC_TABLE_NAME")

# Configure Google Gemini
USE_GEMINI = False # Default to False
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash') # Or your preferred model
        print("Gemini AI configured successfully.")
        USE_GEMINI = True
    except Exception as e:
        print(f"Error configuring Gemini AI: {e}. Explanations will be basic.")
else:
    print("Gemini API Key not found. Explanations will be basic.")

# Constants
ALPHA = 0.05
IF_CONTAMINATION = 'auto'
Z_SCORE_THRESHOLD = 3
IQR_MULTIPLIER = 1.5
MAX_ANOMALIES_TO_LLM = 10 # Reduced for potentially smaller prompts
MAX_DRIFT_COLS_TO_LLM = 7
ID_COLUMN_SUBSTRINGS = ['id', 'key', 'uuid', 'identifier', 'no', 'code'] # Added more common ID parts
ID_UNIQUENESS_THRESHOLD = 0.95 # Stricter for ID

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning) # e.g. from scipy for small samples

# --- Database Helper ---
def get_db_connection():
    if not all([DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, HISTORIC_TABLE_NAME]): # Added HISTORIC_TABLE_NAME check here
        print("Database environment variables (HOST, PORT, NAME, USER, PASSWORD, TABLE_NAME) not fully set. DB operations will be skipped.")
        return None
    try:
        conn = psycopg2.connect(
            host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD
        )
        print("Successfully connected to PostgreSQL database.")
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to PostgreSQL database: {e}")
        return None

def fetch_historic_stats_from_db(db_conn, table_name, columns_info):
    if not db_conn or not table_name:
        print("DB connection or table name missing for fetching historic stats.")
        return {}

    historic_stats = {}
    try:
        with db_conn.cursor() as cursor: # Use 'with' for automatic closing
            for col_name, col_type in columns_info.items():
                try:
                    # Basic quoting for column names (assumes column names don't contain quotes themselves)
                    safe_col_name = f'"{col_name.replace("\"", "\"\"")}"'

                    if col_type == 'numerical':
                        query = f"""
                        SELECT
                            AVG(CAST({safe_col_name} AS NUMERIC)),
                            STDDEV(CAST({safe_col_name} AS NUMERIC)),
                            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY CAST({safe_col_name} AS NUMERIC)),
                            PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY CAST({safe_col_name} AS NUMERIC)),
                            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY CAST({safe_col_name} AS NUMERIC)),
                            MIN(CAST({safe_col_name} AS NUMERIC)),
                            MAX(CAST({safe_col_name} AS NUMERIC)),
                            SUM(CASE WHEN {safe_col_name} IS NULL THEN 1 ELSE 0 END), -- Null count
                            COUNT(*) -- Total count
                        FROM "{table_name}";
                        """
                        cursor.execute(query)
                        stats_row = cursor.fetchone()
                        if stats_row:
                            q1_val = stats_row[2]
                            q3_val = stats_row[4]
                            historic_stats[col_name] = {
                                'mean': stats_row[0], 'std': stats_row[1],
                                'q1': q1_val, 'median': stats_row[3], 'q3': q3_val,
                                'min': stats_row[5], 'max': stats_row[6],
                                'iqr': (q3_val - q1_val) if q1_val is not None and q3_val is not None else None,
                                'null_count': stats_row[7], 'total_count': stats_row[8]
                            }
                    elif col_type == 'categorical':
                        query = f"SELECT CAST({safe_col_name} AS TEXT), COUNT(*) FROM \"{table_name}\" GROUP BY CAST({safe_col_name} AS TEXT);"
                        cursor.execute(query)
                        value_counts_list = cursor.fetchall()
                        historic_stats[col_name] = {'value_counts': dict(value_counts_list)}
                    # print(f"Fetched historic stats for column '{col_name}'.")
                except psycopg2.Error as e:
                    print(f"DB Error fetching stats for column '{col_name}': {e}")
                    db_conn.rollback()
                    historic_stats[col_name] = {'error': f"DB Error: {e}"}
                except Exception as e:
                    print(f"Unexpected error fetching stats for column '{col_name}': {e}")
                    historic_stats[col_name] = {'error': f"Unexpected Error: {e}"}
    except psycopg2.Error as e: # Catch errors related to cursor or connection during loop
        print(f"DB Error during historic stats fetching process: {e}")
        return {'db_process_error': str(e)} # Indicate a broader DB issue
    # cursor is closed by 'with' statement

    return historic_stats

# --- Helper Functions ---
def load_new_data(new_path):
    try:
        if not new_path or not new_path.lower().endswith('.csv'): # Check if path is None or empty
             raise ValueError("Invalid file type or path for new data. Please upload a CSV file.")
        df_new = pd.read_csv(new_path)
        if df_new.empty: raise ValueError("The new dataset CSV file is empty.")
        print(f"Loaded new data: {df_new.shape} from {new_path}")
        return df_new
    except FileNotFoundError: raise ValueError(f"New data file not found: {new_path}")
    except pd.errors.EmptyDataError: raise ValueError("New data CSV file is empty.")
    except Exception as e: raise RuntimeError(f"Unexpected error during new data loading: {e}")

def identify_column_types(df):
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    text_cols = []
    temp_categorical = list(categorical_cols) # Iterate over a copy
    for col in temp_categorical:
        try:
            if len(df[col].dropna()) == 0: continue # Skip all-NaN columns for text checks
            unique_ratio = df[col].nunique(dropna=False) / len(df[col]) if len(df[col]) > 0 else 0
            avg_len = 0
            if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == 'object':
                avg_len = df[col].astype(str).str.len().mean()

            if (unique_ratio > 0.6 and avg_len > 20) or \
               (unique_ratio > 0.3 and avg_len > 40) or \
               (avg_len > 100):
                text_cols.append(col)
                if col in categorical_cols: categorical_cols.remove(col)
        except Exception as e:
            print(f"Warning: Could not fully assess column '{col}' for text type: {e}. Retaining as categorical.")
    print(f"Identified Column Types in New Data: Num: {numerical_cols}, Cat: {categorical_cols}, Text: {text_cols}")
    return numerical_cols, categorical_cols, text_cols

def handle_missing_values(df, numerical_strategy='median', categorical_strategy='mode'):
    df_filled = df.copy()
    missing_info = {} # This isn't used much later but can be for debugging
    for col in df_filled.columns:
        if df_filled[col].isnull().any():
            fill_value = None
            if pd.api.types.is_numeric_dtype(df_filled[col]):
                if numerical_strategy == 'median': fill_value = df_filled[col].median()
                else: fill_value = df_filled[col].mean()
                if pd.isnull(fill_value): fill_value = 0 # Fallback for all-NaN columns
                df_filled[col].fillna(fill_value, inplace=True)
            elif pd.api.types.is_object_dtype(df_filled[col]) or pd.api.types.is_categorical_dtype(df_filled[col]):
                modes = df_filled[col].mode()
                fill_value = modes[0] if not modes.empty else 'Missing_Value_Filled'
                df_filled[col].fillna(fill_value, inplace=True)
            missing_info[col] = fill_value # Store what was used
    return df_filled, missing_info

def get_preview_table_html(df, title="Data Preview"):
    if df is None or df.empty: return f"<p>{title}: No data to display.</p>"
    try: return df.head(10).to_html(classes='preview-table', index=True, border=0, na_rep='NA')
    except Exception as e: return f"<p>Error generating preview for {title}: {e}</p>"

def is_potential_id_column(col_name, series):
    if not isinstance(series, pd.Series): return False
    col_name_lower = str(col_name).lower() # Ensure col_name is string
    if any(sub in col_name_lower for sub in ID_COLUMN_SUBSTRINGS): return True
    try:
        if len(series) == 0: return False # Empty series is not an ID
        # For numeric IDs that might not be 'object' type initially
        is_mostly_int = False
        if pd.api.types.is_numeric_dtype(series.dtype):
            if series.dropna().apply(lambda x: x == int(x)).all(): # Check if all non-NaN are integers
                 is_mostly_int = True

        if series.dtype == 'object' or pd.api.types.is_string_dtype(series.dtype) or is_mostly_int :
            unique_ratio = series.nunique(dropna=False) / len(series)
            if unique_ratio >= ID_UNIQUENESS_THRESHOLD:
                # print(f"Column '{col_name}' flagged as potential ID (uniqueness: {unique_ratio:.2f})")
                return True
    except Exception as e:
        print(f"Warning: Could not assess ID-ness for column '{col_name}': {e}")
    return False

def calculate_new_numerical_summaries(df_new, numerical_cols):
    summaries = {}
    for col in numerical_cols:
        new_col = df_new[col].dropna()
        summary = {k: np.nan for k in ['new_min', 'new_max', 'new_iqr', 'new_mean', 'new_std', 'new_q1', 'new_median', 'new_q3', 'new_null_count', 'new_total_count']}
        try:
            if not new_col.empty:
                summary['new_min'] = new_col.min()
                summary['new_max'] = new_col.max()
                summary['new_iqr'] = scipy_iqr(new_col) if len(new_col) > 0 else np.nan
                summary['new_mean'] = new_col.mean()
                summary['new_std'] = new_col.std()
                summary['new_q1'] = new_col.quantile(0.25)
                summary['new_median'] = new_col.median()
                summary['new_q3'] = new_col.quantile(0.75)
            summary['new_null_count'] = df_new[col].isnull().sum()
            summary['new_total_count'] = len(df_new[col])
            summaries[col] = summary
        except Exception as e:
            print(f"Warning: Could not calculate summary for new data column '{col}': {e}")
            summaries[col] = {'error': str(e)}
    return summaries

# --- Drift Detection Functions (Adapted) ---
def detect_numerical_drift_vs_stats(series_new, historic_col_stats, column_name, alpha=ALPHA):
    default_no_drift = {"column": column_name, "type": "Numerical", "test": "Mean Comparison (vs Historic Stats)",
                        "drift_detected": False, "simple_result": "Comparison N/A", "drift_score": 0, "confidence": 0,
                        "p_value": 1.0, "statistic": None, "simple_test_name": "Average Value Check"}

    if not historic_col_stats or 'mean' not in historic_col_stats or 'std' not in historic_col_stats \
            or pd.isnull(historic_col_stats['mean']) or pd.isnull(historic_col_stats['std']):
        default_no_drift["simple_result"] = "Historic stats unavailable/incomplete"
        return default_no_drift

    series_new_na = series_new.dropna()
    if len(series_new_na) < 5:
        default_no_drift["simple_result"] = "Insufficient new data points"
        return default_no_drift

    hist_mean = historic_col_stats['mean']
    hist_std = historic_col_stats['std']
    p_value, drift_detected, simple_result, statistic = 1.0, False, "Mean is consistent with historic", None

    if hist_std == 0: # Historic data had no variance
        new_mean = series_new_na.mean()
        if not np.isclose(new_mean, hist_mean):
            drift_detected = True
            p_value = 0.0
            simple_result = "Mean changed (historic data had no variance)"
        else:
            simple_result = "Mean consistent (historic data had no variance)"
    else:
        try:
            statistic, p_value = ttest_1samp(series_new_na, hist_mean, nan_policy='omit')
            drift_detected = p_value < alpha
            simple_result = "Significant change in mean" if drift_detected else "Mean is consistent with historic"
        except Exception as e:
            print(f"Warning: t-test failed for {column_name}: {e}. Defaulting to no drift.")
            simple_result = f"Mean comparison test failed ({e})"

    return {"column": column_name, "type": "Numerical", "test": "Mean Comparison (vs Historic Stats)",
            "statistic": statistic, "p_value": p_value, "drift_score": 1 - p_value,
            "drift_detected": drift_detected, "confidence": 1 - p_value,
            "simple_result": simple_result, "simple_test_name": "Average Value Check"}

def detect_categorical_drift_vs_stats(series_new, historic_col_stats, column_name, alpha=ALPHA):
    default_no_drift = {"column": column_name, "type": "Categorical", "test": "Category Proportion Test",
                        "drift_detected": False, "simple_result": "Comparison N/A", "drift_score": 0, "confidence": 0,
                        "p_value": 1.0, "statistic": None, "simple_test_name": "Category Popularity Check"}

    if not historic_col_stats or 'value_counts' not in historic_col_stats or not isinstance(historic_col_stats['value_counts'], dict):
        default_no_drift["simple_result"] = "Historic category counts unavailable/invalid"
        return default_no_drift

    series_new_str = series_new.astype(str).fillna('__NaN__') # Ensure string type for new series
    new_counts_pd = series_new_str.value_counts()
    hist_counts_dict = historic_col_stats['value_counts']
    hist_counts_pd = pd.Series(hist_counts_dict).astype(float) # Ensure float for division

    if new_counts_pd.empty and hist_counts_pd.empty:
        default_no_drift["simple_result"] = "No data in new or historic counts"
        return default_no_drift
    if new_counts_pd.empty:
        default_no_drift["simple_result"] = "No data in new counts to compare"
        return default_no_drift
    if hist_counts_pd.empty:
        # If historic is empty but new is not, it's a definite drift (new categories appeared)
        return {**default_no_drift, "drift_detected": True, "simple_result": "New categories appeared (historic was empty)",
                "drift_score": 1.0, "confidence": 1.0, "p_value": 0.0}


    all_categories = sorted(list(set(new_counts_pd.index) | set(hist_counts_pd.index)))
    new_total = new_counts_pd.sum()
    hist_total = hist_counts_pd.sum()

    if new_total == 0:
        default_no_drift["simple_result"] = "No data in new counts"
        return default_no_drift
    if hist_total == 0: # Should have been caught by hist_counts_pd.empty but as safeguard
        return {**default_no_drift, "drift_detected": True, "simple_result": "Historic counts were zero, new data exists",
                "drift_score": 1.0, "confidence": 1.0, "p_value": 0.0}


    observed_freq = new_counts_pd.reindex(all_categories, fill_value=0).astype(float)
    expected_props = hist_counts_pd.reindex(all_categories, fill_value=0) / hist_total
    expected_freq = expected_props * new_total

    # Filter out categories where expected frequency is effectively zero for chi-square
    # and ensure observed and expected are aligned on the same categories
    valid_mask = (expected_freq > 1e-9) # Use a small epsilon instead of == 0 for float comparisons
    observed_final = observed_freq[valid_mask]
    expected_final = expected_freq[valid_mask]

    if len(observed_final) < 1: # Need at least one category with expected counts
        default_no_drift["simple_result"] = "No comparable categories with expected historic counts"
        return default_no_drift
    if len(observed_final) == 1 and len(all_categories) > 1: # Only one category has expected counts, others are new
        # This implies significant drift if other categories exist in new data but not expected
        if observed_final.sum() < new_total: # New categories exist
            return {**default_no_drift, "drift_detected": True, "simple_result": "New categories/major shift in proportions",
                    "drift_score": 1.0, "confidence": 1.0, "p_value": 0.0}


    # Ensure no zero expected frequencies are passed if they still exist after filtering (should not happen with mask)
    expected_final = np.maximum(expected_final, 1e-9) # Prevent zero division in chisquare if any slip through

    try:
        chi2_stat, p_value = chisquare(f_obs=observed_final, f_exp=expected_final)
        drift_detected = p_value < alpha
        result_interpretation = "Significant change in category proportions" if drift_detected else "Category proportions consistent with historic"
        if np.any(expected_final < 5):
            print(f"Warning: Low expected frequency (<5) in Chi2 test for '{column_name}'. P-value may be less reliable.")
        return {"column": column_name, "type": "Categorical", "test": "Category Proportion Test",
                "statistic": chi2_stat, "p_value": p_value, "drift_score": 1 - p_value,
                "drift_detected": drift_detected, "confidence": 1 - p_value,
                "simple_result": result_interpretation, "simple_test_name": "Category Popularity Check"}
    except ValueError as e: # e.g. if sum of freqs is different or other issues
        print(f"Error during Chi2 G.O.F test calculation for '{column_name}': {e}.")
        default_no_drift["simple_result"] = f"Category proportion test failed ({e})"
        return default_no_drift
    except Exception as e:
         print(f"Unexpected error during Chi2 G.O.F test for '{column_name}': {e}")
         default_no_drift["simple_result"] = f"Unexpected error in proportion test ({e})"
         return default_no_drift

# --- Anomaly Detection Functions (Adapted) ---
def detect_anomalies_isolation_forest(df_historic_sample, df_new, numerical_cols, contamination=IF_CONTAMINATION, random_state=42):
    empty_if_df = pd.DataFrame(columns=['index_in_new', 'anomaly_score', 'is_anomaly', 'severity_score'])
    if df_historic_sample is None or df_historic_sample.empty:
        print("Skipping Isolation Forest: No historic sample data provided.")
        return empty_if_df

    valid_num_cols_hist = [col for col in numerical_cols if col in df_historic_sample.columns]
    # df_new is already processed and its numerical_cols are directly from it
    common_num_cols_for_if = [col for col in valid_num_cols_hist if col in df_new.columns and col in numerical_cols]

    if not common_num_cols_for_if:
        print("Skipping Isolation Forest: No common numerical columns between historic sample and new data for IF.")
        return empty_if_df

    df_hist_num = df_historic_sample[common_num_cols_for_if].copy()
    df_new_num = df_new[common_num_cols_for_if].copy() # Use df_new here, not df_new_filled from main scope

    df_hist_filled, _ = handle_missing_values(df_hist_num, numerical_strategy='median')
    df_new_filled, _ = handle_missing_values(df_new_num, numerical_strategy='median') # Fill based on its own stats

    if df_hist_filled.empty or df_new_filled.empty:
        print("Skipping Isolation Forest: Data empty after filling for IF.")
        return empty_if_df

    scaler = StandardScaler()
    try:
        if df_hist_filled.var().sum() == 0:
             print("Warning: Historic sample numerical data has zero variance for IF. Skipping.")
             return empty_if_df
        df_hist_scaled = scaler.fit_transform(df_hist_filled)
        df_new_scaled = scaler.transform(df_new_filled)
    except ValueError as e: # Catches issues like all-NaN columns after filling if not handled earlier
         print(f"Error during scaling for Isolation Forest: {e}. Skipping IF.")
         return empty_if_df

    try:
        model = IsolationForest(contamination=contamination, random_state=random_state, n_estimators=100)
        model.fit(df_hist_scaled)
        anomaly_scores = model.decision_function(df_new_scaled)
        predictions = model.predict(df_new_scaled)
        severity_scores = np.clip(0.5 - anomaly_scores * 2.0, 0, 1)

        # Ensure index_in_new corresponds to the original df_new's index,
        # especially if df_new_num was a subset or copy
        final_indices = df_new_num.index # Get index from the df_new_num used for prediction

        anomaly_df = pd.DataFrame({
            'index_in_new': final_indices, # Use index of the data fed to predict
            'anomaly_score': anomaly_scores,
            'is_anomaly': predictions == -1,
            'severity_score': severity_scores
        })
        print(f"Isolation Forest flagged {anomaly_df['is_anomaly'].sum()} potential anomalies.")
        return anomaly_df
    except Exception as e:
        print(f"Error during Isolation Forest execution: {e}")
        return empty_if_df

def detect_statistical_anomalies_vs_stats(df_new, historic_stats_all_cols, numerical_cols, z_threshold=Z_SCORE_THRESHOLD, iqr_multiplier=IQR_MULTIPLIER):
    anomalies_by_row = defaultdict(list)
    for col in numerical_cols:
        historic_col_stats = historic_stats_all_cols.get(col)
        if not historic_col_stats or 'error' in historic_col_stats or \
           any(pd.isnull(historic_col_stats.get(k)) for k in ['mean', 'std', 'q1', 'q3']):
            # print(f"Skipping stat anomaly for '{col}': Incomplete historic stats.")
            continue

        hist_mean, hist_std = historic_col_stats['mean'], historic_col_stats['std']
        hist_q1, hist_q3 = historic_col_stats['q1'], historic_col_stats['q3']
        # hist_iqr_val is calculated in historic_stats if q1 and q3 exist
        hist_iqr_val = historic_col_stats.get('iqr', 0 if pd.isnull(hist_q1) or pd.isnull(hist_q3) else hist_q3 - hist_q1)
        if pd.isnull(hist_iqr_val): hist_iqr_val = 0 # Default if still NaN

        iqr_lower_bound = hist_q1 - iqr_multiplier * hist_iqr_val
        iqr_upper_bound = hist_q3 + iqr_multiplier * hist_iqr_val
        z_lower_bound = hist_mean - z_threshold * hist_std if hist_std > 1e-6 else hist_mean
        z_upper_bound = hist_mean + z_threshold * hist_std if hist_std > 1e-6 else hist_mean

        historic_null_ratio = 0
        if 'null_count' in historic_col_stats and 'total_count' in historic_col_stats and historic_col_stats['total_count'] > 0:
            historic_null_ratio = historic_col_stats['null_count'] / historic_col_stats['total_count']

        for idx, value in df_new[col].items(): # Iterate over original df_new
            if pd.isnull(value):
                if historic_null_ratio < 0.01: # If nulls were rare historically
                    anomalies_by_row[idx].append({
                        'column': col, 'value': None, 'method': 'Unexpected Missing Value',
                        'severity_score': 0.8, 'historic_context': f"Historically <1% missing (actual: {historic_null_ratio:.2%})",
                        'simple_method': 'Unexpected Missing'
                    })
            elif pd.api.types.is_numeric_dtype(value):
                is_iqr_outlier = False
                if (value < iqr_lower_bound or value > iqr_upper_bound) and pd.notnull(value):
                    distance = min(abs(value - iqr_lower_bound), abs(value - iqr_upper_bound)) if pd.notnull(iqr_lower_bound) and pd.notnull(iqr_upper_bound) else abs(value)
                    severity = 1.0 - np.exp(-0.1 * abs(distance / (hist_iqr_val + 1e-6))) if hist_iqr_val > 1e-6 else 0.7
                    anomalies_by_row[idx].append({
                        'column': col, 'value': value, 'method': 'IQR Outlier', 'severity_score': max(0.5, severity),
                        'historic_context': f"Typical Range (IQR based): ({iqr_lower_bound:.2f}, {iqr_upper_bound:.2f})",
                        'simple_method': 'Outside Typical Range'
                    })
                    is_iqr_outlier = True
                if hist_std > 1e-6 and pd.notnull(value):
                    z = (value - hist_mean) / hist_std
                    if abs(z) > z_threshold:
                        severity = 1.0 - np.exp(-0.1 * (abs(z) - z_threshold))
                        if not is_iqr_outlier or abs(z) > z_threshold * 1.5:
                             anomalies_by_row[idx].append({
                                 'column': col, 'value': value, 'method': 'Z-score Outlier', 'z_score': f"{z:.2f}",
                                 'severity_score': max(0.5, severity) ,
                                 'historic_context': f"Far from Historic Avg ({hist_mean:.2f}, StdDev: {hist_std:.2f})",
                                 'simple_method': 'Far From Average'
                             })
    # print(f"Statistical methods identified potential anomalies in {len(anomalies_by_row)} rows.")
    return dict(anomalies_by_row)

# --- LLM Integration (Adapted) ---
def format_data_for_llm(df, indices, max_rows=10):
    if not indices or df.empty: return "No relevant data rows to display."
    # Ensure indices is a list and filter for valid indices in df
    valid_indices = [idx for idx in list(indices)[:max_rows] if idx in df.index]
    if not valid_indices: return "Specified indices not found in the dataframe."
    try: return df.loc[valid_indices].to_markdown(index=True)
    except ImportError: return df.loc[valid_indices].to_string() # Fallback

def get_llm_analysis(drift_results, stat_anomalies_by_row, if_anomalies_df, df_new,
                     historic_stats_all_cols, new_numerical_summaries,
                     drift_num_cols, drift_cat_cols, alpha=ALPHA): # Removed max_drift, max_anom, defined globally

    # --- Fallback if LLM is not used ---
    if not USE_GEMINI:
        basic_issues = [] # Basic prioritization logic
        # Drift
        drift_issues_sorted = sorted([r for r in drift_results if r.get('drift_detected')], key=lambda x: x.get('drift_score', 0), reverse=True)
        for item in drift_issues_sorted[:3]:
            basic_issues.append({'issue': f"Significant change in '{item['column']}'", 'reasoning': f"High change severity ({item.get('drift_score',0):.2f}). {item.get('simple_result', '')}", 'recommendation': f"Investigate source of '{item['column']}' changes."})
        # Anomalies
        all_anomalies_basic = []
        for idx, anomalies in stat_anomalies_by_row.items():
            for anom in anomalies: all_anomalies_basic.append({'index': idx, 'score': anom.get('severity_score',0), 'col': anom.get('column','N/A'), 'method':anom.get('simple_method','Unknown')})
        if not if_anomalies_df.empty:
            for _, row in if_anomalies_df[if_anomalies_df['is_anomaly']].iterrows():
                all_anomalies_basic.append({'index': row['index_in_new'], 'score': row.get('severity_score',0), 'col':'Multiple', 'method':'Combined Features'})
        all_anomalies_basic.sort(key=lambda x: x['score'], reverse=True)
        # Summarize top anomaly rows
        top_anom_rows_summary = defaultdict(lambda: {'max_score':0, 'cols':set()})
        for anom in all_anomalies_basic[:MAX_ANOMALIES_TO_LLM*2]: # Consider more for summary
            if anom['score'] > top_anom_rows_summary[anom['index']]['max_score']:
                top_anom_rows_summary[anom['index']]['max_score'] = anom['score']
            top_anom_rows_summary[anom['index']]['cols'].add(anom['col'])
        sorted_anom_summary = sorted(top_anom_rows_summary.items(), key=lambda item: item[1]['max_score'], reverse=True)

        for idx, data in sorted_anom_summary[:3]:
             basic_issues.append({'issue': f"Unusual data in row {idx} (Cols: {', '.join(list(data['cols'])[:2])}{'...' if len(data['cols'])>2 else ''})",
                                  'reasoning': f"High anomaly severity ({data['max_score']:.2f}).",
                                  'recommendation': f"Validate data for row {idx}."})
        # Sort all basic issues by a heuristic (e.g., presence of score)
        basic_issues.sort(key=lambda x: (x['reasoning'].split('(')[1].split(')')[0] if '(' in x['reasoning'] else '0'), reverse=True)

        return {"summary": "LLM analysis skipped. Basic prioritization provided.",
                "prioritized_issues": basic_issues[:5],
                "drift_analysis": "LLM analysis skipped.", "anomaly_analysis": "LLM analysis skipped."}

    # --- Prepare LLM Prompt ---
    # 1. Drift Summary
    significant_drift = sorted([r for r in drift_results if r.get('drift_detected')], key=lambda x: x.get('drift_score', 0), reverse=True)
    drift_summary_prompt = f"Detected Data Distribution Changes (based on significance level {alpha}):\n"
    if not significant_drift: drift_summary_prompt += "No significant distribution changes detected.\n"
    else:
        drift_summary_prompt += "Column | Type | Change Result | Change Severity (0-1) | Confidence (0-1)\n"
        drift_summary_prompt += "-------|------|---------------|-----------------------|-----------------\n"
        for item in significant_drift[:MAX_DRIFT_COLS_TO_LLM]: # Use constant
            drift_summary_prompt += f"{item['column']} | {item.get('type','N/A')} | {item.get('simple_result','N/A')} | {item.get('drift_score',0):.3f} | {item.get('confidence',0):.3f}\n"
        if len(significant_drift) > MAX_DRIFT_COLS_TO_LLM: drift_summary_prompt += f"... (and {len(significant_drift) - MAX_DRIFT_COLS_TO_LLM} more)\n"

    # 2. Numerical Summaries
    summary_prompt = "Numerical Data Summary (Historic Stats from DB vs New Data):\n"
    summary_prompt += "Column | Hist Avg | Hist Spread (StdDev) | Hist Range (Q1-Q3) | New Avg | New Spread (StdDev) | New Range (Q1-Q3)\n"
    summary_prompt += "-------|----------|----------------------|--------------------|---------|---------------------|-------------------\n"
    for col, new_s in new_numerical_summaries.items():
        hist_s = historic_stats_all_cols.get(col, {})
        h_avg = f"{hist_s.get('mean', ''):.2f}" if pd.notna(hist_s.get('mean')) else 'N/A'
        h_std = f"{hist_s.get('std', ''):.2f}" if pd.notna(hist_s.get('std')) else 'N/A'
        h_q1q3 = (f"{hist_s.get('q1', ''):.2f}-{hist_s.get('q3', ''):.2f}"
                  if pd.notna(hist_s.get('q1')) and pd.notna(hist_s.get('q3')) else 'N/A')
        n_avg = f"{new_s.get('new_mean', ''):.2f}" if pd.notna(new_s.get('new_mean')) else 'N/A'
        n_std = f"{new_s.get('new_std', ''):.2f}" if pd.notna(new_s.get('new_std')) else 'N/A'
        n_q1q3 = (f"{new_s.get('new_q1', ''):.2f}-{new_s.get('new_q3', ''):.2f}"
                  if pd.notna(new_s.get('new_q1')) and pd.notna(new_s.get('new_q3')) else 'N/A')
        summary_prompt += f"{col} | {h_avg} | {h_std} | {h_q1q3} | {n_avg} | {n_std} | {n_q1q3}\n"

    # 3. Anomaly Summary
    all_anomalies = []
    for idx, anoms in stat_anomalies_by_row.items():
        for anom in anoms: all_anomalies.append({'index': idx, 'column': anom['column'], 'value': anom.get('value'), 'method': anom.get('simple_method','Stat'), 'score': anom.get('severity_score',0)})
    if not if_anomalies_df.empty:
        for _, row in if_anomalies_df[if_anomalies_df['is_anomaly']].iterrows():
            all_anomalies.append({'index': row['index_in_new'], 'column': 'Multiple', 'value': 'N/A', 'method': 'Combined Features', 'score': row.get('severity_score',0)})
    all_anomalies.sort(key=lambda x: x['score'], reverse=True)
    anomaly_summary_prompt = "Detected Unusual Data Points (Anomalies) in New Dataset:\n"
    if not all_anomalies: anomaly_summary_prompt += "No significant unusual data points detected.\n"
    else:
        anomaly_summary_prompt += f"Found potential unusual data. Top {MAX_ANOMALIES_TO_LLM} instances (by severity):\n" # Use constant
        anomaly_summary_prompt += "Row Index | Column(s) | Reason Flagged | Value | Anomaly Severity (0-1)\n"
        anomaly_summary_prompt += "----------|-----------|----------------|-------|----------------------\n"
        for anom in all_anomalies[:MAX_ANOMALIES_TO_LLM]: # Use constant
            val_str = f"{anom['value']:.2f}" if isinstance(anom['value'], (int, float)) else str(anom['value'])
            val_str = (val_str[:25] + '...') if len(val_str) > 25 else val_str # Shorter truncate
            anomaly_summary_prompt += f"{anom['index']} | {anom['column']} | {anom['method']} | {val_str} | {anom['score']:.3f}\n"
        if len(all_anomalies) > MAX_ANOMALIES_TO_LLM: anomaly_summary_prompt += f"... (and {len(all_anomalies) - MAX_ANOMALIES_TO_LLM} more)\n"

    # 4. Data Snippets
    anomalous_indices_for_snippet = sorted(list(set(a['index'] for a in all_anomalies[:MAX_ANOMALIES_TO_LLM])))
    anomalous_rows_data_prompt = "Data for Top Anomalous Rows (up to {} rows):\n".format(MAX_ANOMALIES_TO_LLM)
    anomalous_rows_data_prompt += format_data_for_llm(df_new, anomalous_indices_for_snippet, max_rows=MAX_ANOMALIES_TO_LLM)


    # 5. Construct Final Prompt
    # (Using the refined prompt structure from previous iterations, ensuring it refers to historic stats from DB)
    prompt = f"""
You are an expert AI data analyst communicating findings about a 'new' dataset.
The historic context comes from statistics (like average, typical spread) stored in a database.
Focus on clear, simple language and actionable insights. Avoid technical jargon unless explaining the concept simply.

--- Analysis Context ---
New dataset shape: {df_new.shape}
Columns analyzed for change (potential IDs excluded): Numerical: {drift_num_cols}, Categorical: {drift_cat_cols}

--- Numerical Data Summary (Historic Stats vs New Data) ---
{summary_prompt}

--- Data Distribution Changes Summary ---
{drift_summary_prompt}
{f"(Note: This means the new data's characteristics (e.g., average value, category popularity) are significantly different from historic patterns.)" if significant_drift else ""}

--- Unusual Data Points (Anomalies) Summary ---
{anomaly_summary_prompt}

--- Anomalous Row Data Snippet ---
{anomalous_rows_data_prompt}

--- Your Task ---
Based ONLY on the information provided above, please provide the following in **simple, business-friendly language**:

1.  **Overall Summary:** A brief (2-3 sentence) summary of the main data quality issues found (distribution changes and unusual data). Highlight the most impactful findings.
2.  **Distribution Change Analysis:**
    *   For the columns with the most significant changes (highest 'Change Severity', up to 3):
        *   Explain *what* changed in simple terms (e.g., "The average 'Age' seems higher now", "The variety of 'Product Types' has shifted", "The typical range of 'Transaction Amount' is wider/narrower than before"). Use the Numerical Data Summary for context.
        *   Suggest a potential business impact (e.g., "This might affect sales forecasts", "Customer demographics may be changing").
3.  **Unusual Data Analysis:**
    *   For the most severe unusual data points (top anomalies):
        *   Explain *why* the data is flagged as unusual in simple terms (e.g., "Row 123 has an 'Income' much higher than the typical range seen historically", "Row 456 has a combination of values that is rare", "Row 789 is missing a value in 'Region' where it was usually present"). Refer to the 'Reason Flagged' and the data snippet.
        *   Suggest potential causes (e.g., data entry mistake, system error, genuinely new customer behavior).
4.  **Prioritized List of Issues (Top 3-5):**
    *   List the most critical issues (can be distribution changes or unusual data).
    *   For each issue:
        *   **Issue:** Brief description in simple terms (e.g., "Major shift in 'TransactionAmount'", "Several rows with extremely high 'SensorReading'", "Increase in missing 'Region' data").
        *   **Reasoning:** Why it's important (e.g., "Could impact financial reporting", "Suggests sensor problems", "May break processes relying on 'Region'").
        *   **Recommendation:** A brief, actionable next step (e.g., "Investigate source system for 'TransactionAmount'", "Check sensors for rows [indices]", "Review why 'Region' data might be missing").

**Output Format:** Provide the response as a JSON object with keys: "summary", "drift_analysis" (string), "anomaly_analysis" (string), "prioritized_issues" (list of objects, each with "issue", "reasoning", "recommendation"). Ensure all explanations use **simple, non-technical language**.
"""
    # 6. Call Gemini API
    print("\n--- Calling Gemini API for simplified analysis and prioritization ---")
    try:
        response = gemini_model.generate_content(prompt)
        cleaned_response = response.text.strip()
        cleaned_response = re.sub(r'^```json\s*', '', cleaned_response, flags=re.IGNORECASE | re.DOTALL)
        cleaned_response = re.sub(r'\s*```$', '', cleaned_response)
        llm_result = json.loads(cleaned_response)
        if not all(k in llm_result for k in ["summary", "drift_analysis", "anomaly_analysis", "prioritized_issues"]):
             raise ValueError("LLM response missing required keys.")
        print("--- Gemini Analysis Parsed Successfully ---")
        return llm_result
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode LLM response as JSON: {e}. Response was: \n{cleaned_response[:500]}...")
        return {"summary": f"LLM Error: Could not parse response. {e}", "prioritized_issues": [], "drift_analysis": "Error parsing LLM response.", "anomaly_analysis": "Error parsing LLM response."}
    except Exception as e: # Includes genai API errors
        print(f"Error calling or processing Gemini API: {e}")
        return {"summary": f"LLM Error: {e}", "prioritized_issues": [], "drift_analysis": "LLM call failed.", "anomaly_analysis": "LLM call failed."}

# --- Reporting Function (Adapted) ---
def generate_report_html(drift_results, stat_anomalies_by_row, if_anomalies_df, llm_analysis, df_new,
                         new_numerical_summaries, historic_stats_all_cols, excluded_id_cols):
    # --- LLM Insights Section ---
    report = "<h2>AI Analysis Summary (Powered by Gemini)</h2>"
    report += f"<p><strong>Overall Summary:</strong> {llm_analysis.get('summary', 'LLM summary not available.')}</p>"
    report += f"<details><summary><strong>Detailed Distribution Change Analysis (AI Insights)</strong></summary><pre>{llm_analysis.get('drift_analysis', 'Not available.')}</pre></details>"
    report += f"<details><summary><strong>Detailed Unusual Data Analysis (AI Insights)</strong></summary><pre>{llm_analysis.get('anomaly_analysis', 'Not available.')}</pre></details>"
    report += "<h3>Prioritized Issues (AI Recommended)</h3>"
    prioritized = llm_analysis.get('prioritized_issues', [])
    if prioritized and isinstance(prioritized, list): # Check if list
        report += "<table class='result-table'><thead><tr><th>Issue</th><th>Why it Matters</th><th>Recommendation</th></tr></thead><tbody>"
        for issue in prioritized:
            report += f"<tr><td>{issue.get('issue', 'N/A')}</td><td>{issue.get('reasoning', 'N/A')}</td><td>{issue.get('recommendation', 'N/A')}</td></tr>"
        report += "</tbody></table>"
    else: report += "<p>No prioritized issues provided by AI or analysis failed.</p>"
    report += "<hr>"

    # --- Numerical Summary Section ---
    report += "<h2>Numerical Data Summary (New Data vs. Historic DB Stats)</h2>"
    if new_numerical_summaries:
        report += "<table class='result-table'><thead><tr><th>Column</th><th>Hist Avg (DB)</th><th>New Avg</th><th>Hist Spread (Q1-Q3, DB)</th><th>New Spread (Q1-Q3)</th><th>Hist Range (Min-Max, DB)</th><th>New Range (Min-Max)</th></tr></thead><tbody>"
        for col, new_s in new_numerical_summaries.items():
            hist_s = historic_stats_all_cols.get(col, {})
            # Formatters with NA handling
            fmt = lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
            h_avg, n_avg = fmt(hist_s.get('mean')), fmt(new_s.get('new_mean'))
            h_q1, h_q3 = hist_s.get('q1'), hist_s.get('q3')
            n_q1, n_q3 = new_s.get('new_q1'), new_s.get('new_q3')
            h_q1q3 = f"{fmt(h_q1)}-{fmt(h_q3)}" if pd.notna(h_q1) and pd.notna(h_q3) else "N/A"
            n_q1q3 = f"{fmt(n_q1)}-{fmt(n_q3)}" if pd.notna(n_q1) and pd.notna(n_q3) else "N/A"
            h_min, h_max = fmt(hist_s.get('min')), fmt(hist_s.get('max'))
            n_min, n_max = fmt(new_s.get('new_min')), fmt(new_s.get('new_max'))
            h_range = f"{h_min} - {h_max}" if h_min != "N/A" else "N/A"
            n_range = f"{n_min} - {n_max}" if n_min != "N/A" else "N/A"
            report += f"<tr><td>{col}</td><td>{h_avg}</td><td>{n_avg}</td><td>{h_q1q3}</td><td>{n_q1q3}</td><td>{h_range}</td><td>{n_range}</td></tr>"
        report += "</tbody></table>"
    else: report += "<p>No numerical summaries calculated for new data.</p>"
    report += "<hr>"

    # --- Data Drift Details ---
    report += "<h2>Data Distribution Change Details</h2>"
    if excluded_id_cols: report += f"<p><i>Note: Potential ID columns excluded from this change analysis: {', '.join(excluded_id_cols)}.</i></p>"
    sig_drift = [r for r in drift_results if r and r.get('drift_detected')]
    if sig_drift:
        report += f"<p>Found {len(sig_drift)} columns where data characteristics likely changed compared to historic patterns:</p>"
        report += "<table class='result-table'><thead><tr><th>Column</th><th>Data Type</th><th>Change Check Method</th><th>Result</th><th>Change Severity (0-1)</th><th>Confidence (0-1)</th></tr></thead><tbody>"
        for r in sorted(sig_drift, key=lambda x: x.get('drift_score', 0), reverse=True):
            report += f"<tr class='highlight-drift'><td>{r['column']}</td><td>{r.get('type','N/A')}</td><td>{r.get('simple_test_name','N/A')}</td><td>{r.get('simple_result','N/A')}</td><td>{r.get('drift_score',0):.3f}</td><td>{r.get('confidence',0):.3f}</td></tr>"
        report += "</tbody></table>"
    else: report += "<p>No significant data distribution changes detected.</p>"
    report += "<hr>"

    # --- Anomaly Details ---
    report += "<h2>Unusual Data Point Details</h2>"
    all_anoms_report = [] # Rebuild for reporting
    for idx, anoms_list in stat_anomalies_by_row.items():
        for anom_item in anoms_list: all_anoms_report.append({'index':idx, 'column':anom_item.get('column'), 'value':anom_item.get('value'), 'method':anom_item.get('simple_method','Stat'), 'score':anom_item.get('severity_score',0), 'context':anom_item.get('historic_context','N/A'), 'type':'Single Column'})
    if not if_anomalies_df.empty:
        for _, row in if_anomalies_df[if_anomalies_df['is_anomaly']].iterrows():
            all_anoms_report.append({'index':row['index_in_new'], 'column':'Multiple', 'value':'N/A', 'method':'Combined Features', 'score':row.get('severity_score',0), 'context':f"IF Score: {row.get('anomaly_score',0):.3f}", 'type':'Multiple Columns'})
    all_anoms_report.sort(key=lambda x: x['score'], reverse=True)

    if all_anoms_report:
        unique_anom_rows = len(set(a['index'] for a in all_anoms_report))
        report += f"<p>Found {len(all_anoms_report)} potential unusual data instances across {unique_anom_rows} unique rows.</p>"
        report += "<h3>Top Unusual Data Instances (by Severity Score)</h3>"
        report += "<table class='result-table'><thead><tr><th>Row Index</th><th>Column(s)</th><th>Value</th><th>Reason Flagged</th><th>Severity (0-1)</th><th>Context/Details</th></tr></thead><tbody>"
        for anom in all_anoms_report[:20]: # Show top 20
             val_str = f"{anom['value']:.2f}" if isinstance(anom['value'], (float, int)) else str(anom['value'])
             val_str = (val_str[:30] + '...') if len(val_str) > 30 else val_str
             hl_class = 'highlight-anomaly-high' if anom['score'] > 0.7 else 'highlight-anomaly-med'
             report += f"<tr class='{hl_class}'><td>{anom['index']}</td><td>{anom['column']}</td><td>{val_str}</td><td>{anom['method']} ({anom['type']})</td><td>{anom['score']:.3f}</td><td>{anom['context']}</td></tr>"
        report += "</tbody></table>"
        if len(all_anoms_report) > 20: report += f"<p>... and {len(all_anoms_report) - 20} more instances.</p>"

        # Collapsible data for anomalous rows
        anom_indices_for_display = sorted(list(set(a['index'] for a in all_anoms_report)))[:MAX_ANOMALIES_TO_LLM]
        report += f"<details><summary><strong>Show Data for Top {len(anom_indices_for_display)} Rows with Unusual Data</strong></summary>"
        if anom_indices_for_display:
            valid_indices_display = [idx for idx in anom_indices_for_display if idx in df_new.index]
            if valid_indices_display:
                 report += "<div class='preview-table-container' style='max-height: 400px;'>"
                 try: report += df_new.loc[valid_indices_display].to_html(classes='preview-table', index=True, border=0, na_rep='NA')
                 except Exception as table_err: report += f"<p>Error generating table: {table_err}</p>"
                 report += "</div>"
            else: report += "<p>Could not retrieve data for anomalous rows.</p>"
        else: report += "<p>No specific anomalous rows to display.</p>"
        report += "</details>"
    else: report += "<p>No significant unusual data points detected.</p>"
    report += "<hr><p><em>End of Report</em></p>"
    return report

# --- Main Analysis Function (Adapted) ---
def run_analysis(new_filepath, historic_sample_filepath=None):
    excluded_id_cols = []
    df_historic_sample = None
    historic_stats_all_cols = {} # Initialize
    db_conn = None # Initialize

    try:
        db_conn = get_db_connection() # Attempt connection first

        # 1. Load New Data
        df_new = load_new_data(new_filepath)
        df_new_filled, _ = handle_missing_values(df_new.copy()) # Use copy for filling

        # 1b. Load Optional Historic Sample Data
        if historic_sample_filepath:
            try:
                df_historic_sample_raw = pd.read_csv(historic_sample_filepath)
                # Basic column harmonization with new_data for IF
                common_cols_sample = df_new.columns.intersection(df_historic_sample_raw.columns)
                if not common_cols_sample.empty:
                    df_historic_sample = df_historic_sample_raw[common_cols_sample].copy()
                    print(f"Loaded and harmonized historic sample data: {df_historic_sample.shape}")
                else:
                     print("Warning: No common columns between new data and historic sample. Sample will not be used effectively.")
                     df_historic_sample = None
            except Exception as e:
                print(f"Warning: Could not load/process historic sample CSV '{historic_sample_filepath}': {e}.")
                df_historic_sample = None

        # 2. Identify Column Types from New Data
        all_num_cols, all_cat_cols, _ = identify_column_types(df_new)

        # 3. Fetch Historic Statistics from Database (if connection and table name are valid)
        if db_conn and HISTORIC_TABLE_NAME:
            columns_for_db_stats = {col: 'numerical' for col in all_num_cols}
            columns_for_db_stats.update({col: 'categorical' for col in all_cat_cols})
            print(f"Fetching historic stats from DB table '{HISTORIC_TABLE_NAME}' for columns: {list(columns_for_db_stats.keys())}")
            historic_stats_all_cols = fetch_historic_stats_from_db(db_conn, HISTORIC_TABLE_NAME, columns_for_db_stats)
        else:
            print("Skipping fetch of historic stats from DB (connection or table name issue).")

        # 4. Exclude Potential ID Columns from Drift Analysis
        drift_num_cols = [col for col in all_num_cols if not is_potential_id_column(col, df_new[col])]
        drift_cat_cols = [col for col in all_cat_cols if not is_potential_id_column(col, df_new[col])]
        excluded_id_cols = [col for col in (all_num_cols + all_cat_cols) if col not in (drift_num_cols + drift_cat_cols)]
        if excluded_id_cols: print(f"Excluding potential ID columns from drift analysis: {excluded_id_cols}")

        # 5. Calculate Numerical Summaries for New Data
        new_numerical_summaries = calculate_new_numerical_summaries(df_new, all_num_cols)

        # 6. Perform Drift Detection
        drift_results = []
        for col in drift_num_cols:
            hist_col_s = historic_stats_all_cols.get(col, {})
            # Prioritize KS test if historic sample is available and valid for this column
            if df_historic_sample is not None and col in df_historic_sample.columns and not df_historic_sample[col].dropna().empty:
                # Using original detect_numerical_drift (KS) if sample available
                # Ensure it's defined or adapt it for two series
                # For now, sticking to DB stats comparison for simplicity of current refactor.
                # To use KS: result_drift = original_ks_drift_function(df_historic_sample[col], df_new_filled[col], col)
                result_drift = detect_numerical_drift_vs_stats(df_new_filled[col], hist_col_s, col)
            else:
                result_drift = detect_numerical_drift_vs_stats(df_new_filled[col], hist_col_s, col)
            if result_drift: drift_results.append(result_drift)

        for col in drift_cat_cols:
            hist_col_s = historic_stats_all_cols.get(col, {})
            result_drift = detect_categorical_drift_vs_stats(df_new_filled[col], hist_col_s, col)
            if result_drift: drift_results.append(result_drift)

        # 7. Perform Anomaly Detection
        # Pass df_new_filled to IF as it's the one IF model should see (matching its potential training on filled historic sample)
        if_anomalies_df = detect_anomalies_isolation_forest(df_historic_sample, df_new_filled, all_num_cols)
        # For stat anomalies, use original df_new to capture true nulls before filling
        stat_anomalies_by_row = detect_statistical_anomalies_vs_stats(df_new, historic_stats_all_cols, all_num_cols)

        # 8. Get LLM Analysis
        llm_analysis = get_llm_analysis(drift_results, stat_anomalies_by_row, if_anomalies_df,
                                        df_new, historic_stats_all_cols, new_numerical_summaries,
                                        drift_num_cols, drift_cat_cols, alpha=ALPHA)

        # 9. Generate Report
        report_html = generate_report_html(drift_results, stat_anomalies_by_row, if_anomalies_df,
                                           llm_analysis, df_new, new_numerical_summaries,
                                           historic_stats_all_cols, excluded_id_cols)

        # 10. Get Data Previews
        new_preview_html = get_preview_table_html(df_new, title="New Data Preview")
        hist_s_preview_title = "Historic Sample Data Preview" if historic_sample_filepath else "No Historic Sample Provided for Preview"
        historic_preview_html = get_preview_table_html(df_historic_sample, title=hist_s_preview_title)

        return report_html, historic_preview_html, new_preview_html

    except (ValueError, RuntimeError, FileNotFoundError) as e:
        print(f"CRITICAL ANALYSIS ERROR: {e}")
        import traceback
        traceback.print_exc()
        error_html = f"<h2>Analysis Failed Critically</h2><hr><p><strong>Error Type: {type(e).__name__}</strong></p><p><strong>Message:</strong> {e}</p><p>Please check inputs and server logs.</p>"
        return error_html, "<p>Preview N/A</p>", "<p>Preview N/A</p>"
    except Exception as e:
        print(f"UNEXPECTED CRITICAL ANALYSIS ERROR: {e}")
        import traceback
        traceback.print_exc()
        error_html = f"<h2>Analysis Failed Unexpectedly</h2><hr><p><strong>An unexpected critical error occurred:</strong> {e}</p><p>Please check server logs.</p>"
        return error_html, "<p>Preview N/A</p>", "<p>Preview N/A</p>"
    finally:
        if db_conn:
            db_conn.close()
            print("Database connection closed.")