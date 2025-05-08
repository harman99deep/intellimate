# analysis_engine.py

import pandas as pd
import numpy as np
# from scipy.stats import ks_2samp, chi2_contingency # Removed as not used
from scipy.stats import iqr as scipy_iqr, ttest_1samp, chisquare
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
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
HISTORIC_TABLE_NAME = os.getenv("HISTORIC_TABLE_NAME")

# Configure Google Gemini
USE_GEMINI = False # Default to False
gemini_model = None # Initialize
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
MAX_ANOMALIES_TO_LLM_SNIPPET = 5
MAX_DRIFT_COLS_TO_LLM_SUMMARY = 7
ID_COLUMN_SUBSTRINGS = ['id', 'key', 'uuid', 'identifier', 'no', 'num', 'code', 'token', 'number'] # Expanded
ID_UNIQUENESS_THRESHOLD = 0.90

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# --- Formatting Helper ---
def format_value_for_display(value):
    if pd.isna(value):
        return "N/A"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, (int, np.integer)):
        return str(value)
    if isinstance(value, (float, np.floating)):
        return f"{value:.2f}"
    return str(value)

# --- Database Helper ---
def get_db_connection():
    if not all([DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, HISTORIC_TABLE_NAME]):
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
        with db_conn.cursor() as cursor:
            for col_name, col_type in columns_info.items():
                try:
                    safe_col_name = f'"{col_name.replace("\"", "\"\"")}"'
                    if col_type == 'numerical':
                        query = f"""
                        SELECT
                            AVG(CAST({safe_col_name} AS NUMERIC)), STDDEV(CAST({safe_col_name} AS NUMERIC)),
                            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY CAST({safe_col_name} AS NUMERIC)),
                            PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY CAST({safe_col_name} AS NUMERIC)),
                            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY CAST({safe_col_name} AS NUMERIC)),
                            MIN(CAST({safe_col_name} AS NUMERIC)), MAX(CAST({safe_col_name} AS NUMERIC)),
                            SUM(CASE WHEN {safe_col_name} IS NULL THEN 1 ELSE 0 END), COUNT(*)
                        FROM "{table_name}";
                        """
                        cursor.execute(query)
                        stats_row = cursor.fetchone()
                        if stats_row:
                            q1_val, q3_val = stats_row[2], stats_row[4]
                            historic_stats[col_name] = {
                                'mean': stats_row[0], 'std': stats_row[1], 'q1': q1_val, 'median': stats_row[3], 'q3': q3_val,
                                'min': stats_row[5], 'max': stats_row[6],
                                'iqr': (q3_val - q1_val) if q1_val is not None and q3_val is not None else None,
                                'null_count': stats_row[7], 'total_count': stats_row[8]
                            }
                    elif col_type == 'categorical': # This now includes former "text" columns
                        query = f"SELECT CAST({safe_col_name} AS TEXT), COUNT(*) FROM \"{table_name}\" GROUP BY CAST({safe_col_name} AS TEXT);"
                        cursor.execute(query)
                        value_counts_list = cursor.fetchall()
                        if value_counts_list: # Ensure there are counts
                             historic_stats[col_name] = {'value_counts': dict(value_counts_list),
                                                         'null_count': None, 'total_count': None} # Add placeholders for consistency
                             # Attempt to get null/total counts for categoricals too if possible, or derive from value_counts sum
                             query_cat_meta = f"""
                             SELECT SUM(CASE WHEN {safe_col_name} IS NULL THEN 1 ELSE 0 END), COUNT(*)
                             FROM "{table_name}";
                             """
                             cursor.execute(query_cat_meta)
                             meta_row = cursor.fetchone()
                             if meta_row:
                                 historic_stats[col_name]['null_count'] = meta_row[0]
                                 historic_stats[col_name]['total_count'] = meta_row[1]
                        else:
                             historic_stats[col_name] = {'value_counts': {}, 'null_count': 0, 'total_count': 0, 'error': 'No categories found or all nulls'}


                except psycopg2.Error as e:
                    print(f"DB Error fetching stats for column '{col_name}': {e}")
                    db_conn.rollback(); historic_stats[col_name] = {'error': f"DB Error: {e}"}
                except Exception as e:
                    print(f"Unexpected error fetching stats for column '{col_name}': {e}")
                    historic_stats[col_name] = {'error': f"Unexpected Error: {e}"}
    except psycopg2.Error as e:
        print(f"DB Error during historic stats fetching process: {e}")
        return {'db_process_error': str(e)}
    return historic_stats

# --- Helper Functions ---
def load_new_data(new_path):
    try:
        if not new_path or not new_path.lower().endswith('.csv'):
             raise ValueError("Invalid file type or path for new data. Please upload a CSV file.")
        df_new = pd.read_csv(new_path)
        if df_new.empty: raise ValueError("The new dataset CSV file is empty.")
        print(f"Loaded new data: {df_new.shape} from {new_path}")
        return df_new
    except FileNotFoundError: raise ValueError(f"New data file not found: {new_path}")
    except pd.errors.EmptyDataError: raise ValueError("New data CSV file is empty.")
    except Exception as e: raise RuntimeError(f"Unexpected error during new data loading: {e}")

def identify_column_types_and_ids(df):
    numerical_cols, categorical_cols, id_cols = [], [], []
    potential_id_cols = []

    for col in df.columns:
        if is_potential_id_column(col, df[col]):
            potential_id_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            numerical_cols.append(col)
        # All non-numeric, non-ID columns are treated as categorical for drift/summary purposes
        # The distinction between "text" and "categorical" is less critical for the current analysis suite
        # if not explicitly used for different processing paths (like NLP).
        elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            categorical_cols.append(col)
        else: # Other specific types (datetime, boolean if not handled above) could be added
            print(f"Column '{col}' has unhandled dtype {df[col].dtype}, treating as categorical for now.")
            categorical_cols.append(col)


    for col in potential_id_cols:
        id_cols.append(col)
        if col in numerical_cols: numerical_cols.remove(col)
        if col in categorical_cols: categorical_cols.remove(col)

    print(f"Identified Column Types: Num: {numerical_cols}, Cat (incl. Text-like): {categorical_cols}, IDs: {id_cols}")
    return numerical_cols, categorical_cols, id_cols


def handle_missing_values(df, numerical_strategy='median', categorical_strategy='mode'):
    df_filled = df.copy()
    for col in df_filled.columns:
        if df_filled[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df_filled[col]):
                fill_value = df_filled[col].median() if numerical_strategy == 'median' else df_filled[col].mean()
                if pd.isnull(fill_value): fill_value = 0 # Fallback
                df_filled[col].fillna(fill_value, inplace=True)
            elif pd.api.types.is_object_dtype(df_filled[col]) or pd.api.types.is_categorical_dtype(df_filled[col]):
                modes = df_filled[col].mode()
                fill_value = modes[0] if not modes.empty else 'Missing_Value_Filled'
                df_filled[col].fillna(fill_value, inplace=True)
    return df_filled

def get_preview_table_html(df, title="Data Preview"):
    if df is None or df.empty: return f"<p>{title}: No data to display.</p>"
    try:
        df_display = df.head(10).copy()
        for col in df_display.select_dtypes(include=[np.number]).columns:
            df_display[col] = df_display[col].apply(format_value_for_display)
        return df_display.to_html(classes='preview-table', index=True, border=0, na_rep='NA', escape=False)
    except Exception as e: return f"<p>Error generating preview for {title}: {e}</p>"

def is_potential_id_column(col_name, series):
    if not isinstance(series, pd.Series): return False
    col_name_lower = str(col_name).lower()
    if any(sub in col_name_lower for sub in ID_COLUMN_SUBSTRINGS): return True
    try:
        if len(series) == 0: return False
        is_mostly_int_or_string_like = False
        # Check if dtype is numeric and all non-NaN values are integers
        if pd.api.types.is_numeric_dtype(series.dtype):
            non_na_numeric = series.dropna()
            if not non_na_numeric.empty and non_na_numeric.apply(lambda x: x == int(x)).all():
                is_mostly_int_or_string_like = True
        elif pd.api.types.is_string_dtype(series.dtype) or series.dtype == 'object':
            is_mostly_int_or_string_like = True

        if is_mostly_int_or_string_like:
            non_na_series = series.dropna()
            if len(non_na_series) > 10 : # Only consider uniqueness for reasonably sized non-null series
                unique_ratio_non_na = non_na_series.nunique() / len(non_na_series)
                if unique_ratio_non_na >= ID_UNIQUENESS_THRESHOLD:
                    return True
            # Fallback for columns with many NaNs or shorter series, check overall uniqueness
            if len(series) > 0:
                 unique_ratio_overall = series.nunique(dropna=False) / len(series)
                 # For shorter columns or high NaN columns, a very high unique ratio is still a strong ID signal
                 if (len(series) <=10 and unique_ratio_overall == 1.0) or \
                    (unique_ratio_overall >= ID_UNIQUENESS_THRESHOLD + 0.05) : # Stricter if using dropna=False
                     return True
    except Exception as e:
        print(f"Warning: Could not assess ID-ness for column '{col_name}': {e}")
    return False

def perform_id_column_checks(df_new, id_cols):
    id_checks_results = {}
    for col in id_cols:
        series = df_new[col]
        res = {
            "column_name": col, "data_type": str(series.dtype),
            "total_records": int(len(series)), "missing_values": int(series.isnull().sum()),
            "missing_percentage": f"{(series.isnull().sum() / len(series) * 100) if len(series) > 0 else 0:.2f}%",
            "unique_values_incl_na": int(series.nunique(dropna=False)),
            "unique_values_excl_na": int(series.nunique(dropna=True)),
        }
        non_null_series = series.dropna()
        if not non_null_series.empty:
            res["duplicate_values_among_non_null"] = int(len(non_null_series) - res["unique_values_excl_na"])
            res["all_unique_among_non_null"] = (res["duplicate_values_among_non_null"] == 0)
            if pd.api.types.is_string_dtype(non_null_series.dtype) or non_null_series.dtype == 'object':
                lengths = non_null_series.astype(str).str.len()
                if not lengths.empty:
                    res["most_common_length"] = int(lengths.mode()[0]) if not lengths.mode().empty else 'N/A'
                    res["min_length"] = int(lengths.min()); res["max_length"] = int(lengths.max())
                    res["consistent_length"] = (res["min_length"] == res["max_length"])
        else:
            res.update({"duplicate_values_among_non_null": 0, "all_unique_among_non_null": True})
        id_checks_results[col] = res
    return id_checks_results

def calculate_new_numerical_summaries(df_new, numerical_cols):
    summaries = {}
    for col in numerical_cols:
        new_col_series = df_new[col].dropna().astype(float)
        summary = {k: np.nan for k in ['new_min', 'new_max', 'new_iqr', 'new_mean', 'new_std', 'new_q1', 'new_median', 'new_q3']}
        try:
            if not new_col_series.empty:
                summary.update({
                    'new_min': new_col_series.min(), 'new_max': new_col_series.max(),
                    'new_iqr': scipy_iqr(new_col_series) if len(new_col_series) > 0 else np.nan,
                    'new_mean': new_col_series.mean(), 'new_std': new_col_series.std(),
                    'new_q1': new_col_series.quantile(0.25), 'new_median': new_col_series.median(),
                    'new_q3': new_col_series.quantile(0.75)
                })
            summary['new_null_count'] = df_new[col].isnull().sum()
            summary['new_total_count'] = len(df_new[col])
            summaries[col] = summary
        except Exception as e:
            print(f"Warning: Could not calculate summary for new data column '{col}': {e}")
            summaries[col] = {'error': str(e), 'new_null_count': df_new[col].isnull().sum(), 'new_total_count': len(df_new[col])}
    return summaries

def calculate_null_value_comparison(df_new, historic_stats_all_cols, all_analyzed_cols):
    null_comparison = {}
    for col in all_analyzed_cols:
        new_null_count = df_new[col].isnull().sum()
        new_total_count = len(df_new[col]) if col in df_new else 0
        new_null_ratio = (new_null_count / new_total_count) if new_total_count > 0 else 0.0

        hist_stats = historic_stats_all_cols.get(col, {})
        hist_null_count = hist_stats.get('null_count')
        hist_total_count = hist_stats.get('total_count')
        hist_null_ratio = None

        if hist_null_count is not None and hist_total_count is not None and hist_total_count > 0:
            hist_null_ratio = float(hist_null_count) / float(hist_total_count)

        change_info = "N/A (no historic data for comparison)"
        if hist_null_ratio is not None:
            diff = new_null_ratio - hist_null_ratio
            change_info = f"{diff:+.2%}"
            if abs(diff) > 0.10: # Threshold for "notable"
                 change_info += " (Notable Increase)" if diff > 0 else " (Notable Decrease)"
        null_comparison[col] = {
            "new_null_count": new_null_count, "new_null_percentage": f"{new_null_ratio:.2%}",
            "historic_null_count": format_value_for_display(hist_null_count),
            "historic_null_percentage": f"{hist_null_ratio:.2%}" if hist_null_ratio is not None else "N/A",
            "change_in_null_percentage_points": change_info
        }
    return null_comparison

# --- Drift Detection Functions ---
def detect_numerical_drift_vs_stats(series_new, historic_col_stats, column_name, alpha=ALPHA):
    default_no_drift = {"column": column_name, "type": "Numerical", "test": "Mean Comparison (vs Historic Stats)",
                        "drift_detected": False, "simple_result": "Comparison N/A or insufficient data", "drift_score": 0, "confidence": 0,
                        "p_value": 1.0, "statistic": None, "simple_test_name": "Average Value Check"}
    if not historic_col_stats or 'mean' not in historic_col_stats or 'std' not in historic_col_stats \
            or pd.isnull(historic_col_stats['mean']) or pd.isnull(historic_col_stats['std']):
        default_no_drift["simple_result"] = "Historic mean/std stats unavailable or incomplete"
        # print(f"Debug: No historic mean/std for {column_name}. Stats: {historic_col_stats}")
        return default_no_drift
    series_new_na = series_new.dropna()
    if len(series_new_na) < 5: # Min sample size for t-test
        default_no_drift["simple_result"] = "Insufficient new non-null data points for comparison"
        return default_no_drift
    hist_mean, hist_std = historic_col_stats['mean'], historic_col_stats['std']
    p_value, drift_detected, simple_result, statistic = 1.0, False, "Mean is consistent with historic", None
    if hist_std == 0: # historic_std can be 0 if all historic values were the same
        new_mean = series_new_na.mean()
        if not np.isclose(new_mean, hist_mean):
            drift_detected = True; p_value = 0.0; simple_result = "Mean changed significantly (historic data had no variance)"
        else: simple_result = "Mean consistent (historic data had no variance)"
    else:
        try:
            statistic, p_value = ttest_1samp(series_new_na, hist_mean, nan_policy='omit')
            drift_detected = p_value < alpha
            simple_result = "Mean changed significantly from historic" if drift_detected else "Mean is consistent with historic"
        except Exception as e:
            print(f"Warning: t-test failed for {column_name}: {e}. Defaulting to no drift.")
            simple_result = f"Mean comparison test failed ({e})"
    return {"column": column_name, "type": "Numerical", "test": "Mean Comparison (vs Historic Stats)",
            "statistic": statistic, "p_value": p_value, "drift_score": 1 - p_value if p_value is not None else 0,
            "drift_detected": drift_detected, "confidence": 1 - p_value if p_value is not None else 0,
            "simple_result": simple_result, "simple_test_name": "Average Value Check"}

def detect_categorical_drift_vs_stats(series_new, historic_col_stats, column_name, alpha=ALPHA):
    default_no_drift = {"column": column_name, "type": "Categorical", "test": "Category Proportion Test",
                        "drift_detected": False, "simple_result": "Comparison N/A or insufficient data", "drift_score": 0, "confidence": 0,
                        "p_value": 1.0, "statistic": None, "simple_test_name": "Category Popularity Check"}

    if not historic_col_stats or 'value_counts' not in historic_col_stats or not isinstance(historic_col_stats['value_counts'], dict) or not historic_col_stats['value_counts']:
        default_no_drift["simple_result"] = "Historic category counts (value_counts) unavailable or empty"
        # print(f"Debug: No historic value_counts for {column_name}. Stats: {historic_col_stats}")
        return default_no_drift

    series_new_str = series_new.astype(str).fillna('__NaN__') # Ensure consistent NaN representation
    new_counts_pd = series_new_str.value_counts()
    hist_counts_dict = historic_col_stats['value_counts']
    hist_counts_pd = pd.Series(hist_counts_dict).astype(float)

    if new_counts_pd.empty and hist_counts_pd.empty: default_no_drift["simple_result"] = "No data in new or historic counts"; return default_no_drift
    if new_counts_pd.empty: default_no_drift["simple_result"] = "No data in new counts to compare"; return default_no_drift
    # hist_counts_pd.empty was checked by historic_col_stats['value_counts'] check earlier

    all_categories = sorted(list(set(new_counts_pd.index) | set(hist_counts_pd.index)))
    new_total, hist_total = new_counts_pd.sum(), hist_counts_pd.sum()

    if new_total < 5 : # Chi-square not reliable with very small new sample
        default_no_drift["simple_result"] = "Insufficient new data points for category proportion test"
        return default_no_drift
    # hist_total should be >0 if hist_counts_pd was not empty

    observed_freq = new_counts_pd.reindex(all_categories, fill_value=0).astype(float)
    expected_props = hist_counts_pd.reindex(all_categories, fill_value=0) / hist_total
    expected_freq = expected_props * new_total

    # Filter out categories where expected frequency is effectively zero for chi-square
    valid_mask = (expected_freq > 1e-9) # Small epsilon for float comparison
    observed_final, expected_final = observed_freq[valid_mask], expected_freq[valid_mask]

    if len(observed_final) < 1: default_no_drift["simple_result"] = "No comparable categories with expected historic counts after filtering"; return default_no_drift
    if len(observed_final) == 1 and len(all_categories) > 1 and observed_final.sum() < new_total: # Only one category has expected counts, others are new/unexpected
        return {**default_no_drift, "drift_detected": True, "simple_result": "Major shift: New categories appeared or historic ones disappeared", "drift_score": 1.0, "confidence": 1.0, "p_value": 0.0}

    # Ensure no zero expected frequencies passed to chisquare if they still exist (should be caught by mask)
    expected_final = np.maximum(expected_final, 1e-9) # Prevent zero division if any slip through

    try:
        # Check degrees of freedom
        ddof = len(observed_final) -1
        if ddof < 1:
            default_no_drift["simple_result"] = "Insufficient categories for reliable Chi-square test (dof < 1)"
            return default_no_drift

        chi2_stat, p_value = chisquare(f_obs=observed_final, f_exp=expected_final)
        drift_detected = p_value < alpha
        result_interpretation = "Category proportions changed significantly from historic" if drift_detected else "Category proportions consistent with historic"
        if np.any(expected_final < 5): # Standard Chi-square warning
            print(f"Warning: Low expected frequency (<5) in Chi2 test for '{column_name}'. P-value may be less reliable.")
            result_interpretation += " (Note: some expected counts were low, p-value less reliable)"
        return {"column": column_name, "type": "Categorical", "test": "Category Proportion Test",
                "statistic": chi2_stat, "p_value": p_value, "drift_score": 1 - p_value,
                "drift_detected": drift_detected, "confidence": 1 - p_value,
                "simple_result": result_interpretation, "simple_test_name": "Category Popularity Check"}
    except ValueError as e:
        print(f"Error during Chi2 G.O.F test calculation for '{column_name}': {e}.")
        default_no_drift["simple_result"] = f"Category proportion test failed ({e})"; return default_no_drift
    except Exception as e:
         print(f"Unexpected error during Chi2 G.O.F test for '{column_name}': {e}")
         default_no_drift["simple_result"] = f"Unexpected error in proportion test ({e})"; return default_no_drift

# --- Anomaly Detection Functions ---
def detect_anomalies_isolation_forest(df_historic_sample, df_new, numerical_cols_for_if, contamination=IF_CONTAMINATION, random_state=42):
    empty_if_df = pd.DataFrame(columns=['index_in_new', 'anomaly_score', 'is_anomaly', 'severity_score'])
    if df_historic_sample is None or df_historic_sample.empty:
        print("Skipping Isolation Forest: No historic sample data provided."); return empty_if_df
    if not numerical_cols_for_if:
        print("Skipping Isolation Forest: No numerical columns provided for IF."); return empty_if_df

    common_num_cols_for_if = [col for col in numerical_cols_for_if if col in df_historic_sample.columns and col in df_new.columns]
    if not common_num_cols_for_if:
        print("Skipping Isolation Forest: No common numerical columns between historic sample and new data for IF."); return empty_if_df

    df_hist_num = df_historic_sample[common_num_cols_for_if].copy()
    df_new_num = df_new[common_num_cols_for_if].copy()
    df_hist_filled, _ = handle_missing_values(df_hist_num, numerical_strategy='median')
    df_new_filled, _ = handle_missing_values(df_new_num, numerical_strategy='median')

    if df_hist_filled.empty or df_new_filled.empty: print("Skipping Isolation Forest: Data empty after filling for IF."); return empty_if_df
    scaler = StandardScaler()
    try:
        if df_hist_filled.var().sum() == 0: print("Warning: Historic sample numerical data has zero variance for IF. Skipping."); return empty_if_df
        df_hist_scaled = scaler.fit_transform(df_hist_filled)
        df_new_scaled = scaler.transform(df_new_filled)
    except ValueError as e: print(f"Error during scaling for Isolation Forest: {e}. Skipping IF."); return empty_if_df

    try:
        model = IsolationForest(contamination=contamination, random_state=random_state, n_estimators=100)
        model.fit(df_hist_scaled)
        anomaly_scores = model.decision_function(df_new_scaled)
        anomaly_df = pd.DataFrame({
            'index_in_new': df_new_num.index, 'anomaly_score': anomaly_scores,
            'is_anomaly': model.predict(df_new_scaled) == -1,
            'severity_score': np.clip(0.5 - anomaly_scores * 2.0, 0, 1) # Rescale: higher is more anomalous
        })
        print(f"Isolation Forest flagged {anomaly_df['is_anomaly'].sum()} potential anomalies.")
        return anomaly_df
    except Exception as e: print(f"Error during Isolation Forest execution: {e}"); return empty_if_df

def detect_statistical_anomalies_vs_stats(df_new, historic_stats_all_cols, numerical_cols_for_stat_anom, z_threshold=Z_SCORE_THRESHOLD, iqr_multiplier=IQR_MULTIPLIER):
    anomalies_by_row = defaultdict(list)
    for col in numerical_cols_for_stat_anom:
        historic_col_stats = historic_stats_all_cols.get(col)
        if not historic_col_stats or 'error' in historic_col_stats or \
           any(pd.isnull(historic_col_stats.get(k)) for k in ['mean', 'std', 'q1', 'q3']):
            continue
        hist_mean, hist_std = historic_col_stats['mean'], historic_col_stats['std']
        hist_q1, hist_q3 = historic_col_stats['q1'], historic_col_stats['q3']
        hist_iqr_val = historic_col_stats.get('iqr', 0 if pd.isnull(hist_q1) or pd.isnull(hist_q3) else hist_q3 - hist_q1)
        if pd.isnull(hist_iqr_val) or hist_iqr_val == 0 : hist_iqr_val = 1e-6 # Avoid division by zero later, ensure small range if IQR is 0

        iqr_lower = hist_q1 - iqr_multiplier * hist_iqr_val
        iqr_upper = hist_q3 + iqr_multiplier * hist_iqr_val
        
        # Ensure std is not zero for z-score bounds
        safe_hist_std = hist_std if hist_std > 1e-6 else 1e-6
        z_lower = hist_mean - z_threshold * safe_hist_std
        z_upper = hist_mean + z_threshold * safe_hist_std

        historic_null_ratio = (float(historic_col_stats.get('null_count',0)) / float(historic_col_stats.get('total_count',1))) if historic_col_stats.get('total_count',0) > 0 else 0.0

        for idx, value in df_new[col].items():
            if pd.isnull(value):
                if historic_null_ratio < 0.01: # Nulls were rare historically
                    anomalies_by_row[idx].append({
                        'column': col, 'value': None, 'method': 'Unexpected Missing Value', 'severity_score': 0.8,
                        'historic_context': f"Historically <1% missing (actual rate: {historic_null_ratio:.2%})", 'simple_method': 'Unexpected Missing'})
            elif pd.api.types.is_numeric_dtype(value):
                value_float = float(value)
                is_iqr_outlier = False
                if (value_float < iqr_lower or value_float > iqr_upper):
                    dist = min(abs(value_float - iqr_lower), abs(value_float - iqr_upper))
                    sev = 1.0 - np.exp(-0.1 * abs(dist / hist_iqr_val)) # hist_iqr_val is now protected from zero
                    anomalies_by_row[idx].append({'column': col, 'value': value_float, 'method': 'IQR Outlier', 'severity_score': max(0.5, min(0.99, sev)),
                                                 'historic_context': f"Typical Range (IQR based): ({format_value_for_display(iqr_lower)}, {format_value_for_display(iqr_upper)})", 'simple_method': 'Outside Typical Range'})
                    is_iqr_outlier = True
                if safe_hist_std > 1e-7: # Redundant check, but safe
                    z = (value_float - hist_mean) / safe_hist_std
                    if abs(z) > z_threshold:
                        sev = 1.0 - np.exp(-0.1 * (abs(z) - z_threshold))
                        if not is_iqr_outlier or abs(z) > z_threshold * 1.5:
                             anomalies_by_row[idx].append({'column': col, 'value': value_float, 'method': 'Z-score Outlier', 'z_score': f"{z:.2f}", 'severity_score': max(0.5, min(0.99,sev)),
                                                          'historic_context': f"Far from Historic Avg ({format_value_for_display(hist_mean)}, StdDev: {format_value_for_display(safe_hist_std)})", 'simple_method': 'Far From Average'})
    return dict(anomalies_by_row)

# --- LLM Integration ---
def format_data_for_llm(df, indices, max_rows=5):
    if not indices or df.empty: return "No relevant data rows to display."
    valid_indices = [idx for idx in list(indices)[:max_rows] if idx in df.index]
    if not valid_indices: return "Specified indices not found in the dataframe."
    df_subset = df.loc[valid_indices].copy()
    for col in df_subset.select_dtypes(include=[np.number]).columns:
        df_subset[col] = df_subset[col].apply(format_value_for_display)
    try: return df_subset.to_markdown(index=True)
    except ImportError: return df_subset.to_string()

def _get_basic_llm_fallback_analysis(
    id_cols_analyzed_results, null_value_summary,
    drift_results, stat_anomalies_by_row, if_anomalies_df,
    # Removed unused df_new, historic_stats_all_cols, new_numerical_summaries, alpha from fallback
    # as they are not directly used to generate basic strings here.
    analyzed_num_cols, analyzed_cat_cols
):
    basic_issues = []
    drift_issues_sorted = sorted([r for r in drift_results if r and r.get('drift_detected')], key=lambda x: x.get('drift_score', 0), reverse=True)
    for item in drift_issues_sorted[:2]:
        basic_issues.append({'issue': f"Significant change in '{item['column']}'", 'reasoning': f"High change severity ({item.get('drift_score',0):.2f}). {item.get('simple_result', '')}", 'recommendation': f"Investigate source of '{item['column']}' changes."})

    if any(stat_anomalies_by_row.values()) or (if_anomalies_df is not None and not if_anomalies_df.empty and if_anomalies_df['is_anomaly'].any()):
         basic_issues.append({'issue': "Potential unusual data points found.", 'reasoning': "Statistical or model-based flags.", 'recommendation': "Review detailed anomaly report for specific rows/columns."})

    for col, null_info in list(null_value_summary.items())[:2]:
        if "Notable Increase" in null_info.get('change_in_null_percentage_points',''):
             basic_issues.append({'issue': f"Notable increase in missing values for '{col}'", 'reasoning': f"New: {null_info['new_null_percentage']}, Change: {null_info['change_in_null_percentage_points']}", 'recommendation': f"Investigate cause of missing data for '{col}'."})
    for col, id_info in list(id_cols_analyzed_results.items())[:1]:
        if id_info.get('missing_values',0) > 0 or not id_info.get('all_unique_among_non_null', True):
            basic_issues.append({'issue': f"Potential issue with ID column '{col}'", 'reasoning': f"Missing: {id_info['missing_values']}, Duplicates: {id_info.get('duplicate_values_among_non_null',0)}", 'recommendation': f"Verify integrity of ID column '{col}'."})

    per_column_fallback = []
    for col in (analyzed_num_cols + analyzed_cat_cols):
        per_column_fallback.append({
            "column_name": col,
            "comparative_analysis": "LLM analysis skipped. Refer to raw drift/summary tables.",
            "anomaly_assessment": {
                "has_anomalies": False, # Default, cannot determine without LLM
                "issue_description": None,
                "suggested_fix": None,
                "status_message": "LLM analysis skipped for detailed anomaly assessment."
            }
        })

    return {"summary": "LLM analysis failed or was skipped. Basic prioritization and raw data sections provided.",
            "prioritized_issues": basic_issues[:5],
            "per_column_analysis": per_column_fallback,
            "id_column_analysis_summary": "LLM analysis skipped. Refer to raw ID checks table.",
            "null_value_analysis_summary": "LLM analysis skipped. Refer to raw null value comparison table."
            }

def get_llm_analysis(
    df_new, id_cols_analyzed_results, null_value_summary,
    drift_results, stat_anomalies_by_row, if_anomalies_df,
    historic_stats_all_cols, new_numerical_summaries,
    analyzed_num_cols, analyzed_cat_cols, alpha=ALPHA
):
    if not USE_GEMINI or not gemini_model:
        return _get_basic_llm_fallback_analysis(
            id_cols_analyzed_results, null_value_summary, drift_results,
            stat_anomalies_by_row, if_anomalies_df, analyzed_num_cols, analyzed_cat_cols
        )

    # --- Prepare LLM Prompt ---
    id_integrity_prompt = "--- ID Column Integrity Checks (New Data) ---\n"
    id_integrity_prompt += "Column | Missing | Missing % | Duplicates (Non-Null) | Consistent Length (if applicable)\n"
    id_integrity_prompt += "-------|---------|-----------|-------------------------|-------------------------------\n"
    if id_cols_analyzed_results:
        for col, res in id_cols_analyzed_results.items():
            id_integrity_prompt += f"{res['column_name']} | {res['missing_values']} | {res['missing_percentage']} | {res.get('duplicate_values_among_non_null', 'N/A')} | {res.get('consistent_length', 'N/A')}\n"
    else: id_integrity_prompt += "No ID columns identified or checked.\n"

    null_summary_prompt = "--- Null Value Analysis (New Data vs. Historic DB Stats) ---\n"
    null_summary_prompt += "Column | New Null % | Hist Null % | Change in Null %\n"
    null_summary_prompt += "-------|--------------|---------------|------------------\n"
    if null_value_summary:
        for col, comp in null_value_summary.items():
            null_summary_prompt += f"{col} | {comp['new_null_percentage']} | {comp['historic_null_percentage']} | {comp['change_in_null_percentage_points']}\n"
    else: null_summary_prompt += "No null value comparison performed.\n"

    num_summary_prompt = "--- Numerical Data Summary (Historic Stats from DB vs New Data) ---\n"
    num_summary_prompt += "Column | Hist Avg | Hist Spread (StdDev) | Hist Range (Min-Max) | New Avg | New Spread (StdDev) | New Range (Min-Max)\n"
    num_summary_prompt += "-------|----------|----------------------|----------------------|---------|---------------------|--------------------\n"
    if new_numerical_summaries:
        for col in analyzed_num_cols:
            new_s = new_numerical_summaries.get(col, {})
            hist_s = historic_stats_all_cols.get(col, {})
            h_avg = format_value_for_display(hist_s.get('mean'))
            h_std = format_value_for_display(hist_s.get('std'))
            h_range = f"{format_value_for_display(hist_s.get('min'))}-{format_value_for_display(hist_s.get('max'))}" if pd.notna(hist_s.get('min')) and pd.notna(hist_s.get('max')) else 'N/A'
            n_avg = format_value_for_display(new_s.get('new_mean'))
            n_std = format_value_for_display(new_s.get('new_std'))
            n_range = f"{format_value_for_display(new_s.get('new_min'))}-{format_value_for_display(new_s.get('new_max'))}" if pd.notna(new_s.get('new_min')) and pd.notna(new_s.get('new_max')) else 'N/A'
            num_summary_prompt += f"{col} | {h_avg} | {h_std} | {h_range} | {n_avg} | {n_std} | {n_range}\n"
    else: num_summary_prompt += "No numerical summaries available for new data.\n"

    significant_drift_for_prompt = sorted([r for r in drift_results if r and r.get('drift_detected')], key=lambda x: x.get('drift_score', 0), reverse=True)
    all_drift_for_prompt = sorted([r for r in drift_results if r], key=lambda x: x.get('column')) # For comprehensive check

    drift_context_prompt = f"--- Data Distribution Changes Summary (Top {MAX_DRIFT_COLS_TO_LLM_SUMMARY} significant changes shown, then others) ---\n"
    drift_context_prompt += "Column | Type | Change Result | Change Severity (0-1 if drifted)\n"
    drift_context_prompt += "-------|------|---------------|--------------------------------\n"
    if not all_drift_for_prompt: drift_context_prompt += "No drift analysis results available for any columns.\n"
    else:
        temp_drift_list = []
        # Add significant drifts first
        for item in significant_drift_for_prompt[:MAX_DRIFT_COLS_TO_LLM_SUMMARY]:
            temp_drift_list.append(f"{item['column']} | {item.get('type','N/A')} | {item.get('simple_result','N/A')} | {item.get('drift_score',0):.3f}\n")
        
        # Add other non-drifting or less significant drifting columns
        other_drift_count = 0
        for item in all_drift_for_prompt:
            if not item.get('drift_detected') or item not in significant_drift_for_prompt[:MAX_DRIFT_COLS_TO_LLM_SUMMARY]:
                 if other_drift_count < (MAX_DRIFT_COLS_TO_LLM_SUMMARY - len(significant_drift_for_prompt[:MAX_DRIFT_COLS_TO_LLM_SUMMARY])) or MAX_DRIFT_COLS_TO_LLM_SUMMARY == 0 : # show some non-drifts too
                    temp_drift_list.append(f"{item['column']} | {item.get('type','N/A')} | {item.get('simple_result','N/A')} | N/A (no drift or below top)\n")
                    other_drift_count +=1
        drift_context_prompt += "".join(temp_drift_list)
        total_drift_items = len(all_drift_for_prompt)
        shown_drift_items = len(temp_drift_list)
        if total_drift_items > shown_drift_items :
             drift_context_prompt += f"... (and {total_drift_items - shown_drift_items} more columns not detailed here)\n"


    all_anomalies_for_llm = []
    for idx, anoms in stat_anomalies_by_row.items():
        for anom in anoms: all_anomalies_for_llm.append({'index': idx, 'column': anom['column'], 'value': format_value_for_display(anom.get('value')), 'method': anom.get('simple_method','Stat'), 'score': anom.get('severity_score',0)})
    if if_anomalies_df is not None and not if_anomalies_df.empty:
        for _, row in if_anomalies_df[if_anomalies_df['is_anomaly']].iterrows():
            all_anomalies_for_llm.append({'index': row['index_in_new'], 'column': 'Multiple (IF)', 'value': 'N/A', 'method': 'Combined Features', 'score': row.get('severity_score',0)})
    all_anomalies_for_llm.sort(key=lambda x: x['score'], reverse=True)
    anomaly_context_prompt = "--- Unusual Data Points (Anomalies) - Context Snippet ---\n"
    if not all_anomalies_for_llm: anomaly_context_prompt += "No significant unusual data points detected by statistical or model-based checks.\n"
    else:
        anomaly_context_prompt += f"Top {MAX_ANOMALIES_TO_LLM_SNIPPET} raw anomaly flags (by severity):\n"
        anomaly_context_prompt += "Row Index | Column(s) | Reason Flagged | Value | Severity (0-1)\n"
        anomaly_context_prompt += "----------|-----------|----------------|-------|----------------\n"
        for anom in all_anomalies_for_llm[:MAX_ANOMALIES_TO_LLM_SNIPPET]:
            val_str = str(anom['value']); val_str = (val_str[:25] + '...') if len(val_str) > 25 else val_str
            anomaly_context_prompt += f"{anom['index']} | {anom['column']} | {anom['method']} | {val_str} | {anom['score']:.3f}\n"

    anomalous_indices_for_snippet = sorted(list(set(a['index'] for a in all_anomalies_for_llm[:MAX_ANOMALIES_TO_LLM_SNIPPET])))
    anomalous_rows_data_prompt = "Data for Top Anomalous Rows (up to {} rows):\n".format(MAX_ANOMALIES_TO_LLM_SNIPPET)
    anomalous_rows_data_prompt += format_data_for_llm(df_new, anomalous_indices_for_snippet, max_rows=MAX_ANOMALIES_TO_LLM_SNIPPET)
    
    columns_for_llm_per_col_analysis = sorted(list(set(analyzed_num_cols + analyzed_cat_cols))) # Ensure unique and sorted

    prompt = f"""
You are an expert AI data analyst. Your goal is to provide clear, actionable insights about a 'new' dataset compared to 'historic' database statistics which is coming from a postgres database. 
Use simple, business-friendly language. Integers should be shown without decimals.

--- Analysis Context & Provided Information ---
New dataset shape: {df_new.shape}
Columns analyzed (excluding IDs): Numerical: {analyzed_num_cols}, Categorical (incl. text-like): {analyzed_cat_cols}
Significance level for statistical tests (alpha): {alpha}

{id_integrity_prompt}
{null_summary_prompt}
{num_summary_prompt}
{drift_context_prompt}
{anomaly_context_prompt}
{anomalous_rows_data_prompt}

--- Your Task ---
Based ONLY on the information provided above, provide the following in the specified JSON format.

1.  **Overall Summary (key: "summary"):**
    *   A very brief (2-3 sentences) headline summary of the MOST CRITICAL data quality issues or changes found.

2.  **ID Column Analysis (key: "id_column_analysis_summary"):**
    *   Briefly summarize key findings from the 'ID Column Integrity Checks'. Highlight any IDs with significant missing values, duplicates, or format inconsistencies. If all good, state that.

3.  **Null Value Analysis (key: "null_value_analysis_summary"):**
    *   Briefly summarize findings from the 'Null Value Analysis'. Highlight columns with notable increases or decreases in missing data compared to historic levels.

4.  **Per-Column Detailed Analysis (key: "per_column_analysis"):**
    *   This should be a list of objects, one for each column in: {columns_for_llm_per_col_analysis}.
    *   For each column, provide an object with keys: "column_name", "comparative_analysis", "anomaly_assessment".
    *   **"column_name"**: The name of the column.
    *   **"comparative_analysis"**:
        *   For ID columns, state if they are unique or not. If not unique, mention the number of duplicates and if they are missing. Also check the format matches the historical data format. If everything is good, state that.
        *   For **numerical columns**, refer to the 'Numerical Data Summary' to explain how the new data's average, spread (StdDev), and min/max range compare to historic data.
        *   For **categorical columns**, refer to the 'Data Distribution Changes Summary' for this column. Check the demographics data on web and see if city matches the state. If not, showcase the rows in a table with anomalies mentioning suggested changes. Explain what its 'Change Result' means in simple terms. For example, if 'State' shows 'Category proportions changed significantly from historic,' you might elaborate by saying 'The popularity of different states has shifted compared to historic patterns.' If new categories appeared or historic ones vanished (indicated in 'Change Result'), mention that.
        *   If a column's 'Change Result' in the 'Data Distribution Changes Summary' indicates consistency (e.g., 'Mean is consistent with historic', 'Category proportions consistent with historic'), state that its distribution appears consistent.
        *   If the 'Change Result' indicates that historic data was unavailable or comparison was not possible (e.g., 'Historic stats unavailable/incomplete', 'Comparison N/A or insufficient data'), then state that a comparison to historic data could not be made for this column.
    *   **"anomaly_assessment"**: An object with keys: "has_anomalies" (boolean), "issue_description" (string), "suggested_fix" (string), "status_message" (string).
        *   **"has_anomalies"**: True if specific anomalies were flagged for THIS column (from 'Anomalous Data Points - Context Snippet') OR if 'Null Value Analysis' showed a 'Notable Increase' for THIS column.
        *   **"issue_description"**: If "has_anomalies" is true, explain *why* data points are unusual for *this specific column* (e.g., 'Values are far outside the historic range of X to Y', 'Unexpectedly high number of missing values compared to history', 'Category Z appeared which was not seen before'). Reference the 'Anomalous Data Points - Context Snippet' and 'Anomalous Row Data Snippet' if helpful. If column names suggest a relationship (e.g., 'City', 'State', 'ZipCode') and anomalies in those columns for the same row seem contradictory (e.g., 'City: Paris, State: Texas' in the 'Anomalous Row Data Snippet'), EXPLICITLY mention this as a potential 'logical data integrity issue'.
        *   **"suggested_fix"**: If "has_anomalies" is true, propose a brief, actionable step (e.g., 'Verify data entry for rows [indices if known, else "flagged rows"] in column [Column Name]', 'Investigate source system for recent changes to [Column Name]', 'If values are correct, update historic benchmarks', 'Review data capture process for increased nulls in [Column Name]').
        *   **"status_message"**: If "has_anomalies" is false, provide a positive confirmation like: "No specific row-level anomalies detected for this column. Data quality appears good regarding outliers or unexpected missingness beyond general null trends reported in the Null Value Analysis section." If there was distribution drift but no specific row anomalies, you can note that the distribution changed but individual data points within the new distribution are not flagged as outliers.

5.  **Prioritized List of Issues (key: "prioritized_issues"):**
    *   A list of 3-5 MOST CRITICAL issues overall (can be ID issues, null value surges, significant distribution changes, or widespread/severe anomalies).
    *   Each item in the list should be an object with keys: "issue" (brief description), "reasoning" (why it's important), "recommendation" (actionable next step).

Output Format: Ensure the entire response is a single, valid JSON object.
"""
    print("\n--- Calling Gemini API for detailed analysis ---")
    try:
        response = gemini_model.generate_content(prompt)
        cleaned_response = response.text.strip()
        match = re.search(r"```json\s*([\s\S]*?)\s*```", cleaned_response)
        json_str = match.group(1) if match else cleaned_response
        if not (json_str.startswith('{') and json_str.endswith('}')) and not (json_str.startswith('[') and json_str.endswith(']')):
            json_str = json_str[json_str.find("{") : json_str.rfind("}") + 1]

        llm_result = json.loads(json_str)
        required_keys = ["summary", "id_column_analysis_summary", "null_value_analysis_summary", "per_column_analysis", "prioritized_issues"]
        if not all(k in llm_result for k in required_keys):
             raise ValueError(f"LLM response missing one or more required top-level keys. Found: {list(llm_result.keys())}")
        if not isinstance(llm_result.get("per_column_analysis"), list):
            raise ValueError("'per_column_analysis' should be a list.")
        if llm_result.get("per_column_analysis"):
            first_col_analysis = llm_result["per_column_analysis"][0]
            if not all(k in first_col_analysis for k in ["column_name", "comparative_analysis", "anomaly_assessment"]):
                raise ValueError("LLM 'per_column_analysis' items missing required keys (column_name, comparative_analysis, anomaly_assessment).")
            if "anomaly_assessment" in first_col_analysis and isinstance(first_col_analysis["anomaly_assessment"], dict):
                if not all(k in first_col_analysis["anomaly_assessment"] for k in ["has_anomalies", "status_message"]): # issue_desc and fix can be null
                     raise ValueError("LLM 'anomaly_assessment' items missing required keys (has_anomalies, status_message).")
            else:
                raise ValueError("LLM 'anomaly_assessment' is missing or not a dictionary in 'per_column_analysis'.")


        print("--- Gemini Analysis Parsed Successfully ---")
        return llm_result
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode LLM response as JSON: {e}. Response was: \n{cleaned_response[:1000]}...")
    except Exception as e: # Includes genai API errors, ValueError from checks
        print(f"Error calling, processing, or validating Gemini API response: {e}")

    # Fallback if any error occurs during LLM interaction or parsing
    print("--- Using basic fallback analysis due to LLM error. ---")
    return _get_basic_llm_fallback_analysis(
        id_cols_analyzed_results, null_value_summary, drift_results,
        stat_anomalies_by_row, if_anomalies_df, analyzed_num_cols, analyzed_cat_cols
    )

# --- Reporting Function ---
def generate_report_html(
    llm_analysis_results, df_new,
    id_cols_analysis_raw, null_value_summary_raw, new_numerical_summaries_raw,
    drift_results_raw, stat_anomalies_by_row_raw, if_anomalies_df_raw,
    historic_stats_all_cols, analyzed_num_cols # Pass analyzed_num_cols for numerical summary
):
    report = "<h2>AI Analysis & Insights (Powered by Gemini)</h2>"
    report += f"<p><strong>Overall Summary:</strong> {llm_analysis_results.get('summary', 'LLM summary not available.')}</p>"

    report += "<h3>Prioritized Issues (AI Recommended)</h3>"
    prioritized = llm_analysis_results.get('prioritized_issues', [])
    if prioritized and isinstance(prioritized, list) and any(p.get('issue') for p in prioritized): # Check if list and has content
        report += "<table class='result-table'><thead><tr><th>Issue</th><th>Why it Matters</th><th>Recommendation</th></tr></thead><tbody>"
        for issue in prioritized:
            report += f"<tr><td>{issue.get('issue', 'N/A')}</td><td>{issue.get('reasoning', 'N/A')}</td><td>{issue.get('recommendation', 'N/A')}</td></tr>"
        report += "</tbody></table>"
    else: report += "<p>No prioritized issues provided by AI, or analysis used fallback. Check raw data sections.</p>"
    report += "<hr>"

    report += "<h2>ID Column Integrity Analysis</h2>"
    id_summary_llm = llm_analysis_results.get('id_column_analysis_summary', '')
    if id_summary_llm and "LLM analysis skipped" not in id_summary_llm : report += f"<p><strong>AI Summary:</strong> {id_summary_llm}</p>"
    else: report += f"<p><i>AI summary for ID columns not available or used fallback. Displaying raw checks:</i></p>"
    if id_cols_analysis_raw:
        report += "<table class='result-table'><thead><tr><th>ID Column</th><th>Missing</th><th>Missing %</th><th>Unique (Excl. NA)</th><th>Duplicates (Non-Null)</th><th>Common Length</th><th>Consistent Length</th></tr></thead><tbody>"
        for col, res in id_cols_analysis_raw.items():
            report += (f"<tr><td>{res['column_name']}</td><td>{res['missing_values']}</td><td>{res['missing_percentage']}</td>"
                       f"<td>{res.get('unique_values_excl_na', 'N/A')}</td><td>{res.get('duplicate_values_among_non_null', 'N/A')}</td>"
                       f"<td>{res.get('most_common_length', 'N/A')}</td><td>{res.get('consistent_length', 'N/A')}</td></tr>")
        report += "</tbody></table>"
    else: report += "<p>No ID columns identified or analyzed.</p>"
    report += "<hr>"

    report += "<h2>Null Value Analysis (New Data vs. Historic)</h2>"
    null_summary_llm = llm_analysis_results.get('null_value_analysis_summary', '')
    if null_summary_llm and "LLM analysis skipped" not in null_summary_llm: report += f"<p><strong>AI Summary:</strong> {null_summary_llm}</p>"
    else: report += f"<p><i>AI summary for null values not available or used fallback. Displaying raw checks:</i></p>"
    if null_value_summary_raw:
        report += "<table class='result-table'><thead><tr><th>Column</th><th>New Null Count</th><th>New Null %</th><th>Historic Null %</th><th>Change in Null % Points</th></tr></thead><tbody>"
        for col, comp in null_value_summary_raw.items():
            hl_class = ''
            if 'Notable Increase' in comp['change_in_null_percentage_points']: hl_class = 'highlight-increase'
            elif 'Notable Decrease' in comp['change_in_null_percentage_points']: hl_class = 'highlight-decrease' # Optional styling for decrease
            report += (f"<tr><td>{col}</td><td>{format_value_for_display(comp['new_null_count'])}</td><td>{comp['new_null_percentage']}</td>"
                       f"<td>{comp['historic_null_percentage']}</td><td class='{hl_class}'>{comp['change_in_null_percentage_points']}</td></tr>")
        report += "</tbody></table>"
    else: report += "<p>No null value comparison performed.</p>"
    report += "<hr>"

    report += "<h2>Per-Column Detailed Analysis & Anomaly Assessment (AI Insights)</h2>"
    per_column_llm = llm_analysis_results.get('per_column_analysis', [])
    if per_column_llm and isinstance(per_column_llm, list) and any("LLM analysis skipped" not in p.get("comparative_analysis","") for p in per_column_llm):
        for col_analysis in per_column_llm:
            col_name = col_analysis.get('column_name', 'N/A')
            report += f"<details><summary><strong>Column: {col_name}</strong></summary>"
            report += "<div class='column-analysis-box'>"
            report += f"<p><strong>Comparative Analysis (New vs. Historic):</strong><br>{col_analysis.get('comparative_analysis', 'Not provided.')}</p>"
            anomaly_assessment = col_analysis.get('anomaly_assessment', {})
            report += "<strong>Anomaly Assessment:</strong><br>"
            if anomaly_assessment.get('has_anomalies'):
                report += f"<p class='highlight-anomaly'><em>Issue Description:</em> {anomaly_assessment.get('issue_description', 'N/A')}</p>"
                report += f"<p class='highlight-fix'><em>Suggested Fix/Action:</em> {anomaly_assessment.get('suggested_fix', 'N/A')}</p>"
            elif anomaly_assessment.get('status_message'): # Check if status_message exists
                 report += f"<p class='highlight-good'>{anomaly_assessment.get('status_message')}</p>"
            else: # Fallback if no status_message and no anomalies
                 report += f"<p class='highlight-good'>No specific anomalies flagged for this column.</p>"

            report += "</div></details>"
    else: report += "<p>Per-column AI analysis not available or used fallback. See raw drift and anomaly sections below for details.</p>"
    report += "<hr>"

    report += "<h2>Numerical Data Summary (New Data vs. Historic DB Stats)</h2>"
    if new_numerical_summaries_raw and analyzed_num_cols: # Check if there are num_cols to display
        report += "<table class='result-table'><thead><tr><th>Column</th><th>Hist Avg</th><th>New Avg</th><th>Hist Spread (StdDev)</th><th>New Spread (StdDev)</th><th>Hist Range (Min-Max)</th><th>New Range (Min-Max)</th></tr></thead><tbody>"
        for col in analyzed_num_cols: # Iterate over the actual numerical columns analyzed
            new_s = new_numerical_summaries_raw.get(col, {})
            hist_s = historic_stats_all_cols.get(col, {})
            h_avg, n_avg = format_value_for_display(hist_s.get('mean')), format_value_for_display(new_s.get('new_mean'))
            h_std, n_std = format_value_for_display(hist_s.get('std')), format_value_for_display(new_s.get('new_std'))
            h_min, h_max = format_value_for_display(hist_s.get('min')), format_value_for_display(hist_s.get('max'))
            n_min, n_max = format_value_for_display(new_s.get('new_min')), format_value_for_display(new_s.get('new_max'))
            h_range = f"{h_min} to {h_max}" if h_min != "N/A" and h_max != "N/A" else "N/A"
            n_range = f"{n_min} to {n_max}" if n_min != "N/A" and n_max != "N/A" else "N/A"
            report += f"<tr><td>{col}</td><td>{h_avg}</td><td>{n_avg}</td><td>{h_std}</td><td>{n_std}</td><td>{h_range}</td><td>{n_range}</td></tr>"
        report += "</tbody></table>"
    elif not analyzed_num_cols: report += "<p>No numerical columns (excluding IDs) were identified for summary.</p>"
    else: report += "<p>No numerical summaries calculated for new data.</p>"
    report += "<hr>"

    report += "<h2>Data Distribution Change Details (Statistical Tests)</h2>"
    sig_drift = [r for r in drift_results_raw if r and r.get('drift_detected')]
    if sig_drift:
        report += f"<p>Found {len(sig_drift)} columns where data characteristics likely changed compared to historic patterns (based on statistical tests):</p>"
        report += "<table class='result-table'><thead><tr><th>Column</th><th>Data Type</th><th>Change Check Method</th><th>Result</th><th>Change Severity (0-1)</th></tr></thead><tbody>"
        for r in sorted(sig_drift, key=lambda x: x.get('drift_score', 0), reverse=True):
            report += f"<tr class='highlight-drift'><td>{r['column']}</td><td>{r.get('type','N/A')}</td><td>{r.get('simple_test_name','N/A')}</td><td>{r.get('simple_result','N/A')}</td><td>{r.get('drift_score',0):.3f}</td></tr>"
        report += "</tbody></table>"
    elif drift_results_raw: # Drift analysis was run but no significant drift
        report += "<p>No significant data distribution changes detected by statistical tests.</p>"
    else: # Drift analysis was not run or produced no results
        report += "<p>Drift analysis did not produce results or was not run for any columns.</p>"

    report += "<hr>"

    report += "<details><summary><strong>Show Raw Anomaly Flags (Technical Detail)</strong></summary>"
    report += "<div>"
    all_anoms_report = []
    for idx, anoms_list in stat_anomalies_by_row_raw.items():
        for anom_item in anoms_list: all_anoms_report.append({'index':idx, 'column':anom_item.get('column'), 'value':format_value_for_display(anom_item.get('value')), 'method':anom_item.get('simple_method','Stat'), 'score':anom_item.get('severity_score',0), 'context':anom_item.get('historic_context','N/A'), 'type':'Single Column'})
    if if_anomalies_df_raw is not None and not if_anomalies_df_raw.empty and 'is_anomaly' in if_anomalies_df_raw.columns:
        for _, row in if_anomalies_df_raw[if_anomalies_df_raw['is_anomaly']].iterrows():
            all_anoms_report.append({'index':row['index_in_new'], 'column':'Multiple (IF)', 'value':'N/A', 'method':'Combined Features', 'score':row.get('severity_score',0), 'context':f"IF Score: {row.get('anomaly_score',0):.3f}", 'type':'Multiple Columns'})
    all_anoms_report.sort(key=lambda x: x.get('score',0), reverse=True) # Add .get for score robustness

    if all_anoms_report:
        unique_anom_rows = len(set(a['index'] for a in all_anoms_report))
        report += f"<p>Found {len(all_anoms_report)} potential raw unusual data instances across {unique_anom_rows} unique rows.</p>"
        report += "<h5>Top Raw Unusual Data Instances (by Severity Score)</h5>"
        report += "<table class='result-table'><thead><tr><th>Row Index</th><th>Column(s)</th><th>Value</th><th>Reason Flagged</th><th>Severity (0-1)</th><th>Context/Details</th></tr></thead><tbody>"
        for anom in all_anoms_report[:15]:
             hl_class = 'highlight-anomaly-high' if anom.get('score',0) > 0.7 else 'highlight-anomaly-med'
             report += f"<tr class='{hl_class}'><td>{anom['index']}</td><td>{anom['column']}</td><td>{anom['value']}</td><td>{anom['method']} ({anom['type']})</td><td>{anom.get('score',0):.3f}</td><td>{anom['context']}</td></tr>"
        report += "</tbody></table>"
        if len(all_anoms_report) > 15: report += f"<p>... and {len(all_anoms_report) - 15} more raw instances.</p>"
    else: report += "<p>No raw unusual data points flagged by statistical or Isolation Forest methods.</p>"
    anom_indices_for_display = sorted(list(set(a['index'] for a in all_anoms_report)))[:MAX_ANOMALIES_TO_LLM_SNIPPET]
    if anom_indices_for_display:
        report += f"<h6>Data for Top {len(anom_indices_for_display)} Rows with Highest Raw Anomaly Flags</h6>"
        valid_indices_display = [idx for idx in anom_indices_for_display if idx in df_new.index]
        if valid_indices_display:
             report += "<div class='preview-table-container' style='max-height: 300px;'>"
             try:
                 df_display_anom = df_new.loc[valid_indices_display].copy()
                 for col_anom_disp in df_display_anom.select_dtypes(include=[np.number]).columns: # Format numbers
                     df_display_anom[col_anom_disp] = df_display_anom[col_anom_disp].apply(format_value_for_display)
                 report += df_display_anom.to_html(classes='preview-table', index=True, border=0, na_rep='NA', escape=False)
             except Exception as table_err: report += f"<p>Error generating table: {table_err}</p>"
             report += "</div>"
    report += "</div></details><hr><p><em>End of Report</em></p>"
    return report

# --- Main Analysis Function ---
def run_analysis(new_filepath, historic_sample_filepath=None):
    db_conn = None
    try:
        db_conn = get_db_connection()
        df_new = load_new_data(new_filepath)
        df_new_filled_for_drift = handle_missing_values(df_new.copy())
        df_historic_sample = None
        if historic_sample_filepath:
            try:
                df_historic_sample_raw = pd.read_csv(historic_sample_filepath)
                common_cols_sample = df_new.columns.intersection(df_historic_sample_raw.columns)
                if not common_cols_sample.empty:
                    df_historic_sample = df_historic_sample_raw[common_cols_sample].copy()
                    print(f"Loaded and harmonized historic sample data: {df_historic_sample.shape}")
                else: print("Warning: No common columns between new data and historic sample.")
            except Exception as e: print(f"Warning: Could not load/process historic sample CSV: {e}.")

        numerical_cols_for_analysis, categorical_cols_for_analysis, id_cols = \
            identify_column_types_and_ids(df_new)
        
        all_analyzed_cols = numerical_cols_for_analysis + categorical_cols_for_analysis
        print(f"Columns for full analysis (non-ID): Numerical: {numerical_cols_for_analysis}, Categorical: {categorical_cols_for_analysis}")

        id_column_analysis_results = perform_id_column_checks(df_new, id_cols)
        if id_cols: print(f"Performed integrity checks on ID columns: {id_cols}")

        historic_stats_all_cols = {}
        if db_conn and HISTORIC_TABLE_NAME:
            columns_for_db_stats = {col: 'numerical' for col in numerical_cols_for_analysis}
            columns_for_db_stats.update({col: 'categorical' for col in categorical_cols_for_analysis}) # Includes text-like
            if columns_for_db_stats:
                print(f"Fetching historic stats from DB for analytical columns: {list(columns_for_db_stats.keys())}")
                historic_stats_all_cols = fetch_historic_stats_from_db(db_conn, HISTORIC_TABLE_NAME, columns_for_db_stats)
                # DEBUG: Print fetched historic stats for a sample categorical column if present
                # sample_cat_col_for_debug = next((c for c in categorical_cols_for_analysis if c in historic_stats_all_cols), None)
                # if sample_cat_col_for_debug:
                #    print(f"DEBUG: Historic stats for '{sample_cat_col_for_debug}': {historic_stats_all_cols[sample_cat_col_for_debug]}")
            else: print("No non-ID numerical or categorical columns identified to fetch historic stats for.")
        else: print("Skipping fetch of historic stats from DB (connection, table name issue, or no relevant columns).")

        new_numerical_summaries = calculate_new_numerical_summaries(df_new, numerical_cols_for_analysis)
        null_value_summary = calculate_null_value_comparison(df_new, historic_stats_all_cols, all_analyzed_cols)

        drift_results = []
        for col in numerical_cols_for_analysis:
            hist_col_s = historic_stats_all_cols.get(col, {})
            result_drift = detect_numerical_drift_vs_stats(df_new_filled_for_drift[col], hist_col_s, col)
            if result_drift: drift_results.append(result_drift)
        for col in categorical_cols_for_analysis: # Includes text-like
            hist_col_s = historic_stats_all_cols.get(col, {})
            result_drift = detect_categorical_drift_vs_stats(df_new_filled_for_drift[col], hist_col_s, col)
            if result_drift: drift_results.append(result_drift)
        
        # DEBUG: Print drift results for a sample categorical column
        sample_cat_col_for_drift_debug = next((c for c in categorical_cols_for_analysis), None)
        if sample_cat_col_for_drift_debug:
           drift_res_debug = [r for r in drift_results if r['column'] == sample_cat_col_for_drift_debug]
           if drift_res_debug:
               print(f"DEBUG: Drift result for '{sample_cat_col_for_drift_debug}': {drift_res_debug[0]}")


        if_anomalies_df = detect_anomalies_isolation_forest(
            df_historic_sample, df_new_filled_for_drift, numerical_cols_for_analysis
        )
        stat_anomalies_by_row = detect_statistical_anomalies_vs_stats(
            df_new, historic_stats_all_cols, numerical_cols_for_analysis
        )

        llm_analysis_results = get_llm_analysis(
            df_new, id_column_analysis_results, null_value_summary,
            drift_results, stat_anomalies_by_row, if_anomalies_df,
            historic_stats_all_cols, new_numerical_summaries,
            numerical_cols_for_analysis, categorical_cols_for_analysis, alpha=ALPHA
        )

        report_html = generate_report_html(
            llm_analysis_results, df_new,
            id_column_analysis_results, null_value_summary, new_numerical_summaries,
            drift_results, stat_anomalies_by_row, if_anomalies_df,
            historic_stats_all_cols, numerical_cols_for_analysis # Pass numerical_cols for report
        )

        new_preview_html = get_preview_table_html(df_new, title="New Data Preview")
        hist_s_preview_title = "Historic Sample Data Preview" if historic_sample_filepath else "No Historic Sample Provided for Preview"
        historic_preview_html = get_preview_table_html(df_historic_sample, title=hist_s_preview_title)

        return report_html, historic_preview_html, new_preview_html

    except (ValueError, RuntimeError, FileNotFoundError) as e:
        print(f"CRITICAL ANALYSIS ERROR: {e}")
        import traceback; traceback.print_exc()
        error_html = f"<h2>Analysis Failed Critically</h2><hr><p><strong>Error Type: {type(e).__name__}</strong></p><p><strong>Message:</strong> {e}</p><p>Please check inputs and server logs.</p>"
        return error_html, "<p>Preview N/A</p>", "<p>Preview N/A</p>"
    except Exception as e:
        print(f"UNEXPECTED CRITICAL ANALYSIS ERROR: {e}")
        import traceback; traceback.print_exc()
        error_html = f"<h2>Analysis Failed Unexpectedly</h2><hr><p><strong>An unexpected critical error occurred:</strong> {e}</p><p>Please check server logs.</p>"
        return error_html, "<p>Preview N/A</p>", "<p>Preview N/A</p>"
    finally:
        if db_conn:
            db_conn.close()
            print("Database connection closed.")