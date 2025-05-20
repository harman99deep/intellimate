# analysis_engine.py

import pandas as pd
import numpy as np
from scipy.stats import iqr as scipy_iqr, ttest_ind, chisquare
import google.generativeai as genai
import os
import warnings
from dotenv import load_dotenv
import json
import re
from collections import defaultdict
import psycopg2
import uuid
from datetime import datetime, timezone, date
import decimal
import html

# --- Configuration ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD = (
    os.getenv("DB_HOST"), os.getenv("DB_PORT"), os.getenv("DB_NAME"),
    os.getenv("DB_USER"), os.getenv("DB_PASSWORD")
)
HISTORIC_TABLE_NAME = os.getenv("HISTORIC_TABLE_NAME")
ANALYSIS_LOG_TABLE_NAME = os.getenv("ANALYSIS_LOG_TABLE_NAME", "data_analysis_logs")

USE_GEMINI = False
gemini_model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        USE_GEMINI = True
        print("Gemini AI configured successfully.")
    except Exception as e:
        print(f"Error configuring Gemini AI: {e}. AI Explanations will be limited.")
else:
    print("Gemini API Key not found. AI Explanations will be limited.")

# --- Constants ---
ALPHA = 0.05
Z_SCORE_THRESHOLD = 3.0
MAX_ROWS_FOR_LLM_DATA_PREVIEW = 5
MAX_ANOMALY_ROWS_TO_SHOW_LLM = 10 
MAX_ANOMALY_ROWS_IN_REPORT_SNIPPET = 5 
MAX_FULL_ANOMALY_ROWS_IN_REPORT = 50 
HISTORIC_SAMPLE_ROWS_FOR_CONTEXT = 50
PREVIEW_TABLE_MAX_ROWS = 10

# --- Anomaly Types ---
ANOMALY_TYPE_MISSING = "missing"
ANOMALY_TYPE_INCORRECT_FORMAT = "incorrect_format"
ANOMALY_TYPE_DATA_MISMATCH = "data_mismatch"
ANOMALY_TYPE_LOCATION_MISMATCH = "location_mismatch"

# Categories for CSS cell/card styling (simplified)
ANOMALY_CATEGORY_MISSING = "missing"  # For missing or null values
ANOMALY_CATEGORY_FORMAT = "format"    # For format/validation issues
ANOMALY_CATEGORY_MISMATCH = "mismatch" # For data mismatches and inconsistencies
ANOMALY_CATEGORY_SUMMARY = "summary"   # For the summary flags column

ANOMALY_TYPE_TO_CSS_CATEGORY_MAP = {
    ANOMALY_TYPE_MISSING: ANOMALY_CATEGORY_MISSING,
    ANOMALY_TYPE_INCORRECT_FORMAT: ANOMALY_CATEGORY_FORMAT,
    ANOMALY_TYPE_DATA_MISMATCH: ANOMALY_CATEGORY_MISMATCH,
    ANOMALY_TYPE_LOCATION_MISMATCH: ANOMALY_CATEGORY_MISMATCH,
}

def get_anomaly_css_category(anomaly_type_str):
    return ANOMALY_TYPE_TO_CSS_CATEGORY_MAP.get(anomaly_type_str, "unknown")

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy.stats._stats_py')

def _to_native_py_type(value):
    if pd.isna(value): return None
    if isinstance(value, (int, float, bool, str, type(None))): return value
    if isinstance(value, (np.integer, np.int64, np.int32, np.int16, np.int8)): return int(value)
    if isinstance(value, (np.floating, np.float64, np.float32)):
        if pd.isna(value): return None
        if value == int(value): return int(value)
        return float(value)
    if isinstance(value, np.bool_): return bool(value)
    if isinstance(value, (datetime, pd.Timestamp, date)): return value.isoformat()
    if isinstance(value, uuid.UUID): return str(value)
    if isinstance(value, decimal.Decimal):
        if value % 1 == 0: return int(value)
        return float(value)
    if isinstance(value, np.ndarray): return [_to_native_py_type(x) for x in value.tolist()]
    try: return str(value)
    except Exception: return f"Unconvertible_Type_{type(value).__name__}"

def _sanitize_for_json(item):
    if isinstance(item, dict):
        return {
            (str(_to_native_py_type(k)) if not isinstance(k, (str, int, float, bool, type(None))) else k): _sanitize_for_json(v)
            for k, v in item.items()
        }
    elif isinstance(item, list):
        return [_sanitize_for_json(i) for i in item]
    return _to_native_py_type(item)

def format_value_for_display(native):
    """Format a value for display in the HTML report, safely escaping HTML but not quotes."""
    if native is None:
        return "NULL"
    if isinstance(native, (int, float)):
        return str(native)
    return html.escape(str(native), quote=False)  # Don't escape quotes, but still escape other HTML characters

def format_data_for_llm(df, max_rows=MAX_ROWS_FOR_LLM_DATA_PREVIEW):
    if df is None or df.empty:
        return "No data sample to display."
    try:
        df_to_format = df.copy()
        if 'original_index' in df_to_format.columns:
            df_to_format['original_index'] = df_to_format['original_index'].apply(_to_native_py_type).astype(str)
        
        df_display = df_to_format.head(max_rows).copy()
        for col in df_display.columns:
            df_display[col] = df_display[col].apply(lambda x: "NULL" if pd.isna(x) else str(_to_native_py_type(x)))
        try:
            import tabulate
            return df_display.to_markdown(index=('original_index' in df_display.columns), tablefmt="pipe")
        except ImportError:
            return df_display.to_string(index=('original_index' in df_display.columns))
    except Exception as e:
        print(f"Error formatting data for LLM (format_data_for_llm): {e}")
        return f"Error formatting data for LLM: {str(e)}"

def get_preview_table_html(df, title="Data Preview", max_rows=PREVIEW_TABLE_MAX_ROWS, table_id=None, categorized_anomalies_for_preview=None):
    if df is None or df.empty:
        return f"<div class='preview-card'><p class='preview-title'>{title}: No data to display.</p></div>"
    try:
        df_display_orig = df.head(max_rows).copy()
        
        orig_idx_col_name_in_df = None
        if 'original_index' in df_display_orig.columns:
            df_display_with_hidden_idx = df_display_orig.copy()
            orig_idx_col_name_in_df = 'original_index'
        elif df_display_orig.index.name == 'original_index':
            df_display_with_hidden_idx = df_display_orig.reset_index()
            orig_idx_col_name_in_df = 'original_index'
        else: 
            df_display_with_hidden_idx = df_display_orig.reset_index().rename(columns={'index': 'original_index_temp'})
            orig_idx_col_name_in_df = 'original_index_temp'
            # print(f"Warning: 'original_index' was not found directly for preview table '{title}'. Using temporary '{orig_idx_col_name_in_df}'.")
        
        df_display_for_html = df_display_with_hidden_idx.drop(columns=[orig_idx_col_name_in_df], errors='ignore')

        if categorized_anomalies_for_preview:
            df_display_for_html['anomaly_flags'] = "No anomalies detected" 
            for display_row_idx, hidden_row_series in df_display_with_hidden_idx.iterrows():
                original_idx_native = _to_native_py_type(hidden_row_series.get(orig_idx_col_name_in_df))
                if original_idx_native is not None and original_idx_native in categorized_anomalies_for_preview:
                    anomalies = categorized_anomalies_for_preview[original_idx_native]
                    if anomalies:
                        # For flags column, use broader categories if desired, or stick to CSS categories.
                        # Let's use CSS categories for flags to align with potential cell highlights.
                        types = sorted(list(set(get_anomaly_css_category(a['type']) for a in anomalies)))
                        if display_row_idx in df_display_for_html.index:
                             df_display_for_html.loc[display_row_idx, 'anomaly_flags'] = ", ".join(t.replace("-", " ").title() for t in types)

        html_rows_content = []
        header_cols = df_display_for_html.columns
        header_html = "<tr>" + "".join(f"<th>{html.escape(str(col))}</th>" for col in header_cols) + "</tr>"

        for display_row_idx, row_series_for_html in df_display_for_html.iterrows():
            original_row_idx_native = _to_native_py_type(df_display_with_hidden_idx.loc[display_row_idx, orig_idx_col_name_in_df])
            html_row = "<tr>"
            for col_name in header_cols:
                cell_value = row_series_for_html.get(col_name)
                cell_classes = ["preview-cell"]
                
                # Skip anomaly styling for city column
                if col_name != 'city' and categorized_anomalies_for_preview and original_idx_native is not None and original_idx_native in categorized_anomalies_for_preview:
                    for anomaly in categorized_anomalies_for_preview[original_idx_native]:
                        anomaly_col = anomaly.get('column')
                        llm_details = anomaly.get('details', {}) 
                        
                        applies_to_col = False
                        if anomaly_col == col_name: applies_to_col = True
                        elif isinstance(llm_details, dict):
                            if isinstance(llm_details.get('involved_columns'), dict) and col_name in llm_details['involved_columns']: applies_to_col = True
                            elif isinstance(llm_details.get('related_columns'), list) and col_name in llm_details['related_columns']: applies_to_col = True
                        
                        if applies_to_col:
                            anomaly_category_for_cell = get_anomaly_css_category(anomaly['type']) # Use CSS category for cell class
                            cell_classes.append(f"cell-anomaly-{anomaly_category_for_cell}")
                
                formatted_val = format_value_for_display(cell_value)
                if col_name == 'anomaly_flags' and formatted_val != "No anomalies detected": 
                    cell_classes.append("cell-anomaly-summary")

                html_row += f"<td class='{' '.join(sorted(list(set(cell_classes))))}'>{formatted_val}</td>"
            html_row += "</tr>"
            html_rows_content.append(html_row)

        table_id_attr = f"id='{table_id}'" if table_id else ""
        final_html = f"<div class='preview-card'><p class='preview-title'>{title}</p>"
        final_html += f"<div class='table-responsive-wrapper'><table {table_id_attr} class='preview-table table table-striped table-sm table-hover'>"
        final_html += f"<thead>{header_html}</thead><tbody>{''.join(html_rows_content)}</tbody></table></div></div>"
        return final_html
    except Exception as e:
        import traceback
        print(f"Error in get_preview_table_html for {title}: {e}\n{traceback.format_exc()}")
        return f"<div class='preview-card error'><p class='preview-title'>Error generating preview for {title}</p><p>{html.escape(str(e))}</p></div>"

def get_db_connection():
    if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD]): return None
    try: return psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD)
    except psycopg2.Error as e: print(f"DB Connection Error: {e}"); return None

def execute_db_query(db_conn, query, params=None, fetch_one=False, fetch_all=False):
    if not db_conn: return None
    try:
        with db_conn.cursor() as cursor:
            cursor.execute(query, params)
            if fetch_one: return cursor.fetchone()
            if fetch_all: return cursor.fetchall()
            db_conn.commit(); return True
    except psycopg2.Error as e: print(f"DB Query Error: {e} (Query: {query[:100]}...)"); db_conn.rollback(); return None

_stats_cache = {}
def _get_cached_or_compute(key, compute_fn, *args, **kwargs):
    if key not in _stats_cache: _stats_cache[key] = compute_fn(*args, **kwargs)
    return _stats_cache[key]

def ensure_log_table_exists(db_conn):
    query = f"""CREATE TABLE IF NOT EXISTS "{ANALYSIS_LOG_TABLE_NAME}" (log_id UUID PRIMARY KEY, run_timestamp TIMESTAMPTZ NOT NULL, status VARCHAR(100) NOT NULL, error_message TEXT, new_data_filename VARCHAR(255), analysis_summary TEXT);"""
    execute_db_query(db_conn, query)

def log_analysis_run_db(db_conn, log_id, status, error_msg=None, filename=None, summary=None):
    query = f"""INSERT INTO "{ANALYSIS_LOG_TABLE_NAME}" (log_id, run_timestamp, status, error_message, new_data_filename, analysis_summary) VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (log_id) DO UPDATE SET run_timestamp = EXCLUDED.run_timestamp, status = EXCLUDED.status, error_message = EXCLUDED.error_message, new_data_filename = EXCLUDED.new_data_filename, analysis_summary = EXCLUDED. ;"""
    params = (str(log_id), datetime.now(timezone.utc), str(status)[:100], str(error_msg)[:10000] if error_msg else None, str(filename)[:255] if filename else None, json.dumps(_sanitize_for_json(summary))[:20000] if summary else None)
    execute_db_query(db_conn, query, params)

def fetch_historic_data(db_conn, table_name, num_sample_rows=HISTORIC_SAMPLE_ROWS_FOR_CONTEXT):
    if not db_conn or not table_name: return None, pd.DataFrame(), {}
    total_rows_res = execute_db_query(db_conn, f'SELECT COUNT(*) FROM "{table_name}";', fetch_one=True)
    historic_total_rows = total_rows_res[0] if total_rows_res else None
    sample_df = pd.DataFrame(); schema_info_rows = execute_db_query(db_conn, f"SELECT column_name, data_type FROM information_schema.columns WHERE table_schema = 'public' AND table_name = %s;", (table_name,), fetch_all=True)
    if not schema_info_rows: print(f"Could not retrieve schema for historic table '{table_name}'."); return historic_total_rows, sample_df, {}
    db_column_names = [row[0] for row in schema_info_rows]; safe_select_cols = ", ".join([f'"{col}"' for col in db_column_names])
    sample_rows_query = f'SELECT {safe_select_cols} FROM "{table_name}" ORDER BY RANDOM() LIMIT %s;'; sample_rows = execute_db_query(db_conn, sample_rows_query, (num_sample_rows,), fetch_all=True)
    if sample_rows:
        sample_df = pd.DataFrame(sample_rows, columns=db_column_names)
        for col_name, col_type_str in schema_info_rows:
            if col_name in sample_df.columns:
                if any(t in col_type_str for t in ['integer', 'numeric', 'real', 'double precision', 'smallint', 'bigint']): sample_df[col_name] = pd.to_numeric(sample_df[col_name], errors='coerce')
                elif any(t in col_type_str for t in ['timestamp', 'date']): sample_df[col_name] = pd.to_datetime(sample_df[col_name], errors='coerce')
    historic_column_stats = {}
    for col_name, col_type_str in schema_info_rows:
        safe_col = f'"{col_name}"'; stats = {'type': col_type_str}
        null_res = execute_db_query(db_conn, f"SELECT SUM(CASE WHEN {safe_col} IS NULL THEN 1 ELSE 0 END), COUNT(*) FROM \"{table_name}\";", fetch_one=True)
        if null_res: stats.update({'null_count': _to_native_py_type(null_res[0]), 'total_in_col': _to_native_py_type(null_res[1])})
        if any(t in col_type_str for t in ['integer', 'numeric', 'real', 'double precision']):
            num_res = execute_db_query(db_conn, f"SELECT AVG(CAST({safe_col} AS NUMERIC)), STDDEV_SAMP(CAST({safe_col} AS NUMERIC)), MIN(CAST({safe_col} AS NUMERIC)), MAX(CAST({safe_col} AS NUMERIC)) FROM \"{table_name}\";", fetch_one=True)
            if num_res: stats.update({'mean': _to_native_py_type(num_res[0]), 'std': _to_native_py_type(num_res[1]), 'min': _to_native_py_type(num_res[2]), 'max': _to_native_py_type(num_res[3])})
        else: 
            cat_res = execute_db_query(db_conn, f"SELECT CAST({safe_col} AS TEXT), COUNT(*) FROM \"{table_name}\" GROUP BY CAST({safe_col} AS TEXT) ORDER BY COUNT(*) DESC LIMIT 5;", fetch_all=True)
            if cat_res: stats['top_values'] = {_to_native_py_type(k): _to_native_py_type(v) for k,v in cat_res}
        historic_column_stats[col_name] = stats
    return historic_total_rows, sample_df, _sanitize_for_json(historic_column_stats)

def load_new_data(filepath):
    try:
        try: df_sample_for_dtypes = pd.read_csv(filepath, nrows=1000, low_memory=False)
        except pd.errors.EmptyDataError: raise ValueError("CSV is empty.")
        except Exception as e: print(f"Warning: Could not read 1000 rows for dtype sniffing, falling back. Error: {e}"); df_sample_for_dtypes = pd.read_csv(filepath, nrows=100, low_memory=False)
        inferred_dtypes = {}
        for col in df_sample_for_dtypes.columns:
            series_sample = df_sample_for_dtypes[col]
            try:
                numeric_series = pd.to_numeric(series_sample, errors='coerce')
                if not numeric_series.isna().all():
                    if numeric_series.dropna().apply(lambda x: isinstance(x, (int, np.integer)) or (isinstance(x, (float, np.floating)) and x.is_integer())).all(): inferred_dtypes[col] = pd.Int64Dtype()
                    else: inferred_dtypes[col] = float
                else: inferred_dtypes[col] = object
            except (ValueError, TypeError): inferred_dtypes[col] = object
        df = pd.read_csv(filepath, dtype=inferred_dtypes, low_memory=False, keep_default_na=True, na_values=['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'n/a', 'nan', 'null'])
        if df.empty: raise ValueError("CSV is empty after loading with inferred dtypes.")
        print(f"Loaded new data: {df.shape} from {filepath}")
        for col in df.columns:
            if df[col].dtype == object:
                try:
                    parsed_dates = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
                    if parsed_dates.notna().sum() > 0.5 * df[col].notna().sum() and df[col].notna().sum() > 0: df[col] = parsed_dates; print(f"Column '{col}' converted to datetime.")
                except Exception: pass
        return df
    except Exception as e: raise RuntimeError(f"Error loading new data from {filepath}: {e}")

def identify_column_types(df):
    num_cols, cat_cols, dt_cols, id_cols = [], [], [], []
    for col in df.columns:
        if col == 'original_index': continue 
        col_data = df[col]; nunique_ratio = col_data.nunique(dropna=False) / len(df) if len(df) > 0 else 0
        is_id_candidate = ((pd.api.types.is_string_dtype(col_data) or pd.api.types.is_numeric_dtype(col_data)) and (any(sub in col.lower() for sub in ['id', 'key', 'uuid', 'identifier', 'number']) and nunique_ratio > 0.9) or (nunique_ratio > 0.98))
        if is_id_candidate: id_cols.append(col)
        elif pd.api.types.is_numeric_dtype(col_data) and not pd.api.types.is_bool_dtype(col_data): num_cols.append(col)
        elif pd.api.types.is_datetime64_any_dtype(col_data) or pd.api.types.is_timedelta64_dtype(col_data): dt_cols.append(col)
        else: cat_cols.append(col)
    id_cols = list(set(id_cols)); num_cols = [c for c in num_cols if c not in id_cols]; cat_cols = [c for c in cat_cols if c not in id_cols]; dt_cols = [c for c in dt_cols if c not in id_cols]
    primary_id = id_cols[0] if id_cols else (df.index.name if df.index.name and df.index.name not in df.columns and df.index.name != 'original_index' else None)
    if not primary_id and id_cols: primary_id = id_cols[0]
    print(f"Identified Types -> Primary ID: {primary_id}, Other IDs: {id_cols}, Num: {num_cols}, Cat: {cat_cols}, DT: {dt_cols}")
    return num_cols, cat_cols, dt_cols, id_cols, primary_id

def compare_schemas(new_df_cols, historic_cols_list_or_dictkeys):
    set_new = set(c for c in map(str,new_df_cols) if c != 'original_index'); set_hist = set(map(str,historic_cols_list_or_dictkeys))
    return {'new_cols': sorted(list(set_new - set_hist)), 'missing_cols': sorted(list(set_hist - set_new)), 'common_cols': sorted(list(set_new & set_hist))}

def calculate_basic_stats(df, num_cols, cat_cols_and_dt):
    stats = {};
    if df is None or df.empty: return stats
    for col_list, is_numeric in [(num_cols, True), (cat_cols_and_dt, False)]:
        for col in col_list:
            if col == 'original_index': continue 
            if col in df.columns:
                series = df[col]; col_stat = {'null_count': series.isnull().sum(), 'total_in_col': len(series)}
                series_dropna = series.dropna()
                if not series_dropna.empty:
                    if is_numeric: col_stat.update({'mean': series_dropna.mean(), 'std': series_dropna.std(), 'min': series_dropna.min(), 'max': series_dropna.max(), 'median': series_dropna.median()})
                    else:
                        col_stat.update({'top_values': dict(series_dropna.value_counts(normalize=False).nlargest(5)), 'unique_count': series_dropna.nunique()})
                        if pd.api.types.is_datetime64_any_dtype(series_dropna) or pd.api.types.is_timedelta64_dtype(series_dropna):
                             min_val = series_dropna.min(); max_val = series_dropna.max()
                             col_stat.update({'min_date': pd.Timestamp(min_val).isoformat() if pd.notna(min_val) else None, 'max_date': pd.Timestamp(max_val).isoformat() if pd.notna(max_val) else None})
                stats[col] = col_stat
    return _sanitize_for_json(stats)

def detect_row_level_anomalies(new_df_with_original_idx, historic_stats, num_cols, primary_id_col): 
    categorized_anomalies = defaultdict(list)
    if new_df_with_original_idx is None or new_df_with_original_idx.empty: return {}
    if 'original_index' not in new_df_with_original_idx.columns: print("Error: 'original_index' column missing in detect_row_level_anomalies."); return {}
    _perform_basic_semantic_validation(new_df_with_original_idx, categorized_anomalies)
    for col in new_df_with_original_idx.columns:
        if col == 'original_index': continue
        if col not in historic_stats: continue
        hist_col_stats = historic_stats[col]; hist_mean = _to_native_py_type(hist_col_stats.get('mean')); hist_std = _to_native_py_type(hist_col_stats.get('std')); hist_null_count = _to_native_py_type(hist_col_stats.get('null_count', 0)); hist_total_rows_for_col_metric = _to_native_py_type(hist_col_stats.get('total_in_col', 0)) 
        hist_null_frac = (hist_null_count / hist_total_rows_for_col_metric) if hist_total_rows_for_col_metric > 0 else 0.0
        current_col_data_series = new_df_with_original_idx[col]; original_indices_series = new_df_with_original_idx['original_index']
        if col in num_cols:
            if hist_mean is not None and hist_std is not None:
                safe_hist_std = hist_std if hist_std > 1e-9 else 1e-9 
                if pd.api.types.is_numeric_dtype(current_col_data_series):
                    non_null_mask = current_col_data_series.notna()
                    if non_null_mask.sum() > 0:
                        values_to_check = current_col_data_series[non_null_mask]; z_scores_series = (values_to_check - hist_mean) / safe_hist_std; outlier_mask_on_z_scores = abs(z_scores_series) > Z_SCORE_THRESHOLD
                        if outlier_mask_on_z_scores.any():
                            outlier_df_indices = values_to_check[outlier_mask_on_z_scores].index 
                            for df_idx in outlier_df_indices:
                                original_idx_val = original_indices_series.loc[df_idx]; val = current_col_data_series.loc[df_idx]; z = z_scores_series.loc[df_idx]; native_orig_idx = _to_native_py_type(original_idx_val)
                                categorized_anomalies[native_orig_idx].append({'type': ANOMALY_TYPE_DATA_MISMATCH, 'column': col, 'value': _to_native_py_type(val), 'message': f"Outlier: value {format_value_for_display(val)} (Z={z:.1f}). Hist mean {format_value_for_display(format(hist_mean, '.2f'))}, std {format_value_for_display(format(hist_std, '.2f'))}."})
        current_col_null_count = current_col_data_series.isnull().sum(); current_col_total = len(current_col_data_series)
        current_col_null_frac = (current_col_null_count / current_col_total) if current_col_total > 0 else 0
        if (hist_null_frac < 0.05 and current_col_null_frac > (hist_null_frac + 0.10) and current_col_null_count > 0) or \
           (hist_null_frac == 0 and current_col_null_frac > 0.01 and current_col_null_count > 0) :
            null_df_indices = current_col_data_series[current_col_data_series.isnull()].index
            for df_idx in null_df_indices: 
                original_idx_val = original_indices_series.loc[df_idx]; native_orig_idx = _to_native_py_type(original_idx_val)
                if not any(a['type'] == ANOMALY_TYPE_MISSING and a['column'] == col for a in categorized_anomalies[native_orig_idx]):
                    categorized_anomalies[native_orig_idx].append({'type': ANOMALY_TYPE_MISSING, 'column': col, 'value': None, 'message': f"Unexpected NULL. Column's current null rate: {current_col_null_frac:.1%}, historic: {hist_null_frac:.1%}."})
    return _sanitize_for_json(dict(categorized_anomalies))

def _validate_email_value(value, col, original_idx_map_key, categorized_anomalies):
    email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    if pd.notna(value) and isinstance(value, str) and not email_pattern.match(value): categorized_anomalies[original_idx_map_key].append({'type': ANOMALY_TYPE_INCORRECT_FORMAT, 'column': col, 'value': _to_native_py_type(value), 'message': f"Invalid email format: '{format_value_for_display(value)}'."})

def _validate_date_value_str(value_str, col, col_lower, original_idx_map_key, categorized_anomalies):
    if pd.notna(value_str) and isinstance(value_str, str) and re.search(r'\d', value_str): 
        try: 
            parsed_dt = pd.to_datetime(value_str)
            if 'birth' in col_lower or 'dob' in col_lower:
                if parsed_dt.tz_localize(None) > datetime.now(timezone.utc).tz_localize(None).replace(tzinfo=None): categorized_anomalies[original_idx_map_key].append({'type': ANOMALY_TYPE_INCORRECT_FORMAT, 'column': col, 'value': _to_native_py_type(value_str), 'message': f"Future date for birth date: {format_value_for_display(value_str)} (parsed as {parsed_dt.date()})."})
        except (ValueError, TypeError): categorized_anomalies[original_idx_map_key].append({'type': ANOMALY_TYPE_INCORRECT_FORMAT, 'column': col, 'value': _to_native_py_type(value_str), 'message': f"Unparseable date format: '{format_value_for_display(value_str)}'."})

def _validate_future_birth_date_value(value_dt, col, original_idx_map_key, categorized_anomalies):
    if pd.notna(value_dt) and isinstance(value_dt, (pd.Timestamp, datetime, date)):
        current_dt_for_comp = value_dt; now_naive = datetime.now(timezone.utc).tz_localize(None)
        if hasattr(value_dt, 'tzinfo') and value_dt.tzinfo is not None: current_dt_for_comp = value_dt.tz_localize(None)
        if isinstance(current_dt_for_comp, date) and not isinstance(current_dt_for_comp, datetime): current_dt_for_comp = datetime.combine(current_dt_for_comp, datetime.min.time())
        col_lower = col.lower()
        if ('birth' in col_lower or 'dob' in col_lower) and current_dt_for_comp > now_naive: categorized_anomalies[original_idx_map_key].append({'type': ANOMALY_TYPE_INCORRECT_FORMAT, 'column': col, 'value': _to_native_py_type(value_dt), 'message': f"Future date for birth date: {format_value_for_display(value_dt)}."})

def _validate_phone_value(value, col, original_idx_map_key, categorized_anomalies):
    if pd.notna(value) and isinstance(value, str) and re.search(r'[a-zA-Z]', value): categorized_anomalies[original_idx_map_key].append({'type': ANOMALY_TYPE_INCORRECT_FORMAT, 'column': col, 'value': _to_native_py_type(value), 'message': f"Invalid phone number (contains letters): '{format_value_for_display(value)}'."})

def _validate_location_pair(value1, col1, value2, col2, original_idx_map_key, categorized_anomalies):
    if pd.notna(value1) and pd.notna(value2) and isinstance(value1, str) and isinstance(value2, str):
        city_col, state_col = (col1, col2) if 'city' in col1.lower() else (col2, col1)
        city_val, state_val = (value1, value2) if city_col == col1 else (value2, value1)

        # Load US city/state data (in a real application, this would be loaded once from a database/file)
        # This is a small subset for demonstration. In production, use a complete database of valid city/state combinations.
        valid_city_state_pairs = {
            ('New York', 'New York'), ('Los Angeles', 'California'), ('Chicago', 'Illinois'),
            ('Houston', 'Texas'), ('Phoenix', 'Arizona'), ('Philadelphia', 'Pennsylvania'),
            ('San Antonio', 'Texas'), ('San Diego', 'California'), ('Dallas', 'Texas'),
            ('San Jose', 'California'), ('Austin', 'Texas'), ('Jacksonville', 'Florida'),
            ('Fort Worth', 'Texas'), ('Columbus', 'Ohio'), ('Charlotte', 'North Carolina'),
            ('Indianapolis', 'Indiana'), ('Seattle', 'Washington'), ('Denver', 'Colorado'),
            ('Boston', 'Massachusetts'), ('Nashville', 'Tennessee'), ('Portland', 'Oregon'),
            ('Las Vegas', 'Nevada'), ('Detroit', 'Michigan'), ('Memphis', 'Tennessee')
        }

        city_state_pair = (city_val.strip(), state_val.strip())
        if city_state_pair not in valid_city_state_pairs:
            categorized_anomalies[original_idx_map_key].append({
                'type': ANOMALY_TYPE_LOCATION_MISMATCH,
                'column': city_col,
                'value': _to_native_py_type(city_val),
                'related_columns': {state_col: _to_native_py_type(state_val)},
                'message': f"Possible city/state mismatch: '{format_value_for_display(city_val)}' in '{format_value_for_display(state_val)}'"
            })

def _perform_basic_semantic_validation(df_with_original_idx, categorized_anomalies):
    if df_with_original_idx is None or df_with_original_idx.empty: return
    if 'original_index' not in df_with_original_idx.columns: return 
    
    # Find city and state columns
    city_col = None
    state_col = None
    for col in df_with_original_idx.columns:
        col_lower = col.lower()
        if 'city' in col_lower: city_col = col
        elif 'state' in col_lower: state_col = col

    # Check for city/state mismatches
    if city_col and state_col:
        for idx in df_with_original_idx.index:
            city_val = df_with_original_idx.loc[idx, city_col]
            state_val = df_with_original_idx.loc[idx, state_col]
            original_idx_map_key = _to_native_py_type(df_with_original_idx.loc[idx, 'original_index'])
            _validate_location_pair(city_val, city_col, state_val, state_col, original_idx_map_key, categorized_anomalies)

    # Perform other validations
    for col in df_with_original_idx.columns:
        if col == 'original_index': continue 
        col_lower = col.lower(); current_col_series = df_with_original_idx[col]; original_indices_series = df_with_original_idx['original_index']
        for df_idx, value in current_col_series.items(): 
            original_idx_map_key = _to_native_py_type(original_indices_series.loc[df_idx])
            if 'email' in col_lower: _validate_email_value(value, col, original_idx_map_key, categorized_anomalies)
            if pd.api.types.is_datetime64_any_dtype(current_col_series.dtype):
                if 'birth' in col_lower or 'dob' in col_lower: _validate_future_birth_date_value(value, col, original_idx_map_key, categorized_anomalies)
            elif isinstance(value, str) and any(term in col_lower for term in ['date', 'time', 'day', 'month', 'year']): _validate_date_value_str(value, col, col_lower, original_idx_map_key, categorized_anomalies)
            elif any(term in col_lower for term in ['phone', 'mobile', 'cell', 'tel']): _validate_phone_value(value, col, original_idx_map_key, categorized_anomalies)

def get_llm_analysis(log_id, new_df_sample_with_orig_idx, historic_df_sample, new_stats, historic_stats, schema_comparison, anomalous_rows_sample_messages, total_new_rows, total_historic_rows, primary_id_col):
    raw_llm_response_for_debug = "LLM not called (not configured or error before call)."; default_llm_response = {"overall_assessment": {"quality_summary": "LLM Analysis skipped.", "key_findings_summary": []}, "schema_analysis": {"detected_changes_summary": "N/A", "new_columns_impact": [], "missing_columns_impact": [], "root_cause_hypothesis": "N/A", "suggested_remediation_ddl": "N/A"}, "data_drift_and_anomaly_analysis": {"significant_distribution_shifts": [], "anomaly_patterns_observed": "N/A", "root_cause_hypothesis": "N/A", "suggested_investigation_actions": []}, "semantic_analysis": {"inferred_column_meanings": [], "location_inconsistencies": [], "format_inconsistencies": [], "relational_inconsistencies": [], "suspicious_values": []}, "general_actionable_recommendations": ["Review configuration if LLM analysis is desired."]}
    if not USE_GEMINI or not gemini_model: return default_llm_response, raw_llm_response_for_debug
    formatted_new_data_preview = format_data_for_llm(new_df_sample_with_orig_idx); formatted_historic_data_preview = format_data_for_llm(historic_df_sample)
    prompt = f"""
You are IngestMate Edge, an expert data pipeline observability and remediation agent.
Analyze data drift, anomalies, and semantic inconsistencies between a new dataset and historic baseline. Provide detailed analysis, root causes, and actionable remediation. The new data sample includes an 'original_index' column. When reporting row-specific issues from the sample, please reference this 'original_index' value in the 'row_identifier' field of your JSON response. Output your response STRICTLY as a JSON object matching the structure at the end.
CONTEXT:
- New Data Rows Total: {total_new_rows}, Historic Rows Total: {total_historic_rows if total_historic_rows is not None else 'N/A'}
- Primary ID Column (if identified, might differ from original_index): {primary_id_col or 'Not Identified'}
NEW DATA SAMPLE ({MAX_ROWS_FOR_LLM_DATA_PREVIEW} rows, includes 'original_index'):
{formatted_new_data_preview}
HISTORIC DATA SAMPLE ({MAX_ROWS_FOR_LLM_DATA_PREVIEW} rows, if available):
{formatted_historic_data_preview if historic_df_sample is not None and not historic_df_sample.empty else "No historic data sample."}
SCHEMA COMPARISON:
- New Columns: {schema_comparison.get('new_cols', [])}
- Missing Columns: {schema_comparison.get('missing_cols', [])}
- Common Columns Count: {len(schema_comparison.get('common_cols', []))}
AGGREGATE STATISTICS (New vs. Historic):
New Dataset Stats: {json.dumps(new_stats, indent=1, default=str)}
Historic Dataset Stats: {json.dumps(historic_stats, indent=1, default=str) if historic_stats else "No historic stats."}
PRELIMINARY ANOMALOUS ROW MESSAGES (from statistical/basic checks, keyed by original_index):
{json.dumps(anomalous_rows_sample_messages, indent=1, default=str) if anomalous_rows_sample_messages else "No preliminary anomalies from basic checks, or sample is empty."}
CRITICAL SEMANTIC ANALYSIS (Your primary focus beyond stats): For each finding, specify the 'original_index' from the sample in the 'row_identifier' field.
1.  INFERRED MEANINGS: Infer semantic meaning for each column (e.g., 'email', 'city', 'transaction_amount').
2.  FORMAT INCONSISTENCIES: Check standard formats (emails, phones, dates, URLs, postal codes). For each inconsistent row in the sample, detail: 'row_identifier' (the 'original_index' value), the problematic column(s) with their values, and a clear description.
3.  **LOCATION INCONSISTENCY ANALYSIS (VERY HIGH PRIORITY - populate `semantic_analysis.location_inconsistencies`):** Identify city, state/province, country, postal_code columns. For EACH ROW in the sample, rigorously validate if these components form a plausible geographic unit. **FLAG ANY MISMATCHES (e.g., "Las Vegas, California", "Chicago, Texas, USA", "Paris, France, Germany"). Check postal codes against city/state.** For each inconsistent row found in the sample, detail: 'row_identifier' (the 'original_index' value), the problematic columns with their values, and a clear description of the inconsistency.
4.  RELATIONAL INCONSISTENCIES (General): Check other logical relationships (e.g., start_date after end_date). Detail 'row_identifier' and issue.
5.  SUSPICIOUS VALUES: Identify syntactically valid but semantically unlikely values (e.g., age=200, city_name="Test Data Input"). Detail 'row_identifier' and issue.
6.  PATTERN SHIFTS: Note breakdowns in column correlations or unexpected categorical distribution changes not captured by simple stats.
JSON OUTPUT STRUCTURE (Strictly adhere to this. Use 'original_index' value for `row_identifier` where applicable):
{{
  "overall_assessment": {{"quality_summary": "Concise (1-2 sentences) opinion on data quality change.","key_findings_summary": ["2-3 most critical findings (schema, drift, semantic)."]}},
  "schema_analysis": {{"detected_changes_summary": "e.g., 'Added 2 columns, removed 1'.","new_columns_impact": [{{ "name": "col", "potential_purpose": "..", "data_implications": ".." }}],"missing_columns_impact": [{{ "name": "col", "potential_reason": "..", "impact": ".." }}],"root_cause_hypothesis": "Hypothesis for schema changes.","suggested_remediation_ddl": "EXAMPLE PostgreSQL DDL (e.g., 'ALTER TABLE ... ADD COLUMN ...;'). 'N/A' if none."}},
  "data_drift_and_anomaly_analysis": {{"significant_distribution_shifts": [{{ "column": "col", "observation": "..", "historic_metric_value": "..", "new_metric_value": "..", "potential_cause_hypothesis": "..", "potential_impact": ".." }}],"anomaly_patterns_observed": "Key patterns in statistical/basic anomalies. 'No significant patterns' if none.","root_cause_hypothesis": "Hypothesis for major drifts/anomalies.","suggested_investigation_actions": ["Actionable step 1.", "Step 2"]}},
  "semantic_analysis": {{ "inferred_column_meanings": [ {{ "column": "col", "inferred_type": "type", "confidence": "high/medium/low" }} ], "location_inconsistencies": [ {{ "row_identifier": "original_index_value_from_sample", "involved_columns": {{ "city": "CityVal", "state": "StateVal" }}, "issue_description": "e.g., City 'X' not in State 'Y'." }} ], "format_inconsistencies": [ {{ "row_identifier": "original_index_value_from_sample", "column": "col", "expected_format": "..", "inconsistent_examples": ["ex1"], "issue_description": "Specific problem observed." }} ], "relational_inconsistencies": [ {{ "row_identifier": "original_index_value_from_sample", "related_columns": ["c1", "c2"], "issue_description": "..", "example_values": {{}} }} ], "suspicious_values": [ {{ "row_identifier": "original_index_value_from_sample", "column": "col", "suspicious_values_sample": ["val1"], "reason": ".." }} ] }},
  "general_actionable_recommendations": [ "High-level recommendation 1." ]
}}
Be concise, thorough, and ensure valid JSON. Location consistency and referencing 'original_index' values in 'row_identifier' are paramount.
"""
    try:
        response = gemini_model.generate_content(prompt); raw_llm_response_for_debug = response.text.strip() if hasattr(response, 'text') and response.text else "{}"
        # print(f"DEBUG (Log ID: {log_id}): Raw LLM Response Snippet:\n{raw_llm_response_for_debug[:1000]}...")
        json_str_match = re.search(r"```json\s*([\s\S]*?)\s*```", raw_llm_response_for_debug); json_to_parse = json_str_match.group(1) if json_str_match else raw_llm_response_for_debug
        llm_result_from_api = json.loads(json_to_parse); merged_llm_response = json.loads(json.dumps(default_llm_response))
        def _deep_update(target, source): 
            for key, value in source.items():
                if key in target:
                    if isinstance(target[key], dict) and isinstance(value, dict): _deep_update(target[key], value)
                    elif isinstance(target[key], list) and isinstance(value, list): 
                        if value: target[key] = value 
                        elif not target[key]: target[key] = [] 
                    elif value is not None: target[key] = value
        _deep_update(merged_llm_response, llm_result_from_api)
        current_quality_summary = merged_llm_response.get("overall_assessment", {}).get("quality_summary", "LLM Analysis skipped.")
        if not current_quality_summary or "LLM Analysis skipped." in current_quality_summary:
            sa = merged_llm_response.get("schema_analysis", {}); da = merged_llm_response.get("data_drift_and_anomaly_analysis", {}); sema = merged_llm_response.get("semantic_analysis", {})
            if sa.get("detected_changes_summary", "N/A") == "N/A" and da.get("anomaly_patterns_observed", "N/A") == "N/A" and not any(sema.get(k) for k in ["location_inconsistencies", "format_inconsistencies", "relational_inconsistencies", "suspicious_values"]):
                merged_llm_response["overall_assessment"]["quality_summary"] = (f"LLM response structure might be partially incorrect or incomplete. (Original LLM summary: '{current_quality_summary or 'Not provided'}')")
        return merged_llm_response, raw_llm_response_for_debug
    except json.JSONDecodeError as je: print(f"LLM JSONDecodeError (Log ID: {log_id}): {je}. Raw: {raw_llm_response_for_debug[:500]}"); error_response = json.loads(json.dumps(default_llm_response)); error_response["overall_assessment"]["quality_summary"] = f"LLM analysis failed: JSON parsing error. ({je})"; return error_response, raw_llm_response_for_debug
    except Exception as e: print(f"LLM Analysis Error (Log ID: {log_id}): {e}. Raw: {raw_llm_response_for_debug[:500]}"); error_response = json.loads(json.dumps(default_llm_response)); error_response["overall_assessment"]["quality_summary"] = f"LLM analysis failed: {e}"; return error_response, raw_llm_response_for_debug

def get_original_index_from_llm_row_id(llm_row_id_str, df_sample_original_indices):
    if llm_row_id_str is None or df_sample_original_indices.empty: return None
    if isinstance(llm_row_id_str, str):
        matches = df_sample_original_indices[df_sample_original_indices.astype(str) == llm_row_id_str]
        if not matches.empty: return _to_native_py_type(matches.iloc[0])
    try:
        target_dtype = df_sample_original_indices.dtype; converted_id = None
        if pd.api.types.is_integer_dtype(target_dtype): converted_id = int(llm_row_id_str)
        elif pd.api.types.is_float_dtype(target_dtype): converted_id = float(llm_row_id_str)
        if converted_id is not None:
            matches = df_sample_original_indices[df_sample_original_indices == converted_id]
            if not matches.empty: return _to_native_py_type(matches.iloc[0])
    except (ValueError, TypeError): pass # print(f"Could not convert LLM row_identifier '{llm_row_id_str}' to match original_index dtype {df_sample_original_indices.dtype}")
    # print(f"DEBUG: Could not find native original_index for LLM row_identifier '{llm_row_id_str}' in sample indices: {list(df_sample_original_indices)}")
    return None

def _process_llm_semantic_anomalies_into_map(llm_semantic_analysis_dict, categorized_anomalies_map, new_df_llm_sample_original_indices):
    if not llm_semantic_analysis_dict or not isinstance(llm_semantic_analysis_dict, dict): return
    semantic_map = {"location_inconsistencies": ANOMALY_TYPE_LOCATION_MISMATCH, "format_inconsistencies": ANOMALY_TYPE_INCORRECT_FORMAT, "relational_inconsistencies": ANOMALY_TYPE_DATA_MISMATCH, "suspicious_values": ANOMALY_TYPE_DATA_MISMATCH}
    # print(f"DEBUG _process_llm_semantic_anomalies_into_map: Starting. LLM semantic_analysis: {json.dumps(llm_semantic_analysis_dict, indent=2)}")
    for llm_key, anomaly_type_const in semantic_map.items():
        llm_anomalies = llm_semantic_analysis_dict.get(llm_key, [])
        if not isinstance(llm_anomalies, list): 
            # print(f"DEBUG _process_llm_semantic_anomalies_into_map: LLM key '{llm_key}' is not a list. Found: {type(llm_anomalies)}")
            continue
        # print(f"DEBUG _process_llm_semantic_anomalies_into_map: Processing LLM key '{llm_key}', found {len(llm_anomalies)} items.")
        for llm_anomaly_item in llm_anomalies:
            if not isinstance(llm_anomaly_item, dict): 
                # print(f"DEBUG _process_llm_semantic_anomalies_into_map: Item in LLM key '{llm_key}' is not a dict. Item: {llm_anomaly_item}")
                continue
            llm_row_id_str = llm_anomaly_item.get("row_identifier")
            native_original_idx = get_original_index_from_llm_row_id(llm_row_id_str, new_df_llm_sample_original_indices)
            if native_original_idx is None:
                # print(f"DEBUG _process_llm_semantic_anomalies_into_map: Could not map LLM row_identifier '{llm_row_id_str}' for anomaly: {llm_anomaly_item}")
                continue
            # print(f"DEBUG _process_llm_semantic_anomalies_into_map: Mapped LLM row_id '{llm_row_id_str}' to native_original_idx '{native_original_idx}' for item: {llm_anomaly_item}")
            message = llm_anomaly_item.get("issue_description") or llm_anomaly_item.get("reason", "AI detected issue.")
            column_from_llm = llm_anomaly_item.get("column"); involved_cols_dict = llm_anomaly_item.get("involved_columns"); related_cols_list = llm_anomaly_item.get("related_columns")
            primary_col_for_anomaly = column_from_llm
            if not primary_col_for_anomaly:
                if isinstance(involved_cols_dict, dict) and involved_cols_dict: primary_col_for_anomaly = list(involved_cols_dict.keys())[0] 
                elif isinstance(related_cols_list, list) and related_cols_list: primary_col_for_anomaly = related_cols_list[0]
            value_display = "See details" 
            if isinstance(involved_cols_dict, dict): value_display = str({k:v for k,v in list(involved_cols_dict.items())[:2]})
            elif llm_anomaly_item.get("inconsistent_examples"): value_display = str(llm_anomaly_item.get("inconsistent_examples")[:1])
            elif llm_anomaly_item.get("suspicious_values_sample"): value_display = str(llm_anomaly_item.get("suspicious_values_sample")[:1])
            categorized_anomalies_map[native_original_idx].append({'type': anomaly_type_const, 'column': primary_col_for_anomaly or "Row-level", 'value': value_display, 'message': message, 'details': llm_anomaly_item})
            # print(f"DEBUG _process_llm_semantic_anomalies_into_map: Added AI anomaly for original_index '{native_original_idx}': Type '{anomaly_type_const}', Col '{primary_col_for_anomaly}', Msg '{message}'")
    # print("DEBUG _process_llm_semantic_anomalies_into_map: Finished.")

def _render_llm_overall_assessment(d): return f"<div class='report-section card llm-overall-assessment-card alert-info'><h3 class='card-title'><span class='emoji'>💡</span> Overall AI Assessment</h3><p><strong>Quality Summary:</strong> {html.escape(d.get('quality_summary', 'N/A'))}</p>{('<strong>Key Findings:</strong><ul>' + ''.join(f'<li>{html.escape(str(f))}</li>' for f in d.get('key_findings_summary', []) if str(f).strip()) + '</ul>') if d.get('key_findings_summary') and isinstance(d.get('key_findings_summary'), list) and any(str(f).strip() for f in d.get('key_findings_summary')) else '<p><em>No specific key findings highlighted by AI.</em></p>'}</div>" if d else ""
def _render_llm_schema_analysis(d):
    if not d: return ""
    h = f"<div class='report-section card llm-schema-analysis-card'><h3 class='card-title'><span class='emoji'>🧬</span> Schema Drift: AI Analysis & Remediation</h3><p><strong>Detected Changes Summary:</strong> {html.escape(d.get('detected_changes_summary', 'N/A'))}</p>"
    for impact_type, impacts_list_key in [("New Columns", 'new_columns_impact'), ("Missing Columns", 'missing_columns_impact')]:
        impacts = d.get(impacts_list_key, [])
        if impacts and isinstance(impacts, list) and any(impacts):
            h += f"<strong>Impact of {impact_type}:</strong><ul>"; dk1,dv1k=('potential_purpose','data_implications') if impact_type=="New Columns" else ('potential_reason','impact')
            for i in impacts: h+=f"<li><strong>{html.escape(i.get('name','Unknown'))}:</strong> {html.escape(dk1.replace('_',' ').title())}: {html.escape(i.get(dk1,'N/A'))}. {html.escape(dv1k.replace('_',' ').title())}: {html.escape(i.get(dv1k,'N/A'))}</li>" if isinstance(i,dict) else ""
            h += "</ul>"
    h += f"<p><strong>Potential Root Cause:</strong> {html.escape(d.get('root_cause_hypothesis', 'N/A'))}</p>"; ddl=d.get('suggested_remediation_ddl','N/A')
    h += f"<strong>Suggested DDL (Example for Review):</strong><pre><code class='language-sql'>{html.escape(str(ddl))}</code></pre>" if ddl and str(ddl).strip().lower() not in ['n/a','none',''] else "<p><strong>Suggested DDL:</strong> N/A</p>"
    h += "</div>"; return h
def _render_llm_data_drift_anomaly_analysis(d):
    if not d: return ""
    h = f"<div class='report-section card llm-data-drift-card'><h3 class='card-title'><span class='emoji'>📉</span> Data Drift & Statistical Anomalies: AI Analysis</h3>"
    shifts = d.get('significant_distribution_shifts', [])
    if shifts and isinstance(shifts, list) and any(shifts):
        h += "<strong>Significant Distribution Shifts:</strong><ul>"
        for s in shifts: h+=f"<li><strong>Column '{html.escape(s.get('column','N/A'))}':</strong> {html.escape(s.get('observation','N/A'))} (Historic: {html.escape(str(s.get('historic_metric_value','N/A')))}, New: {html.escape(str(s.get('new_metric_value','N/A')))})<br><em>Potential Cause:</em> {html.escape(s.get('potential_cause_hypothesis','N/A'))}<br><em>Potential Impact:</em> {html.escape(s.get('potential_impact','N/A'))}</li>" if isinstance(s,dict) else ""
        h += "</ul>"
    h += f"<p><strong>Statistical Anomaly Patterns Observed:</strong> {html.escape(d.get('anomaly_patterns_observed', 'N/A'))}</p><p><strong>Potential Root Cause for Drifts/Anomalies:</strong> {html.escape(d.get('root_cause_hypothesis', 'N/A'))}</p>"
    actions = d.get('suggested_investigation_actions', [])
    if actions and isinstance(actions,list) and any(str(a).strip() for a in actions): h+="<strong>Suggested Investigation Actions:</strong><ul>" + "".join(f"<li>{html.escape(str(a))}</li>" for a in actions if str(a).strip()) + "</ul>"
    else: h+="<p><em>No specific investigation actions suggested by AI for statistical drift.</em></p>"
    h += "</div>"; return h
def _render_llm_semantic_analysis(d): 
    if not d: return ""
    h = "<div class='report-section card llm-semantic-analysis-card'><h3 class='card-title'><span class='emoji'>🧩</span> Semantic & Contextual AI Analysis (Summary)</h3><p><em>Note: Detailed row-level AI findings (if any) are listed in separate tables below and used for cell highlighting in previews. This section provides the AI's narrative summary.</em></p>"
    for key, title, lk in [('location_inconsistencies','📍 Location Data Inconsistencies','location_inconsistencies'),('format_inconsistencies','📝 Format Inconsistencies','format_inconsistencies'),('relational_inconsistencies','🔗 Relational Inconsistencies','relational_inconsistencies'),('suspicious_values','🕵️ Suspicious Values','suspicious_values')]:
        data = d.get(lk, [])
        if data and isinstance(data,list) and any(data):
            h+=f"<h4><span class='emoji'>{title} (AI Summary):</span></h4><ul>"
            for item in data[:3]: 
                if isinstance(item,dict): rid=html.escape(str(item.get('row_identifier','N/A')));issue=html.escape(item.get('issue_description',item.get('reason','N/A')));cips,vips=[],[]
                if item.get('column'):cips.append(f"Col: '{html.escape(item.get('column'))}'")
                if item.get('involved_columns') and isinstance(item.get('involved_columns'),dict):cips.append(f"Involved: {html.escape(str(item.get('involved_columns')))}")
                if item.get('related_columns') and isinstance(item.get('related_columns'),list):cips.append(f"Related: {html.escape(str(item.get('related_columns')))}")
                if item.get('inconsistent_examples'):vips.append(f"Examples: {html.escape(str(item.get('inconsistent_examples')[:2]))}")
                if item.get('example_values') and isinstance(item.get('example_values'),dict):vips.append(f"Example Values: {html.escape(str(item.get('example_values')))}")
                if item.get('suspicious_values_sample'):vips.append(f"Samples: {html.escape(str(item.get('suspicious_values_sample')[:2]))}")
                cis="; ".join(cips);vis="; ".join(vips);h+=f"<li>Row ID/Context: {rid} -> {issue}<br/><em>{cis} {vis}</em></li>"
            h+="</ul>"
    im=d.get('inferred_column_meanings',[])
    if im and isinstance(im,list) and any(im):
        h+="<details class='collapsible-section'><summary>Inferred Column Meanings by AI</summary><div class='collapsible-content'><ul>"
        for i in im: h+=f"<li><strong>{html.escape(i.get('column','N/A'))}:</strong> {html.escape(i.get('inferred_type','N/A'))} (Confidence: {html.escape(i.get('confidence','N/A'))})</li>" if isinstance(i,dict) else ""
        h+="</ul></div></details>"
    h+="</div>"; return h
def _render_llm_general_recommendations(r):
    if not r or not isinstance(r,list): return ""
    fr=[str(rec) for rec in r if str(rec).strip() and "Review LLM" not in str(rec) and "skipped" not in str(rec).lower() and "N/A" != str(rec)]
    if not fr: return ""
    h="<div class='report-section card llm-recommendations-card alert-success'><h3 class='card-title'><span class='emoji'>🛠️</span> General AI Recommendations</h3><ul>"; h+="".join(f"<li>{html.escape(rec)}</li>" for rec in fr); h+="</ul></div>"; return h

def _render_categorized_anomalies_report(categorized_anomalies_map, new_df_with_original_idx, title, anomaly_type_filter, max_rows_snippet, max_rows_full, section_id_prefix):
    if not categorized_anomalies_map: return ""
    relevant_rows_data = [] 
    df_for_lookup = new_df_with_original_idx.copy()
    if 'original_index' not in df_for_lookup.columns:
        if df_for_lookup.index.name == 'original_index': df_for_lookup.reset_index(inplace=True)
        else: return f"<p>Error rendering {title}: Missing original_index for lookup.</p>"
    for original_idx_native_key, anomalies_for_row in categorized_anomalies_map.items():
        current_type_anomalies = [a for a in anomalies_for_row if a['type'] == anomaly_type_filter]
        if current_type_anomalies:
            try:
                match_series = pd.Series([False] * len(df_for_lookup)); orig_idx_col_dtype = df_for_lookup['original_index'].dtype
                try:
                    comp_key = original_idx_native_key
                    if pd.api.types.is_numeric_dtype(orig_idx_col_dtype) or pd.api.types.is_integer_dtype(orig_idx_col_dtype) :
                        if not isinstance(comp_key, (int, float)): comp_key = pd.to_numeric(original_idx_native_key, errors='ignore')
                        if not pd.isna(comp_key): match_series = (df_for_lookup['original_index'] == comp_key)
                    elif pd.api.types.is_string_dtype(orig_idx_col_dtype) or orig_idx_col_dtype == 'object': match_series = (df_for_lookup['original_index'].astype(str) == str(original_idx_native_key))
                    else: match_series = (df_for_lookup['original_index'] == original_idx_native_key)
                except Exception: pass
                matching_df_rows = df_for_lookup[match_series]
                if not matching_df_rows.empty: full_row_dict = matching_df_rows.iloc[0].drop(labels=['original_index'], errors='ignore').to_dict(); relevant_rows_data.append((original_idx_native_key, current_type_anomalies, full_row_dict))
                else: relevant_rows_data.append((original_idx_native_key, current_type_anomalies, {"error": f"Row data for index '{original_idx_native_key}' not found."}))
            except Exception: relevant_rows_data.append((original_idx_native_key, current_type_anomalies, {"error": "Row data lookup failed."}))
    if not relevant_rows_data: return ""
    
    css_category_for_card = get_anomaly_css_category(anomaly_type_filter) # Get the CSS category for card styling
    html_str = f"<div class='report-section card anomaly-category-card anomaly-{css_category_for_card}-card'>"
    html_str += f"<h3 class='card-title'><span class='emoji'>⚠️</span> {html.escape(title)} ({len(relevant_rows_data)} detected)</h3>"
    html_str += "<div class='table-responsive-wrapper'><table class='result-table table table-sm table-bordered table-hover'>"
    html_str += "<thead><tr><th>Original Row Index</th><th>Anomalous Column(s)</th><th>Value(s)/Example(s)</th><th>Description(s)</th></tr></thead><tbody>"
    
    for i, (original_idx, anomalies, _) in enumerate(relevant_rows_data[:max_rows_snippet]):
        cols_involved = sorted(list(set([a.get('column', 'N/A') for a in anomalies])))
        vals_display = []
        for a in anomalies:
            if a['type'].endswith("_ai"):
                details = a.get('details', {}); display_val = "AI Anomaly"
                if isinstance(details, dict):
                    if details.get('involved_columns') and isinstance(details['involved_columns'], dict) and details['involved_columns']: display_val = ", ".join([f"{k}: {v}" for k,v in list(details['involved_columns'].items())[:2]])
                    elif details.get('inconsistent_examples') and isinstance(details['inconsistent_examples'], list) and details['inconsistent_examples']: display_val = str(details['inconsistent_examples'][0])
                    elif details.get('suspicious_values_sample') and isinstance(details['suspicious_values_sample'], list) and details['suspicious_values_sample']: display_val = str(details['suspicious_values_sample'][0])
                    else: display_val = format_value_for_display(a.get('value', "Details in row context"))
                else: display_val = format_value_for_display(a.get('value', "Details in row context"))
                vals_display.append(display_val)
            else: vals_display.append(format_value_for_display(a.get('value', 'N/A')))
        descs_html = "".join([f"<li>{html.escape(a['message'])}</li>" for a in anomalies])
        html_str += f"<tr><td>{html.escape(str(original_idx))}</td><td>{html.escape(', '.join(cols_involved))}</td><td>{html.escape(', '.join(vals_display[:3]))}{'...' if len(vals_display) > 3 else ''}</td><td><ul>{descs_html}</ul></td></tr>"
    html_str += "</tbody></table></div>"

    if len(relevant_rows_data) > max_rows_snippet:
        full_list_id = f"{section_id_prefix}-full-list-{anomaly_type_filter.replace('_','-')}" 
        html_str += f"<p><a href='#' class='show-more-anomalies-link' data-target-id='{full_list_id}'>Showing {max_rows_snippet} of {len(relevant_rows_data)} {title.lower()}. Click to see up to {max_rows_full} more.</a></p>"
        html_str += f"<div id='{full_list_id}' class='hidden-anomaly-details table-responsive-wrapper' style='display:none; margin-top:15px;'>"
        html_str += f"<h4>Full List (up to {max_rows_full} more examples of {html.escape(title.lower())}):</h4>"
        html_str += "<table class='result-table table table-sm table-bordered table-hover'>"
        html_str += "<thead><tr><th>Original Row Index</th><th>Anomalous Column(s)</th><th>Value(s)/Example(s)</th><th>Description(s)</th><th>Full Row Context (JSON snippet)</th></tr></thead><tbody>"
        for i, (original_idx, anomalies, full_row_data) in enumerate(relevant_rows_data[max_rows_snippet : max_rows_snippet + max_rows_full]):
            cols_involved = sorted(list(set([a.get('column','N/A') for a in anomalies])))
            vals_display_full = [] # Re-calculate for full list with same logic as above
            for a in anomalies:
                if a['type'].endswith("_ai"):
                    details = a.get('details', {}); display_val_full = "AI Anomaly"
                    if isinstance(details, dict):
                        if details.get('involved_columns') and isinstance(details['involved_columns'], dict) and details['involved_columns']: display_val_full = ", ".join([f"{k}: {v}" for k,v in list(details['involved_columns'].items())[:2]])
                        elif details.get('inconsistent_examples') and isinstance(details['inconsistent_examples'], list) and details['inconsistent_examples']: display_val_full = str(details['inconsistent_examples'][0])
                        elif details.get('suspicious_values_sample') and isinstance(details['suspicious_values_sample'], list) and details['suspicious_values_sample']: display_val_full = str(details['suspicious_values_sample'][0])
                        else: display_val_full = format_value_for_display(a.get('value', "Details in row context"))
                    else: display_val_full = format_value_for_display(a.get('value', "Details in row context"))
                    vals_display_full.append(display_val_full)
                else: vals_display_full.append(format_value_for_display(a.get('value', 'N/A')))

            descs_html = "".join([f"<li>{html.escape(a['message'])}</li>" for a in anomalies])
            row_context_snippet = {k: format_value_for_display(v) for k, v in list(full_row_data.items())[:7] if k != 'original_index'} if isinstance(full_row_data, dict) and "error" not in full_row_data else {"error": full_row_data.get("error", "Context unavailable")}
            html_str += f"<tr><td>{html.escape(str(original_idx))}</td><td>{html.escape(', '.join(cols_involved))}</td><td>{html.escape(', '.join(vals_display_full[:3]))}{'...' if len(vals_display_full) > 3 else ''}</td><td><ul>{descs_html}</ul></td><td><pre style='font-size:0.8em; max-height:100px; overflow:auto;'>{html.escape(json.dumps(row_context_snippet, indent=2))}</pre></td></tr>"
        html_str += "</tbody></table></div>"
    html_str += "</div>"; return html_str

def generate_report_html(log_id, new_df_with_original_idx, historic_sample_df, schema_comp,
                         new_stats, historic_stats, categorized_anomalies_map, llm_response,
                         new_filename, historic_filename_desc):
    report_html = f"<div class='report-container card'>"
    report_html += f"<h2 class='card-title main-report-title'>IngestMate Edge: Data Drift & Anomaly Report</h2>"
    report_html += f"<p class='log-id-display'><strong>Log ID:</strong> {log_id}</p><hr class='thin-hr'>"
    new_rows_count = len(new_df_with_original_idx) if new_df_with_original_idx is not None else 'N/A'
    hist_rows_sample_count = len(historic_sample_df) if historic_sample_df is not None else 'N/A'
    report_html += f"<div class='report-section card overview-card'><h3 class='card-title'><span class='emoji'>📄</span> Analysis Overview</h3><p><strong>New Dataset:</strong> {html.escape(new_filename)} ({new_rows_count} rows)</p><p><strong>Historic Baseline:</strong> {html.escape(historic_filename_desc)} ({hist_rows_sample_count} sample rows shown for preview)</p></div>"
    report_html += _render_llm_overall_assessment(llm_response.get("overall_assessment"))
    report_html += _render_llm_schema_analysis(llm_response.get("schema_analysis"))
    report_html += _render_llm_data_drift_anomaly_analysis(llm_response.get("data_drift_and_anomaly_analysis"))
    report_html += _render_llm_semantic_analysis(llm_response.get("semantic_analysis")) 
    report_html += _render_llm_general_recommendations(llm_response.get("general_actionable_recommendations"))
    report_html += "<div class='report-section card dataset-comparison-card'><h3 class='card-title'><span class='emoji'>🆚</span> Detected Schema Comparison (Raw)</h3><div class='table-responsive-wrapper'><table class='result-table table table-sm table-bordered'>"
    new_cols_str = ', '.join(schema_comp.get('new_cols',[])); missing_cols_str = ', '.join(schema_comp.get('missing_cols',[]))
    report_html += f"<tr><td>New Columns</td><td class='column-list'><code>{html.escape(new_cols_str) if new_cols_str else 'None'}</code>{' (Count: ' + str(len(schema_comp.get('new_cols',[]))) + ')' if schema_comp.get('new_cols') else ''}</td></tr>"
    report_html += f"<tr><td>Missing Columns</td><td class='column-list'><code>{html.escape(missing_cols_str) if missing_cols_str else 'None'}</code>{' (Count: ' + str(len(schema_comp.get('missing_cols',[]))) + ')' if schema_comp.get('missing_cols') else ''}</td></tr>"
    report_html += f"<tr><td>Common Columns Count</td><td>{len(schema_comp.get('common_cols',[]))}</td></tr></tbody></table></div></div>"
    
    report_html += "<div class='report-section card data-preview-card'><h3 class='card-title'><span class='emoji'>📊</span> Data Previews (with Anomaly Cell Highlighting)</h3>"
    # Updated Legend Structure with simplified categories
    report_html += """
        <div class="anomaly-legend card">
            <h5 class="card-title legend-title">Anomaly Legend (Cell Highlighting Guide):</h5>
            <ul>
                <li>
                    <span class="legend-color-box cell-anomaly-missing"></span>
                    <strong>Missing Values</strong>
                </li>
                <li>
                    <span class="legend-color-box cell-anomaly-format"></span>
                    <strong>Format & Validation Issues</strong>
                </li>
                <li>
                    <span class="legend-color-box cell-anomaly-mismatch"></span>
                    <strong>Data Mismatches & Inconsistencies</strong>
                </li>
                <li>
                    <span class="legend-color-box cell-anomaly-summary"></span>
                    <strong>Row Summary Details</strong>
                </li>
            </ul>
        </div>
    """
    report_html += f"<details class='collapsible-section' open><summary>New Data (First {PREVIEW_TABLE_MAX_ROWS} rows)</summary>"
    report_html += get_preview_table_html(new_df_with_original_idx, " New Data Preview", PREVIEW_TABLE_MAX_ROWS, table_id="new_data_preview_table", categorized_anomalies_for_preview=categorized_anomalies_map)
    report_html += "</details>"
    if historic_sample_df is not None and not historic_sample_df.empty:
        report_html += f"<details class='collapsible-section'><summary>Historic Data Sample (First {PREVIEW_TABLE_MAX_ROWS} Rows)</summary>"
        report_html += get_preview_table_html(historic_sample_df, "Historic Data Preview", PREVIEW_TABLE_MAX_ROWS) 
        report_html += "</details>"
    else: report_html += "<p class='no-preview-note'>No historic data sample available for preview.</p>"
    report_html += "</div>"

    # Render anomaly sections - titles can be adjusted here if needed for broader categories,
    # but the anomaly_type_filter remains specific for data retrieval.
    report_html += _render_categorized_anomalies_report(categorized_anomalies_map, new_df_with_original_idx, "Statistical Outliers", ANOMALY_TYPE_DATA_MISMATCH, MAX_ANOMALY_ROWS_IN_REPORT_SNIPPET, MAX_FULL_ANOMALY_ROWS_IN_REPORT, "statout")
    report_html += _render_categorized_anomalies_report(categorized_anomalies_map, new_df_with_original_idx, "Unexpected Null/Missing Values", ANOMALY_TYPE_MISSING, MAX_ANOMALY_ROWS_IN_REPORT_SNIPPET, MAX_FULL_ANOMALY_ROWS_IN_REPORT, "statnull") # Title adjusted
    report_html += _render_categorized_anomalies_report(categorized_anomalies_map, new_df_with_original_idx, "Email Format Issues", ANOMALY_TYPE_INCORRECT_FORMAT, MAX_ANOMALY_ROWS_IN_REPORT_SNIPPET, MAX_FULL_ANOMALY_ROWS_IN_REPORT, "emailfmt")
    # report_html += _render_categorized_anomalies_report(categorized_anomalies_map, new_df_with_original_idx, "Phone Format Issues", ANOMALY_TYPE_INCORRECT_FORMAT, MAX_ANOMALY_ROWS_IN_REPORT_SNIPPET, MAX_FULL_ANOMALY_ROWS_IN_REPORT, "phonefmt")
    # report_html += _render_categorized_anomalies_report(categorized_anomalies_map, new_df_with_original_idx, "Invalid Future Dates (Logical Error)", ANOMALY_TYPE_INCORRECT_FORMAT, MAX_ANOMALY_ROWS_IN_REPORT_SNIPPET, MAX_FULL_ANOMALY_ROWS_IN_REPORT, "futuredt") # Title adjusted
    # report_html += _render_categorized_anomalies_report(categorized_anomalies_map, new_df_with_original_idx, "Unparseable Date Strings", ANOMALY_TYPE_INCORRECT_FORMAT, MAX_ANOMALY_ROWS_IN_REPORT_SNIPPET, MAX_FULL_ANOMALY_ROWS_IN_REPORT, "unparsedt")
    
    # AI Detected Sections
    report_html += _render_categorized_anomalies_report(categorized_anomalies_map, new_df_with_original_idx, "AI Detected: Location Data Mismatches", ANOMALY_TYPE_LOCATION_MISMATCH, MAX_ANOMALY_ROWS_IN_REPORT_SNIPPET, MAX_FULL_ANOMALY_ROWS_IN_REPORT, "ailloc")
    report_html += _render_categorized_anomalies_report(categorized_anomalies_map, new_df_with_original_idx, "AI Detected: General Semantic Format/Validation Issues", ANOMALY_TYPE_INCORRECT_FORMAT, MAX_ANOMALY_ROWS_IN_REPORT_SNIPPET, MAX_FULL_ANOMALY_ROWS_IN_REPORT, "aisemfmt")
    report_html += _render_categorized_anomalies_report(categorized_anomalies_map, new_df_with_original_idx, "AI Detected: General Semantic Relational Issues", ANOMALY_TYPE_DATA_MISMATCH, MAX_ANOMALY_ROWS_IN_REPORT_SNIPPET, MAX_FULL_ANOMALY_ROWS_IN_REPORT, "aisemrel")
    report_html += _render_categorized_anomalies_report(categorized_anomalies_map, new_df_with_original_idx, "AI Detected: General Semantically Suspicious Values", ANOMALY_TYPE_DATA_MISMATCH, MAX_ANOMALY_ROWS_IN_REPORT_SNIPPET, MAX_FULL_ANOMALY_ROWS_IN_REPORT, "aisemsus")
    
    report_html += "<hr class='thin-hr'><p style='text-align:center; font-style:italic; margin-top:20px;'>End of IngestMate Edge Report</p></div>"
    return report_html

def run_analysis(new_filepath, historic_csv_filepath=None):
    log_id = uuid.uuid4(); db_conn = get_db_connection();
    if db_conn: ensure_log_table_exists(db_conn)
    analysis_output = {'log_id': str(log_id)}; new_df_raw, historic_sample_df, historic_column_stats = None, pd.DataFrame(), {}; new_df_with_original_idx = None; historic_total_rows_db = None
    llm_response_structure_template = get_llm_analysis("template_log_id", pd.DataFrame(), pd.DataFrame(), {}, {}, {}, {}, 0,0,None)[0] 
    llm_response = llm_response_structure_template; raw_llm_text_for_debug = "LLM not called."
    new_filename = os.path.basename(new_filepath) if new_filepath else "N/A"
    report_html = f"<div class='card error-card'><h2>Analysis Pending (Log ID: {log_id})</h2><p>Processing...</p></div>"
    historic_preview_html = "<p>Historic data preview pending.</p>"; new_data_preview_html = "<p>New data preview pending.</p>" 
    try:
        global _stats_cache; _stats_cache = {}
        new_df_raw = load_new_data(new_filepath); analysis_output['new_data_shape'] = new_df_raw.shape if new_df_raw is not None else (0,0)
        new_df_with_original_idx = new_df_raw.copy()
        if new_df_with_original_idx.index.name == 'original_index': new_df_with_original_idx.reset_index(inplace=True)
        elif 'original_index' not in new_df_with_original_idx.columns: new_df_with_original_idx.reset_index(inplace=True); new_df_with_original_idx.rename(columns={'index':'original_index'}, inplace=True, errors='ignore')
        new_data_preview_html = get_preview_table_html(new_df_with_original_idx, "New Data Preview (Initial)", table_id="new_data_preview_table_main")
        num_cols, cat_cols, dt_cols, id_cols, primary_id = identify_column_types(new_df_with_original_idx); analysis_output.update({'num_cols':num_cols, 'cat_cols':cat_cols, 'dt_cols':dt_cols, 'id_cols':id_cols, 'primary_id':primary_id})
        new_df_stats = calculate_basic_stats(new_df_with_original_idx, num_cols, cat_cols + dt_cols); analysis_output['new_data_stats'] = new_df_stats
        historic_filename_description = "N/A"
        if HISTORIC_TABLE_NAME and db_conn: historic_total_rows_db, historic_sample_df, historic_column_stats = _get_cached_or_compute(f"historic_data_{HISTORIC_TABLE_NAME}", fetch_historic_data, db_conn, HISTORIC_TABLE_NAME); historic_filename_description = f"Database Table: {HISTORIC_TABLE_NAME}"; analysis_output['historic_data_source'] = "Database"
        elif historic_csv_filepath and os.path.exists(historic_csv_filepath): historic_sample_df = load_new_data(historic_csv_filepath); historic_total_rows_db = len(historic_sample_df) if historic_sample_df is not None else 0; h_num, h_cat, h_dt, _, _ = identify_column_types(historic_sample_df); historic_column_stats = calculate_basic_stats(historic_sample_df, h_num, h_cat + h_dt); historic_filename_description = f"CSV File: {os.path.basename(historic_csv_filepath)}"; analysis_output['historic_data_source'] = "CSV Fallback"
        else: analysis_output['historic_data_source'] = "None Available"; historic_filename_description = "No Historic Baseline Available"
        if historic_sample_df is not None and not historic_sample_df.empty: analysis_output['historic_sample_shape'] = historic_sample_df.shape; historic_preview_html = get_preview_table_html(historic_sample_df.head(PREVIEW_TABLE_MAX_ROWS), "Historic Data Sample") 
        analysis_output['historic_column_stats'] = historic_column_stats; analysis_output['historic_total_rows'] = historic_total_rows_db
        schema_comp = compare_schemas(new_df_with_original_idx.columns, list(historic_column_stats.keys()) if historic_column_stats else (list(historic_sample_df.columns) if historic_sample_df is not None and not historic_sample_df.empty else [])); analysis_output['schema_comparison'] = schema_comp
        categorized_anomalies_map = detect_row_level_anomalies(new_df_with_original_idx, historic_column_stats, num_cols, primary_id) 
        llm_anomaly_sample_messages = {}; count = 0
        for idx_key_native, anomalies_list in categorized_anomalies_map.items():
            if count >= MAX_ANOMALY_ROWS_TO_SHOW_LLM: break
            llm_anomaly_sample_messages[str(idx_key_native)] = [a['message'] for a in anomalies_list[:2]]; count +=1
        new_df_llm_sample = new_df_with_original_idx.head(MAX_ROWS_FOR_LLM_DATA_PREVIEW)
        hist_df_llm_sample = historic_sample_df.head(MAX_ROWS_FOR_LLM_DATA_PREVIEW) if historic_sample_df is not None and not historic_sample_df.empty else pd.DataFrame()
        llm_response, raw_llm_text_for_debug = get_llm_analysis(str(log_id), new_df_llm_sample, hist_df_llm_sample, new_df_stats, historic_column_stats, schema_comp, llm_anomaly_sample_messages, len(new_df_with_original_idx), historic_total_rows_db, primary_id)
        analysis_output['llm_full_response'] = llm_response; analysis_output['llm_raw_text_response'] = raw_llm_text_for_debug
        if llm_response and 'semantic_analysis' in llm_response:
            _process_llm_semantic_anomalies_into_map(llm_response['semantic_analysis'], categorized_anomalies_map, new_df_llm_sample['original_index'] if 'original_index' in new_df_llm_sample else pd.Series(dtype='object'))
        analysis_output['categorized_anomalies_map'] = categorized_anomalies_map
        log_summary_text = llm_response.get("overall_assessment", {}).get("quality_summary", "AI summary unavailable.")
        status_for_log = "Success";
        if "partially incorrect" in log_summary_text or "failed" in log_summary_text or "skipped" in log_summary_text.lower(): status_for_log = "Success with AI issues"
        analysis_output['llm_summary_text_for_log'] = log_summary_text
        report_html = generate_report_html(str(log_id), new_df_with_original_idx, historic_sample_df.head(PREVIEW_TABLE_MAX_ROWS) if historic_sample_df is not None else pd.DataFrame(), schema_comp, new_df_stats, historic_column_stats, categorized_anomalies_map, llm_response, new_filename, historic_filename_description)
        log_analysis_run_db(db_conn, log_id, status_for_log, filename=new_filename, summary=log_summary_text)
        return report_html, historic_preview_html, new_data_preview_html, str(log_id), _sanitize_for_json(analysis_output)
    except Exception as e:
        import traceback; tb_str = traceback.format_exc(); print(f"Critical Error in run_analysis (Log ID: {log_id}): {e}\n{tb_str}")
        error_message_for_display = f"Error: {type(e).__name__} - {html.escape(str(e))}. Check server logs."
        report_html = f"<div class='card error-card'><h2>Analysis Failed Critically (Log ID: {log_id})</h2><p>{error_message_for_display}</p><details><summary>Traceback (partial)</summary><pre style='white-space: pre-wrap; word-break: break-all;'>{html.escape(tb_str[:1500])}</pre></details></div>"
        log_summary = analysis_output.get('llm_summary_text_for_log', "Critical failure before AI stage.")
        log_analysis_run_db(db_conn, log_id, "Critical Failure", f"{type(e).__name__}: {str(e)}\nTrace: {tb_str[:5000]}", filename=new_filename, summary=log_summary)
        if 'log_id' not in analysis_output: analysis_output['log_id'] = str(log_id)
        if 'llm_full_response' not in analysis_output or not analysis_output['llm_full_response'].get('overall_assessment'): analysis_output['llm_full_response'] = llm_response_structure_template; analysis_output['llm_raw_text_response'] = raw_llm_text_for_debug if raw_llm_text_for_debug else "LLM not called due to critical error."
        return report_html, historic_preview_html, new_data_preview_html, str(log_id), _sanitize_for_json(analysis_output)
    finally:
        if db_conn:
            try: db_conn.close()
            except Exception as e_close: print(f"Error closing DB connection: {e_close}")