# analysis_engine.py

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, ttest_ind
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

# Supabase PostgreSQL Configuration
SUPABASE_DB_HOST = os.getenv("HOST")  # Supabase pooler host
SUPABASE_DB_PORT = os.getenv("PORT", "6543")  # Supabase pooler port
SUPABASE_DB_NAME = os.getenv("DBNAME", "postgres")
SUPABASE_DB_USER = os.getenv("USER")  # Supabase connection string user
SUPABASE_DB_PASSWORD = os.getenv("PASSWORD")
HISTORIC_TABLE_NAME = os.getenv("HISTORIC_TABLE_NAME")
ANALYSIS_LOG_TABLE_NAME = os.getenv("ANALYSIS_LOG_TABLE_NAME", "data_analysis_logs")

# Gemini AI Configuration
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

# --- Constants ---
DRIFT_THRESHOLD = 0.15  # 15% change threshold for drift detection
VOLUME_THRESHOLD = 0.10  # 10% change threshold for volume anomalies
SIGNIFICANCE_LEVEL = 0.05
MAX_ROWS_FOR_PREVIEW = 10 # Consistent with overview.html expectation
HISTORIC_SAMPLE_ROWS_FOR_CONTEXT = 50

warnings.filterwarnings('ignore')

def _to_native_py_type(value):
    """Convert pandas/numpy types to native Python types"""
    if pd.isna(value): 
        return None
    if isinstance(value, (int, float, bool, str, type(None))): 
        return value
    if isinstance(value, (np.integer, np.int64, np.int32, np.int16, np.int8)): 
        return int(value)
    if isinstance(value, (np.floating, np.float64, np.float32)):
        if pd.isna(value): return None
        return float(value)
    if isinstance(value, np.bool_): 
        return bool(value)
    if isinstance(value, (datetime, pd.Timestamp, date)): 
        return value.isoformat()
    if isinstance(value, uuid.UUID): 
        return str(value)
    if isinstance(value, decimal.Decimal):
        return float(value)
    try: 
        return str(value)
    except Exception: 
        return f"Unconvertible_Type_{type(value).__name__}"

def _sanitize_for_json(item):
    """Sanitize data structures for JSON serialization"""
    if isinstance(item, dict):
        return {str(k): _sanitize_for_json(v) for k, v in item.items()}
    elif isinstance(item, list):
        return [_sanitize_for_json(i) for i in item]
    return _to_native_py_type(item)

def get_db_connection():
    """Get a connection to the Supabase PostgreSQL database using connection pooler"""
    if not all([SUPABASE_DB_HOST, SUPABASE_DB_NAME, SUPABASE_DB_USER, SUPABASE_DB_PASSWORD]):
        print("Missing Supabase database configuration. Check your environment variables.")
        return None
    try:
        conn = psycopg2.connect(
            host=SUPABASE_DB_HOST,
            port=SUPABASE_DB_PORT,
            dbname=SUPABASE_DB_NAME,
            user=SUPABASE_DB_USER,
            password=SUPABASE_DB_PASSWORD,
            # Supabase recommended connection settings
            sslmode='require',
            connect_timeout=10
        )
        print("Successfully connected to Supabase database")
        return conn
    except psycopg2.Error as e:
        print(f"Failed to connect to Supabase database: {e}")
        return None

def execute_db_query(db_conn, query, params=None, fetch_one=False, fetch_all=False):
    """Execute a query on Supabase PostgreSQL database with proper error handling"""
    if not db_conn: 
        return None
    try:
        with db_conn.cursor() as cursor:
            cursor.execute(query, params)
            if fetch_one: 
                return cursor.fetchone()
            if fetch_all: 
                return cursor.fetchall()
            db_conn.commit()
            return True
    except psycopg2.OperationalError as e:
        print(f"Supabase connection error: {e}")
        db_conn.rollback()
        return None
    except psycopg2.Error as e: 
        print(f"Supabase query error: {e}")
        db_conn.rollback()
        return None

def ensure_log_table_exists(db_conn):
    """Ensure the analysis log table exists in Supabase"""
    if not db_conn:
        return
    
    try:
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS "{ANALYSIS_LOG_TABLE_NAME}" (
            log_id UUID PRIMARY KEY,
            run_timestamp TIMESTAMPTZ NOT NULL,
            status VARCHAR(100) NOT NULL,
            error_message TEXT,
            new_data_filename VARCHAR(255),
            historic_table_name VARCHAR(255),
            analysis_summary TEXT,
            analysis_data JSONB 
        );
        """
        execute_db_query(db_conn, create_table_query)
        print(f"Ensured log table '{ANALYSIS_LOG_TABLE_NAME}' exists with required schema")
                
    except Exception as e:
        print(f"Error ensuring log table schema: {e}")


def log_analysis_to_db(db_conn, log_id, status, error_msg=None, filename=None, 
                      historic_table=None, summary=None, analyzed_data=None): # Added analyzed_data
    """Log analysis results to database, including structured data."""
    if not db_conn:
        return
        
    try:
        ensure_log_table_exists(db_conn)
        
        json_analyzed_data = None
        if analyzed_data:
            try:
                json_analyzed_data = json.dumps(_sanitize_for_json(analyzed_data))
            except Exception as json_e:
                print(f"Error serializing analyzed_data to JSON: {json_e}")
                error_msg = (error_msg or "") + f"; JSON serialization error: {json_e}"

        insert_query = f"""
        INSERT INTO "{ANALYSIS_LOG_TABLE_NAME}" 
        (log_id, run_timestamp, status, error_message, new_data_filename, 
         historic_table_name, analysis_summary, analysis_data)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (log_id) DO UPDATE SET
        run_timestamp = EXCLUDED.run_timestamp,
        status = EXCLUDED.status,
        error_message = EXCLUDED.error_message,
        new_data_filename = EXCLUDED.new_data_filename,
        historic_table_name = EXCLUDED.historic_table_name,
        analysis_summary = EXCLUDED.analysis_summary,
        analysis_data = EXCLUDED.analysis_data; 
        """
        
        params = (
            str(log_id),
            datetime.now(timezone.utc),
            str(status)[:100],
            str(error_msg)[:10000] if error_msg else None,
            str(filename)[:255] if filename else None,
            str(historic_table)[:255] if historic_table else None,
            str(summary)[:20000] if summary else None,
            json_analyzed_data # Add json_analyzed_data
        )
        
        execute_db_query(db_conn, insert_query, params)
        print(f"Logged analysis results to database: {log_id}")
        
    except Exception as e:
        print(f"Failed to log to database: {e}")

def fetch_historic_data(db_conn, table_name, num_sample_rows=HISTORIC_SAMPLE_ROWS_FOR_CONTEXT):
    """Fetch historic data from PostgreSQL table"""
    if not db_conn or not table_name: 
        return None, pd.DataFrame(), {}
    
    try:
        total_rows_res = execute_db_query(db_conn, f'SELECT COUNT(*) FROM "{table_name}";', fetch_one=True)
        historic_total_rows = total_rows_res[0] if total_rows_res else 0
        
        schema_info_rows = execute_db_query(
            db_conn, 
            f"SELECT column_name, data_type FROM information_schema.columns WHERE table_schema = 'public' AND table_name = %s ORDER BY ordinal_position;",
            (table_name,), 
            fetch_all=True
        )
        
        if not schema_info_rows:
            print(f"Could not retrieve schema for historic table '{table_name}'.")
            return historic_total_rows, pd.DataFrame(), {}
        
        db_column_names = [row[0] for row in schema_info_rows]
        safe_select_cols = ", ".join([f'"{col}"' for col in db_column_names])
        sample_rows_query = f'SELECT {safe_select_cols} FROM "{table_name}" ORDER BY RANDOM() LIMIT %s;'
        sample_rows = execute_db_query(db_conn, sample_rows_query, (num_sample_rows,), fetch_all=True)
        
        sample_df = pd.DataFrame()
        if sample_rows:
            sample_df = pd.DataFrame(sample_rows, columns=db_column_names)
            for col_name, col_type_str in schema_info_rows:
                if col_name in sample_df.columns:
                    if any(t in col_type_str for t in ['integer', 'numeric', 'real', 'double precision', 'smallint', 'bigint']):
                        sample_df[col_name] = pd.to_numeric(sample_df[col_name], errors='coerce')
                    elif any(t in col_type_str for t in ['timestamp', 'date']):
                        sample_df[col_name] = pd.to_datetime(sample_df[col_name], errors='coerce')
        
        historic_column_stats = {}
        for col_name, col_type_str in schema_info_rows:
            safe_col = f'"{col_name}"'
            stats = {'type': col_type_str, 'column': col_name}
            
            null_res = execute_db_query(
                db_conn, 
                f"SELECT COUNT(*) - COUNT({safe_col}), COUNT(*) FROM \"{table_name}\";",
                fetch_one=True
            )
            if null_res:
                stats.update({
                    'null_count': _to_native_py_type(null_res[0]), 
                    'total_count': _to_native_py_type(null_res[1]),
                    'null_percentage': (_to_native_py_type(null_res[0]) / _to_native_py_type(null_res[1]) * 100) if null_res[1] > 0 else 0
                })
            
            if any(t in col_type_str for t in ['integer', 'numeric', 'real', 'double precision']):
                num_res = execute_db_query(
                    db_conn,
                    f"""SELECT 
                        AVG(CAST({safe_col} AS NUMERIC)), 
                        STDDEV_SAMP(CAST({safe_col} AS NUMERIC)),
                        MIN(CAST({safe_col} AS NUMERIC)), 
                        MAX(CAST({safe_col} AS NUMERIC)),
                        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY CAST({safe_col} AS NUMERIC)),
                        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY CAST({safe_col} AS NUMERIC)),
                        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY CAST({safe_col} AS NUMERIC))
                        FROM \"{table_name}\" 
                        WHERE {safe_col} IS NOT NULL;""",
                    fetch_one=True
                )
                if num_res:
                    stats.update({
                        'mean': _to_native_py_type(num_res[0]) if num_res[0] is not None else 0,
                        'std': _to_native_py_type(num_res[1]) if num_res[1] is not None else 0,
                        'min': _to_native_py_type(num_res[2]) if num_res[2] is not None else 0,
                        'max': _to_native_py_type(num_res[3]) if num_res[3] is not None else 0,
                        'median': _to_native_py_type(num_res[4]) if num_res[4] is not None else 0,
                        'q25': _to_native_py_type(num_res[5]) if num_res[5] is not None else 0,
                        'q75': _to_native_py_type(num_res[6]) if num_res[6] is not None else 0
                    })
            else:
                cat_res = execute_db_query(
                    db_conn,
                    f"""SELECT COUNT(DISTINCT {safe_col}) as unique_count,
                        (SELECT CAST({safe_col} AS TEXT) 
                         FROM \"{table_name}\" 
                         WHERE {safe_col} IS NOT NULL 
                         GROUP BY {safe_col} 
                         ORDER BY COUNT(*) DESC 
                         LIMIT 1) as most_frequent,
                        (SELECT COUNT(*) 
                         FROM \"{table_name}\" 
                         WHERE {safe_col} IS NOT NULL 
                         GROUP BY {safe_col} 
                         ORDER BY COUNT(*) DESC 
                         LIMIT 1) as most_frequent_count
                        FROM \"{table_name}\" 
                        WHERE {safe_col} IS NOT NULL;""",
                    fetch_one=True
                )
                if cat_res:
                    stats.update({
                        'unique_count': _to_native_py_type(cat_res[0]) if cat_res[0] is not None else 0,
                        'most_frequent': _to_native_py_type(cat_res[1]) if cat_res[1] is not None else None,
                        'most_frequent_count': _to_native_py_type(cat_res[2]) if cat_res[2] is not None else 0
                    })
            
            historic_column_stats[col_name] = stats
        
        return historic_total_rows, sample_df, _sanitize_for_json(historic_column_stats)
        
    except Exception as e:
        print(f"Error fetching historic data from table '{table_name}': {e}")
        return None, pd.DataFrame(), {}

def load_new_data(filepath):
    try:
        df = pd.read_csv(filepath, low_memory=False)
        if df.empty:
            raise ValueError("CSV file is empty")
        print(f"Loaded current data: {df.shape} from {filepath}")
        for col in df.columns:
            if df[col].dtype == object:
                try:
                    parsed_dates = pd.to_datetime(df[col], errors='coerce')
                    if parsed_dates.notna().sum() > 0.5 * len(df):
                        df[col] = parsed_dates
                        print(f"Column '{col}' converted to datetime.")
                except Exception:
                    pass
        return df
    except Exception as e:
        raise RuntimeError(f"Error loading data from {filepath}: {e}")

def calculate_new_stats(df):
    stats = {}
    for column in df.columns:
        col_stats = {
            'column': column,
            'dtype': str(df[column].dtype),
            'null_count': int(df[column].isnull().sum()),
            'total_count': len(df[column]),
            'null_percentage': float(df[column].isnull().sum() / len(df[column]) * 100) if len(df[column]) > 0 else 0
        }
        non_null_data = df[column].dropna()
        if pd.api.types.is_numeric_dtype(df[column]):
            if len(non_null_data) > 0:
                col_stats.update({
                    'mean': float(non_null_data.mean()),
                    'std': float(non_null_data.std()),
                    'min': float(non_null_data.min()),
                    'max': float(non_null_data.max()),
                    'median': float(non_null_data.median()),
                    'q25': float(non_null_data.quantile(0.25)),
                    'q75': float(non_null_data.quantile(0.75))
                })
        else:
            if len(non_null_data) > 0:
                col_stats.update({
                    'unique_count': int(non_null_data.nunique()),
                    'most_frequent': str(non_null_data.mode().iloc[0]) if len(non_null_data.mode()) > 0 else None,
                    'most_frequent_count': int(non_null_data.value_counts().iloc[0]) if len(non_null_data.value_counts()) > 0 else 0
                })
        stats[column] = col_stats
    return stats

def compare_datasets(new_stats, historic_stats, new_df_len, historic_df_len):
    comparison_results = {
        'schema_changes': [],
        'statistical_changes': [], # For mean, std, etc.
        'distribution_changes': [], # For KS-test like (not fully implemented here)
        'volume_changes': [],
        'data_quality_issues': []
    }
    
    new_cols = set(new_stats.keys())
    historic_cols = set(historic_stats.keys())
    
    # Schema changes
    for col in new_cols - historic_cols:
        comparison_results['schema_changes'].append({
            'type': 'column_added',
            'column': col,
            'severity': 'Medium', # Default severity
            'description': f"New column '{col}' found in current dataset."
        })
    
    for col in historic_cols - new_cols:
        comparison_results['schema_changes'].append({
            'type': 'column_removed',
            'column': col,
            'severity': 'Critical', # Default severity
            'description': f"Column '{col}' missing from current dataset (was in historic)."
        })
    
    # Volume changes (Row count)
    if historic_df_len is not None and historic_df_len > 0 :
        row_count_change_pct = ((new_df_len - historic_df_len) / historic_df_len) * 100
        direction = "increase" if row_count_change_pct > 0 else "decrease"
        severity = "Low"
        if abs(row_count_change_pct) > VOLUME_THRESHOLD * 100 * 2: # e.g. > 20%
            severity = "Critical"
        elif abs(row_count_change_pct) > VOLUME_THRESHOLD * 100: # e.g. > 10%
            severity = "Medium"

        if abs(row_count_change_pct) > 0.1: # Minimal change to report
            comparison_results['volume_changes'].append({
                'metric': 'Row Count',
                'type': 'row_count_anomaly',
                'baseline_value': historic_df_len,
                'current_value': new_df_len,
                'change_percentage': row_count_change_pct,
                'direction': direction,
                'severity': severity,
                'description': f"Row count changed by {row_count_change_pct:.1f}% from {historic_df_len:,} to {new_df_len:,}."
            })
    elif historic_df_len is None:
         comparison_results['volume_changes'].append({
            'metric': 'Row Count',
            'type': 'row_count_anomaly',
            'baseline_value': 0, # Placeholder
            'current_value': new_df_len,
            'change_percentage': 0, # Cannot calculate
            'direction': "N/A",
            'severity': "Low",
            'description': f"Current dataset has {new_df_len:,} rows. Historic row count not available for comparison."
        })


    # Statistical and Data Quality changes for common columns
    common_cols = new_cols & historic_cols
    for col in common_cols:
        new_col_stat = new_stats[col]
        hist_col_stat = historic_stats.get(col, {}) # Use .get for safety
        
        # Null percentage change
        new_null_pct = new_col_stat.get('null_percentage', 0)
        hist_null_pct = hist_col_stat.get('null_percentage', 0)
        null_diff = abs(new_null_pct - hist_null_pct)
        if null_diff > 10: # 10% absolute difference in null percentage
            severity = "Critical" if null_diff > 30 else "Medium" if null_diff > 15 else "Low"
            comparison_results['data_quality_issues'].append({
                'type': 'null_percentage_drift',
                'column': col,
                'baseline_value': hist_null_pct,
                'current_value': new_null_pct,
                'change_percentage': null_diff, # This is absolute diff, not relative change for this metric
                'severity': severity,
                'description': f"Null percentage in '{col}' changed from {hist_null_pct:.1f}% to {new_null_pct:.1f}% (Diff: {null_diff:.1f}%)."
            })

        # Numeric stats drift (mean, std)
        if 'mean' in new_col_stat and 'mean' in hist_col_stat:
            new_mean = new_col_stat['mean']
            hist_mean = hist_col_stat['mean']
            mean_drift_score = 0
            mean_change_pct = 0

            if hist_mean is not None and new_mean is not None:
                if abs(hist_mean) > 1e-9: # Avoid division by zero or near-zero
                    mean_change_pct = ((new_mean - hist_mean) / hist_mean) * 100
                    mean_drift_score = abs(mean_change_pct) / 100.0 # Normalize to 0-1+ range
                elif abs(new_mean - hist_mean) > 1e-9 : # if hist_mean is zero, check absolute diff
                     mean_change_pct = float('inf') if new_mean > hist_mean else float('-inf')
                     mean_drift_score = 1.0 # Max drift if baseline was zero and current is not

            if abs(mean_change_pct) > DRIFT_THRESHOLD * 100 : # e.g. > 15%
                severity = "Critical" if abs(mean_change_pct) > (DRIFT_THRESHOLD * 2 * 100) else "Medium"
                comparison_results['statistical_changes'].append({
                    'type': 'mean_drift',
                    'column': col,
                    'baseline_value': hist_mean,
                    'current_value': new_mean,
                    'change_percentage': mean_change_pct,
                    'drift_score': min(mean_drift_score, 1.0), # Cap at 1.0
                    'severity': severity,
                    'description': f"Mean of '{col}' changed by {mean_change_pct:.1f}% (from {hist_mean:.2f} to {new_mean:.2f})."
                })

            new_std = new_col_stat.get('std')
            hist_std = hist_col_stat.get('std')
            std_drift_score = 0
            std_change_pct = 0

            if hist_std is not None and new_std is not None:
                if abs(hist_std) > 1e-9:
                    std_change_pct = ((new_std - hist_std) / hist_std) * 100
                    std_drift_score = abs(std_change_pct) / 100.0
                elif abs(new_std - hist_std) > 1e-9:
                    std_change_pct = float('inf') if new_std > hist_std else float('-inf')
                    std_drift_score = 1.0
            
            if abs(std_change_pct) > DRIFT_THRESHOLD * 100 * 1.33: # e.g. > 20% for std
                severity = "Critical" if abs(std_change_pct) > (DRIFT_THRESHOLD * 2.66 * 100) else "Medium"
                comparison_results['statistical_changes'].append({
                    'type': 'variance_drift', # Or std_drift
                    'column': col,
                    'baseline_value': hist_std,
                    'current_value': new_std,
                    'change_percentage': std_change_pct,
                    'drift_score': min(std_drift_score, 1.0),
                    'severity': severity,
                    'description': f"Std deviation of '{col}' changed by {std_change_pct:.1f}% (from {hist_std:.2f} to {new_std:.2f})."
                })
        
        # Placeholder for distribution drifts (e.g., KS test)
        # This would typically involve comparing distributions from new_df[col] and historic_df[col]
        # For now, this part of comparison_results will remain empty from this function
        # comparison_results['distribution_changes'].append({...})

    return comparison_results

def generate_html_report(log_id, new_df, historic_df_sample, new_stats, historic_stats, 
                        comparison_results_list, new_filename, historic_table_name):
    # This function generates the simple embedded HTML, not the main page structure
    # The main page structure is handled by Flask templates using the full analyzed_data
    
    report_html = f"""
    <div class="report-container card">
        <h3 class="main-title">Embedded Analysis Snippet</h3>
        <p class="log-id-display"><strong>Analysis ID:</strong> {html.escape(str(log_id))}</p>
        <hr>
        <p>This is a basic report snippet. Full details are available in the respective tabs.</p>
    """
    
    schema_changes_items = comparison_results_list.get('schema_changes', [])
    if schema_changes_items:
        report_html += "<h4>Schema Changes Detected:</h4><ul>"
        for item in schema_changes_items[:3]: # Show a few
            report_html += f"<li>{html.escape(item['description'])}</li>"
        if len(schema_changes_items) > 3:
            report_html += "<li>...and more.</li>"
        report_html += "</ul>"

    stat_changes_items = comparison_results_list.get('statistical_changes', [])
    if stat_changes_items:
        report_html += "<h4>Statistical Drifts Detected:</h4><ul>"
        for item in stat_changes_items[:3]:
             report_html += f"<li>{html.escape(item['description'])}</li>"
        if len(stat_changes_items) > 3:
            report_html += "<li>...and more.</li>"
        report_html += "</ul>"
    
    report_html += "</div>"
    return report_html


def run_analysis(new_filepath, historic_csv_filepath=None): # historic_csv_filepath not used
    log_id = uuid.uuid4()
    db_conn = get_db_connection()
    
    new_df = pd.DataFrame()
    historic_df_sample = pd.DataFrame() # Sample for context, not full comparison here
    historic_total_rows = None
    new_filename = os.path.basename(new_filepath) if new_filepath else "N/A"
    historic_table_used = HISTORIC_TABLE_NAME or "Not Specified"

    try:
        new_df = load_new_data(new_filepath)
        new_stats = calculate_new_stats(new_df)
        
        if HISTORIC_TABLE_NAME and db_conn:
            historic_total_rows, historic_df_sample, historic_stats = fetch_historic_data(
                db_conn, HISTORIC_TABLE_NAME, num_sample_rows=HISTORIC_SAMPLE_ROWS_FOR_CONTEXT
            )
        else:
            historic_stats = {} # No historic data to compare against stats-wise

        comparison_details = compare_datasets(
            new_stats, historic_stats, len(new_df), historic_total_rows
        )
        
        # Prepare detailed lists for templates from comparison_details
        # These are already structured lists from compare_datasets
        statistical_drifts = comparison_details.get('statistical_changes', [])
        distribution_drifts = comparison_details.get('distribution_changes', []) # Likely empty
        volume_anomalies = comparison_details.get('volume_changes', [])
        schema_changes_list = comparison_details.get('schema_changes', [])
        data_quality_issues = comparison_details.get('data_quality_issues', [])

        # Generate alerts for critical issues
        alerts = []
        alert_counter = 0
        def add_alert(title, type, severity, description, column=None, drift_score=None):
            nonlocal alert_counter
            alerts.append({
                'id': f"alert_{alert_counter}",
                'title': title, 'type': type, 'severity': severity, 'status': 'active',
                'description': description, 'column': column, 'drift_score': drift_score,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
            alert_counter += 1

        for drift in statistical_drifts:
            if drift['severity'] == 'Critical':
                add_alert(f"Critical Statistical Drift: {drift['column']}", "Statistical Drift", "Critical", 
                          drift['description'], drift['column'], drift['drift_score'])
        for anomaly in volume_anomalies:
            if anomaly['severity'] == 'Critical':
                 add_alert(f"Critical Volume Anomaly: {anomaly['metric']}", "Volume Anomaly", "Critical",
                           anomaly['description'])
        for change in schema_changes_list:
            if change['severity'] == 'Critical': # e.g., column_removed
                add_alert(f"Critical Schema Change: {change['type']} '{change['column']}'", "Schema Change", "Critical",
                          change['description'], change['column'])
        for issue in data_quality_issues:
            if issue['severity'] == 'Critical':
                 add_alert(f"Critical Data Quality Issue: {issue['type']} in '{issue['column']}'", issue['type'], "Critical",
                           issue['description'], issue['column'])


        # Prepare the full analyzed_data structure
        analyzed_data = {
            'log_id': str(log_id),
            'new_filename': new_filename,
            'historic_table_name': historic_table_used,
            'overview': {
                'current_dataset': {'rows': len(new_df), 'columns': len(new_df.columns)},
                'baseline_dataset': {'rows': historic_total_rows if historic_total_rows is not None else 0, 
                                     'columns': len(historic_stats.keys())} # Num columns in historic stats
            },
            'current_preview': new_df.head(MAX_ROWS_FOR_PREVIEW).to_dict(orient='records'),
            # detailed data for specific pages
            'statistical_drifts': statistical_drifts,
            'distribution_drifts': distribution_drifts, # Likely empty
            'volume_anomalies': volume_anomalies,
            'schema_changes': schema_changes_list,
            'data_quality_issues': data_quality_issues, # Add this
            'alerts': alerts,
            # Storing raw stats can be large, consider if needed or summarize
            # 'new_stats': new_stats, 
            # 'historic_stats': historic_stats 
        }
        
        # Generate a simple HTML report snippet (optional, as main views are template-driven)
        html_report_snippet = generate_html_report(
            log_id, new_df, historic_df_sample, new_stats, historic_stats,
            comparison_details, new_filename, historic_table_used
        )
        analyzed_data['report_html'] = html_report_snippet # Embed this basic snippet
        
        # Summarize for logging
        critical_issues_count = len(alerts) # Number of generated alerts
        total_issues_count = (len(statistical_drifts) + len(distribution_drifts) +
                             len(volume_anomalies) + len(schema_changes_list) + len(data_quality_issues))

        status_msg = "Success" if total_issues_count == 0 else f"Completed with {total_issues_count} issues ({critical_issues_count} critical)"
        summary_text = (f"Analysis found {len(statistical_drifts)} stat. drifts, "
                        f"{len(volume_anomalies)} vol. anomalies, "
                        f"{len(schema_changes_list)} schema changes, "
                        f"{len(data_quality_issues)} DQ issues. "
                        f"{critical_issues_count} critical alerts generated.")

        if db_conn:
            log_analysis_to_db(
                db_conn, log_id, status_msg, filename=new_filename,
                historic_table=historic_table_used, summary=summary_text,
                analyzed_data=analyzed_data # Pass the full structured data
            )
        
        # The main Flask app will use analyzed_data to pass to templates.
        # The html_report here is the basic snippet, not the main page.
        return html_report_snippet, "", "", str(log_id), analyzed_data 
        
    except Exception as e:
        error_message = f"Analysis failed: {str(e)}"
        print(f"Error in run_analysis: {error_message}")
        if db_conn:
            log_analysis_to_db(
                db_conn, log_id, "Failed", error_msg=str(e),
                filename=new_filename,
                historic_table=historic_table_used
            )
        error_html = f"""<div class="error-report card"><h2>Analysis Failed</h2>
                         <p><strong>Error:</strong> {html.escape(str(e))}</p>
                         <p><strong>Log ID:</strong> {log_id}</p></div>"""
        error_data = {'log_id': str(log_id), 'error': str(e), 'status': 'failed', 'data':{}} # Add data for consistency
        return error_html, "", "", str(log_id), error_data
        
    finally:
        if db_conn:
            db_conn.close()