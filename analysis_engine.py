# analysis_engine.py

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, ttest_ind # ttest_ind not currently used but kept
import google.generativeai as genai
import os
import warnings
from dotenv import load_dotenv
import json
import re
from collections import defaultdict
import psycopg2
import uuid # CORRECTED: was 'uu' before, now 'uuid'
from datetime import datetime, timezone, date
import decimal
import html

# --- Configuration ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Supabase PostgreSQL Configuration
SUPABASE_DB_HOST = os.getenv("HOST")
SUPABASE_DB_PORT = os.getenv("PORT", "6543") # Default to Supabase pooler port
SUPABASE_DB_NAME = os.getenv("DBNAME", "postgres")
SUPABASE_DB_USER = os.getenv("USER")
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
DRIFT_THRESHOLD = 0.15
VOLUME_THRESHOLD = 0.10
SIGNIFICANCE_LEVEL = 0.05 # Not directly used in current comparisons but good to have
MAX_ROWS_FOR_PREVIEW = 10
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
    if isinstance(value, uuid.UUID): # Check for uuid.UUID
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
        print("Missing Supabase database configuration. Check your environment variables: HOST, DBNAME, USER, PASSWORD.")
        return None
    try:
        conn = psycopg2.connect(
            host=SUPABASE_DB_HOST,
            port=SUPABASE_DB_PORT, # Ensure this is the pooler port (e.g., 6543)
            dbname=SUPABASE_DB_NAME,
            user=SUPABASE_DB_USER,
            password=SUPABASE_DB_PASSWORD,
            sslmode='require', # Essential for Supabase
            connect_timeout=10 # Good practice
        )
        print(f"Attempting to connect to Supabase: Host={SUPABASE_DB_HOST}, Port={SUPABASE_DB_PORT}, DB={SUPABASE_DB_NAME}, User={SUPABASE_DB_USER}")
        print("Successfully connected to Supabase database.")
        return conn
    except psycopg2.Error as e:
        print(f"Failed to connect to Supabase database: {e}")
        return None

def execute_db_query(db_conn, query, params=None, fetch_one=False, fetch_all=False):
    """Execute a query on Supabase PostgreSQL database with proper error handling"""
    if not db_conn:
        print("execute_db_query: No database connection provided.")
        return None
    try:
        with db_conn.cursor() as cursor:
            cursor.execute(query, params)
            if fetch_one:
                return cursor.fetchone()
            if fetch_all:
                return cursor.fetchall()
            db_conn.commit() # Commit only if not fetching (i.e., for INSERT, UPDATE, CREATE)
            return True
    except psycopg2.OperationalError as e: # Specific error for connection issues
        print(f"Supabase OperationalError during query execution: {e}")
        # It's often good to try to re-establish connection or handle this gracefully
        # For now, we'll just rollback and return None
        try:
            db_conn.rollback()
        except psycopg2.Error: # If rollback fails (e.g. connection truly gone)
            pass 
        return None
    except psycopg2.Error as e:
        print(f"Supabase Query Error: {e}")
        try:
            db_conn.rollback()
        except psycopg2.Error:
            pass
        return None
    except Exception as e: # Catch any other unexpected errors
        print(f"Unexpected error during DB query: {e}")
        try:
            db_conn.rollback()
        except psycopg2.Error:
            pass
        return None


def ensure_log_table_exists(db_conn):
    """Ensure the analysis log table exists and has the analysis_data column."""
    if not db_conn:
        print("ensure_log_table_exists: No database connection provided.")
        return
    
    try:
        # Get connection info for logging (BE CAREFUL NOT TO LOG PASSWORDS)
        conn_info = db_conn.get_dsn_parameters()
        print(f"ensure_log_table_exists: Verifying log table on host '{conn_info.get('host')}', port '{conn_info.get('port')}', dbname '{conn_info.get('dbname')}', user '{conn_info.get('user')}'")

        # Check if table exists
        table_exists_query = f"""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name = %s
        );
        """
        table_exists_result = execute_db_query(db_conn, table_exists_query, (ANALYSIS_LOG_TABLE_NAME,), fetch_one=True)

        if not (table_exists_result and table_exists_result[0]):
            print(f"Log table '{ANALYSIS_LOG_TABLE_NAME}' does not exist. Creating...")
            create_table_query = f"""
            CREATE TABLE "{ANALYSIS_LOG_TABLE_NAME}" (
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
            if execute_db_query(db_conn, create_table_query):
                print(f"Log table '{ANALYSIS_LOG_TABLE_NAME}' created successfully with analysis_data column on host '{conn_info.get('host')}'.")
            else:
                print(f"Failed to create log table '{ANALYSIS_LOG_TABLE_NAME}' on host '{conn_info.get('host')}'.")
                return # Stop if table creation fails
        else:
            print(f"Log table '{ANALYSIS_LOG_TABLE_NAME}' already exists on host '{conn_info.get('host')}'. Checking for 'analysis_data' column...")
            # Check if analysis_data column exists
            column_exists_query = f"""
            SELECT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_schema = 'public' 
                AND table_name = %s
                AND column_name = 'analysis_data'
            );
            """
            column_exists_result = execute_db_query(db_conn, column_exists_query, (ANALYSIS_LOG_TABLE_NAME,), fetch_one=True)
            if not (column_exists_result and column_exists_result[0]):
                print(f"'analysis_data' column missing in '{ANALYSIS_LOG_TABLE_NAME}'. Altering table...")
                alter_table_query = f"""
                ALTER TABLE "{ANALYSIS_LOG_TABLE_NAME}"
                ADD COLUMN analysis_data JSONB;
                """
                if execute_db_query(db_conn, alter_table_query):
                    print(f"Added 'analysis_data' JSONB column to '{ANALYSIS_LOG_TABLE_NAME}' on host '{conn_info.get('host')}'.")
                else:
                     print(f"Failed to add 'analysis_data' column to '{ANALYSIS_LOG_TABLE_NAME}' on host '{conn_info.get('host')}'.")
            else:
                print(f"'analysis_data' column already exists in '{ANALYSIS_LOG_TABLE_NAME}' on host '{conn_info.get('host')}'. Schema is up to date.")
                
    except Exception as e:
        print(f"Error in ensure_log_table_exists: {e}")


def log_analysis_to_db(db_conn, log_id, status, error_msg=None, filename=None, 
                      historic_table=None, summary=None, analyzed_data=None):
    """Log analysis results to database, including structured data."""
    if not db_conn:
        print("log_analysis_to_db: No database connection provided.")
        return
        
    try:
        # Ensure the log table exists and has the correct schema BEFORE attempting to insert
        # ensure_log_table_exists is now more robust and will be called at app startup or before first log
        
        json_analyzed_data = None
        if analyzed_data:
            try:
                # Ensure all nested data is also sanitized
                sanitized_data_for_json = _sanitize_for_json(analyzed_data)
                json_analyzed_data = json.dumps(sanitized_data_for_json)
            except Exception as json_e:
                print(f"Error serializing analyzed_data to JSON: {json_e}")
                # Log a simplified version or just the error if serialization fails completely
                simplified_error_data = {'error': f"JSON serialization failed: {str(json_e)}", 'original_log_id': str(log_id)}
                json_analyzed_data = json.dumps(simplified_error_data)
                error_msg = (error_msg or "") + f"; JSON serialization error for detailed data: {json_e}"


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
            log_id, # Pass UUID object directly if your psycopg2 version handles it, else str(log_id)
            datetime.now(timezone.utc),
            str(status)[:100],
            str(error_msg)[:10000] if error_msg else None,
            str(filename)[:255] if filename else None,
            str(historic_table)[:255] if historic_table else None,
            str(summary)[:20000] if summary else None,
            json_analyzed_data
        )
        
        if execute_db_query(db_conn, insert_query, params):
            print(f"Logged analysis results to database: {log_id}")
        else:
            print(f"Failed to log analysis results to database for log ID: {log_id}")
        
    except Exception as e:
        print(f"Unexpected error in log_analysis_to_db: {e}")


def fetch_historic_data(db_conn, table_name, num_sample_rows=HISTORIC_SAMPLE_ROWS_FOR_CONTEXT):
    if not db_conn or not table_name:
        print("fetch_historic_data: No DB connection or table name.")
        return None, pd.DataFrame(), {}
    
    try:
        print(f"Fetching historic data for table: {table_name}")
        total_rows_res = execute_db_query(db_conn, f'SELECT COUNT(*) FROM "{html.escape(table_name)}";', fetch_one=True) # Escape table name
        historic_total_rows = total_rows_res[0] if total_rows_res else 0
        print(f"Historic table '{table_name}' has {historic_total_rows} rows.")

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
        safe_select_cols = ", ".join([f'"{col}"' for col in db_column_names]) # Already safe due to selection from schema
        sample_rows_query = f'SELECT {safe_select_cols} FROM "{html.escape(table_name)}" ORDER BY RANDOM() LIMIT %s;'
        sample_rows = execute_db_query(db_conn, sample_rows_query, (num_sample_rows,), fetch_all=True)
        
        sample_df = pd.DataFrame()
        if sample_rows:
            sample_df = pd.DataFrame(sample_rows, columns=db_column_names)
            for col_name, col_type_str in schema_info_rows:
                if col_name in sample_df.columns:
                    # Basic type conversion based on PostgreSQL types
                    if 'int' in col_type_str or 'serial' in col_type_str:
                        sample_df[col_name] = pd.to_numeric(sample_df[col_name], errors='coerce').astype('Int64') # Nullable Int
                    elif 'numeric' in col_type_str or 'decimal' in col_type_str or 'real' in col_type_str or 'double precision' in col_type_str:
                        sample_df[col_name] = pd.to_numeric(sample_df[col_name], errors='coerce')
                    elif 'timestamp' in col_type_str or 'date' in col_type_str:
                        sample_df[col_name] = pd.to_datetime(sample_df[col_name], errors='coerce')
                    elif 'boolean' in col_type_str:
                         sample_df[col_name] = sample_df[col_name].astype('boolean') # Nullable Boolean
        
        historic_column_stats = {}
        for col_name, col_type_str in schema_info_rows:
            safe_col = f'"{col_name}"'
            stats = {'type': col_type_str, 'column': col_name}
            
            null_res = execute_db_query(db_conn, f"SELECT COUNT(*) - COUNT({safe_col}), COUNT(*) FROM \"{html.escape(table_name)}\";", fetch_one=True)
            if null_res and null_res[1] is not None: # Ensure total count is not None
                null_count = null_res[0] if null_res[0] is not None else 0
                total_count = null_res[1] if null_res[1] > 0 else 0 # Avoid division by zero
                stats.update({
                    'null_count': _to_native_py_type(null_count), 
                    'total_count': _to_native_py_type(total_count),
                    'null_percentage': (_to_native_py_type(null_count) / _to_native_py_type(total_count) * 100) if total_count > 0 else 0
                })
            
            if any(t in col_type_str for t in ['integer', 'numeric', 'real', 'double precision', 'smallint', 'bigint', 'decimal']):
                num_query = f"""SELECT 
                        AVG(CAST({safe_col} AS NUMERIC)), STDDEV_SAMP(CAST({safe_col} AS NUMERIC)),
                        MIN(CAST({safe_col} AS NUMERIC)), MAX(CAST({safe_col} AS NUMERIC)),
                        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY CAST({safe_col} AS NUMERIC)),
                        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY CAST({safe_col} AS NUMERIC)),
                        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY CAST({safe_col} AS NUMERIC))
                        FROM \"{html.escape(table_name)}\" WHERE {safe_col} IS NOT NULL;"""
                num_res = execute_db_query(db_conn, num_query, fetch_one=True)
                if num_res:
                    stats.update({
                        'mean': _to_native_py_type(num_res[0]), 'std': _to_native_py_type(num_res[1]),
                        'min': _to_native_py_type(num_res[2]), 'max': _to_native_py_type(num_res[3]),
                        'median': _to_native_py_type(num_res[4]), 'q25': _to_native_py_type(num_res[5]),
                        'q75': _to_native_py_type(num_res[6])
                    })
            else: # For non-numeric types (text, varchar, date, boolean etc.)
                cat_query = f"""SELECT COUNT(DISTINCT {safe_col}) as unique_count,
                        (SELECT CAST({safe_col} AS TEXT) FROM \"{html.escape(table_name)}\" WHERE {safe_col} IS NOT NULL GROUP BY {safe_col} ORDER BY COUNT(*) DESC LIMIT 1) as most_frequent,
                        (SELECT COUNT(*) FROM \"{html.escape(table_name)}\" WHERE {safe_col} IS NOT NULL GROUP BY {safe_col} ORDER BY COUNT(*) DESC LIMIT 1) as most_frequent_count
                        FROM \"{html.escape(table_name)}\" WHERE {safe_col} IS NOT NULL;"""
                cat_res = execute_db_query(db_conn, cat_query, fetch_one=True)
                if cat_res:
                    stats.update({
                        'unique_count': _to_native_py_type(cat_res[0]),
                        'most_frequent': _to_native_py_type(cat_res[1]),
                        'most_frequent_count': _to_native_py_type(cat_res[2])
                    })
            historic_column_stats[col_name] = stats
        
        return historic_total_rows, sample_df, _sanitize_for_json(historic_column_stats)
        
    except Exception as e:
        print(f"Error in fetch_historic_data for table '{table_name}': {e}")
        return None, pd.DataFrame(), {}


def load_new_data(filepath):
    try:
        df = pd.read_csv(filepath, low_memory=False)
        if df.empty:
            raise ValueError("CSV file is empty")
        print(f"Loaded current data: {df.shape} from {filepath}")
        for col in df.columns:
            if df[col].dtype == object: # Try to infer types more broadly
                try: # Attempt datetime conversion
                    parsed_dates = pd.to_datetime(df[col], errors='coerce')
                    # If a significant portion converted, assume it's a date column
                    if parsed_dates.notna().sum() > 0.5 * len(df[col].dropna()):
                        df[col] = parsed_dates
                        print(f"Column '{col}' converted to datetime.")
                        continue # Move to next column
                except Exception:
                    pass # Failed datetime conversion, try numeric
                
                try: # Attempt numeric conversion
                    parsed_numeric = pd.to_numeric(df[col], errors='coerce')
                    if parsed_numeric.notna().sum() > 0.5 * len(df[col].dropna()):
                        df[col] = parsed_numeric
                        print(f"Column '{col}' converted to numeric.")
                except Exception:
                    pass # Failed numeric conversion, leave as object
        return df
    except Exception as e:
        raise RuntimeError(f"Error loading data from {filepath}: {e}")


def calculate_new_stats(df):
    stats = {}
    if df.empty: return stats
    for column in df.columns:
        col_data = df[column].dropna()
        col_stats = {
            'column': column,
            'dtype': str(df[column].dtype), # Original dtype from pandas
            'null_count': int(df[column].isnull().sum()),
            'total_count': len(df[column]),
            'null_percentage': float(df[column].isnull().sum() / len(df[column]) * 100) if len(df[column]) > 0 else 0
        }
        
        if pd.api.types.is_numeric_dtype(df[column]): # Check original dtype
            if not col_data.empty:
                col_stats.update({
                    'mean': float(col_data.mean()), 'std': float(col_data.std()),
                    'min': float(col_data.min()), 'max': float(col_data.max()),
                    'median': float(col_data.median()),
                    'q25': float(col_data.quantile(0.25)), 'q75': float(col_data.quantile(0.75))
                })
        elif pd.api.types.is_datetime64_any_dtype(df[column]): # Check for datetime
             if not col_data.empty:
                col_stats.update({
                    'min': col_data.min().isoformat(), # Min date
                    'max': col_data.max().isoformat(), # Max date
                    'unique_count': int(col_data.nunique())
                })
        else: # Categorical / Object
            if not col_data.empty:
                col_stats.update({
                    'unique_count': int(col_data.nunique()),
                    'most_frequent': str(col_data.mode().iloc[0]) if not col_data.mode().empty else None,
                    'most_frequent_count': int(col_data.value_counts().iloc[0]) if not col_data.value_counts().empty else 0
                })
        stats[column] = col_stats
    return stats


def compare_datasets(new_stats, historic_stats, new_df_len, historic_df_len):
    comparison_results = {
        'schema_changes': [], 'statistical_changes': [], 'distribution_changes': [],
        'volume_changes': [], 'data_quality_issues': []
    }
    new_cols = set(new_stats.keys())
    historic_cols = set(historic_stats.keys())

    for col in new_cols - historic_cols:
        comparison_results['schema_changes'].append({'type': 'column_added', 'column': col, 'severity': 'Medium', 'description': f"New column '{col}' found."})
    for col in historic_cols - new_cols:
        comparison_results['schema_changes'].append({'type': 'column_removed', 'column': col, 'severity': 'Critical', 'description': f"Column '{col}' missing."})

    if historic_df_len is not None and historic_df_len > 0:
        row_count_change_pct = ((new_df_len - historic_df_len) / historic_df_len) * 100
        direction = "increase" if row_count_change_pct > 0 else "decrease"
        severity = "Low"
        if abs(row_count_change_pct) > VOLUME_THRESHOLD * 200: severity = "Critical" # >20%
        elif abs(row_count_change_pct) > VOLUME_THRESHOLD * 100: severity = "Medium" # >10%
        if abs(row_count_change_pct) > 0.1:
            comparison_results['volume_changes'].append({
                'metric': 'Row Count', 'type': 'row_count_anomaly', 'baseline_value': historic_df_len,
                'current_value': new_df_len, 'change_percentage': row_count_change_pct,
                'direction': direction, 'severity': severity,
                'description': f"Row count changed by {row_count_change_pct:.1f}% (from {historic_df_len:,} to {new_df_len:,})."
            })
    elif historic_df_len is None: # historic_df_len could be 0 if table was empty
         comparison_results['volume_changes'].append({
            'metric': 'Row Count', 'type': 'row_count_info', 'baseline_value': 0,
            'current_value': new_df_len, 'change_percentage': 0, 'direction': "N/A", 'severity': "Info",
            'description': f"Current dataset: {new_df_len:,} rows. Baseline row count not available or zero."
        })


    common_cols = new_cols & historic_cols
    for col in common_cols:
        new_s, hist_s = new_stats[col], historic_stats.get(col, {})
        new_null_pct, hist_null_pct = new_s.get('null_percentage', 0), hist_s.get('null_percentage', 0)
        null_diff = abs(new_null_pct - hist_null_pct)
        if null_diff > 10:
            dq_sev = "Critical" if null_diff > 30 else "Medium" if null_diff > 15 else "Low"
            comparison_results['data_quality_issues'].append({
                'type': 'null_percentage_drift', 'column': col, 'baseline_value': hist_null_pct,
                'current_value': new_null_pct, 'change_percentage': null_diff, 'severity': dq_sev,
                'description': f"Nulls in '{col}': {hist_null_pct:.1f}% to {new_null_pct:.1f}%."
            })

        if all(k in new_s and k in hist_s for k in ['mean', 'std']):
            new_mean, hist_mean = new_s['mean'], hist_s['mean']
            mean_change_pct, mean_drift_score = 0, 0
            if hist_mean is not None and new_mean is not None:
                if abs(hist_mean) > 1e-9:
                    mean_change_pct = ((new_mean - hist_mean) / hist_mean) * 100
                    mean_drift_score = abs(mean_change_pct) / 100.0
                elif abs(new_mean - hist_mean) > 1e-9: mean_drift_score = 1.0

            if abs(mean_change_pct) > DRIFT_THRESHOLD * 100:
                m_sev = "Critical" if abs(mean_change_pct) > DRIFT_THRESHOLD * 200 else "Medium"
                comparison_results['statistical_changes'].append({
                    'type': 'mean_drift', 'column': col, 'baseline_value': hist_mean, 'current_value': new_mean,
                    'change_percentage': mean_change_pct, 'drift_score': min(mean_drift_score, 1.0), 'severity': m_sev,
                    'description': f"Mean of '{col}' changed by {mean_change_pct:.1f}%."
                })

            new_std, hist_std = new_s.get('std'), hist_s.get('std')
            std_change_pct, std_drift_score = 0, 0
            if hist_std is not None and new_std is not None:
                if abs(hist_std) > 1e-9:
                    std_change_pct = ((new_std - hist_std) / hist_std) * 100
                    std_drift_score = abs(std_change_pct) / 100.0
                elif abs(new_std - hist_std) > 1e-9: std_drift_score = 1.0
            
            if abs(std_change_pct) > DRIFT_THRESHOLD * 133: # ~20% for std
                s_sev = "Critical" if abs(std_change_pct) > DRIFT_THRESHOLD * 266 else "Medium"
                comparison_results['statistical_changes'].append({
                    'type': 'variance_drift', 'column': col, 'baseline_value': hist_std, 'current_value': new_std,
                    'change_percentage': std_change_pct, 'drift_score': min(std_drift_score, 1.0), 'severity': s_sev,
                    'description': f"Std Dev of '{col}' changed by {std_change_pct:.1f}%."
                })
    return comparison_results

def generate_html_report(*args): # Placeholder, not the primary report generator
    return "<div class='card'><p>Basic analysis snippet. See tabs for full details.</p></div>"


def run_analysis(new_filepath, historic_csv_filepath=None):
    log_id = uuid.uuid4() # Use the imported uuid
    db_conn = get_db_connection()
    
    # Ensure log table exists early, especially if this is the first run or schema might change
    if db_conn:
        ensure_log_table_exists(db_conn) # Call it here
    else:
        print("Cannot ensure log table exists: No DB connection.")


    new_df = pd.DataFrame()
    historic_df_sample = pd.DataFrame()
    historic_total_rows = 0 # Initialize to 0
    new_filename = os.path.basename(new_filepath) if new_filepath else "N/A"
    historic_table_used = HISTORIC_TABLE_NAME or "Not Specified"
    historic_stats = {} # Initialize

    try:
        new_df = load_new_data(new_filepath)
        new_stats = calculate_new_stats(new_df)
        
        if HISTORIC_TABLE_NAME and db_conn:
            # fetch_historic_data returns: total_rows, sample_df, column_stats
            historic_total_rows, historic_df_sample, historic_stats_from_db = fetch_historic_data(
                db_conn, HISTORIC_TABLE_NAME, num_sample_rows=HISTORIC_SAMPLE_ROWS_FOR_CONTEXT
            )
            if historic_stats_from_db: # Check if stats were successfully fetched
                 historic_stats = historic_stats_from_db
            if historic_total_rows is None: historic_total_rows = 0 # Ensure it's not None
        
        comparison_details = compare_datasets(new_stats, historic_stats, len(new_df), historic_total_rows)
        
        statistical_drifts = comparison_details.get('statistical_changes', [])
        distribution_drifts = comparison_details.get('distribution_changes', [])
        volume_anomalies = comparison_details.get('volume_changes', [])
        schema_changes_list = comparison_details.get('schema_changes', [])
        data_quality_issues = comparison_details.get('data_quality_issues', [])

        alerts = []
        alert_counter = 0
        def add_alert(title, type, severity, description, column=None, drift_score=None):
            nonlocal alert_counter
            alerts.append({
                'id': f"alert_{alert_counter}_{str(uuid.uuid4())[:8]}", # Make ID more unique
                'title': title, 'type': type, 'severity': severity, 'status': 'active',
                'description': description, 'column': column, 'drift_score': drift_score,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
            alert_counter += 1

        for item_list, item_type_prefix in [
            (statistical_drifts, "StatDrift"), (volume_anomalies, "VolAnomaly"),
            (schema_changes_list, "SchemaChg"), (data_quality_issues, "DQIssue")
        ]:
            for item in item_list:
                if item.get('severity') == 'Critical':
                    title = f"Critical {item.get('type', 'Issue')}"
                    if item.get('column'): title += f": {item.get('column')}"
                    elif item.get('metric'): title += f": {item.get('metric')}" # For volume
                    
                    add_alert(title, item.get('type', 'Unknown Type'), "Critical", 
                              item['description'], item.get('column'), item.get('drift_score'))
        
        analyzed_data = {
            'log_id': str(log_id), 'new_filename': new_filename, 'historic_table_name': historic_table_used,
            'overview': {
                'current_dataset': {'rows': len(new_df), 'columns': len(new_df.columns)},
                'baseline_dataset': {'rows': historic_total_rows, 'columns': len(historic_stats.keys())}
            },
            'current_preview': new_df.head(MAX_ROWS_FOR_PREVIEW).to_dict(orient='records'),
            'statistical_drifts': statistical_drifts, 'distribution_drifts': distribution_drifts,
            'volume_anomalies': volume_anomalies, 'schema_changes': schema_changes_list,
            'data_quality_issues': data_quality_issues, 'alerts': alerts,
            'report_html': generate_html_report(log_id, new_df, historic_df_sample, new_stats, historic_stats, comparison_details, new_filename, historic_table_used)
        }
        
        critical_issues_count = len([a for a in alerts if a['severity'] == 'Critical'])
        total_issues_identified = sum(len(lst) for lst in [statistical_drifts, distribution_drifts, volume_anomalies, schema_changes_list, data_quality_issues])
        status_msg = "Success" if total_issues_identified == 0 else f"Completed with {total_issues_identified} issues ({critical_issues_count} critical)"
        summary_text = f"Analysis: {len(statistical_drifts)} stat drifts, {len(volume_anomalies)} vol anomalies, {len(schema_changes_list)} schema changes, {len(data_quality_issues)} DQ issues. {critical_issues_count} critical alerts."

        if db_conn:
            log_analysis_to_db(db_conn, log_id, status_msg, filename=new_filename, historic_table=historic_table_used, summary=summary_text, analyzed_data=analyzed_data)
        
        return analyzed_data['report_html'], "", "", str(log_id), analyzed_data 
        
    except Exception as e:
        error_message = f"Analysis failed: {str(e)}"
        print(f"Error in run_analysis for {new_filename}: {error_message}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        if db_conn:
            log_analysis_to_db(db_conn, log_id, "Failed", error_msg=error_message, filename=new_filename, historic_table=historic_table_used)
        
        error_html = f"<div class='error-report card'><h2>Analysis Failed</h2><p><strong>Error:</strong> {html.escape(error_message)}</p><p><strong>Log ID:</strong> {log_id}</p></div>"
        error_data = {'log_id': str(log_id), 'error': error_message, 'status': 'failed', 'data':{}}
        return error_html, "", "", str(log_id), error_data
        
    finally:
        if db_conn:
            db_conn.close()
            print("Database connection closed.")