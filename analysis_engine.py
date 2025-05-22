# analysis_engine.py - Restored with PostgreSQL Historical Data Functionality

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

# --- Constants ---
DRIFT_THRESHOLD = 0.15  # 15% change threshold for drift detection
VOLUME_THRESHOLD = 0.10  # 10% change threshold for volume anomalies
SIGNIFICANCE_LEVEL = 0.05
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
    if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD]): 
        return None
    try: 
        return psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME, 
                               user=DB_USER, password=DB_PASSWORD)
    except psycopg2.Error as e: 
        print(f"DB Connection Error: {e}")
        return None

def execute_db_query(db_conn, query, params=None, fetch_one=False, fetch_all=False):
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
    except psycopg2.Error as e: 
        print(f"DB Query Error: {e}")
        db_conn.rollback()
        return None

def ensure_log_table_exists(db_conn):
    """Ensure the analysis log table exists"""
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
            analysis_summary TEXT
        );
        """
        execute_db_query(db_conn, create_table_query)
        print(f"Log table '{ANALYSIS_LOG_TABLE_NAME}' ensured to exist.")
    except Exception as e:
        print(f"Error creating log table: {e}")

def log_analysis_to_db(db_conn, log_id, status, error_msg=None, filename=None, 
                      historic_table=None, summary=None):
    """Log analysis results to database"""
    if not db_conn:
        return
        
    try:
        # Ensure log table exists
        ensure_log_table_exists(db_conn)
        
        # Insert log entry
        insert_query = f"""
        INSERT INTO "{ANALYSIS_LOG_TABLE_NAME}" 
        (log_id, run_timestamp, status, error_message, new_data_filename, 
         historic_table_name, analysis_summary)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (log_id) DO UPDATE SET
        run_timestamp = EXCLUDED.run_timestamp,
        status = EXCLUDED.status,
        error_message = EXCLUDED.error_message,
        new_data_filename = EXCLUDED.new_data_filename,
        historic_table_name = EXCLUDED.historic_table_name,
        analysis_summary = EXCLUDED.analysis_summary;
        """
        
        params = (
            str(log_id),
            datetime.now(timezone.utc),
            str(status)[:100],
            str(error_msg)[:10000] if error_msg else None,
            str(filename)[:255] if filename else None,
            str(historic_table)[:255] if historic_table else None,
            str(summary)[:20000] if summary else None
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
        # Get total row count
        total_rows_res = execute_db_query(db_conn, f'SELECT COUNT(*) FROM "{table_name}";', fetch_one=True)
        historic_total_rows = total_rows_res[0] if total_rows_res else 0
        
        # Get column information
        schema_info_rows = execute_db_query(
            db_conn, 
            f"SELECT column_name, data_type FROM information_schema.columns WHERE table_schema = 'public' AND table_name = %s ORDER BY ordinal_position;",
            (table_name,), 
            fetch_all=True
        )
        
        if not schema_info_rows:
            print(f"Could not retrieve schema for historic table '{table_name}'.")
            return historic_total_rows, pd.DataFrame(), {}
        
        # Get sample data
        db_column_names = [row[0] for row in schema_info_rows]
        safe_select_cols = ", ".join([f'"{col}"' for col in db_column_names])
        sample_rows_query = f'SELECT {safe_select_cols} FROM "{table_name}" ORDER BY RANDOM() LIMIT %s;'
        sample_rows = execute_db_query(db_conn, sample_rows_query, (num_sample_rows,), fetch_all=True)
        
        sample_df = pd.DataFrame()
        if sample_rows:
            sample_df = pd.DataFrame(sample_rows, columns=db_column_names)
            
            # Convert data types
            for col_name, col_type_str in schema_info_rows:
                if col_name in sample_df.columns:
                    if any(t in col_type_str for t in ['integer', 'numeric', 'real', 'double precision', 'smallint', 'bigint']):
                        sample_df[col_name] = pd.to_numeric(sample_df[col_name], errors='coerce')
                    elif any(t in col_type_str for t in ['timestamp', 'date']):
                        sample_df[col_name] = pd.to_datetime(sample_df[col_name], errors='coerce')
        
        # Calculate column statistics from database
        historic_column_stats = {}
        for col_name, col_type_str in schema_info_rows:
            safe_col = f'"{col_name}"'
            stats = {'type': col_type_str, 'column': col_name}
            
            # Null count and total
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
            
            # Numeric statistics
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
                # Categorical statistics
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
    """Load current dataset from CSV file"""
    try:
        # Infer data types
        df_sample = pd.read_csv(filepath, nrows=1000, low_memory=False)
        
        # Read full dataset
        df = pd.read_csv(filepath, low_memory=False)
        
        if df.empty:
            raise ValueError("CSV file is empty")
        
        print(f"Loaded current data: {df.shape} from {filepath}")
        
        # Convert date columns if possible
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
    """Calculate statistical summary for current dataset"""
    stats = {}
    
    for column in df.columns:
        col_stats = {
            'column': column,
            'dtype': str(df[column].dtype),
            'null_count': int(df[column].isnull().sum()),
            'total_count': len(df[column]),
            'null_percentage': float(df[column].isnull().sum() / len(df[column]) * 100)
        }
        
        if pd.api.types.is_numeric_dtype(df[column]):
            non_null_data = df[column].dropna()
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
            non_null_data = df[column].dropna()
            if len(non_null_data) > 0:
                col_stats.update({
                    'unique_count': int(non_null_data.nunique()),
                    'most_frequent': str(non_null_data.mode().iloc[0]) if len(non_null_data.mode()) > 0 else None,
                    'most_frequent_count': int(non_null_data.value_counts().iloc[0]) if len(non_null_data.value_counts()) > 0 else 0
                })
        
        stats[column] = col_stats
    
    return stats

def compare_datasets(new_stats, historic_stats):
    """Compare current dataset with historic baseline"""
    comparison_results = {
        'schema_changes': [],
        'statistical_changes': [],
        'volume_changes': [],
        'data_quality_issues': []
    }
    
    new_cols = set(new_stats.keys())
    historic_cols = set(historic_stats.keys())
    
    # Schema changes
    new_columns = new_cols - historic_cols
    missing_columns = historic_cols - new_cols
    
    for col in new_columns:
        comparison_results['schema_changes'].append({
            'type': 'column_added',
            'column': col,
            'description': f"New column '{col}' found in current dataset"
        })
    
    for col in missing_columns:
        comparison_results['schema_changes'].append({
            'type': 'column_removed',
            'column': col,
            'description': f"Column '{col}' missing from current dataset"
        })
    
    # Statistical changes for common columns
    common_cols = new_cols & historic_cols
    
    for col in common_cols:
        new_col = new_stats[col]
        historic_col = historic_stats[col]
        
        # Check for significant changes in null percentage
        new_null_pct = new_col.get('null_percentage', 0)
        historic_null_pct = historic_col.get('null_percentage', 0)
        null_change = abs(new_null_pct - historic_null_pct)
        
        if null_change > 10:  # More than 10% change in null percentage
            comparison_results['data_quality_issues'].append({
                'type': 'null_percentage_change',
                'column': col,
                'historic_null_pct': historic_null_pct,
                'new_null_pct': new_null_pct,
                'change': null_change,
                'description': f"Null percentage changed by {null_change:.1f}% in column '{col}'"
            })
        
        # Statistical drift for numeric columns
        if 'mean' in new_col and 'mean' in historic_col:
            new_mean = new_col['mean']
            historic_mean = historic_col['mean']
            
            if historic_mean != 0:
                mean_change_pct = abs((new_mean - historic_mean) / historic_mean) * 100
                
                if mean_change_pct > 15:  # 15% threshold
                    comparison_results['statistical_changes'].append({
                        'type': 'mean_drift',
                        'column': col,
                        'historic_mean': historic_mean,
                        'new_mean': new_mean,
                        'change_percentage': mean_change_pct,
                        'description': f"Mean changed by {mean_change_pct:.1f}% in column '{col}'"
                    })
            
            # Standard deviation drift
            new_std = new_col.get('std', 0)
            historic_std = historic_col.get('std', 0)
            
            if historic_std > 0:
                std_change_pct = abs((new_std - historic_std) / historic_std) * 100
                
                if std_change_pct > 20:  # 20% threshold for std
                    comparison_results['statistical_changes'].append({
                        'type': 'std_drift',
                        'column': col,
                        'historic_std': historic_std,
                        'new_std': new_std,
                        'change_percentage': std_change_pct,
                        'description': f"Standard deviation changed by {std_change_pct:.1f}% in column '{col}'"
                    })
    
    return comparison_results

def generate_html_report(log_id, new_df, historic_df, new_stats, historic_stats, 
                        comparison_results, new_filename, historic_table_name):
    """Generate comprehensive HTML report"""
    
    report_html = f"""
    <div class="report-container">
        <h2 class="main-title">Data Analysis Report</h2>
        <p class="log-id-display"><strong>Analysis ID:</strong> {log_id}</p>
        <hr>
        
        <div class="dataset-overview card">
            <h3>Dataset Overview</h3>
            <div class="dataset-comparison">
                <div class="dataset-info current">
                    <h4>Current Dataset</h4>
                    <p><strong>Source:</strong> {html.escape(new_filename)}</p>
                    <p><strong>Rows:</strong> {len(new_df):,}</p>
                    <p><strong>Columns:</strong> {len(new_df.columns)}</p>
                </div>
                <div class="dataset-info historic">
                    <h4>Historic Dataset</h4>
                    <p><strong>Source:</strong> PostgreSQL Table: {html.escape(historic_table_name)}</p>
                    <p><strong>Sample Rows:</strong> {len(historic_df):,}</p>
                    <p><strong>Columns:</strong> {len(historic_df.columns) if not historic_df.empty else 0}</p>
                </div>
            </div>
        </div>
    """
    
    # Schema Changes
    schema_changes = comparison_results.get('schema_changes', [])
    if schema_changes:
        report_html += """
        <div class="schema-changes card">
            <h3>Schema Changes</h3>
            <ul>
        """
        for change in schema_changes:
            report_html += f"<li class='{change['type']}'>{html.escape(change['description'])}</li>"
        report_html += "</ul></div>"
    
    # Statistical Changes
    stat_changes = comparison_results.get('statistical_changes', [])
    if stat_changes:
        report_html += """
        <div class="statistical-changes card">
            <h3>Statistical Changes</h3>
            <table class="changes-table">
                <thead>
                    <tr>
                        <th>Column</th>
                        <th>Type</th>
                        <th>Historic Value</th>
                        <th>New Value</th>
                        <th>Change %</th>
                    </tr>
                </thead>
                <tbody>
        """
        for change in stat_changes:
            if change['type'] == 'mean_drift':
                report_html += f"""
                <tr>
                    <td>{html.escape(change['column'])}</td>
                    <td>Mean Drift</td>
                    <td>{change['historic_mean']:.2f}</td>
                    <td>{change['new_mean']:.2f}</td>
                    <td>{change['change_percentage']:.1f}%</td>
                </tr>
                """
            elif change['type'] == 'std_drift':
                report_html += f"""
                <tr>
                    <td>{html.escape(change['column'])}</td>
                    <td>Std Dev Drift</td>
                    <td>{change['historic_std']:.2f}</td>
                    <td>{change['new_std']:.2f}</td>
                    <td>{change['change_percentage']:.1f}%</td>
                </tr>
                """
        report_html += "</tbody></table></div>"
    
    # Data Quality Issues
    dq_issues = comparison_results.get('data_quality_issues', [])
    if dq_issues:
        report_html += """
        <div class="data-quality-issues card">
            <h3>Data Quality Issues</h3>
            <ul>
        """
        for issue in dq_issues:
            report_html += f"<li>{html.escape(issue['description'])}</li>"
        report_html += "</ul></div>"
    
    # Data Previews
    report_html += f"""
        <div class="data-previews card">
            <h3>Data Previews</h3>
            <details open>
                <summary>Current Data (First 10 rows)</summary>
                <div class="table-wrapper">
                    {new_df.head(10).to_html(classes='preview-table', escape=False, table_id='current-data-preview')}
                </div>
            </details>
    """
    
    if not historic_df.empty:
        report_html += f"""
            <details>
                <summary>Historic Data Sample (First 10 rows)</summary>
                <div class="table-wrapper">
                    {historic_df.head(10).to_html(classes='preview-table', escape=False, table_id='historic-data-preview')}
                </div>
            </details>
        """
    
    report_html += "</div>"
    
    # Summary
    total_issues = len(schema_changes) + len(stat_changes) + len(dq_issues)
    if total_issues == 0:
        report_html += """
        <div class="summary card success">
            <h3>Summary</h3>
            <p>✅ No significant issues detected. Your data appears consistent with the historic baseline.</p>
        </div>
        """
    else:
        report_html += f"""
        <div class="summary card warning">
            <h3>Summary</h3>
            <p>⚠️ {total_issues} issue(s) detected that require attention:</p>
            <ul>
                <li>Schema Changes: {len(schema_changes)}</li>
                <li>Statistical Changes: {len(stat_changes)}</li>
                <li>Data Quality Issues: {len(dq_issues)}</li>
            </ul>
        </div>
        """
    
    report_html += "</div>"
    return report_html

def run_analysis(new_filepath, historic_csv_filepath=None):
    """
    Main function to run data analysis comparing current data with historic baseline
    
    Args:
        new_filepath: Path to current CSV file
        historic_csv_filepath: Not used, kept for compatibility
        
    Returns:
        Tuple: (html_report, historic_preview_html, new_preview_html, log_id, analyzed_data)
    """
    log_id = uuid.uuid4()
    db_conn = get_db_connection()
    
    try:
        # Load current dataset
        new_df = load_new_data(new_filepath)
        new_filename = os.path.basename(new_filepath)
        
        # Calculate current dataset statistics
        new_stats = calculate_new_stats(new_df)
        
        # Load historic data from PostgreSQL
        historic_total_rows = None
        historic_df = pd.DataFrame()
        historic_stats = {}
        historic_table_name = HISTORIC_TABLE_NAME or "Not Specified"
        
        if HISTORIC_TABLE_NAME and db_conn:
            historic_total_rows, historic_df, historic_stats = fetch_historic_data(
                db_conn, HISTORIC_TABLE_NAME
            )
        
        # Compare datasets
        comparison_results = compare_datasets(new_stats, historic_stats)
        
        # Generate HTML report
        html_report = generate_html_report(
            str(log_id), new_df, historic_df, new_stats, historic_stats,
            comparison_results, new_filename, historic_table_name
        )
        
        # Generate preview HTML
        new_preview_html = new_df.head(10).to_html(
            classes='preview-table', escape=False, table_id='current-preview'
        )
        
        historic_preview_html = ""
        if not historic_df.empty:
            historic_preview_html = historic_df.head(10).to_html(
                classes='preview-table', escape=False, table_id='historic-preview'
            )
        else:
            historic_preview_html = "<p>No historic data available for preview.</p>"
        
        # Prepare analyzed data with the structure expected by templates
        # Extract statistical drifts
        statistical_drifts = []
        for col, changes in comparison_results.get('statistical_changes', {}).items():
            if changes.get('significant_change', False):
                drift_score = abs(changes.get('percent_change', 0)) / 100
                severity = 'High' if drift_score > 0.5 else 'Medium' if drift_score > 0.2 else 'Low'
                statistical_drifts.append({
                    'column': col,
                    'type': 'statistical_drift',
                    'drift_score': drift_score,
                    'severity': severity,
                    'description': f"Statistical drift detected in {col}. Changed by {changes.get('percent_change', 0):.2f}%."
                })
        
        # Extract distribution drifts
        distribution_drifts = []
        for col, changes in comparison_results.get('distribution_changes', {}).items():
            if changes.get('significant_change', False):
                p_value = changes.get('p_value', 0.5)
                test_statistic = changes.get('test_statistic', 0)
                drift_score = 1 - p_value if p_value <= 1 else 0
                severity = 'High' if p_value < 0.01 else 'Medium' if p_value < 0.05 else 'Low'
                distribution_drifts.append({
                    'column': col,
                    'type': 'distribution_drift',
                    'p_value': p_value,
                    'test_statistic': test_statistic,
                    'drift_score': drift_score,
                    'severity': severity,
                    'description': f"Distribution drift detected in {col}. P-value: {p_value:.4f}."
                })
        
        # Extract volume anomalies
        volume_anomalies = []
        row_count_change = comparison_results.get('row_count_change', 0)
        if abs(row_count_change) > 10:
            severity = 'High' if abs(row_count_change) > 50 else 'Medium' if abs(row_count_change) > 20 else 'Low'
            volume_anomalies.append({
                'metric': 'Row Count',
                'type': 'volume_anomaly',
                'change_percent': row_count_change,
                'severity': severity,
                'description': f"Row count changed by {row_count_change:.2f}%."
            })
        
        # Extract schema changes
        schema_changes = []
        for change in comparison_results.get('schema_changes', []):
            severity = 'High' if change.get('type') == 'column_removed' else 'Medium'
            schema_changes.append({
                'column': change.get('column', ''),
                'type': change.get('type', ''),
                'severity': severity,
                'description': change.get('description', '')
            })
        
        # Generate alerts for critical issues
        alerts = []
        # Add high severity statistical drifts to alerts
        for drift in statistical_drifts:
            if drift['severity'] == 'High':
                alerts.append({
                    'id': f"stat_{len(alerts)}",
                    'title': f"Critical Statistical Drift in {drift['column']}",
                    'type': 'Statistical Drift',
                    'severity': 'Critical',
                    'status': 'active',
                    'column': drift['column'],
                    'drift_score': drift['drift_score'],
                    'description': drift['description'],
                    'timestamp': datetime.now().isoformat()
                })
        
        # Add high severity distribution drifts to alerts
        for drift in distribution_drifts:
            if drift['severity'] == 'High':
                alerts.append({
                    'id': f"dist_{len(alerts)}",
                    'title': f"Critical Distribution Drift in {drift['column']}",
                    'type': 'Distribution Drift',
                    'severity': 'Critical',
                    'status': 'active',
                    'column': drift['column'],
                    'drift_score': drift['drift_score'],
                    'description': drift['description'],
                    'timestamp': datetime.now().isoformat()
                })
        
        # Add high severity volume anomalies to alerts
        for anomaly in volume_anomalies:
            if anomaly['severity'] == 'High':
                alerts.append({
                    'id': f"vol_{len(alerts)}",
                    'title': f"Critical Volume Change in {anomaly['metric']}",
                    'type': 'Volume Anomaly',
                    'severity': 'Critical',
                    'status': 'active',
                    'description': anomaly['description'],
                    'timestamp': datetime.now().isoformat()
                })
        
        # Add schema changes to alerts
        for change in schema_changes:
            if change['severity'] == 'High':
                alerts.append({
                    'id': f"schema_{len(alerts)}",
                    'title': f"Critical Schema Change: {change['type'].replace('_', ' ').title()}",
                    'type': 'Schema Change',
                    'severity': 'Critical',
                    'status': 'active',
                    'column': change.get('column', ''),
                    'description': change['description'],
                    'timestamp': datetime.now().isoformat()
                })
        
        analyzed_data = {
            'log_id': str(log_id),
            'new_filename': new_filename,
            'historic_table_name': historic_table_name,
            'new_shape': new_df.shape,
            'historic_shape': historic_df.shape,
            'new_stats': new_stats,
            'historic_stats': historic_stats,
            'comparison_results': comparison_results,
            'statistical_drifts': statistical_drifts,
            'distribution_drifts': distribution_drifts,
            'volume_anomalies': volume_anomalies,
            'schema_changes': schema_changes,
            'alerts': alerts
        }
        
        # Log to database
        total_issues = (len(comparison_results.get('schema_changes', [])) + 
                       len(comparison_results.get('statistical_changes', [])) + 
                       len(comparison_results.get('data_quality_issues', [])))
        
        status = "Success" if total_issues == 0 else f"Completed with {total_issues} issues"
        summary = f"Analysis completed. Found {total_issues} issues requiring attention."
        
        if db_conn:
            log_analysis_to_db(
                db_conn, log_id, status, filename=new_filename,
                historic_table=historic_table_name, summary=summary
            )
        
        return html_report, historic_preview_html, new_preview_html, str(log_id), analyzed_data
        
    except Exception as e:
        error_message = f"Analysis failed: {str(e)}"
        print(f"Error in run_analysis: {error_message}")
        
        # Log error to database
        if db_conn:
            log_analysis_to_db(
                db_conn, log_id, "Failed", error_msg=str(e),
                filename=os.path.basename(new_filepath) if new_filepath else None,
                historic_table=HISTORIC_TABLE_NAME
            )
        
        error_html = f"""
        <div class="error-report card">
            <h2>Analysis Failed</h2>
            <p><strong>Error:</strong> {html.escape(str(e))}</p>
            <p><strong>Log ID:</strong> {log_id}</p>
        </div>
        """
        
        error_data = {
            'log_id': str(log_id),
            'error': str(e),
            'status': 'failed'
        }
        
        return error_html, "", "", str(log_id), error_data
        
    finally:
        if db_conn:
            db_conn.close()