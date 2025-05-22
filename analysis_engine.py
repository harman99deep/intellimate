# analysis_engine.py - Refactored for Data Drift Detection

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

# --- Drift Types ---
DRIFT_TYPE_STATISTICAL = "statistical_drift"
DRIFT_TYPE_DISTRIBUTION = "distribution_drift"
DRIFT_TYPE_VOLUME = "volume_drift"
DRIFT_TYPE_SCHEMA = "schema_drift"

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

class DataDriftAnalyzer:
    """Main class for detecting data drift between baseline and current datasets"""
    
    def __init__(self, drift_threshold=DRIFT_THRESHOLD, volume_threshold=VOLUME_THRESHOLD, 
                 significance_level=SIGNIFICANCE_LEVEL):
        self.drift_threshold = drift_threshold
        self.volume_threshold = volume_threshold
        self.significance_level = significance_level
        
    def analyze_datasets(self, current_df, baseline_stats, baseline_total_rows, 
                        current_filename, baseline_filename):
        """
        Perform comprehensive data drift analysis
        
        Args:
            current_df: Current dataset to analyze
            baseline_stats: Statistical summary of baseline dataset
            baseline_total_rows: Total rows in baseline dataset
            current_filename: Name of current dataset file
            baseline_filename: Name of baseline dataset source
            
        Returns:
            Dictionary containing analysis results organized by sections
        """
        run_id = f"RUN_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Calculate current dataset statistics
        current_stats = self._calculate_dataset_stats(current_df)
        current_total_rows = len(current_df)
        
        # Detect different types of drift
        statistical_drifts = self._detect_statistical_drift(current_stats, baseline_stats)
        distribution_drifts = self._detect_distribution_drift(current_df, baseline_stats)
        volume_anomalies = self._detect_volume_anomalies(
            current_total_rows, baseline_total_rows, 
            len(current_df.columns), len(baseline_stats.keys()) if baseline_stats else 0
        )
        schema_changes = self._detect_schema_changes(
            list(current_df.columns), list(baseline_stats.keys()) if baseline_stats else []
        )
        
        # Generate alerts for critical issues
        alerts = self._generate_alerts(statistical_drifts, distribution_drifts, 
                                     volume_anomalies, schema_changes)
        
        # Create summary
        summary = self._create_summary(statistical_drifts, distribution_drifts, 
                                     volume_anomalies, schema_changes)
        
        return {
            'run_id': run_id,
            'timestamp': timestamp,
            'status': 'completed',
            'current_filename': current_filename,
            'baseline_filename': baseline_filename,
            'summary': summary,
            'overview': {
                'current_dataset': {
                    'filename': current_filename,
                    'rows': current_total_rows,
                    'columns': len(current_df.columns)
                },
                'baseline_dataset': {
                    'filename': baseline_filename,
                    'rows': baseline_total_rows,
                    'columns': len(baseline_stats.keys()) if baseline_stats else 0
                },
                'drift_summary': summary
            },
            'data_drift': {
                'statistical_drifts': statistical_drifts,
                'distribution_drifts': distribution_drifts
            },
            'volume_anomalies': volume_anomalies,
            'schema_changes': schema_changes,
            'alerts': alerts
        }
    
    def _calculate_dataset_stats(self, df):
        """Calculate statistical summary for a dataset"""
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
    
    def _detect_statistical_drift(self, current_stats, baseline_stats):
        """Detect statistical drift in numerical columns"""
        drifts = []
        
        for column, current_col_stats in current_stats.items():
            if column not in baseline_stats:
                continue
                
            baseline_col_stats = baseline_stats[column]
            
            # Only analyze numerical columns for statistical drift
            if 'mean' in current_col_stats and 'mean' in baseline_col_stats:
                # Mean drift
                current_mean = current_col_stats['mean']
                baseline_mean = baseline_col_stats['mean']
                
                if baseline_mean != 0:
                    mean_change = abs((current_mean - baseline_mean) / baseline_mean)
                    if mean_change > self.drift_threshold:
                        severity = self._get_severity(mean_change)
                        drifts.append({
                            'type': 'mean_drift',
                            'column': column,
                            'severity': severity,
                            'description': f'Mean shifted from {baseline_mean:.2f} to {current_mean:.2f} ({mean_change*100:.1f}% change)',
                            'baseline_value': baseline_mean,
                            'current_value': current_mean,
                            'change_percentage': mean_change * 100,
                            'drift_score': mean_change
                        })
                
                # Standard deviation drift
                current_std = current_col_stats.get('std', 0)
                baseline_std = baseline_col_stats.get('std', 0)
                
                if baseline_std != 0:
                    std_change = abs((current_std - baseline_std) / baseline_std)
                    if std_change > self.drift_threshold:
                        severity = self._get_severity(std_change)
                        drifts.append({
                            'type': 'variance_drift',
                            'column': column,
                            'severity': severity,
                            'description': f'Standard deviation changed from {baseline_std:.2f} to {current_std:.2f} ({std_change*100:.1f}% change)',
                            'baseline_value': baseline_std,
                            'current_value': current_std,
                            'change_percentage': std_change * 100,
                            'drift_score': std_change
                        })
                
                # Range drift
                current_range = current_col_stats['max'] - current_col_stats['min']
                baseline_range = baseline_col_stats['max'] - baseline_col_stats['min']
                
                if baseline_range != 0:
                    range_change = abs((current_range - baseline_range) / baseline_range)
                    if range_change > self.drift_threshold:
                        severity = self._get_severity(range_change)
                        drifts.append({
                            'type': 'range_drift',
                            'column': column,
                            'severity': severity,
                            'description': f'Value range changed from {baseline_range:.2f} to {current_range:.2f} ({range_change*100:.1f}% change)',
                            'baseline_value': baseline_range,
                            'current_value': current_range,
                            'change_percentage': range_change * 100,
                            'drift_score': range_change
                        })
        
        return drifts
    
    def _detect_distribution_drift(self, current_df, baseline_stats):
        """Detect distribution drift using statistical tests"""
        drifts = []
        
        for column in current_df.columns:
            if column not in baseline_stats:
                continue
                
            baseline_col_stats = baseline_stats[column]
            
            # Only test numerical columns with sufficient data
            if pd.api.types.is_numeric_dtype(current_df[column]) and 'mean' in baseline_col_stats:
                current_data = current_df[column].dropna()
                
                if len(current_data) < 10:  # Need minimum sample size
                    continue
                
                # Generate synthetic baseline data for comparison
                # In a real implementation, you'd store actual baseline data
                baseline_mean = baseline_col_stats['mean']
                baseline_std = baseline_col_stats.get('std', 1)
                
                if baseline_std > 0:
                    # Create synthetic baseline sample
                    np.random.seed(42)  # For reproducible results
                    synthetic_baseline = np.random.normal(
                        baseline_mean, baseline_std, len(current_data)
                    )
                    
                    # Kolmogorov-Smirnov test
                    ks_stat, p_value = ks_2samp(synthetic_baseline, current_data)
                    
                    if p_value < self.significance_level:
                        severity = self._get_severity(ks_stat)
                        drifts.append({
                            'type': 'distribution_drift',
                            'column': column,
                            'severity': severity,
                            'description': f'Distribution significantly different (KS statistic: {ks_stat:.3f}, p-value: {p_value:.4f})',
                            'test_statistic': ks_stat,
                            'p_value': p_value,
                            'drift_score': ks_stat
                        })
        
        return drifts
    
    def _detect_volume_anomalies(self, current_rows, baseline_rows, current_cols, baseline_cols):
        """Detect volume-based anomalies"""
        anomalies = []
        
        # Row count anomaly
        if baseline_rows and baseline_rows > 0:
            row_change = abs(current_rows - baseline_rows) / baseline_rows
            if row_change > self.volume_threshold:
                severity = self._get_severity(row_change)
                direction = "increase" if current_rows > baseline_rows else "decrease"
                
                anomalies.append({
                    'type': 'row_count_anomaly',
                    'metric': 'Row Count',
                    'severity': severity,
                    'description': f'Expected ~{baseline_rows:,} rows, received {current_rows:,} rows ({row_change*100:.1f}% {direction})',
                    'baseline_value': baseline_rows,
                    'current_value': current_rows,
                    'change_percentage': row_change * 100,
                    'direction': direction,
                    'chart_width': min(((current_rows / baseline_rows) * 50), 100) if baseline_rows > 0 else 50
                })
        
        # Column count anomaly
        if baseline_cols != current_cols:
            col_change = abs(current_cols - baseline_cols) / baseline_cols if baseline_cols > 0 else 1
            severity = self._get_severity(col_change)
            direction = "increase" if current_cols > baseline_cols else "decrease"
            
            anomalies.append({
                'type': 'column_count_anomaly',
                'metric': 'Column Count',
                'severity': severity,
                'description': f'Expected {baseline_cols} columns, found {current_cols} columns ({col_change*100:.1f}% {direction})',
                'baseline_value': baseline_cols,
                'current_value': current_cols,
                'change_percentage': col_change * 100,
                'direction': direction
            })
        
        return anomalies
    
    def _detect_schema_changes(self, current_columns, baseline_columns):
        """Detect schema changes between datasets"""
        changes = []
        
        current_cols_set = set(current_columns)
        baseline_cols_set = set(baseline_columns)
        
        # New columns
        added_cols = current_cols_set - baseline_cols_set
        for col in added_cols:
            changes.append({
                'type': 'column_added',
                'column': col,
                'severity': 'Medium',
                'description': f"New column '{col}' detected that was not in baseline schema"
            })
        
        # Removed columns
        removed_cols = baseline_cols_set - current_cols_set
        for col in removed_cols:
            changes.append({
                'type': 'column_removed',
                'column': col,
                'severity': 'Critical',
                'description': f"Column '{col}' was present in baseline but missing in current dataset"
            })
        
        return changes
    
    def _generate_alerts(self, statistical_drifts, distribution_drifts, volume_anomalies, schema_changes):
        """Generate alerts for critical drift detection results"""
        alerts = []
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Critical statistical drifts
        critical_stat_drifts = [d for d in statistical_drifts if d['severity'] == 'Critical']
        for drift in critical_stat_drifts:
            alerts.append({
                'id': f"ALT_{uuid.uuid4().hex[:8].upper()}",
                'type': 'Statistical Drift',
                'severity': 'Critical',
                'title': f"{drift['column']} {drift['type'].replace('_', ' ').title()}",
                'description': drift['description'],
                'timestamp': timestamp,
                'status': 'active',
                'column': drift['column'],
                'drift_score': drift['drift_score']
            })
        
        # Critical distribution drifts
        critical_dist_drifts = [d for d in distribution_drifts if d['severity'] == 'Critical']
        for drift in critical_dist_drifts:
            alerts.append({
                'id': f"ALT_{uuid.uuid4().hex[:8].upper()}",
                'type': 'Distribution Drift',
                'severity': 'Critical',
                'title': f"{drift['column']} Distribution Change",
                'description': drift['description'],
                'timestamp': timestamp,
                'status': 'active',
                'column': drift['column'],
                'drift_score': drift['drift_score']
            })
        
        # Critical volume anomalies
        critical_volume = [v for v in volume_anomalies if v['severity'] == 'Critical']
        for anomaly in critical_volume:
            alerts.append({
                'id': f"ALT_{uuid.uuid4().hex[:8].upper()}",
                'type': 'Volume Anomaly',
                'severity': 'Critical',
                'title': f"{anomaly['metric']} Anomaly",
                'description': anomaly['description'],
                'timestamp': timestamp,
                'status': 'active'
            })
        
        # Critical schema changes
        critical_schema = [s for s in schema_changes if s['severity'] == 'Critical']
        for change in critical_schema:
            alerts.append({
                'id': f"ALT_{uuid.uuid4().hex[:8].upper()}",
                'type': 'Schema Change',
                'severity': 'Critical',
                'title': f"{change['type'].replace('_', ' ').title()}: {change['column']}",
                'description': change['description'],
                'timestamp': timestamp,
                'status': 'active',
                'column': change['column']
            })
        
        return alerts
    
    def _create_summary(self, statistical_drifts, distribution_drifts, volume_anomalies, schema_changes):
        """Create summary statistics for the analysis"""
        all_issues = statistical_drifts + distribution_drifts + volume_anomalies + schema_changes
        
        severity_counts = {"Critical": 0, "Medium": 0, "Low": 0}
        for issue in all_issues:
            severity_counts[issue.get('severity', 'Low')] += 1
        
        # Calculate row count change if available
        row_count_change = 0
        for anomaly in volume_anomalies:
            if anomaly.get('type') == 'row_count_anomaly':
                row_count_change = anomaly.get('change_percentage', 0)
                break
        
        return {
            "total_issues": len(all_issues),
            "critical_issues": severity_counts["Critical"],
            "medium_issues": severity_counts["Medium"],
            "low_issues": severity_counts["Low"],
            "row_count_change": row_count_change,
            "statistical_drift_count": len(statistical_drifts),
            "distribution_drift_count": len(distribution_drifts),
            "volume_anomaly_count": len(volume_anomalies),
            "schema_change_count": len(schema_changes)
        }
    
    def _get_severity(self, score):
        """Determine severity based on drift score"""
        if score >= 0.5:  # 50%+ change
            return "Critical"
        elif score >= 0.25:  # 25%+ change
            return "Medium"
        else:
            return "Low"

# Database and utility functions (kept from original)
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

def fetch_baseline_data(db_conn, table_name):
    """Fetch baseline dataset statistics from database"""
    if not db_conn or not table_name: 
        return None, {}
    
    # Get total row count
    total_rows_res = execute_db_query(db_conn, f'SELECT COUNT(*) FROM "{table_name}";', fetch_one=True)
    total_rows = total_rows_res[0] if total_rows_res else 0
    
    # Get column information
    schema_info_rows = execute_db_query(
        db_conn, 
        f"SELECT column_name, data_type FROM information_schema.columns WHERE table_schema = 'public' AND table_name = %s;",
        (table_name,), 
        fetch_all=True
    )
    
    if not schema_info_rows:
        return total_rows, {}
    
    baseline_stats = {}
    for col_name, col_type in schema_info_rows:
        stats = {'column': col_name, 'dtype': col_type}
        
        # Calculate statistics based on column type
        safe_col = f'"{col_name}"'
        
        # Null count
        null_res = execute_db_query(
            db_conn, 
            f"SELECT COUNT(*) - COUNT({safe_col}), COUNT(*) FROM \"{table_name}\";",
            fetch_one=True
        )
        if null_res:
            stats['null_count'] = null_res[0]
            stats['total_count'] = null_res[1]
            stats['null_percentage'] = (null_res[0] / null_res[1] * 100) if null_res[1] > 0 else 0
        
        # Numeric statistics
        if any(t in col_type.lower() for t in ['integer', 'numeric', 'real', 'double', 'decimal']):
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
                    'mean': float(num_res[0]) if num_res[0] is not None else 0,
                    'std': float(num_res[1]) if num_res[1] is not None else 0,
                    'min': float(num_res[2]) if num_res[2] is not None else 0,
                    'max': float(num_res[3]) if num_res[3] is not None else 0,
                    'median': float(num_res[4]) if num_res[4] is not None else 0,
                    'q25': float(num_res[5]) if num_res[5] is not None else 0,
                    'q75': float(num_res[6]) if num_res[6] is not None else 0
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
                     LIMIT 1) as most_frequent
                    FROM \"{table_name}\" 
                    WHERE {safe_col} IS NOT NULL;""",
                fetch_one=True
            )
            if cat_res:
                stats.update({
                    'unique_count': cat_res[0] if cat_res[0] is not None else 0,
                    'most_frequent': cat_res[1] if cat_res[1] is not None else None
                })
        
        baseline_stats[col_name] = stats
    
    return total_rows, _sanitize_for_json(baseline_stats)

def load_current_data(filepath):
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

def run_drift_analysis(current_filepath, baseline_csv_filepath=None):
    """
    Main function to run data drift analysis
    
    Args:
        current_filepath: Path to current CSV file
        baseline_csv_filepath: Optional path to baseline CSV file (fallback)
        
    Returns:
        Tuple: (results_dict, html_content, log_id)
    """
    log_id = uuid.uuid4()
    db_conn = get_db_connection()
    
    try:
        # Initialize analyzer
        analyzer = DataDriftAnalyzer()
        
        # Load current dataset
        current_df = load_current_data(current_filepath)
        current_filename = os.path.basename(current_filepath)
        
        # Load baseline data
        baseline_total_rows = None
        baseline_stats = {}
        baseline_filename = "No baseline available"
        
        if HISTORIC_TABLE_NAME and db_conn:
            baseline_total_rows, baseline_stats = fetch_baseline_data(db_conn, HISTORIC_TABLE_NAME)
            baseline_filename = f"Database: {HISTORIC_TABLE_NAME}"
        elif baseline_csv_filepath and os.path.exists(baseline_csv_filepath):
            baseline_df = load_current_data(baseline_csv_filepath)
            baseline_stats = analyzer._calculate_dataset_stats(baseline_df)
            baseline_total_rows = len(baseline_df)
            baseline_filename = f"CSV: {os.path.basename(baseline_csv_filepath)}"
        
        # Run analysis
        results = analyzer.analyze_datasets(
            current_df=current_df,
            baseline_stats=baseline_stats,
            baseline_total_rows=baseline_total_rows,
            current_filename=current_filename,
            baseline_filename=baseline_filename
        )
        
        # Generate preview data
        current_preview = current_df.head(MAX_ROWS_FOR_PREVIEW)
        
        # Add preview data to results
        results['current_preview'] = current_preview.to_dict('records')
        
        # Create simple HTML output (detailed HTML generation will be in templates)
        html_content = f"""
        <div class="drift-analysis-summary">
            <h2>Data Drift Analysis Complete</h2>
            <p><strong>Run ID:</strong> {results['run_id']}</p>
            <p><strong>Status:</strong> {results['status']}</p>
            <p><strong>Total Issues Found:</strong> {results['summary']['total_issues']}</p>
            <p><strong>Critical Issues:</strong> {results['summary']['critical_issues']}</p>
        </div>
        """
        
        # Log to database if available
        if db_conn:
            log_analysis_to_db(db_conn, log_id, results, current_filename)
        
        return results, html_content, str(log_id)
        
    except Exception as e:
        error_msg = f"Analysis failed: {str(e)}"
        print(f"Error in run_drift_analysis: {error_msg}")
        
        error_results = {
            'run_id': f"ERROR_{uuid.uuid4().hex[:8]}",
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'status': 'failed',
            'error': error_msg,
            'summary': {'total_issues': 0, 'critical_issues': 0, 'medium_issues': 0, 'low_issues': 0}
        }
        
        error_html = f"<div class='error-summary'><h2>Analysis Failed</h2><p>{html.escape(error_msg)}</p></div>"
        
        return error_results, error_html, str(log_id)
    
    finally:
        if db_conn:
            db_conn.close()

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
            current_filename VARCHAR(255),
            baseline_filename VARCHAR(255),
            total_issues INTEGER DEFAULT 0,
            critical_issues INTEGER DEFAULT 0,
            summary_json TEXT
        );
        """
        execute_db_query(db_conn, create_table_query)
        print(f"Log table '{ANALYSIS_LOG_TABLE_NAME}' ensured to exist.")
    except Exception as e:
        print(f"Error creating log table: {e}")
        raise
def log_analysis_to_db(db_conn, log_id, results, filename):
    """Log analysis results to database"""
    try:
        # Ensure log table exists
        ensure_log_table_exists(db_conn)
        
        # Insert log entry
        insert_query = f"""
        INSERT INTO "{ANALYSIS_LOG_TABLE_NAME}" 
        (log_id, run_timestamp, status, current_filename, baseline_filename, 
         total_issues, critical_issues, summary_json)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (log_id) DO UPDATE SET
        run_timestamp = EXCLUDED.run_timestamp,
        status = EXCLUDED.status,
        total_issues = EXCLUDED.total_issues,
        critical_issues = EXCLUDED.critical_issues,
        summary_json = EXCLUDED.summary_json;
        """
        
        params = (
            str(log_id),
            datetime.now(timezone.utc),
            results.get('status', 'unknown'),
            filename,
            results.get('baseline_filename', 'unknown'),
            results.get('summary', {}).get('total_issues', 0),
            results.get('summary', {}).get('critical_issues', 0),
            json.dumps(_sanitize_for_json(results.get('summary', {})))
        )
        
        execute_db_query(db_conn, insert_query, params)
        print(f"Logged analysis results to database: {log_id}")
        
    except Exception as e:
        print(f"Failed to log to database: {e}")

# Keep the main run_analysis function for backward compatibility
def run_analysis(new_filepath, historic_csv_filepath=None):
    """Backward compatibility wrapper"""
    try:
        results, html_content, log_id = run_drift_analysis(new_filepath, historic_csv_filepath)
        
        # Format for the original template structure
        return (
            html_content,  # report_html
            "",  # historic_preview_html (empty for now)
            "",  # new_data_preview_html (empty for now)
            log_id,
            results  # analyzed_data_for_template
        )
    except Exception as e:
        error_html = f"<div class='error-card'><h2>Critical Error</h2><p>{html.escape(str(e))}</p></div>"
        error_results = {'error': str(e), 'log_id': str(uuid.uuid4())}
        return error_html, "", "", error_results['log_id'], error_results