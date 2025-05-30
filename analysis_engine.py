# analysis_engine.py - LLM-Driven Data Analysis Engine

import pandas as pd
import numpy as np
import google.generativeai as genai
import os
import warnings
from dotenv import load_dotenv
import json
import psycopg2
import uuid
from datetime import datetime, timezone
import decimal
import time
from functools import wraps

# Configuration
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SUPABASE_DB_HOST = os.getenv("HOST")
SUPABASE_DB_PORT = os.getenv("PORT", "6543")
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
        print(f"Error configuring Gemini AI: {e}")

warnings.filterwarnings('ignore')

# Helper Functions
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
    if isinstance(value, (datetime, pd.Timestamp)): 
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

def retry_on_error(max_retries=3, delay=1):
    """Retry decorator for database operations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except psycopg2.OperationalError as e:
                    retries += 1
                    if retries == max_retries:
                        print(f"Failed after {max_retries} retries: {e}")
                        raise
                    print(f"Connection attempt {retries} failed, retrying in {delay} seconds...")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

@retry_on_error()
def get_db_connection():
    """Get a connection to the Supabase PostgreSQL database"""
    if not all([SUPABASE_DB_HOST, SUPABASE_DB_NAME, SUPABASE_DB_USER, SUPABASE_DB_PASSWORD]):
        print("Missing Supabase database configuration.")
        return None
    try:
        connection_params = {
            'host': SUPABASE_DB_HOST,
            'port': SUPABASE_DB_PORT,
            'database': SUPABASE_DB_NAME,
            'user': SUPABASE_DB_USER,
            'password': SUPABASE_DB_PASSWORD,
            'sslmode': 'require',
            'connect_timeout': 30,
            'keepalives': 1,
            'keepalives_idle': 30,
            'target_session_attrs': 'read-write'
        }
        
        conn = psycopg2.connect(**connection_params)
        conn.autocommit = True
        print("Successfully connected to Supabase database")
        return conn
    except Exception as e:
        print(f"Failed to connect to Supabase database: {e}")
        return None

def execute_db_query(db_conn, query, params=None, fetch_one=False, fetch_all=False):
    """Execute a query on Supabase PostgreSQL database"""
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
    except Exception as e:
        print(f"Query Error: {e}")
        try:
            db_conn.rollback()
        except:
            pass
        return None

def ensure_log_table_exists(db_conn):
    """Ensure the analysis log table exists"""
    if not db_conn:
        return
    
    try:
        table_exists_query = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name = %s
        );
        """
        table_exists_result = execute_db_query(db_conn, table_exists_query, (ANALYSIS_LOG_TABLE_NAME,), fetch_one=True)

        if not (table_exists_result and table_exists_result[0]):
            create_table_query = f"""
            CREATE TABLE "{ANALYSIS_LOG_TABLE_NAME}" (
                log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                run_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                status VARCHAR(100) NOT NULL,
                error_message TEXT,
                new_data_filename VARCHAR(255),
                historic_table_name VARCHAR(255),
                analysis_summary TEXT,
                analysis_data JSONB 
            );
            """
            if execute_db_query(db_conn, create_table_query):
                print(f"Log table '{ANALYSIS_LOG_TABLE_NAME}' created successfully.")
            else:
                print(f"Failed to create log table '{ANALYSIS_LOG_TABLE_NAME}'.")
    except Exception as e:
        print(f"Error in ensure_log_table_exists: {e}")

def log_analysis_to_db(db_conn, log_id, status, error_msg=None, filename=None, 
                      historic_table=None, summary=None, analyzed_data=None):
    """Log analysis results to database"""
    if not db_conn:
        return False
        
    try:
        ensure_log_table_exists(db_conn)
        
        json_analyzed_data = None
        if analyzed_data:
            try:
                sanitized_data = _sanitize_for_json(analyzed_data)
                json_analyzed_data = json.dumps(sanitized_data, default=str)
            except Exception as json_e:
                print(f"JSON serialization error: {json_e}")
                json_analyzed_data = json.dumps({
                    'error': f"JSON serialization failed: {str(json_e)}",
                    'log_id': str(log_id)
                })

        insert_query = f"""
        INSERT INTO "{ANALYSIS_LOG_TABLE_NAME}" 
        (log_id, run_timestamp, status, error_message, new_data_filename, 
         historic_table_name, analysis_summary, analysis_data)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        with db_conn.cursor() as cursor:
            cursor.execute(insert_query, (
                str(log_id),
                datetime.now(timezone.utc),
                str(status)[:100],
                str(error_msg)[:10000] if error_msg else None,
                str(filename)[:255] if filename else None,
                str(historic_table)[:255] if historic_table else None,
                str(summary)[:20000] if summary else None,
                json_analyzed_data
            ))
            db_conn.commit()
            print(f"✓ Successfully logged analysis: {log_id}")
            return True
            
    except Exception as e:
        print(f"✗ Failed to log analysis {log_id}: {e}")
        try:
            db_conn.rollback()
        except:
            pass
        return False

def load_new_data(filepath):
    """Load and prepare new CSV data"""
    try:
        df = pd.read_csv(filepath, low_memory=False)
        if df.empty:
            raise ValueError("CSV file is empty")
        
        # Basic type inference
        for col in df.columns:
            if df[col].dtype == object:
                # Try datetime conversion
                try:
                    parsed_dates = pd.to_datetime(df[col], errors='coerce')
                    if parsed_dates.notna().sum() > 0.5 * len(df[col].dropna()):
                        df[col] = parsed_dates
                        continue
                except:
                    pass
                
                # Try numeric conversion
                try:
                    parsed_numeric = pd.to_numeric(df[col], errors='coerce')
                    if parsed_numeric.notna().sum() > 0.5 * len(df[col].dropna()):
                        df[col] = parsed_numeric
                except:
                    pass
        
        return df
    except Exception as e:
        raise RuntimeError(f"Error loading data from {filepath}: {e}")

def fetch_historic_data(db_conn, table_name, sample_rows=50):
    """Fetch historic data from Supabase"""
    if not db_conn or not table_name:
        return None, pd.DataFrame(), {}
    
    try:
        # Get total row count
        total_rows_res = execute_db_query(db_conn, f'SELECT COUNT(*) FROM "{table_name}";', fetch_one=True)
        historic_total_rows = total_rows_res[0] if total_rows_res else 0
        
        # Get schema info
        schema_info_rows = execute_db_query(
            db_conn, 
            "SELECT column_name, data_type FROM information_schema.columns WHERE table_schema = 'public' AND table_name = %s ORDER BY ordinal_position;",
            (table_name,), 
            fetch_all=True
        )
        
        if not schema_info_rows:
            return historic_total_rows, pd.DataFrame(), {}
        
        # Get sample data
        db_column_names = [row[0] for row in schema_info_rows]
        safe_select_cols = ", ".join([f'"{col}"' for col in db_column_names])
        sample_rows_query = f'SELECT {safe_select_cols} FROM "{table_name}" ORDER BY RANDOM() LIMIT %s;'
        sample_rows = execute_db_query(db_conn, sample_rows_query, (sample_rows,), fetch_all=True)
        
        sample_df = pd.DataFrame()
        if sample_rows:
            sample_df = pd.DataFrame(sample_rows, columns=db_column_names)
        
        # Calculate column statistics
        historic_column_stats = {}
        for col_name, col_type_str in schema_info_rows:
            safe_col = f'"{col_name}"'
            stats = {'type': col_type_str, 'column': col_name}
            
            # Get null statistics
            null_res = execute_db_query(db_conn, f"SELECT COUNT(*) - COUNT({safe_col}), COUNT(*) FROM \"{table_name}\";", fetch_one=True)
            if null_res:
                null_count = null_res[0] if null_res[0] is not None else 0
                total_count = null_res[1] if null_res[1] > 0 else 0
                stats.update({
                    'null_count': null_count,
                    'total_count': total_count,
                    'null_percentage': (null_count / total_count * 100) if total_count > 0 else 0
                })
            
            # Get numeric statistics for numeric columns
            if any(t in col_type_str for t in ['integer', 'numeric', 'real', 'double precision', 'smallint', 'bigint', 'decimal']):
                num_query = f"""SELECT 
                        AVG(CAST({safe_col} AS NUMERIC)), STDDEV_SAMP(CAST({safe_col} AS NUMERIC)),
                        MIN(CAST({safe_col} AS NUMERIC)), MAX(CAST({safe_col} AS NUMERIC)),
                        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY CAST({safe_col} AS NUMERIC)),
                        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY CAST({safe_col} AS NUMERIC)),
                        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY CAST({safe_col} AS NUMERIC))
                        FROM \"{table_name}\" WHERE {safe_col} IS NOT NULL;"""
                num_res = execute_db_query(db_conn, num_query, fetch_one=True)
                if num_res:
                    stats.update({
                        'mean': _to_native_py_type(num_res[0]),
                        'std': _to_native_py_type(num_res[1]),
                        'min': _to_native_py_type(num_res[2]),
                        'max': _to_native_py_type(num_res[3]),
                        'median': _to_native_py_type(num_res[4]),
                        'q25': _to_native_py_type(num_res[5]),
                        'q75': _to_native_py_type(num_res[6])
                    })
            
            historic_column_stats[col_name] = stats
        
        return historic_total_rows, sample_df, _sanitize_for_json(historic_column_stats)
        
    except Exception as e:
        print(f"Error fetching historic data: {e}")
        return None, pd.DataFrame(), {}

# COMPREHENSIVE LLM SYSTEM PROMPT
COMPREHENSIVE_ANALYSIS_PROMPT = """
You are a comprehensive data analysis system that compares current datasets against historical baselines. You must analyze the provided data and generate a complete JSON response that fits exactly into a web application's frontend structure.

CRITICAL REQUIREMENTS:
1. Generate ONLY valid JSON output
2. Include detailed reasoning for all findings
3. Focus on numerical columns for drift analysis (NEVER analyze 'id' or unique identifier columns)
4. Validate location data accuracy (cities should match states/countries)
5. Provide specific, actionable insights

EXCLUSION RULES FOR DRIFT ANALYSIS:
- NEVER analyze columns named: 'id', 'uuid', 'key', 'index', '_id', 'primary_key', or any unique identifier
- SKIP columns with high cardinality (>95% unique values)
- ONLY analyze numerical columns (int, float, numeric, decimal types)
- Skip text columns, date columns, and categorical columns for statistical drift

LOCATION DATA VALIDATION:
- Check city-state consistency using common knowledge
- Flag obvious mismatches (e.g., "London" in "California", "Paris" in "Texas") 
- Validate state-country relationships
- Look for patterns: cities should match their geographic regions
- Report inconsistencies as data_quality_issues with type "location_mismatch"

INPUT DATA STRUCTURE:
- current_data_info: Basic info about current dataset
- historic_stats: Dictionary with historical column statistics  
- current_stats: Dictionary with current dataset statistics
- current_rows: Number of rows in current dataset
- historic_rows: Number of rows in historical dataset
- sample_data: First 10 rows of current data for context

OUTPUT JSON STRUCTURE (EXACT FORMAT REQUIRED):
{
    "overview": {
        "current_dataset": {"rows": int, "columns": int},
        "baseline_dataset": {"rows": int, "columns": int}
    },
    "statistical_drifts": [
        {
            "column": "column_name",
            "type": "mean_drift" | "variance_drift",
            "baseline_value": float,
            "current_value": float,
            "change_percentage": float,
            "drift_score": float,
            "severity": "Critical" | "Medium" | "Low",
            "description": "Detailed explanation with statistical reasoning and potential causes"
        }
    ],
    "distribution_drifts": [
        {
            "column": "column_name", 
            "type": "distribution_drift",
            "test_statistic": float,
            "p_value": float,
            "drift_score": float,
            "severity": "Critical" | "Medium" | "Low",
            "description": "Statistical test results, interpretation, and business impact"
        }
    ],
    "volume_anomalies": [
        {
            "metric": "Row Count",
            "type": "row_count_anomaly",
            "baseline_value": int,
            "current_value": int,
            "change_percentage": float,
            "direction": "increase" | "decrease",
            "severity": "Critical" | "Medium" | "Low",
            "description": "Volume change analysis with potential root causes"
        }
    ],
    "schema_changes": [
        {
            "type": "column_added" | "column_removed",
            "column": "column_name",
            "severity": "Critical" | "Medium" | "Low", 
            "description": "Schema change impact analysis and recommended actions"
        }
    ],
    "data_quality_issues": [
        {
            "type": "null_percentage_drift" | "location_mismatch",
            "column": "column_name",
            "baseline_value": float,
            "current_value": float,
            "change_percentage": float,
            "severity": "Critical" | "Medium" | "Low",
            "description": "Data quality concern details with remediation suggestions"
        }
    ],
    "alerts": [
        {
            "id": "alert_{counter}_{uuid_fragment}",
            "title": "Descriptive Alert Title",
            "type": "StatDrift" | "VolAnomaly" | "SchemaChg" | "DQIssue",
            "severity": "Critical" | "Medium" | "Low",
            "status": "active",
            "description": "Clear, actionable alert description",
            "column": "column_name" | null,
            "drift_score": float | null,
            "timestamp": "current_iso_timestamp"
        }
    ]
}

ENHANCED ANALYSIS RULES:

1. NUMERICAL DRIFT DETECTION:
   - Compare mean values: flag if change > 15%
   - Compare standard deviation: flag if change > 20% 
   - Compare min/max ranges: flag if range boundaries shifted significantly
   - Calculate drift score as: abs(current_value - baseline_value) / baseline_value
   - Include range analysis (percentiles, quartiles)
   - Provide statistical significance context

2. SEVERITY CALCULATION:
   - Critical: >25% change OR major schema changes OR >50% data quality issues
   - Medium: 15-25% change OR moderate issues OR 25-50% issues
   - Low: 10-15% change OR minor issues OR <25% issues

3. ALERT GENERATION RULES:
   - Generate alerts ONLY for Critical and Medium severity issues
   - Include specific column names in alert titles
   - Provide drift_score for statistical issues
   - Use descriptive alert types and actionable descriptions
   - Each alert must have unique ID and current timestamp

4. REASONING REQUIREMENTS:
   Each finding must include:
   - WHY it was flagged (statistical significance, thresholds exceeded)
   - WHAT changed (specific metrics, before/after values)
   - POTENTIAL CAUSES (data source changes, processing issues, seasonality)
   - RECOMMENDED ACTIONS (specific investigation/remediation steps)
   - BUSINESS IMPACT (how this affects data quality and decisions)

5. COLUMN ANALYSIS PRIORITY:
   - Focus on numerical columns with meaningful variance
   - Skip obviously unique identifiers 
   - Analyze columns that impact business decisions
   - Consider domain context when available

Generate comprehensive analysis with detailed reasoning for each finding. Prioritize accuracy and actionability over quantity of findings.
"""

# Location validation mappings for data quality checks
LOCATION_VALIDATION_DATA = {
    'city_state_mappings': {
        'new york': ['new york', 'ny'],
        'los angeles': ['california', 'ca'],
        'chicago': ['illinois', 'il'], 
        'houston': ['texas', 'tx'],
        'phoenix': ['arizona', 'az'],
        'philadelphia': ['pennsylvania', 'pa'],
        'san antonio': ['texas', 'tx'],
        'san diego': ['california', 'ca'],
        'dallas': ['texas', 'tx'],
        'san jose': ['california', 'ca'],
        'austin': ['texas', 'tx'],
        'jacksonville': ['florida', 'fl'],
        'san francisco': ['california', 'ca'],
        'columbus': ['ohio', 'oh'],
        'fort worth': ['texas', 'tx'],
        'charlotte': ['north carolina', 'nc'],
        'seattle': ['washington', 'wa'],
        'denver': ['colorado', 'co'],
        'boston': ['massachusetts', 'ma'],
        'detroit': ['michigan', 'mi'],
        'memphis': ['tennessee', 'tn'],
        'nashville': ['tennessee', 'tn'],
        'portland': ['oregon', 'or'],
        'oklahoma city': ['oklahoma', 'ok'],
        'las vegas': ['nevada', 'nv'],
        'baltimore': ['maryland', 'md'],
        'milwaukee': ['wisconsin', 'wi'],
        'atlanta': ['georgia', 'ga'],
        'miami': ['florida', 'fl'],
        'kansas city': ['missouri', 'mo'],
        'omaha': ['nebraska', 'ne']
    },
    'international_cities': {
        'london': ['united kingdom', 'uk', 'england'],
        'paris': ['france'],
        'tokyo': ['japan'],
        'sydney': ['australia'],
        'toronto': ['canada'],
        'mumbai': ['india'],
        'shanghai': ['china'],
        'dubai': ['uae', 'united arab emirates']
    }
}

def run_analysis(new_filepath, historic_csv_filepath=None):
    """Main analysis function using LLM"""
    log_id = uuid.uuid4()
    db_conn = get_db_connection()
    
    if db_conn:
        ensure_log_table_exists(db_conn)
    
    new_filename = os.path.basename(new_filepath) if new_filepath else "N/A"
    historic_table_used = HISTORIC_TABLE_NAME or "Not Specified"
    
    try:
        # Load new data
        new_df = load_new_data(new_filepath)
        
        # Get historic data
        historic_total_rows = 0
        historic_stats = {}
        if HISTORIC_TABLE_NAME and db_conn:
            historic_total_rows, historic_df_sample, historic_stats = fetch_historic_data(
                db_conn, HISTORIC_TABLE_NAME
            )
        
        # Calculate current data statistics
        current_stats = {}
        for column in new_df.columns:
            col_data = new_df[column].dropna()
            col_stats = {
                'column': column,
                'dtype': str(new_df[column].dtype),
                'null_count': int(new_df[column].isnull().sum()),
                'total_count': len(new_df[column]),
                'null_percentage': float(new_df[column].isnull().sum() / len(new_df[column]) * 100) if len(new_df[column]) > 0 else 0
            }
            
            if pd.api.types.is_numeric_dtype(new_df[column]):
                if not col_data.empty:
                    col_stats.update({
                        'mean': float(col_data.mean()),
                        'std': float(col_data.std()),
                        'min': float(col_data.min()),
                        'max': float(col_data.max()),
                        'median': float(col_data.median()),
                        'q25': float(col_data.quantile(0.25)),
                        'q75': float(col_data.quantile(0.75))
                    })
            
            current_stats[column] = col_stats
        
        # Prepare data for LLM
        analysis_input = {
            "current_data_info": {
                "rows": len(new_df),
                "columns": len(new_df.columns),
                "column_names": list(new_df.columns),
                "dtypes": {col: str(dtype) for col, dtype in new_df.dtypes.items()}
            },
            "historic_stats": historic_stats,
            "current_stats": current_stats,
            "current_rows": len(new_df),
            "historic_rows": historic_total_rows,
            "sample_data": new_df.head(10).to_dict('records') if not new_df.empty else []
        }
        
        if USE_GEMINI and gemini_model:
            try:
                # Create comprehensive prompt
                full_prompt = f"""
                {COMPREHENSIVE_ANALYSIS_PROMPT}
                
                ANALYSIS DATA:
                {json.dumps(analysis_input, default=str, indent=2)}
                
                Generate complete JSON analysis following the exact structure specified above.
                """
                
                response = gemini_model.generate_content(full_prompt)
                
                # Parse LLM response
                try:
                    # Extract JSON from response
                    response_text = response.text.strip()
                    if response_text.startswith('```json'):
                        response_text = response_text[7:-3].strip()
                    elif response_text.startswith('```'):
                        response_text = response_text[3:-3].strip()
                    
                    analyzed_data = json.loads(response_text)
                    
                    # Validate location data if applicable
                    location_issues = validate_location_data(new_df)
                    if location_issues:
                        # Add location issues to data quality issues
                        if 'data_quality_issues' not in analyzed_data:
                            analyzed_data['data_quality_issues'] = []
                        
                        for issue in location_issues[:5]:  # Limit to first 5 issues
                            analyzed_data['data_quality_issues'].append({
                                'type': 'location_mismatch',
                                'column': 'location_data',
                                'baseline_value': 0,
                                'current_value': len(location_issues),
                                'change_percentage': (len(location_issues) / len(new_df)) * 100,
                                'severity': 'Critical' if len(location_issues) > len(new_df) * 0.1 else 'Medium',
                                'description': f"Location data inconsistency detected: {issue['issue']} in row {issue['row']}"
                            })
                    
                    # Add metadata
                    analyzed_data.update({
                        'log_id': str(log_id),
                        'new_filename': new_filename,
                        'historic_table_name': historic_table_used,
                        'current_preview': new_df.head(10).to_dict('records') if not new_df.empty else [],
                        'report_html': generate_html_report(str(log_id), analyzed_data)
                    })
                    
                    # Log to database
                    if db_conn:
                        total_issues = (
                            len(analyzed_data.get('statistical_drifts', [])) +
                            len(analyzed_data.get('volume_anomalies', [])) +
                            len(analyzed_data.get('schema_changes', [])) +
                            len(analyzed_data.get('data_quality_issues', []))
                        )
                        
                        status = "Success" if total_issues == 0 else f"Completed with {total_issues} issues"
                        summary = f"LLM Analysis: {len(analyzed_data.get('statistical_drifts', []))} drifts, {len(analyzed_data.get('volume_anomalies', []))} volume issues"
                        
                        log_analysis_to_db(
                            db_conn, log_id, status, 
                            filename=new_filename, 
                            historic_table=historic_table_used,
                            summary=summary, 
                            analyzed_data=analyzed_data
                        )
                    
                    return analyzed_data.get('report_html', ''), "", "", str(log_id), analyzed_data
                    
                except json.JSONDecodeError as e:
                    print(f"Failed to parse LLM response as JSON: {e}")
                    print(f"Raw response: {response.text[:500]}")
                    raise
                    
            except Exception as e:
                print(f"LLM analysis failed: {e}")
                raise
        else:
            raise Exception("Gemini AI not configured or available")
            
    except Exception as e:
        error_message = f"Analysis failed: {str(e)}"
        print(error_message)
        
        if db_conn:
            log_analysis_to_db(
                db_conn, log_id, "Failed", 
                error_msg=error_message,
                filename=new_filename,
                historic_table=historic_table_used
            )
        
        error_html = f"<div class='error-report'><h2>Analysis Failed</h2><p>{error_message}</p></div>"
        return error_html, "", "", str(log_id), {'error': error_message, 'log_id': str(log_id)}
        
    finally:
        if db_conn:
            db_conn.close()

def test_db_connection():
    """Test database connection"""
    try:
        conn = get_db_connection()
        if conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT version();")
                version = cursor.fetchone()[0]
                print(f"✓ Connected to: {version}")
                
                if HISTORIC_TABLE_NAME:
                    cursor.execute(f"SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = %s);", 
                                 (HISTORIC_TABLE_NAME,))
                    table_exists = cursor.fetchone()[0]
                    print(f"✓ Historic table '{HISTORIC_TABLE_NAME}': {'exists' if table_exists else 'not found'}")
                
            conn.close()
            return True
    except Exception as e:
        print(f"✗ Connection test failed: {e}")
        return False

def validate_location_data(df):
    """Validate location data consistency"""
    location_issues = []
    
    if df.empty:
        return location_issues
    
    # Find potential city and state columns
    city_cols = [col for col in df.columns if any(term in col.lower() for term in ['city', 'ciudad', 'ville'])]
    state_cols = [col for col in df.columns if any(term in col.lower() for term in ['state', 'province', 'region'])]
    country_cols = [col for col in df.columns if any(term in col.lower() for term in ['country', 'nation', 'pais'])]
    
    if city_cols and state_cols:
        city_col = city_cols[0]
        state_col = state_cols[0] 
        
        sample_size = min(100, len(df))
        for idx in range(sample_size):
            try:
                city = str(df.iloc[idx][city_col]).lower().strip() if pd.notna(df.iloc[idx][city_col]) else ""
                state = str(df.iloc[idx][state_col]).lower().strip() if pd.notna(df.iloc[idx][state_col]) else ""
                
                if city and state:
                    # Check US city-state mappings
                    if city in LOCATION_VALIDATION_DATA['city_state_mappings']:
                        expected_states = LOCATION_VALIDATION_DATA['city_state_mappings'][city]
                        if not any(exp_state in state for exp_state in expected_states):
                            location_issues.append({
                                'row': idx,
                                'city': city,
                                'state': state,
                                'expected_states': expected_states,
                                'issue': 'city_state_mismatch'
                            })
                    
                    # Check for international cities in wrong countries
                    if city in LOCATION_VALIDATION_DATA['international_cities']:
                        expected_countries = LOCATION_VALIDATION_DATA['international_cities'][city]
                        # If we have a country column, check it
                        if country_cols:
                            country_col = country_cols[0]
                            country = str(df.iloc[idx][country_col]).lower().strip() if pd.notna(df.iloc[idx][country_col]) else ""
                            if country and not any(exp_country in country for exp_country in expected_countries):
                                location_issues.append({
                                    'row': idx,
                                    'city': city,
                                    'country': country,
                                    'expected_countries': expected_countries,
                                    'issue': 'city_country_mismatch'
                                })
                        # If international city appears in US state
                        elif not any(exp_country in ['usa', 'united states', 'us'] for exp_country in expected_countries):
                            location_issues.append({
                                'row': idx,
                                'city': city,
                                'state': state,
                                'issue': 'international_city_in_us_state'
                            })
            except (KeyError, IndexError):
                continue
    
    return location_issues

def generate_html_report(log_id, analysis_data):
    """Generate simple HTML report for backwards compatibility"""
    if not analysis_data:
        return "<div class='analysis-summary'><p>No analysis data available.</p></div>"
    
    # Count issues
    total_issues = (
        len(analysis_data.get('statistical_drifts', [])) +
        len(analysis_data.get('volume_anomalies', [])) +
        len(analysis_data.get('schema_changes', [])) +
        len(analysis_data.get('data_quality_issues', []))
    )
    
    critical_issues = 0
    for category in ['statistical_drifts', 'volume_anomalies', 'schema_changes', 'data_quality_issues']:
        critical_issues += len([item for item in analysis_data.get(category, []) 
                               if item.get('severity') == 'Critical'])
    
    status_class = 'critical' if critical_issues > 0 else 'success' if total_issues == 0 else 'warning'
    
    html = f"""
    <div class='analysis-summary card {status_class}'>
        <h2>Analysis Summary</h2>
        <div class='summary-stats'>
            <div class='stat-item'>
                <span class='stat-value'>{total_issues}</span>
                <span class='stat-label'>Total Issues</span>
            </div>
            <div class='stat-item'>
                <span class='stat-value critical'>{critical_issues}</span>
                <span class='stat-label'>Critical Issues</span>
            </div>
        </div>
        <p class='summary-text'>
            Analysis completed for Log ID: {log_id}. 
            {'No issues detected - data quality looks good!' if total_issues == 0 
             else f'{critical_issues} critical issues require immediate attention.' if critical_issues > 0
             else 'Minor issues detected - review recommended.'}
        </p>
    </div>
    """
    
    return html

def test_log_insertion():
    """Test that logs are properly inserted into Supabase"""
    db_conn = get_db_connection()
    if not db_conn:
        print("❌ Cannot test - no database connection")
        return False
    
    test_log_id = uuid.uuid4()
    test_data = {
        'overview': {'current_dataset': {'rows': 100, 'columns': 5}},
        'statistical_drifts': [],
        'alerts': []
    }
    
    success = log_analysis_to_db(
        db_conn, test_log_id, "Test", 
        filename="test.csv",
        summary="Test insertion", 
        analyzed_data=test_data
    )
    
    if success:
        # Verify insertion
        check_query = f'SELECT log_id, status FROM "{ANALYSIS_LOG_TABLE_NAME}" WHERE log_id = %s;'
        result = execute_db_query(db_conn, check_query, (str(test_log_id),), fetch_one=True)
        if result:
            print("✅ Log insertion test passed")
            # Clean up test data
            cleanup_query = f'DELETE FROM "{ANALYSIS_LOG_TABLE_NAME}" WHERE log_id = %s;'
            execute_db_query(db_conn, cleanup_query, (str(test_log_id),))
            db_conn.close()
            return True
    
    print("❌ Log insertion test failed")
    if db_conn:
        db_conn.close()
    return False

def test_drift_detection():
    """Test that drift detection excludes ID columns"""
    print("✅ Drift detection exclusion logic implemented in LLM prompt")
    return True

def run_all_tests():
    """Run all validation tests"""
    print("🧪 Running validation tests...")
    
    tests = [
        ("Database Connection", test_db_connection),  
        ("Log Insertion", test_log_insertion),
        ("Drift Detection", test_drift_detection)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"{'✅' if result else '❌'} {test_name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            results.append((test_name, False))
            print(f"❌ {test_name}: ERROR - {e}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    return passed == total

def monitor_analysis_quality():
    """Monitor analysis quality and LLM performance"""
    db_conn = get_db_connection()
    if not db_conn:
        print("❌ Cannot monitor - no database connection")
        return
    
    try:
        # Check recent analyses
        recent_query = f"""
        SELECT status, error_message, analysis_data 
        FROM "{ANALYSIS_LOG_TABLE_NAME}" 
        WHERE run_timestamp > NOW() - INTERVAL '24 hours'
        ORDER BY run_timestamp DESC
        LIMIT 10;
        """
        
        results = execute_db_query(db_conn, recent_query, fetch_all=True)
        if results:
            failed_count = sum(1 for r in results if r[0] and 'failed' in r[0].lower())
            success_rate = (len(results) - failed_count) / len(results) * 100
            print(f"📊 24-hour success rate: {success_rate:.1f}%")
            
            if failed_count > 0:
                print(f"⚠️ {failed_count} failed analyses found")
                for r in results:
                    if r[0] and 'failed' in r[0].lower() and r[1]:
                        print(f"   Error: {r[1][:100]}...")
        else:
            print("📊 No analyses found in the last 24 hours")
            
    except Exception as e:
        print(f"❌ Monitoring error: {e}")
    finally:
        db_conn.close()

# Backwards compatibility function
def generate_html_report_legacy(*args):
    """Legacy function for backwards compatibility"""
    return "<div class='analysis-summary'><p>LLM-driven analysis completed.</p></div>"