# analysis_engine.py - Refactored LLM-Driven Data Analysis Engine

import pandas as pd
import numpy as np
import google.generativeai as genai
import os
import json
import psycopg2
import uuid
from datetime import datetime, timezone
from dotenv import load_dotenv

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

# Initialize Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
else:
    gemini_model = None

# COMPREHENSIVE LLM SYSTEM PROMPT
SYSTEM_PROMPT = """
You are a comprehensive data analysis system. Analyze the provided current dataset against historical baseline and generate a complete JSON response.

CRITICAL RULES:
1. NEVER analyze 'id', 'uuid', 'key', 'index', or unique identifier columns for drift
2. ONLY analyze numerical columns for statistical drift (int, float, numeric types)
3. Validate location data: check city-state consistency (e.g., flag "London" in "California")
4. Provide detailed reasoning for ALL findings
5. Generate ONLY valid JSON output

SEVERITY THRESHOLDS:
- Critical: >25% change, major schema issues, >50% data quality problems
- Medium: 15-25% change, moderate issues
- Low: <15% change, minor issues

OUTPUT STRUCTURE (EXACT FORMAT):
{
    "overview": {
        "current_dataset": {"rows": int, "columns": int},
        "baseline_dataset": {"rows": int, "columns": int}
    },
    "statistical_drifts": [
        {
            "column": "column_name",
            "type": "mean_drift|variance_drift",
            "baseline_value": float,
            "current_value": float,
            "change_percentage": float,
            "drift_score": float,
            "severity": "Critical|Medium|Low",
            "description": "Detailed explanation with statistical reasoning, potential causes, and recommended actions"
        }
    ],
    "distribution_drifts": [
        {
            "column": "column_name",
            "type": "distribution_drift",
            "test_statistic": float,
            "p_value": float,
            "drift_score": float,
            "severity": "Critical|Medium|Low",
            "description": "Statistical test interpretation with business impact and next steps"
        }
    ],
    "volume_anomalies": [
        {
            "metric": "Row Count",
            "type": "row_count_anomaly",
            "baseline_value": int,
            "current_value": int,
            "change_percentage": float,
            "direction": "increase|decrease",
            "severity": "Critical|Medium|Low",
            "description": "Volume change analysis with root cause suggestions"
        }
    ],
    "schema_changes": [
        {
            "type": "column_added|column_removed",
            "column": "column_name",
            "severity": "Critical|Medium|Low",
            "description": "Schema impact analysis with remediation steps"
        }
    ],
    "data_quality_issues": [
        {
            "type": "null_percentage_drift|location_mismatch",
            "column": "column_name",
            "baseline_value": float,
            "current_value": float,
            "change_percentage": float,
            "severity": "Critical|Medium|Low",
            "description": "Data quality issue details with correction recommendations"
        }
    ],
    "alerts": [
        {
            "id": "alert_unique_id",
            "title": "Descriptive Alert Title",
            "type": "StatDrift|VolAnomaly|SchemaChg|DQIssue",
            "severity": "Critical|Medium|Low",
            "status": "active",
            "description": "Clear actionable alert description",
            "column": "column_name|null",
            "drift_score": float|null,
            "timestamp": "current_iso_timestamp"
        }
    ]
}

ANALYSIS REQUIREMENTS:
- Include WHY each issue was flagged
- Explain WHAT changed with before/after values
- Suggest POTENTIAL CAUSES
- Provide RECOMMENDED ACTIONS
- Focus on business impact and actionability
- Generate alerts ONLY for Critical and Medium severity issues

Analyze the data and provide comprehensive findings with detailed reasoning.
"""

def get_db_connection():
    """Get Supabase database connection"""
    try:
        conn = psycopg2.connect(
            host=SUPABASE_DB_HOST, port=SUPABASE_DB_PORT, database=SUPABASE_DB_NAME,
            user=SUPABASE_DB_USER, password=SUPABASE_DB_PASSWORD, sslmode='require'
        )
        conn.autocommit = True
        return conn
    except Exception as e:
        print(f"DB connection failed: {e}")
        return None

def sanitize_for_json(item):
    """Convert data types for JSON serialization"""
    if isinstance(item, dict):
        return {str(k): sanitize_for_json(v) for k, v in item.items()}
    elif isinstance(item, list):
        return [sanitize_for_json(i) for i in item]
    elif pd.isna(item):
        return None
    elif isinstance(item, (np.integer, np.int64, np.int32)):
        return int(item)
    elif isinstance(item, (np.floating, np.float64, np.float32)):
        return float(item) if not pd.isna(item) else None
    elif isinstance(item, (datetime, pd.Timestamp)):
        return item.isoformat()
    elif isinstance(item, uuid.UUID):
        return str(item)
    else:
        return str(item)

def ensure_log_table(db_conn):
    """Ensure analysis log table exists"""
    if not db_conn:
        return
    try:
        create_query = f'''
        CREATE TABLE IF NOT EXISTS "{ANALYSIS_LOG_TABLE_NAME}" (
            log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            run_timestamp TIMESTAMPTZ DEFAULT NOW(),
            status VARCHAR(200),
            error_message TEXT,
            new_data_filename VARCHAR(255),
            historic_table_name VARCHAR(255),
            analysis_summary TEXT,
            analysis_data JSONB
        )'''
        with db_conn.cursor() as cursor:
            cursor.execute(create_query)
    except Exception as e:
        print(f"Table creation error: {e}")

def log_to_db(db_conn, log_id, status, filename, historic_table, summary, data, error=None):
    """Log analysis results to database"""
    if not db_conn:
        return False
    try:
        ensure_log_table(db_conn)
        json_data = json.dumps(sanitize_for_json(data), default=str) if data else None
        
        query = f'''INSERT INTO "{ANALYSIS_LOG_TABLE_NAME}" 
                   (log_id, status, error_message, new_data_filename, historic_table_name, analysis_summary, analysis_data)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)'''
        
        with db_conn.cursor() as cursor:
            cursor.execute(query, (str(log_id), status, error, filename, historic_table, summary, json_data))
        return True
    except Exception as e:
        print(f"Logging failed: {e}")
        return False

def fetch_historic_data(db_conn, table_name):
    """Fetch historic data statistics"""
    if not db_conn or not table_name:
        return 0, {}
    
    try:
        # Get row count
        with db_conn.cursor() as cursor:
            cursor.execute(f'SELECT COUNT(*) FROM "{table_name}"')
            total_rows = cursor.fetchone()[0]
            
            # Get column info and stats
            cursor.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_schema = 'public' AND table_name = %s 
                ORDER BY ordinal_position
            """, (table_name,))
            columns = cursor.fetchall()
            
            stats = {}
            for col_name, col_type in columns:
                col_stats = {'column': col_name, 'type': col_type}
                
                # Get basic stats
                cursor.execute(f'SELECT COUNT(*) - COUNT("{col_name}"), COUNT(*) FROM "{table_name}"')
                null_count, total = cursor.fetchone()
                col_stats.update({
                    'null_count': null_count,
                    'total_count': total,
                    'null_percentage': (null_count/total*100) if total > 0 else 0
                })
                
                # Get numeric stats if numeric column
                if any(t in col_type for t in ['integer', 'numeric', 'real', 'double', 'decimal']):
                    cursor.execute(f'''
                        SELECT AVG(CAST("{col_name}" AS NUMERIC)), 
                               STDDEV_SAMP(CAST("{col_name}" AS NUMERIC)),
                               MIN(CAST("{col_name}" AS NUMERIC)), 
                               MAX(CAST("{col_name}" AS NUMERIC))
                        FROM "{table_name}" WHERE "{col_name}" IS NOT NULL
                    ''')
                    num_stats = cursor.fetchone()
                    if num_stats and num_stats[0] is not None:
                        col_stats.update({
                            'mean': float(num_stats[0]),
                            'std': float(num_stats[1]) if num_stats[1] else 0,
                            'min': float(num_stats[2]),
                            'max': float(num_stats[3])
                        })
                
                stats[col_name] = col_stats
            
            return total_rows, stats
    except Exception as e:
        print(f"Historic data fetch error: {e}")
        return 0, {}

def calculate_current_stats(df):
    """Calculate current dataset statistics"""
    stats = {}
    for col in df.columns:
        col_data = df[col].dropna()
        col_stats = {
            'column': col,
            'dtype': str(df[col].dtype),
            'null_count': int(df[col].isnull().sum()),
            'total_count': len(df[col]),
            'null_percentage': float(df[col].isnull().sum() / len(df[col]) * 100) if len(df[col]) > 0 else 0
        }
        
        if pd.api.types.is_numeric_dtype(df[col]) and not col_data.empty:
            col_stats.update({
                'mean': float(col_data.mean()),
                'std': float(col_data.std()),
                'min': float(col_data.min()),
                'max': float(col_data.max())
            })
        
        stats[col] = col_stats
    return stats

def generate_summary(data, filename):
    """Generate comprehensive analysis summary"""
    if not data:
        return "Analysis failed - no data available"
    
    # Count issues by severity
    critical_count = sum(
        len([item for item in data.get(cat, []) if item.get('severity') == 'Critical'])
        for cat in ['statistical_drifts', 'volume_anomalies', 'schema_changes', 'data_quality_issues']
    )
    
    total_issues = sum(
        len(data.get(cat, []))
        for cat in ['statistical_drifts', 'volume_anomalies', 'schema_changes', 'data_quality_issues']
    )
    
    # Get dataset info
    overview = data.get('overview', {})
    current_rows = overview.get('current_dataset', {}).get('rows', 0)
    baseline_rows = overview.get('baseline_dataset', {}).get('rows', 0)
    
    # Build summary
    summary_parts = [f"📊 ANALYSIS: {filename} ({current_rows:,} rows vs {baseline_rows:,} baseline)"]
    
    if baseline_rows > 0:
        row_change = ((current_rows - baseline_rows) / baseline_rows) * 100
        if abs(row_change) > 1:
            direction = "↗️ INCREASED" if row_change > 0 else "↘️ DECREASED"
            summary_parts.append(f"Volume: {direction} by {abs(row_change):.1f}%")
    
    # Critical issues
    if critical_count > 0:
        summary_parts.append(f"\n🚨 CRITICAL ISSUES ({critical_count}):")
        for cat in ['statistical_drifts', 'volume_anomalies', 'schema_changes', 'data_quality_issues']:
            critical_items = [item for item in data.get(cat, []) if item.get('severity') == 'Critical']
            for item in critical_items[:2]:  # Show top 2
                change = f" ({item.get('change_percentage', 0):+.1f}%)" if item.get('change_percentage') else ""
                summary_parts.append(f"• {item.get('column', 'N/A')}: {item.get('type', 'unknown')}{change}")
    
    # Result assessment
    if total_issues == 0:
        summary_parts.append("\n✅ RESULT: No issues detected - data quality excellent")
    elif critical_count > 0:
        summary_parts.append(f"\n❌ RESULT: IMMEDIATE ACTION REQUIRED - {critical_count} critical issues")
    else:
        summary_parts.append(f"\n⚠️ RESULT: {total_issues} moderate issues found - review recommended")
    
    summary_parts.append(f"\n📈 SCOPE: {len(set(item.get('column') for cat in data.values() if isinstance(cat, list) for item in cat if item.get('column')))} columns analyzed")
    
    return '\n'.join(summary_parts)

def generate_html_report(log_id, data):
    """Generate simple HTML report"""
    if not data:
        return "<div class='error'>Analysis failed</div>"
    
    total_issues = sum(len(data.get(cat, [])) for cat in ['statistical_drifts', 'volume_anomalies', 'schema_changes', 'data_quality_issues'])
    critical_issues = sum(len([item for item in data.get(cat, []) if item.get('severity') == 'Critical']) for cat in ['statistical_drifts', 'volume_anomalies', 'schema_changes', 'data_quality_issues'])
    
    status_class = 'critical' if critical_issues > 0 else 'success' if total_issues == 0 else 'warning'
    
    return f"""
    <div class='analysis-summary {status_class}'>
        <h2>Analysis Complete</h2>
        <div class='summary-stats'>
            <span class='stat'>Total Issues: {total_issues}</span>
            <span class='stat critical'>Critical: {critical_issues}</span>
        </div>
        <p>Analysis ID: {log_id}</p>
    </div>
    """

def run_analysis(csv_filepath, historic_csv_filepath=None):
    """Main analysis function using LLM"""
    log_id = uuid.uuid4()
    db_conn = get_db_connection()
    filename = os.path.basename(csv_filepath)
    
    try:
        # Load and analyze data
        new_df = pd.read_csv(csv_filepath, low_memory=False)
        if new_df.empty:
            raise ValueError("CSV file is empty")
        
        # Get historic data
        historic_rows, historic_stats = fetch_historic_data(db_conn, HISTORIC_TABLE_NAME)
        current_stats = calculate_current_stats(new_df)
        
        # Prepare LLM input
        analysis_input = {
            "current_data_info": {
                "rows": len(new_df),
                "columns": len(new_df.columns),
                "column_names": list(new_df.columns),
                "sample_data": new_df.head(5).to_dict('records')
            },
            "historic_stats": historic_stats,
            "current_stats": current_stats,
            "historic_rows": historic_rows
        }
        
        if not gemini_model:
            raise Exception("Gemini AI not configured - check GEMINI_API_KEY")
        
        # Call LLM with system prompt
        full_prompt = f"{SYSTEM_PROMPT}\n\nDATA TO ANALYZE:\n{json.dumps(analysis_input, default=str, indent=2)}"
        response = gemini_model.generate_content(full_prompt)
        
        # Parse response
        response_text = response.text.strip()
        if response_text.startswith('```json'):
            response_text = response_text[7:-3].strip()
        elif response_text.startswith('```'):
            response_text = response_text[3:-3].strip()
        
        analyzed_data = json.loads(response_text)
        
        # Add metadata
        analyzed_data.update({
            'log_id': str(log_id),
            'new_filename': filename,
            'historic_table_name': HISTORIC_TABLE_NAME,
            'current_preview': new_df.head(10).to_dict('records'),
            'report_html': generate_html_report(str(log_id), analyzed_data)
        })
        
        # Generate status and summary
        total_issues = sum(len(analyzed_data.get(cat, [])) for cat in ['statistical_drifts', 'volume_anomalies', 'schema_changes', 'data_quality_issues'])
        critical_count = sum(len([item for item in analyzed_data.get(cat, []) if item.get('severity') == 'Critical']) for cat in ['statistical_drifts', 'volume_anomalies', 'schema_changes', 'data_quality_issues'])
        
        if total_issues == 0:
            status = "✅ Success - No Issues Detected"
        elif critical_count > 0:
            status = f"🚨 Critical Issues Found ({critical_count} critical)"
        else:
            status = f"⚠️ Issues Detected ({total_issues} total)"
        
        summary = generate_summary(analyzed_data, filename)
        
        # Log to database
        if db_conn:
            log_to_db(db_conn, log_id, status, filename, HISTORIC_TABLE_NAME, summary, analyzed_data)
        
        return analyzed_data['report_html'], "", "", str(log_id), analyzed_data
        
    except Exception as e:
        error_msg = f"Analysis failed: {str(e)}"
        error_summary = f"❌ ANALYSIS FAILED: {filename}\nError: {str(e)[:200]}\nRecommendation: Check data format and API connectivity"
        
        if db_conn:
            log_to_db(db_conn, log_id, "❌ Analysis Failed", filename, HISTORIC_TABLE_NAME, error_summary, None, str(e))
        
        return f"<div class='error'>Analysis failed: {error_msg}</div>", "", "", str(log_id), {'error': error_msg, 'log_id': str(log_id)}
    
    finally:
        if db_conn:
            db_conn.close()

def test_db_connection():
    """Test database connection"""
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT version()")
                print(f"✅ Database connected: {cursor.fetchone()[0][:50]}...")
            conn.close()
            return True
        except Exception as e:
            print(f"❌ Database test failed: {e}")
            return False
    return False