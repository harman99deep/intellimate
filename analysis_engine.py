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
    """Get Supabase database connection with enhanced debugging"""
    if not all([SUPABASE_DB_HOST, SUPABASE_DB_NAME, SUPABASE_DB_USER, SUPABASE_DB_PASSWORD]):
        print("❌ Missing database configuration. Check environment variables:")
        print(f"  HOST: {'✅' if SUPABASE_DB_HOST else '❌'}")
        print(f"  DBNAME: {'✅' if SUPABASE_DB_NAME else '❌'}")
        print(f"  USER: {'✅' if SUPABASE_DB_USER else '❌'}")
        print(f"  PASSWORD: {'✅' if SUPABASE_DB_PASSWORD else '❌'}")
        return None
    
    try:
        print(f"🔌 Connecting to Supabase: {SUPABASE_DB_USER}@{SUPABASE_DB_HOST}:{SUPABASE_DB_PORT}/{SUPABASE_DB_NAME}")
        
        conn = psycopg2.connect(
            host=SUPABASE_DB_HOST, 
            port=SUPABASE_DB_PORT, 
            database=SUPABASE_DB_NAME,
            user=SUPABASE_DB_USER, 
            password=SUPABASE_DB_PASSWORD, 
            sslmode='require',
            connect_timeout=30,
            keepalives=1,
            keepalives_idle=30
        )
        conn.autocommit = True
        print("✅ Database connection established")
        return conn
        
    except psycopg2.OperationalError as op_err:
        print(f"❌ Database connection failed (Operational): {op_err}")
        print("💡 Check: Network connectivity, credentials, firewall settings")
        return None
    except psycopg2.Error as db_err:
        print(f"❌ Database connection failed (PostgreSQL): {db_err}")
        return None
    except Exception as e:
        print(f"❌ Database connection failed (Unexpected): {e}")
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
    """Ensure analysis log table exists with proper schema"""
    if not db_conn:
        print("❌ No database connection for table creation")
        return False
    
    try:
        # Check if table exists first
        check_query = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name = %s
        )
        """
        
        with db_conn.cursor() as cursor:
            cursor.execute(check_query, (ANALYSIS_LOG_TABLE_NAME,))
            table_exists = cursor.fetchone()[0]
            
            if not table_exists:
                print(f"📝 Creating log table: {ANALYSIS_LOG_TABLE_NAME}")
                create_query = f'''
                CREATE TABLE "{ANALYSIS_LOG_TABLE_NAME}" (
                    log_id UUID PRIMARY KEY,
                    run_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    status VARCHAR(200),
                    error_message TEXT,
                    new_data_filename VARCHAR(255),
                    historic_table_name VARCHAR(255),
                    analysis_summary TEXT,
                    analysis_data JSONB
                )'''
                cursor.execute(create_query)
                db_conn.commit()
                print(f"✅ Created log table: {ANALYSIS_LOG_TABLE_NAME}")
            else:
                print(f"✅ Log table already exists: {ANALYSIS_LOG_TABLE_NAME}")
                
            # Verify table structure
            cursor.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_schema = 'public' AND table_name = %s 
                ORDER BY ordinal_position
            """, (ANALYSIS_LOG_TABLE_NAME,))
            columns = cursor.fetchall()
            print(f"📋 Table columns: {[col[0] for col in columns]}")
            
        return True
        
    except psycopg2.Error as db_err:
        print(f"❌ Database error creating table: {db_err}")
        try:
            db_conn.rollback()
        except:
            pass
        return False
    except Exception as e:
        print(f"❌ Unexpected error creating table: {e}")
        return False

def log_to_db(db_conn, log_id, status, filename, historic_table, summary, data, error=None):
    """Log analysis results to database"""
    if not db_conn:
        print("❌ No database connection for logging")
        return False
    
    try:
        ensure_log_table(db_conn)
        
        # Prepare JSON data with better error handling
        json_data = None
        if data:
            try:
                sanitized_data = sanitize_for_json(data)
                json_data = json.dumps(sanitized_data, default=str, ensure_ascii=False)
                print(f"✅ JSON data prepared: {len(json_data)} characters")
            except Exception as json_err:
                print(f"⚠️ JSON serialization error: {json_err}")
                json_data = json.dumps({"error": f"Serialization failed: {str(json_err)}", "log_id": str(log_id)})
        
        # Insert with explicit transaction and better error handling
        query = f'''INSERT INTO "{ANALYSIS_LOG_TABLE_NAME}" 
                   (log_id, run_timestamp, status, error_message, new_data_filename, historic_table_name, analysis_summary, analysis_data)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)'''
        
        params = (
            str(log_id),
            datetime.now(timezone.utc),
            str(status)[:200] if status else None,
            str(error)[:5000] if error else None,
            str(filename)[:255] if filename else None,
            str(historic_table)[:255] if historic_table else None,
            str(summary)[:10000] if summary else None,
            json_data
        )
        
        with db_conn.cursor() as cursor:
            cursor.execute(query, params)
            db_conn.commit()  # Explicit commit
            print(f"✅ Successfully logged to database: {log_id}")
            
        # Verify insertion
        verify_query = f'SELECT log_id FROM "{ANALYSIS_LOG_TABLE_NAME}" WHERE log_id = %s'
        with db_conn.cursor() as cursor:
            cursor.execute(verify_query, (str(log_id),))
            result = cursor.fetchone()
            if result:
                print(f"✅ Log insertion verified: {result[0]}")
                return True
            else:
                print(f"❌ Log insertion verification failed for: {log_id}")
                return False
                
    except psycopg2.Error as db_err:
        print(f"❌ Database error during logging: {db_err}")
        try:
            db_conn.rollback()
        except:
            pass
        return False
    except Exception as e:
        print(f"❌ Unexpected logging error: {e}")
        import traceback
        traceback.print_exc()
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
    """Main analysis function using LLM with enhanced logging"""
    log_id = uuid.uuid4()
    filename = os.path.basename(csv_filepath)
    
    print(f"🚀 Starting analysis for: {filename} (Log ID: {log_id})")
    
    # Test database connection first
    db_conn = get_db_connection()
    if not db_conn:
        error_msg = "Database connection failed - cannot log results"
        print(f"❌ {error_msg}")
        return f"<div class='error'>Database Error: {error_msg}</div>", "", "", str(log_id), {'error': error_msg, 'log_id': str(log_id)}
    
    try:
        print("📂 Loading CSV data...")
        # Load and analyze data
        new_df = pd.read_csv(csv_filepath, low_memory=False)
        if new_df.empty:
            raise ValueError("CSV file is empty")
        print(f"✅ Loaded CSV: {len(new_df)} rows, {len(new_df.columns)} columns")
        
        print("📊 Fetching historic data...")
        # Get historic data
        historic_rows, historic_stats = fetch_historic_data(db_conn, HISTORIC_TABLE_NAME)
        current_stats = calculate_current_stats(new_df)
        print(f"✅ Historic data: {historic_rows} rows, {len(historic_stats)} columns")
        
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
            raise Exception("Gemini AI not configured - check GEMINI_API_KEY in environment")
        
        print("🤖 Calling Gemini AI for analysis...")
        # Call LLM with system prompt
        full_prompt = f"{SYSTEM_PROMPT}\n\nDATA TO ANALYZE:\n{json.dumps(analysis_input, default=str, indent=2)}"
        response = gemini_model.generate_content(full_prompt)
        print("✅ Received LLM response")
        
        # Parse response
        response_text = response.text.strip()
        if response_text.startswith('```json'):
            response_text = response_text[7:-3].strip()
        elif response_text.startswith('```'):
            response_text = response_text[3:-3].strip()
        
        print("📝 Parsing LLM response...")
        analyzed_data = json.loads(response_text)
        print("✅ Successfully parsed analysis data")
        
        # Add metadata
        analyzed_data.update({
            'log_id': str(log_id),
            'new_filename': filename,
            'historic_table_name': HISTORIC_TABLE_NAME or 'Not configured',
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
        print(f"📊 Analysis complete: {status}")
        
        # Log to database with verification
        print("💾 Logging results to database...")
        log_success = log_to_db(db_conn, log_id, status, filename, HISTORIC_TABLE_NAME, summary, analyzed_data)
        
        if not log_success:
            print("⚠️ Database logging failed, but analysis completed")
        
        print(f"🎉 Analysis completed successfully: {log_id}")
        return analyzed_data['report_html'], "", "", str(log_id), analyzed_data
        
    except json.JSONDecodeError as json_err:
        error_msg = f"LLM response parsing failed: {str(json_err)}"
        print(f"❌ {error_msg}")
        print(f"Raw LLM response: {response.text[:500] if 'response' in locals() else 'No response received'}")
        
        error_summary = f"❌ JSON PARSING FAILED: {filename}\nError: {str(json_err)[:200]}\nLLM Response Preview: {response.text[:200] if 'response' in locals() else 'N/A'}"
        log_to_db(db_conn, log_id, "❌ JSON Parse Failed", filename, HISTORIC_TABLE_NAME, error_summary, None, str(json_err))
        
        return f"<div class='error'>Analysis failed: {error_msg}</div>", "", "", str(log_id), {'error': error_msg, 'log_id': str(log_id)}
        
    except Exception as e:
        error_msg = f"Analysis failed: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        
        error_summary = f"❌ ANALYSIS FAILED: {filename}\nError: {str(e)[:200]}\nRecommendation: Check data format, API connectivity, and database access"
        log_to_db(db_conn, log_id, "❌ Analysis Failed", filename, HISTORIC_TABLE_NAME, error_summary, None, str(e))
        
        return f"<div class='error'>Analysis failed: {error_msg}</div>", "", "", str(log_id), {'error': error_msg, 'log_id': str(log_id)}
    
    finally:
        if db_conn:
            db_conn.close()
            print("🔌 Database connection closed")

def test_db_connection():
    """Test database connection and log table access"""
    print("🧪 Testing database connection and logging...")
    
    conn = get_db_connection()
    if not conn:
        print("❌ Database connection failed")
        return False
        
    try:
        with conn.cursor() as cursor:
            # Test basic connection
            cursor.execute("SELECT version()")
            version_info = cursor.fetchone()[0]
            print(f"✅ Database connected: {version_info[:50]}...")
            
            # Test log table
            ensure_log_table(conn)
            
            # Test a sample log insertion
            test_log_id = uuid.uuid4()
            test_data = {
                'overview': {'current_dataset': {'rows': 100, 'columns': 5}},
                'statistical_drifts': [],
                'alerts': []
            }
            
            print(f"🧪 Testing log insertion with ID: {test_log_id}")
            success = log_to_db(
                conn, test_log_id, "✅ Test Successful", 
                "test_file.csv", "test_table", 
                "Test log insertion successful", 
                test_data
            )
            
            if success:
                # Verify the log was inserted
                cursor.execute(f'SELECT COUNT(*) FROM "{ANALYSIS_LOG_TABLE_NAME}" WHERE log_id = %s', (str(test_log_id),))
                count = cursor.fetchone()[0]
                if count > 0:
                    print(f"✅ Test log verified in database")
                    
                    # Clean up test data
                    cursor.execute(f'DELETE FROM "{ANALYSIS_LOG_TABLE_NAME}" WHERE log_id = %s', (str(test_log_id),))
                    conn.commit()
                    print("🧹 Test data cleaned up")
                else:
                    print("❌ Test log not found in database")
                    return False
            else:
                print("❌ Test log insertion failed")
                return False
            
            # Check recent logs
            cursor.execute(f'SELECT COUNT(*) FROM "{ANALYSIS_LOG_TABLE_NAME}"')
            total_logs = cursor.fetchone()[0]
            print(f"📊 Total logs in database: {total_logs}")
            
        conn.close()
        print("✅ Database connection test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        import traceback
        traceback.print_exc()
        if conn:
            conn.close()
        return False

def debug_recent_logs():
    """Debug function to check recent logs in database"""
    print("🔍 Debugging recent logs...")
    
    conn = get_db_connection()
    if not conn:
        print("❌ Cannot debug - no database connection")
        return
    
    try:
        with conn.cursor() as cursor:
            # Check if table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_name = %s
                )
            """, (ANALYSIS_LOG_TABLE_NAME,))
            
            if not cursor.fetchone()[0]:
                print(f"❌ Table '{ANALYSIS_LOG_TABLE_NAME}' does not exist")
                return
            
            # Get recent logs
            cursor.execute(f'''
                SELECT log_id, run_timestamp, status, new_data_filename, error_message
                FROM "{ANALYSIS_LOG_TABLE_NAME}" 
                ORDER BY run_timestamp DESC 
                LIMIT 5
            ''')
            
            logs = cursor.fetchall()
            if logs:
                print(f"📋 Found {len(logs)} recent logs:")
                for i, (log_id, timestamp, status, filename, error) in enumerate(logs, 1):
                    print(f"  {i}. {log_id} | {timestamp} | {status}")
                    print(f"     File: {filename}")
                    if error:
                        print(f"     Error: {error[:100]}...")
                    print()
            else:
                print("📭 No logs found in database")
                
            # Check total count
            cursor.execute(f'SELECT COUNT(*) FROM "{ANALYSIS_LOG_TABLE_NAME}"')
            total = cursor.fetchone()[0]
            print(f"📊 Total logs in database: {total}")
            
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()