# main.py
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from werkzeug.utils import secure_filename
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv, set_key
from datetime import datetime, timezone
import json
import os
import uuid
import html as pyhtml
import analysis_engine

load_dotenv()

# Upload configuration
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
ALLOWED_EXTENSIONS = {'csv'}
MAX_FILE_UPLOAD_MB = int(os.getenv('MAX_FILE_UPLOAD_MB', 32))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'dev_secret_key_change_this_in_prod')
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_UPLOAD_MB * 1024 * 1024

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Helper function to load analysis data from database
def load_analysis_from_db(log_id_str):
    """Load analysis data from database by log_id, including structured data."""
    if not log_id_str:
        return None
        
    db_conn = None
    try:
        db_conn = analysis_engine.get_db_connection()
        if not db_conn:
            app.logger.error("Database connection failed when loading analysis")
            return None
            
        analysis_engine.ensure_log_table_exists(db_conn) 
        
        with db_conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
            query = f"""
            SELECT log_id, run_timestamp, status, new_data_filename, 
                   historic_table_name, error_message, analysis_summary, analysis_data
            FROM "{analysis_engine.ANALYSIS_LOG_TABLE_NAME}"
            WHERE log_id = %s
            """
            cursor.execute(query, (uuid.UUID(log_id_str),))
            log_db_data = cursor.fetchone()
            
            if not log_db_data:
                return None

            results = {
                'log_id': str(log_db_data['log_id']),
                'timestamp': log_db_data['run_timestamp'].isoformat() if log_db_data['run_timestamp'] else datetime.now(timezone.utc).isoformat(),
                'status': log_db_data['status'],
                'filename': log_db_data['new_data_filename'],
                'current_filename': log_db_data['new_data_filename'], # For overview.html
                'baseline_filename': log_db_data['historic_table_name'], # For overview.html
                'historic_table': log_db_data['historic_table_name'],
                'error_message': log_db_data['error_message'],
                'summary': {
                    'text': log_db_data['analysis_summary'] or "No summary available.",
                    'statistical_drift_count': 0,
                    'distribution_drift_count': 0,
                    'volume_anomaly_count': 0,
                    'schema_change_count': 0,
                    'alert_count': 0,
                    'critical_issues': 0,
                    'total_issues': 0 # Initialize
                },
                'data': {}, 
                'report_html': f"<p>Basic log entry: {log_db_data['status']}</p>", 
                'overview': {'current_dataset': {}, 'baseline_dataset': {}}, 
                'current_preview': [] 
            }

            if log_db_data['analysis_data']:
                try:
                    loaded_structured_data = json.loads(log_db_data['analysis_data'])
                    if isinstance(loaded_structured_data, dict):
                        results['data'] = loaded_structured_data 
                        results['overview'] = loaded_structured_data.get('overview', results['overview'])
                        results['current_preview'] = loaded_structured_data.get('current_preview', [])
                        results['report_html'] = loaded_structured_data.get('report_html', results['report_html'])
                        
                        s_drift_count = len(loaded_structured_data.get('statistical_drifts', []))
                        d_drift_count = len(loaded_structured_data.get('distribution_drifts', []))
                        v_anomaly_count = len(loaded_structured_data.get('volume_anomalies', []))
                        s_change_count = len(loaded_structured_data.get('schema_changes', []))
                        dq_issues_count = len(loaded_structured_data.get('data_quality_issues', [])) # Consider if this contributes to total_issues
                        alert_list = loaded_structured_data.get('alerts', [])
                        
                        results['summary']['statistical_drift_count'] = s_drift_count
                        results['summary']['distribution_drift_count'] = d_drift_count
                        results['summary']['volume_anomaly_count'] = v_anomaly_count
                        results['summary']['schema_change_count'] = s_change_count
                        results['summary']['alert_count'] = len(alert_list)
                        results['summary']['critical_issues'] = len([a for a in alert_list if a.get('severity','').lower() == 'critical'])
                        # Calculate total_issues based on individual issue types
                        results['summary']['total_issues'] = s_drift_count + d_drift_count + v_anomaly_count + s_change_count + dq_issues_count

                except (json.JSONDecodeError, TypeError) as e:
                    app.logger.error(f"Error parsing analysis_data JSON for log {log_id_str}: {e}")
                    results['error_message'] = (results['error_message'] or "") + "; Failed to parse detailed results."
            
            return results
            
    except Exception as e:
        app.logger.error(f"Error loading analysis from database for log {log_id_str}: {e}")
        return None
    finally:
        if db_conn:
            db_conn.close()

# --- Routes ---
@app.route('/')
def index():
    log_id_from_query = request.args.get('log_id')
    if log_id_from_query:
        flash(f"Analysis completed. Log ID: {pyhtml.escape(log_id_from_query)}", "info")
    
    return render_template('index.html', 
                         max_file_upload_mb=MAX_FILE_UPLOAD_MB,
                         historic_table_name=analysis_engine.HISTORIC_TABLE_NAME or '')

@app.route('/set_historic_table', methods=['POST'])
def set_historic_table():
    new_table = request.form.get('historic_table_name', '').strip()
    
    if not new_table:
        flash('Please enter a valid table name.', 'error')
        return redirect(url_for('index'))
    
    try:
        dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
        if not os.path.exists(dotenv_path):
            with open(dotenv_path, 'w') as f:
                f.write(f"HISTORIC_TABLE_NAME={new_table}\n")
        else:
            set_key(dotenv_path, 'HISTORIC_TABLE_NAME', new_table)
        
        analysis_engine.HISTORIC_TABLE_NAME = new_table
        
        db_conn = analysis_engine.get_db_connection()
        if db_conn:
            try:
                with db_conn.cursor() as cursor:
                    cursor.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema = 'public' AND table_name = %s);", (new_table,))
                    table_exists = cursor.fetchone()[0]
                    if table_exists:
                        cursor.execute(f'SELECT COUNT(*) FROM "{new_table}";')
                        row_count = cursor.fetchone()[0]
                        flash(f"Historic table '{new_table}' selected successfully! ({row_count:,} rows)", "success")
                    else:
                        flash(f"Warning: Table '{new_table}' does not exist in the database. Please verify.", "warning")
            except Exception as e:
                flash(f"Error accessing table '{new_table}': {str(e)}", "error")
            finally:
                db_conn.close()
        else:
            flash(f"Table '{new_table}' set, but couldn't verify (DB connection failed).", "warning")
    except Exception as e:
        flash(f"Error setting historic table: {str(e)}", "error")
    return redirect(url_for('index'))

@app.route('/api/db_tables')
def list_db_tables_api(): 
    db_conn = None
    try:
        db_conn = analysis_engine.get_db_connection()
        if not db_conn:
            return jsonify({"status": "error", "message": "Database connection failed"}), 500
        with db_conn.cursor() as cursor:
            cursor.execute("""
                SELECT table_name, 
                       (SELECT COUNT(*) FROM information_schema.columns 
                        WHERE table_name = t.table_name AND table_schema = 'public') as column_count
                FROM information_schema.tables t
                WHERE table_schema = 'public' AND table_type = 'BASE TABLE' ORDER BY table_name;
            """)
            tables_data = cursor.fetchall()
            tables = [{"name": row[0], "columns": row[1]} for row in tables_data]
        return jsonify({"status": "ok", "tables": tables})
    except Exception as e:
        app.logger.error(f"Error fetching table names: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        if db_conn: db_conn.close()

@app.route('/api/table_info/<table_name>')
def get_table_info(table_name):
    db_conn = None
    try:
        db_conn = analysis_engine.get_db_connection()
        if not db_conn:
            return jsonify({"status": "error", "message": "Database connection failed"}), 500
        with db_conn.cursor() as cursor:
            cursor.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema = 'public' AND table_name = %s);", (table_name,))
            if not cursor.fetchone()[0]:
                return jsonify({"status": "error", "message": f"Table '{table_name}' does not exist"}), 404
            cursor.execute(f'SELECT COUNT(*) FROM "{table_name}";')
            row_count = cursor.fetchone()[0]
            cursor.execute("SELECT column_name, data_type, is_nullable FROM information_schema.columns WHERE table_schema = 'public' AND table_name = %s ORDER BY ordinal_position;", (table_name,))
            columns = [{"name": col[0], "type": col[1], "nullable": col[2] == 'YES'} for col in cursor.fetchall()]
            return jsonify({"status": "ok", "table_name": table_name, "row_count": row_count, "column_count": len(columns), "columns": columns})
    except Exception as e:
        app.logger.error(f"Error getting table info for {table_name}: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        if db_conn: db_conn.close()

@app.route('/analyze', methods=['POST'])
def analyze_data():
    if 'new_file' not in request.files:
        flash('Current dataset file (CSV) is required.', 'error')
        return redirect(url_for('index'))
    
    current_file = request.files['new_file']
    if current_file.filename == '':
        flash('No file selected. Please upload a CSV.', 'error')
        return redirect(url_for('index'))
    if not allowed_file(current_file.filename):
        flash(f'Invalid file type. Only CSV files are allowed.', 'error')
        return redirect(url_for('index'))
    if not analysis_engine.HISTORIC_TABLE_NAME:
        flash('Please set a historic table name before running analysis.', 'error')
        return redirect(url_for('index'))
    
    filename_secure = secure_filename(current_file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename_secure)
    
    try:
        current_file.save(filepath)
    except Exception as e:
        app.logger.error(f"Error saving uploaded file {filename_secure}: {e}")
        flash(f"Error saving file: {pyhtml.escape(str(e))}", "error")
        return redirect(url_for('index'))
    
    try:
        _, _, _, log_id, analyzed_data_dict = analysis_engine.run_analysis(filepath) 
        
        s_drift_count = len(analyzed_data_dict.get('statistical_drifts', []))
        d_drift_count = len(analyzed_data_dict.get('distribution_drifts', []))
        v_anomaly_count = len(analyzed_data_dict.get('volume_anomalies', []))
        s_change_count = len(analyzed_data_dict.get('schema_changes', []))
        dq_issues_count = len(analyzed_data_dict.get('data_quality_issues', []))
        alert_list = analyzed_data_dict.get('alerts', [])
        total_issues_val = s_drift_count + d_drift_count + v_anomaly_count + s_change_count + dq_issues_count

        # Get row count change from volume anomalies if present
        row_count_change_val = 0.0
        volume_anomalies_list = analyzed_data_dict.get('volume_anomalies', [])
        if volume_anomalies_list:
            rc_anomaly = next((item for item in volume_anomalies_list if item["metric"] == "Row Count"), None)
            if rc_anomaly:
                row_count_change_val = rc_anomaly.get("change_percentage", 0.0)


        session['last_analysis'] = {
            'log_id': log_id,
            'report_html': analyzed_data_dict.get('report_html', ''), 
            'filename': analyzed_data_dict.get('new_filename', filename_secure),
            'current_filename': analyzed_data_dict.get('new_filename', filename_secure), # For overview.html
            'baseline_filename': analyzed_data_dict.get('historic_table_name', analysis_engine.HISTORIC_TABLE_NAME), # For overview.html
            'historic_table': analyzed_data_dict.get('historic_table_name', analysis_engine.HISTORIC_TABLE_NAME),
            'status': analyzed_data_dict.get('status', 'Completed'), 
            'timestamp': analyzed_data_dict.get('timestamp', datetime.now(timezone.utc).isoformat()),
            'summary': {
                'text': f"Analysis of {filename_secure} completed.", 
                'statistical_drift_count': s_drift_count,
                'distribution_drift_count': d_drift_count,
                'volume_anomaly_count': v_anomaly_count,
                'schema_change_count': s_change_count,
                'alert_count': len(alert_list),
                'critical_issues': len([a for a in alert_list if a.get('severity','').lower() == 'critical']),
                'total_issues': total_issues_val, # Added
                'row_count_change': row_count_change_val # Added for overview.html
            },
            'data': analyzed_data_dict, 
            'overview': analyzed_data_dict.get('overview', {}),
            'current_preview': analyzed_data_dict.get('current_preview', [])
        }
        
        flash(f"Analysis completed successfully! Log ID: {log_id}", "success")
        return redirect(url_for('view_overview', run_id=log_id)) 
        
    except Exception as e:
        app.logger.error(f"Analysis failed for file {filename_secure}: {e}", exc_info=True)
        flash(f"Analysis failed: {pyhtml.escape(str(e))}", "error")
        return redirect(url_for('index'))

@app.route('/database_tables_list') 
def list_db_tables_page(): 
    logs = []
    db_conn = None
    error_message = None
    
    results_for_nav = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'summary': {
            'statistical_drift_count': 0, 'distribution_drift_count': 0,
            'volume_anomaly_count': 0, 'schema_change_count': 0, 'alert_count': 0, 'critical_issues':0,
            'total_issues': 0 # Added for nav if overview.html is used as a base
        },
        'data': {} 
    }
    run_id_for_nav = None

    try:
        db_conn = analysis_engine.get_db_connection()
        if not db_conn:
            flash("Database connection failed. Cannot display logs.", "error")
            error_message = "Database connection failed"
        else:
            analysis_engine.ensure_log_table_exists(db_conn)
            with db_conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                query = f"""
                SELECT log_id, run_timestamp, status, new_data_filename, 
                       historic_table_name, error_message, analysis_summary, analysis_data 
                FROM "{analysis_engine.ANALYSIS_LOG_TABLE_NAME}"
                ORDER BY run_timestamp DESC LIMIT 50;
                """
                cursor.execute(query)
                logs_data = cursor.fetchall()
                logs = [dict(row) for row in logs_data]

                if logs and logs[0].get('analysis_data'): 
                    try:
                        most_recent_structured = json.loads(logs[0]['analysis_data'])
                        s_drift_count = len(most_recent_structured.get('statistical_drifts', []))
                        d_drift_count = len(most_recent_structured.get('distribution_drifts', []))
                        v_anomaly_count = len(most_recent_structured.get('volume_anomalies', []))
                        s_change_count = len(most_recent_structured.get('schema_changes', []))
                        dq_issues_count = len(most_recent_structured.get('data_quality_issues', []))
                        alert_list = most_recent_structured.get('alerts', [])

                        results_for_nav['summary']['statistical_drift_count'] = s_drift_count
                        results_for_nav['summary']['distribution_drift_count'] = d_drift_count
                        results_for_nav['summary']['volume_anomaly_count'] = v_anomaly_count
                        results_for_nav['summary']['schema_change_count'] = s_change_count
                        results_for_nav['summary']['alert_count'] = len(alert_list)
                        results_for_nav['summary']['critical_issues'] = len([a for a in alert_list if a.get('severity','').lower() == 'critical'])
                        results_for_nav['summary']['total_issues'] = s_drift_count + d_drift_count + v_anomaly_count + s_change_count + dq_issues_count
                        results_for_nav['timestamp'] = logs[0]['run_timestamp'].isoformat() if logs[0]['run_timestamp'] else results_for_nav['timestamp']
                        run_id_for_nav = str(logs[0]['log_id'])
                    except (json.JSONDecodeError, TypeError) as e:
                        app.logger.error(f"Error parsing analysis_data for nav from recent log: {e}")
                elif logs: 
                    run_id_for_nav = str(logs[0]['log_id'])
                    results_for_nav['timestamp'] = logs[0]['run_timestamp'].isoformat() if logs[0]['run_timestamp'] else results_for_nav['timestamp']
    except psycopg2.Error as e:
        app.logger.error(f"Database error fetching logs: {e}")
        error_message = f"Database error: {str(e).split('DETAIL:')[0].strip()}"
        flash("Error fetching analysis logs from database.", "error")
    except Exception as e:
        app.logger.error(f"Unexpected error fetching logs: {e}")
        error_message = f"Unexpected error: {str(e)}"
        flash("Unexpected error occurred while fetching logs.", "error")
    finally:
        if db_conn: db_conn.close()
    
    return render_template('runs_list.html', 
                         logs=logs, 
                         error_message=error_message,
                         results=results_for_nav, 
                         run_id=run_id_for_nav,    
                         current_tab='history') 


# --- Analysis Detail Routes ---
def get_analysis_results_for_view(run_id_str):
    last_analysis = session.get('last_analysis')
    if last_analysis and last_analysis.get('log_id') == run_id_str:
        return last_analysis
    
    analysis_data_from_db = load_analysis_from_db(run_id_str)
    if analysis_data_from_db:
        return analysis_data_from_db
    
    flash(f"Analysis with ID {run_id_str} not found.", "error")
    return None

@app.route('/results/<run_id>') 
def view_results(run_id):
    results_data = get_analysis_results_for_view(run_id)
    if results_data:
        return render_template('overview.html', 
                             run_id=run_id,
                             current_tab='overview',
                             results=results_data)
    return redirect(url_for('list_db_tables_page'))


@app.route('/overview/<run_id>')
def view_overview(run_id):
    results_data = get_analysis_results_for_view(run_id)
    if results_data:
        # Make sure overview.html has all it needs from results.summary
        if 'row_count_change' not in results_data['summary']:
            volume_anomalies_list = results_data.get('data',{}).get('volume_anomalies', [])
            rc_anomaly = next((item for item in volume_anomalies_list if item["metric"] == "Row Count"), None)
            results_data['summary']['row_count_change'] = rc_anomaly.get("change_percentage", 0.0) if rc_anomaly else 0.0
        
        # For current_filename and baseline_filename in overview.html
        results_data['current_filename'] = results_data.get('filename', 'N/A')
        results_data['baseline_filename'] = results_data.get('historic_table', 'N/A')


        return render_template('overview.html', 
                             run_id=run_id,
                             current_tab='overview',
                             results=results_data)
    return redirect(url_for('list_db_tables_page'))


@app.route('/data-drift/<run_id>')
def view_data_drift(run_id):
    results_data = get_analysis_results_for_view(run_id)
    if results_data:
        return render_template('data_drift.html',
                             run_id=run_id,
                             current_tab='data-drift',
                             results=results_data,
                             data_drift=results_data.get('data', {})) 
    return redirect(url_for('list_db_tables_page'))

@app.route('/volume-anomalies/<run_id>')
def view_volume_anomalies(run_id):
    results_data = get_analysis_results_for_view(run_id)
    if results_data:
        return render_template('volume_anomalies.html',
                             run_id=run_id,
                             current_tab='volume-anomalies',
                             results=results_data,
                             volume_anomalies=results_data.get('data', {}).get('volume_anomalies', []))
    return redirect(url_for('list_db_tables_page'))

@app.route('/schema-changes/<run_id>')
def view_schema_changes(run_id):
    results_data = get_analysis_results_for_view(run_id)
    if results_data:
        return render_template('schema_changes.html',
                             run_id=run_id,
                             current_tab='schema-changes',
                             results=results_data,
                             schema_changes=results_data.get('data', {}).get('schema_changes', []))
    return redirect(url_for('list_db_tables_page'))

@app.route('/alerts/<run_id>')
def view_alerts(run_id):
    results_data = get_analysis_results_for_view(run_id)
    if results_data:
        return render_template('alerts.html',
                             run_id=run_id,
                             current_tab='alerts',
                             results=results_data)
    return redirect(url_for('list_db_tables_page'))


@app.route('/api/health')
def health_check():
    health_info = {
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'components': {
            'database': {
                'status': 'ok',
                'message': 'Database connection successful',
                'details': {}
            },
            'tables': {
                'status': 'ok',
                'message': 'All required tables present',
                'details': {}
            }
        }
    }
    
    db_conn = None
    try:
        db_conn = analysis_engine.get_db_connection()
        if not db_conn:
            health_info['components']['database'].update({
                'status': 'error',
                'message': 'Database connection failed'
            })
            health_info['status'] = 'error'
        else:
            with db_conn.cursor() as cursor:
                # Basic connectivity test
                cursor.execute("SELECT version();")
                version = cursor.fetchone()[0]
                health_info['components']['database']['details']['version'] = version
                
                # Check required tables
                tables_to_check = [analysis_engine.ANALYSIS_LOG_TABLE_NAME]
                if analysis_engine.HISTORIC_TABLE_NAME:
                    tables_to_check.append(analysis_engine.HISTORIC_TABLE_NAME)
                
                for table in tables_to_check:
                    cursor.execute(
                        "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema = 'public' AND table_name = %s);",
                        (table,)
                    )
                    if cursor.fetchone()[0]:
                        cursor.execute(f'SELECT COUNT(*) FROM "{table}";')
                        count = cursor.fetchone()[0]
                        health_info['components']['tables']['details'][table] = {
                            'exists': True,
                            'records': count
                        }
                    else:
                        health_info['components']['tables'].update({
                            'status': 'warning',
                            'message': f'Table {table} does not exist'
                        })
                        health_info['components']['tables']['details'][table] = {
                            'exists': False
                        }
                        if health_info['status'] == 'ok':
                            health_info['status'] = 'warning'
                
    except Exception as e:
        health_info.update({
            'status': 'error',
            'error': str(e)
        })
    finally:
        if db_conn:
            db_conn.close()
    
    return jsonify(health_info)
@app.context_processor
def inject_global_vars():
    return {
        'SCRIPT_LOAD_TIME': datetime.now(timezone.utc),
        'current_year': datetime.now().year
    }

if __name__ == '__main__':
    # Test database connection before starting the app
    print("\nTesting Supabase database connection...")
    if analysis_engine.test_db_connection():
        print("Database connection test passed. Starting application...\n")
        app.run(debug=True, host='0.0.0.0', port=int(os.getenv("PORT", 8000)))
    else:
        print("ERROR: Failed to establish database connection. Please check your configuration.")
        print("Hint: Verify your Supabase credentials and connection details in .env file.")