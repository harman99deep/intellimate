# main.py - Refactored for Multi-Page Data Drift Interface

import os
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from werkzeug.utils import secure_filename
import analysis_engine
import psycopg2
import psycopg2.extras
import uuid
from dotenv import load_dotenv, set_key
from datetime import datetime, timezone
import html as pyhtml

load_dotenv()

UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
ALLOWED_EXTENSIONS = {'csv'}
MAX_FILE_UPLOAD_MB = int(os.getenv('MAX_FILE_UPLOAD_MB', 32))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'dev_secret_key_change_this_in_prod')
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_UPLOAD_MB * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main dashboard - Upload interface"""
    log_id_from_query = request.args.get('log_id')
    if log_id_from_query:
        flash(f"Analysis completed. Log ID: {pyhtml.escape(log_id_from_query)}", "info")
    
    return render_template('dashboard.html', 
                         max_file_upload_mb=MAX_FILE_UPLOAD_MB,
                         historic_table_name=os.getenv('HISTORIC_TABLE_NAME', ''))

@app.route('/api/baseline-info')
def get_baseline_info():
    """Get information about the baseline table"""
    if not analysis_engine.HISTORIC_TABLE_NAME:
        return jsonify({'error': 'No baseline table configured'}), 400
    
    db_conn = None
    try:
        db_conn = analysis_engine.get_db_connection()
        if not db_conn:
            return jsonify({'error': 'Database connection failed'}), 500
        
        # Get table information
        with db_conn.cursor() as cursor:
            # Check if table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = %s
                );
            """, (analysis_engine.HISTORIC_TABLE_NAME,))
            
            table_exists = cursor.fetchone()[0]
            
            if not table_exists:
                return jsonify({'error': f'Table "{analysis_engine.HISTORIC_TABLE_NAME}" does not exist'}), 404
            
            # Get table statistics
            cursor.execute(f'SELECT COUNT(*) FROM "{analysis_engine.HISTORIC_TABLE_NAME}";')
            row_count = cursor.fetchone()[0]
            
            # Get column information
            cursor.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_schema = 'public' 
                AND table_name = %s 
                ORDER BY ordinal_position;
            """, (analysis_engine.HISTORIC_TABLE_NAME,))
            
            columns = cursor.fetchall()
            
            return jsonify({
                'table_name': analysis_engine.HISTORIC_TABLE_NAME,
                'exists': True,
                'row_count': row_count,
                'column_count': len(columns),
                'columns': [{'name': col[0], 'type': col[1]} for col in columns]
            })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if db_conn:
            db_conn.close()

@app.route('/analyze', methods=['POST'])
def analyze_data():
    """Process uploaded file and run drift analysis"""
    if 'current_file' not in request.files:
        flash('Current dataset file (CSV) is required.', 'error')
        return redirect(url_for('index'))
    
    current_file = request.files['current_file']
    
    if current_file.filename == '':
        flash('No file selected. Please upload a CSV.', 'error')
        return redirect(url_for('index'))
    
    if not allowed_file(current_file.filename):
        flash(f'Invalid file type. Only CSV files are allowed.', 'error')
        return redirect(url_for('index'))
    
    # Save uploaded file
    filename_secure = secure_filename(current_file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename_secure)
    
    try:
        current_file.save(filepath)
    except Exception as e:
        app.logger.error(f"Error saving uploaded file {filename_secure}: {e}")
        flash(f"Error saving file: {pyhtml.escape(str(e))}", "error")
        return redirect(url_for('index'))
    
    try:
        # Run drift analysis
        results, html_content, log_id = analysis_engine.run_drift_analysis(filepath)
        
        # Store results in session for multi-page access
        session['current_analysis'] = {
            'log_id': log_id,
            'results': results,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        flash(f"Analysis completed successfully! Run ID: {log_id}", "success")
        return redirect(url_for('view_overview', run_id=log_id))
        
    except Exception as e:
        app.logger.error(f"Analysis failed for file {filename_secure}: {e}", exc_info=True)
        flash(f"Analysis failed: {pyhtml.escape(str(e))}", "error")
        return redirect(url_for('index'))

@app.route('/analysis/<run_id>')
def view_overview(run_id):
    """Overview page - main dashboard for a specific run"""
    analysis_data = session.get('current_analysis', {})
    
    if not analysis_data or analysis_data.get('log_id') != run_id:
        # Try to load from database if not in session
        analysis_data = load_analysis_from_db(run_id)
        if not analysis_data:
            flash(f"Analysis run {run_id} not found.", "error")
            return redirect(url_for('index'))
    
    results = analysis_data.get('results', {})
    
    return render_template('overview.html', 
                         run_id=run_id,
                         results=results,
                         current_tab='overview')

@app.route('/analysis/<run_id>/data-drift')
def view_data_drift(run_id):
    """Data Drift page - statistical and distribution changes"""
    analysis_data = session.get('current_analysis', {})
    
    if not analysis_data or analysis_data.get('log_id') != run_id:
        analysis_data = load_analysis_from_db(run_id)
        if not analysis_data:
            flash(f"Analysis run {run_id} not found.", "error")
            return redirect(url_for('index'))
    
    results = analysis_data.get('results', {})
    data_drift = results.get('data_drift', {})
    
    return render_template('data_drift.html',
                         run_id=run_id,
                         results=results,
                         data_drift=data_drift,
                         current_tab='data-drift')

@app.route('/analysis/<run_id>/volume-anomalies')
def view_volume_anomalies(run_id):
    """Volume Anomalies page - row count and data volume changes"""
    analysis_data = session.get('current_analysis', {})
    
    if not analysis_data or analysis_data.get('log_id') != run_id:
        analysis_data = load_analysis_from_db(run_id)
        if not analysis_data:
            flash(f"Analysis run {run_id} not found.", "error")
            return redirect(url_for('index'))
    
    results = analysis_data.get('results', {})
    volume_anomalies = results.get('volume_anomalies', [])
    
    return render_template('volume_anomalies.html',
                         run_id=run_id,
                         results=results,
                         volume_anomalies=volume_anomalies,
                         current_tab='volume-anomalies')

@app.route('/analysis/<run_id>/schema-changes')
def view_schema_changes(run_id):
    """Schema Changes page - column additions/removals"""
    analysis_data = session.get('current_analysis', {})
    
    if not analysis_data or analysis_data.get('log_id') != run_id:
        analysis_data = load_analysis_from_db(run_id)
        if not analysis_data:
            flash(f"Analysis run {run_id} not found.", "error")
            return redirect(url_for('index'))
    
    results = analysis_data.get('results', {})
    schema_changes = results.get('schema_changes', [])
    
    return render_template('schema_changes.html',
                         run_id=run_id,
                         results=results,
                         schema_changes=schema_changes,
                         current_tab='schema-changes')

@app.route('/analysis/<run_id>/alerts')
def view_alerts(run_id):
    """Alerts page - critical notifications"""
    analysis_data = session.get('current_analysis', {})
    
    if not analysis_data or analysis_data.get('log_id') != run_id:
        analysis_data = load_analysis_from_db(run_id)
        if not analysis_data:
            flash(f"Analysis run {run_id} not found.", "error")
            return redirect(url_for('index'))
    
    results = analysis_data.get('results', {})
    alerts = results.get('alerts', [])
    
    return render_template('alerts.html',
                         run_id=run_id,
                         results=results,
                         alerts=alerts,
                         current_tab='alerts')

@app.route('/runs')
def list_runs():
    """List all analysis runs"""
    runs = []
    db_conn = None
    error_message = None
    
    try:
        db_conn = analysis_engine.get_db_connection()
        if not db_conn:
            flash("Database connection failed. Cannot display runs.", "error")
            return render_template('runs_list.html', runs=runs)
        
        # Ensure the log table exists
        try:
            analysis_engine.ensure_log_table_exists(db_conn)
        except Exception as e:
            print(f"Error creating log table: {e}")
        
        # Try to fetch runs
        with db_conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
            # Check if table exists first
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = %s
                );
            """, (analysis_engine.ANALYSIS_LOG_TABLE_NAME,))
            
            table_exists = cursor.fetchone()[0]
            
            if not table_exists:
                print(f"Table {analysis_engine.ANALYSIS_LOG_TABLE_NAME} does not exist, creating it...")
                analysis_engine.ensure_log_table_exists(db_conn)
                # Return empty list for now since no runs exist yet
                return render_template('runs_list.html', runs=[])
            
            # Fetch runs from the table
            query = f"""
            SELECT log_id, run_timestamp, status, current_filename, baseline_filename,
                   total_issues, critical_issues
            FROM "{analysis_engine.ANALYSIS_LOG_TABLE_NAME}"
            ORDER BY run_timestamp DESC
            LIMIT 50;
            """
            cursor.execute(query)
            runs_data = cursor.fetchall()
            runs = [dict(row) for row in runs_data]
    
    except psycopg2.Error as e:
        app.logger.error(f"Database error fetching runs: {e}")
        error_message = f"Database error: {str(e).split('DETAIL:')[0].strip()}"
        flash("Error fetching analysis runs from database.", "error")
    except Exception as e:
        app.logger.error(f"Unexpected error fetching runs: {e}")
        error_message = f"Unexpected error: {str(e)}"
        flash("Unexpected error occurred while fetching runs.", "error")
    finally:
        if db_conn:
            db_conn.close()
    
    return render_template('runs_list.html', runs=runs, error_message=error_message)

@app.route('/api/runs/<run_id>')
def api_get_run(run_id):
    """API endpoint to get run details"""
    analysis_data = session.get('current_analysis', {})
    
    if analysis_data and analysis_data.get('log_id') == run_id:
        return jsonify(analysis_data.get('results', {}))
    
    # Try to load from database
    analysis_data = load_analysis_from_db(run_id)
    if analysis_data:
        return jsonify(analysis_data.get('results', {}))
    
    return jsonify({'error': 'Run not found'}), 404

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    db_status = "ok"
    db_message = "Database connection successful."
    table_status = "ok"
    table_message = ""
    
    try:
        db_conn = analysis_engine.get_db_connection()
        if not db_conn:
            db_status = "error"
            db_message = "Database connection failed."
        else:
            with db_conn.cursor() as cursor:
                cursor.execute("SELECT 1;")
                result = cursor.fetchone()
                if not result or result[0] != 1:
                    db_status = "error"
                    db_message = "Database query test failed."
                else:
                    # Check if log table exists and is accessible
                    try:
                        cursor.execute(f"""
                            SELECT EXISTS (
                                SELECT FROM information_schema.tables 
                                WHERE table_schema = 'public' 
                                AND table_name = %s
                            );
                        """, (analysis_engine.ANALYSIS_LOG_TABLE_NAME,))
                        table_exists = cursor.fetchone()[0]
                        
                        if table_exists:
                            cursor.execute(f'SELECT COUNT(*) FROM "{analysis_engine.ANALYSIS_LOG_TABLE_NAME}";')
                            row_count = cursor.fetchone()[0]
                            table_message = f"Log table exists with {row_count} records."
                        else:
                            table_status = "warning"
                            table_message = "Log table does not exist (will be created on first analysis)."
                    except Exception as e:
                        table_status = "error"
                        table_message = f"Error checking log table: {str(e)}"
            
            db_conn.close()
    except Exception as e:
        db_status = "error"
        db_message = f"Database health check failed: {str(e)}"
    
    baseline_config_status = "ok" if analysis_engine.HISTORIC_TABLE_NAME else "warning"
    baseline_config_message = f"Baseline table: {analysis_engine.HISTORIC_TABLE_NAME}" if analysis_engine.HISTORIC_TABLE_NAME else "No baseline table configured"
    
    overall_status = "ok"
    if db_status == "error":
        overall_status = "error"
    elif table_status == "error":
        overall_status = "error"
    elif baseline_config_status == "warning" or table_status == "warning":
        overall_status = "degraded"
    
    return jsonify({
        "service_status": overall_status,
        "message": "Data Drift Analysis Service",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "components": {
            "database": {"status": db_status, "message": db_message},
            "log_table": {"status": table_status, "message": table_message},
            "baseline_config": {"status": baseline_config_status, "message": baseline_config_message}
        }
    })

def load_analysis_from_db(run_id):
    """Load analysis results from database"""
    db_conn = None
    try:
        db_conn = analysis_engine.get_db_connection()
        if not db_conn:
            return None
        
        with db_conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
            query = f"""
            SELECT log_id, run_timestamp, status, current_filename, baseline_filename,
                   total_issues, critical_issues, summary_json
            FROM "{analysis_engine.ANALYSIS_LOG_TABLE_NAME}"
            WHERE log_id = %s;
            """
            cursor.execute(query, (run_id,))
            row = cursor.fetchone()
            
            if row:
                # Reconstruct basic results structure
                import json
                summary = json.loads(row['summary_json']) if row['summary_json'] else {}
                
                return {
                    'log_id': str(row['log_id']),
                    'results': {
                        'run_id': str(row['log_id']),
                        'timestamp': row['run_timestamp'].isoformat() if row['run_timestamp'] else None,
                        'status': row['status'],
                        'current_filename': row['current_filename'],
                        'baseline_filename': row['baseline_filename'],
                        'summary': summary,
                        # Note: Detailed drift data not stored in this simple schema
                        # In production, you'd want to store the full results JSON
                        'data_drift': {'statistical_drifts': [], 'distribution_drifts': []},
                        'volume_anomalies': [],
                        'schema_changes': [],
                        'alerts': []
                    }
                }
    except Exception as e:
        app.logger.error(f"Error loading analysis from DB: {e}")
        return None
    finally:
        if db_conn:
            db_conn.close()
    
    return None

@app.context_processor
def inject_global_vars():
    """Inject global template variables"""
    return {
        'SCRIPT_LOAD_TIME': datetime.now(timezone.utc),
        'current_year': datetime.now().year
    }

# Add custom Jinja2 filters
@app.template_filter('min_value')
def min_value_filter(value, minimum):
    """Return the minimum between value and minimum"""
    return min(value, minimum)

@app.template_filter('max_value') 
def max_value_filter(value, maximum):
    """Return the maximum between value and maximum"""
    return max(value, maximum)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.getenv("PORT", 8000)))