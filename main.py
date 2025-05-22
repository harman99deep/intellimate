# main.py

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
    """Main dashboard - Upload interface with historic table configuration"""
    log_id_from_query = request.args.get('log_id')
    if log_id_from_query:
        flash(f"Analysis completed. Log ID: {pyhtml.escape(log_id_from_query)}", "info")
    
    return render_template('index.html', 
                         max_file_upload_mb=MAX_FILE_UPLOAD_MB,
                         historic_table_name=analysis_engine.HISTORIC_TABLE_NAME or '')

@app.route('/set_historic_table', methods=['POST'])
def set_historic_table():
    """Set the historic table name for analysis"""
    new_table = request.form.get('historic_table_name', '').strip()
    
    if not new_table:
        flash('Please enter a valid table name.', 'error')
        return redirect(url_for('index'))
    
    try:
        # Update environment variable
        dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
        set_key(dotenv_path, 'HISTORIC_TABLE_NAME', new_table)
        
        # Update the module's global variable
        analysis_engine.HISTORIC_TABLE_NAME = new_table
        
        # Test if table exists
        db_conn = analysis_engine.get_db_connection()
        if db_conn:
            try:
                with db_conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_schema = 'public' 
                            AND table_name = %s
                        );
                    """, (new_table,))
                    
                    table_exists = cursor.fetchone()[0]
                    
                    if table_exists:
                        # Get row count for confirmation
                        cursor.execute(f'SELECT COUNT(*) FROM "{new_table}";')
                        row_count = cursor.fetchone()[0]
                        flash(f"Historic table '{new_table}' selected successfully! ({row_count:,} rows)", "success")
                    else:
                        flash(f"Warning: Table '{new_table}' does not exist in the database. Please verify the table name.", "warning")
                        
            except Exception as e:
                flash(f"Error accessing table '{new_table}': {str(e)}", "error")
            finally:
                db_conn.close()
        else:
            flash(f"Table '{new_table}' set, but couldn't verify existence (database connection failed).", "warning")
            
    except Exception as e:
        flash(f"Error setting historic table: {str(e)}", "error")
    
    return redirect(url_for('index'))

@app.route('/api/db_tables')
def list_db_tables():
    """API endpoint to list available database tables"""
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
                WHERE table_schema = 'public' 
                AND table_type = 'BASE TABLE'
                ORDER BY table_name;
            """)
            
            tables_data = cursor.fetchall()
            tables = [{"name": row[0], "columns": row[1]} for row in tables_data]
            
        return jsonify({"status": "ok", "tables": tables})
        
    except Exception as e:
        app.logger.error(f"Error fetching table names: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        if db_conn:
            db_conn.close()

@app.route('/api/table_info/<table_name>')
def get_table_info(table_name):
    """Get detailed information about a specific table"""
    try:
        db_conn = analysis_engine.get_db_connection()
        if not db_conn:
            return jsonify({"status": "error", "message": "Database connection failed"}), 500
            
        with db_conn.cursor() as cursor:
            # Check if table exists and get basic info
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = %s
                );
            """, (table_name,))
            
            table_exists = cursor.fetchone()[0]
            
            if not table_exists:
                return jsonify({"status": "error", "message": f"Table '{table_name}' does not exist"}), 404
            
            # Get table statistics
            cursor.execute(f'SELECT COUNT(*) FROM "{table_name}";')
            row_count = cursor.fetchone()[0]
            
            # Get column information
            cursor.execute("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns 
                WHERE table_schema = 'public' 
                AND table_name = %s 
                ORDER BY ordinal_position;
            """, (table_name,))
            
            columns = [{"name": col[0], "type": col[1], "nullable": col[2] == 'YES'} 
                      for col in cursor.fetchall()]
            
            return jsonify({
                "status": "ok",
                "table_name": table_name,
                "row_count": row_count,
                "column_count": len(columns),
                "columns": columns
            })
            
    except Exception as e:
        app.logger.error(f"Error getting table info for {table_name}: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        if db_conn:
            db_conn.close()

@app.route('/analyze', methods=['POST'])
def analyze_data():
    """Process uploaded file and run analysis against historic baseline"""
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
    
    if not analysis_engine.HISTORIC_TABLE_NAME:
        flash('Please set a historic table name before running analysis.', 'error')
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
        # Run analysis
        report_html, historic_preview_html, current_preview_html, log_id, analyzed_data = \
            analysis_engine.run_analysis(filepath)
        
        # Store results in session
        session['last_analysis'] = {
            'log_id': log_id,
            'report_html': report_html,
            'historic_preview_html': historic_preview_html,
            'current_preview_html': current_preview_html,
            'analyzed_data': analyzed_data,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        flash(f"Analysis completed successfully! Log ID: {log_id}", "success")
        return redirect(url_for('view_results', log_id=log_id))
        
    except Exception as e:
        app.logger.error(f"Analysis failed for file {filename_secure}: {e}", exc_info=True)
        flash(f"Analysis failed: {pyhtml.escape(str(e))}", "error")
        return redirect(url_for('index'))

@app.route('/results/<log_id>')
def view_results(log_id):
    """View analysis results"""
    # Try to get from session first
    last_analysis = session.get('last_analysis')
    
    if last_analysis and last_analysis.get('log_id') == log_id:
        return render_template('results.html',
                             log_id=log_id,
                             report_html=last_analysis['report_html'],
                             historic_preview_html=last_analysis['historic_preview_html'],
                             current_preview_html=last_analysis['current_preview_html'],
                             analyzed_data=last_analysis['analyzed_data'])
    
    # Try to load from database
    try:
        analysis_data = load_analysis_from_db(log_id)
        if analysis_data:
            return render_template('results.html',
                                 log_id=log_id,
                                 report_html=analysis_data.get('report_html', '<p>Report not available</p>'),
                                 historic_preview_html=analysis_data.get('historic_preview_html', ''),
                                 current_preview_html=analysis_data.get('current_preview_html', ''),
                                 analyzed_data=analysis_data)
        else:
            flash(f"Analysis with Log ID {log_id} not found.", "error")
            return redirect(url_for('analysis_logs'))
            
    except Exception as e:
        app.logger.error(f"Error loading analysis {log_id}: {e}")
        flash("Error loading analysis results.", "error")
        return redirect(url_for('analysis_logs'))

@app.route('/analysis_logs')
def analysis_logs():
    """Display analysis logs from database"""
    logs = []
    db_conn = None
    error_message = None
    
    try:
        db_conn = analysis_engine.get_db_connection()
        if not db_conn:
            flash("Database connection failed. Cannot display logs.", "error")
            error_message = "Database connection failed"
            return render_template('analysis_logs.html', logs=logs, error_message=error_message)
        
        # Ensure the log table exists
        analysis_engine.ensure_log_table_exists(db_conn)
        
        with db_conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
            query = f"""
            SELECT log_id, run_timestamp, status, new_data_filename, 
                   historic_table_name, error_message, analysis_summary
            FROM "{analysis_engine.ANALYSIS_LOG_TABLE_NAME}"
            ORDER BY run_timestamp DESC
            LIMIT 50;
            """
            cursor.execute(query)
            logs_data = cursor.fetchall()
            logs = [dict(row) for row in logs_data]
    
    except psycopg2.Error as e:
        app.logger.error(f"Database error fetching logs: {e}")
        error_message = f"Database error: {str(e).split('DETAIL:')[0].strip()}"
        flash("Error fetching analysis logs from database.", "error")
    except Exception as e:
        app.logger.error(f"Unexpected error fetching logs: {e}")
        error_message = f"Unexpected error: {str(e)}"
        flash("Unexpected error occurred while fetching logs.", "error")
    finally:
        if db_conn:
            db_conn.close()
    
    return render_template('analysis_logs.html', logs=logs, error_message=error_message)

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
    
    historic_table_status = "ok" if analysis_engine.HISTORIC_TABLE_NAME else "warning"
    historic_table_message = f"Historic table: {analysis_engine.HISTORIC_TABLE_NAME}" if analysis_engine.HISTORIC_TABLE_NAME else "No historic table configured"
    
    overall_status = "ok"
    if db_status == "error":
        overall_status = "error"
    elif table_status == "error":
        overall_status = "error"
    elif historic_table_status == "warning" or table_status == "warning":
        overall_status = "degraded"
    
    return jsonify({
        "service_status": overall_status,
        "message": "Data Analysis Service",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "components": {
            "database": {"status": db_status, "message": db_message},
            "log_table": {"status": table_status, "message": table_message},
            "historic_table_config": {"status": historic_table_status, "message": historic_table_message}
        }
    })

def load_analysis_from_db(log_id):
    """Load analysis results from database"""
    db_conn = None
    try:
        db_conn = analysis_engine.get_db_connection()
        if not db_conn:
            return None
        
        with db_conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
            query = f"""
            SELECT log_id, run_timestamp, status, new_data_filename, 
                   historic_table_name, error_message, analysis_summary
            FROM "{analysis_engine.ANALYSIS_LOG_TABLE_NAME}"
            WHERE log_id = %s;
            """
            cursor.execute(query, (log_id,))
            row = cursor.fetchone()
            
            if row:
                return {
                    'log_id': str(row['log_id']),
                    'timestamp': row['run_timestamp'].isoformat() if row['run_timestamp'] else None,
                    'status': row['status'],
                    'filename': row['new_data_filename'],
                    'historic_table': row['historic_table_name'],
                    'error_message': row['error_message'],
                    'summary': row['analysis_summary'],
                    'report_html': f"<div class='card'><h3>Log Entry Details</h3><p><strong>Status:</strong> {row['status']}</p><p><strong>Summary:</strong> {row['analysis_summary'] or 'No summary available'}</p></div>"
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.getenv("PORT", 8000)))