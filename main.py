# main.py
import os
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from werkzeug.utils import secure_filename
import analysis_engine
import psycopg2
import psycopg2.extras
import uuid
from dotenv import load_dotenv, set_key, dotenv_values
from datetime import datetime, timezone
import html as pyhtml # For escaping in main.py if ever needed for flash messages etc.

load_dotenv()

UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
ALLOWED_EXTENSIONS = {'csv'}
MAX_FILE_UPLOAD_MB = int(os.getenv('MAX_FILE_UPLOAD_MB', 32)) # Define it once here

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'dev_secret_key_change_this_in_prod')
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_UPLOAD_MB * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/set_historic_table', methods=['POST'])
def set_historic_table():
    new_table = request.form['historic_table_name']
    dotenv_path = os.path.join(os.path.dirname(__file__), '.env')

    set_key(dotenv_path, 'HISTORIC_TABLE_NAME', new_table)
    load_dotenv(dotenv_path, override=True)

    # **Here** we forcibly update the imported module’s global
    analysis_engine.HISTORIC_TABLE_NAME = new_table

    flash(f"PostgresDB table selected: '{new_table}'", "success")
    return redirect(url_for('index'))


# @app.route('/api/db_tables', methods=['GET']) #new endpoint
# def list_db_tables():
#     try:
#         conn = analysis_engine.get_db_connection()
#         with conn.cursor() as cursor:
#             cursor.execute("""
#                 SELECT table_name
#                 FROM information_schema.tables
#                 WHERE table_schema = 'public'
#                 ORDER BY table_name;
#             """)
#             tables = [row[0] for row in cursor.fetchall()]
#         return jsonify({"status": "ok", "tables": tables})
#     except Exception as e:
#         app.logger.error(f"Error fetching table names: {e}", exc_info=True)
#         return jsonify({"status": "error", "message": str(e)}), 500
#     finally:
#         if conn:
#             conn.close()

@app.route('/', methods=['GET'])
def index():
    log_id_from_query = request.args.get('log_id')
    if log_id_from_query:
        flash(f"Viewing analysis options. Previous Log ID for reference (if any): {pyhtml.escape(log_id_from_query)}", "info")

    env_config = dotenv_values(".env")  # Reads fresh every time
    historic_table = env_config.get('HISTORIC_TABLE_NAME', '')

    return render_template('index.html',
                           max_file_upload_mb=MAX_FILE_UPLOAD_MB,
                           historic_table_name=historic_table)


@app.route('/analyze', methods=['POST'])
def analyze_files():
    if 'new_file' not in request.files:
        flash('New data file (CSV) is required.', 'error')
        return redirect(url_for('index'))

    new_file = request.files['new_file']

    if new_file.filename == '':
        flash('No new data file selected. Please upload a CSV.', 'error')
        return redirect(url_for('index'))

    if not allowed_file(new_file.filename):
        flash(f'Invalid file type. Only {", ".join(ALLOWED_EXTENSIONS)} files are allowed.', 'error')
        return redirect(url_for('index'))

    new_filename_secure = secure_filename(new_file.filename)
    new_filepath = os.path.join(app.config['UPLOAD_FOLDER'], new_filename_secure)
    try:
        new_file.save(new_filepath)
    except Exception as e:
        app.logger.error(f"Error saving uploaded file {new_filename_secure}: {e}")
        flash(f"Error saving file: {pyhtml.escape(str(e))}", "error")
        return redirect(url_for('index'))

    analysis_report_html = "<div class='card error-card'><h3>Analysis did not run or encountered an unexpected issue.</h3></div>"
    historic_preview_html = "<p>Historic data preview could not be generated.</p>"
    new_data_preview_html = "<p>New data preview could not be generated.</p>"
    current_run_log_id = "N/A_ENGINE_FAILURE"
    analyzed_data_for_template = {}
    historic_filename_desc_for_session = "(From Database or Historic Sample CSV)"


    try:
        analysis_report_html, historic_preview_html, new_data_preview_html, engine_log_id_returned, analyzed_data_for_template = \
            analysis_engine.run_analysis(new_filepath)

        current_run_log_id = engine_log_id_returned
        
        # Try to get a more descriptive historic filename if available in the results
        if analyzed_data_for_template and 'historic_data_source' in analyzed_data_for_template:
            if analyzed_data_for_template['historic_data_source'] == "Database":
                historic_filename_desc_for_session = f"Database Table: {analysis_engine.HISTORIC_TABLE_NAME}"
            elif analyzed_data_for_template['historic_data_source'] == "CSV Fallback" and analyzed_data_for_template.get('historic_filename_description'):
                 historic_filename_desc_for_session = analyzed_data_for_template.get('historic_filename_description', "CSV File")
            elif analyzed_data_for_template['historic_data_source'] == "None Available":
                 historic_filename_desc_for_session = "No Historic Baseline Available"


        if any(err_indicator in analysis_report_html for err_indicator in ["Analysis Failed Critically", "Critical Analysis Error"]): # More specific check
             flash(f"Analysis completed with critical errors. Log ID: {current_run_log_id}", "error") # Changed to error
        elif "Analysis Failed" in analysis_report_html: # General failure
             flash(f"Analysis failed. Log ID: {current_run_log_id}", "warning")
        else:
             flash(f"Analysis successful! Log ID: {current_run_log_id}", "success")

        session['last_log_id'] = current_run_log_id
        session['last_report_data'] = {
            'report_content': analysis_report_html,
            'historic_preview': historic_preview_html,
            'new_preview': new_data_preview_html,
            'historic_filename': historic_filename_desc_for_session,
            'new_filename': new_filename_secure,
            'analyzed_results': analyzed_data_for_template
        }

    except Exception as e:
        app.logger.error(f"Critical error in /analyze for file {new_filename_secure}: {e}", exc_info=True)
        flash(f"A critical server error occurred during analysis: {pyhtml.escape(str(e))}", "error")
        # Try to get log_id if engine assigned one before crashing (unlikely for top-level crashes here)
        # For instance, if run_analysis itself raises an error not caught internally.
        current_run_log_id = analyzed_data_for_template.get('log_id', "N/A_CRITICAL_SERVER_ERROR")

        analysis_report_html = f"<div class='card error-card'><h2>Critical Server Error During Analysis</h2><p>{pyhtml.escape(str(e))}</p><p>Log ID (if available): {current_run_log_id}</p></div>"
        if not analyzed_data_for_template:
            analyzed_data_for_template = {
                'log_id': current_run_log_id,
                'error': str(e)
            }
        # Ensure session has some error info if user tries to view "latest result"
        session['last_log_id'] = current_run_log_id
        session['last_report_data'] = {
            'report_content': analysis_report_html, # Show the error report
            'new_filename': new_filename_secure,
            'analyzed_results': analyzed_data_for_template
        }

    return render_template('results.html',
                           log_id=current_run_log_id,
                           report_content=analysis_report_html,
                           historic_preview=historic_preview_html,
                           new_preview=new_data_preview_html,
                           historic_filename=historic_filename_desc_for_session,
                           new_filename=new_filename_secure,
                           analyzed_results=analyzed_data_for_template,
                           is_direct_log_view=False)

@app.route('/results/latest', methods=['GET'])
def view_latest_result():
    last_log_id = session.get('last_log_id')
    last_report_data = session.get('last_report_data')

    if last_log_id and last_report_data:
        return render_template('results.html',
                               log_id=last_log_id,
                               report_content=last_report_data.get('report_content'),
                               historic_preview=last_report_data.get('historic_preview'),
                               new_preview=last_report_data.get('new_preview'),
                               historic_filename=last_report_data.get('historic_filename', "(Historic data details not available)"),
                               new_filename=last_report_data.get('new_filename', "(New file details not available)"),
                               analyzed_results=last_report_data.get('analyzed_results'),
                               is_direct_log_view=False)
    else:
        flash("No previous analysis result found in this session. Please run a new analysis.", "info")
        return redirect(url_for('index'))

@app.route('/results/<uuid:log_id_str>', methods=['GET'])
def view_specific_result(log_id_str):
    db_conn = None
    log_details = None
    log_table_name = analysis_engine.ANALYSIS_LOG_TABLE_NAME

    try:
        db_conn = analysis_engine.get_db_connection()
        if not db_conn:
            flash("Database connection failed. Cannot display log details.", "error")
            return redirect(url_for('analysis_logs'))

        with db_conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
            sql_query = f'SELECT log_id, run_timestamp, status, new_data_filename, error_message, analysis_summary FROM "{log_table_name}" WHERE log_id = %s;'
            cursor.execute(sql_query, (str(log_id_str),))
            log_details = cursor.fetchone()

        if not log_details:
            flash(f"Log ID {log_id_str} not found.", "error")
            return redirect(url_for('analysis_logs'))

        report_content_for_log_view = None
        # If status is critical failure, the error_message from log is most relevant.
        # The analysis_summary might be a generic "critical failure" message.
        summary_or_error = log_details['analysis_summary']
        if "Critical Failure" in log_details['status'] and log_details['error_message']:
            summary_or_error = log_details['error_message'] # Prefer detailed error message for critical failures
            # Construct a basic error report HTML
            report_content_for_log_view = f"<div class='card error-card'><h2>Analysis Log: Critical Failure</h2><p><strong>Log ID:</strong> {log_details['log_id']}</p><p><strong>Timestamp:</strong> {log_details['run_timestamp'].strftime('%Y-%m-%d %H:%M:%S UTC') if log_details['run_timestamp'] else 'N/A'}</p><h4>Error Details:</h4><pre style='white-space: pre-wrap; word-break: break-all;'>{pyhtml.escape(summary_or_error)}</pre></div>"


        return render_template('results.html',
                               log_id=str(log_details['log_id']),
                               report_content=report_content_for_log_view, # This will be minimal for non-critical failures or the error HTML for critical ones
                               historic_preview=None,
                               new_preview=None,
                               historic_filename=None,
                               new_filename=log_details['new_data_filename'],
                               analyzed_results=None, # Full dict not stored in log
                               log_summary=summary_or_error, # Use the determined summary or error
                               status=log_details['status'],
                               timestamp=log_details['run_timestamp'].strftime('%Y-%m-%d %H:%M:%S UTC') if log_details['run_timestamp'] else 'N/A',
                               is_direct_log_view=True)

    except psycopg2.Error as e:
        app.logger.error(f"Database error fetching log {log_id_str}: {e}")
        flash(f"Database error accessing log {log_id_str}: {pyhtml.escape(str(e).splitlines()[0])}", "error")
        return redirect(url_for('analysis_logs'))
    except Exception as e:
        app.logger.error(f"Unexpected error fetching log {log_id_str}: {e}", exc_info=True)
        flash(f"Unexpected error fetching log {log_id_str}.", "error")
        return redirect(url_for('analysis_logs'))
    finally:
        if db_conn:
            db_conn.close()


@app.route('/analysis_logs', methods=['GET'])
def analysis_logs():
    logs = []
    db_conn = None
    log_table_name = analysis_engine.ANALYSIS_LOG_TABLE_NAME
    error_msg_page = None

    if not log_table_name or not isinstance(log_table_name, str) or not log_table_name.strip():
        flash("Log table name is not configured or is invalid.", "error")
        app.logger.error("ANALYSIS_LOG_TABLE_NAME is not set or is invalid.")
        error_msg_page = "Log table name configuration error."
        return render_template('analysis_logs.html', logs=logs, error_message=error_msg_page, SCRIPT_LOAD_TIME=datetime.now(timezone.utc))

    try:
        db_conn = analysis_engine.get_db_connection()
        if not db_conn:
            flash("Database connection failed. Cannot display logs.", "error")
            app.logger.error("Failed to establish database connection for fetching logs.")
            error_msg_page = "Database connection error."
            return render_template('analysis_logs.html', logs=logs, error_message=error_msg_page, SCRIPT_LOAD_TIME=datetime.now(timezone.utc))

        analysis_engine.ensure_log_table_exists(db_conn)

        with db_conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
            sql_query = f'SELECT log_id, run_timestamp, status, new_data_filename, error_message, analysis_summary FROM "{log_table_name}" ORDER BY run_timestamp DESC LIMIT 100;'
            cursor.execute(sql_query)
            logs_data = cursor.fetchall()
            logs = [dict(row) for row in logs_data]
            if not logs_data and not error_msg_page :
                 flash("No analysis logs found in the database.", "info")
    except psycopg2.Error as e:
        app.logger.error(f"Database error while accessing logs from table '{log_table_name}': {e}", exc_info=True)
        db_error_str = str(e).split('\n')[0]
        if "relation" in str(e).lower() and "does not exist" in str(e).lower():
            flash(f"Log table '{log_table_name}' does not exist. It should be auto-created on first analysis.", "error")
            error_msg_page = f"Log table '{log_table_name}' not found."
        else:
            flash(f"Database error accessing logs: {pyhtml.escape(db_error_str)}", "error")
            error_msg_page = f"Database error: {pyhtml.escape(db_error_str)}"
    except Exception as e:
        app.logger.error(f"Unexpected error fetching analysis logs: {e}", exc_info=True)
        error_msg_page = "An unexpected error occurred while fetching logs."
        flash(error_msg_page, "error")
    finally:
        if db_conn:
            db_conn.close()

    return render_template('analysis_logs.html', logs=logs, error_message=error_msg_page, SCRIPT_LOAD_TIME=datetime.now(timezone.utc))

@app.route('/api/health', methods=['GET'])
def health_check():
    db_status = "ok"
    db_message = "Database connection successful."
    db_conn = None
    log_table_config_status = "ok"
    log_table_config_message = "Log table name is configured."
    log_table_name_value = analysis_engine.ANALYSIS_LOG_TABLE_NAME

    if not log_table_name_value or not isinstance(log_table_name_value, str) or not log_table_name_value.strip():
        log_table_config_status = "error"
        log_table_config_message = "ANALYSIS_LOG_TABLE_NAME environment variable is missing or empty."
        log_table_name_value = "Not configured"

    try:
        db_conn = analysis_engine.get_db_connection()
        if not db_conn:
            db_status = "error"
            db_message = "Database connection failed (check credentials, network, or server status)."
        else:
            with db_conn.cursor() as cursor:
                cursor.execute("SELECT 1;")
                if cursor.fetchone()[0] != 1:
                    db_status = "error"
                    db_message = "Database query test (SELECT 1) failed."

                if log_table_config_status == "ok":
                    try:
                        cursor.execute(f"SELECT 1 FROM \"{analysis_engine.ANALYSIS_LOG_TABLE_NAME}\" LIMIT 1;")
                        cursor.fetchone()
                        db_message += f" Log table '{analysis_engine.ANALYSIS_LOG_TABLE_NAME}' is accessible."
                    except psycopg2.Error as e_table_check:
                        if "relation" in str(e_table_check).lower() and "does not exist" in str(e_table_check).lower():
                            db_status = "warning"
                            db_message += f" Log table '{analysis_engine.ANALYSIS_LOG_TABLE_NAME}' does not exist (should be auto-created)."
                        else:
                            db_status = "error"
                            db_message += f" Error accessing log table '{analysis_engine.ANALYSIS_LOG_TABLE_NAME}': {pyhtml.escape(str(e_table_check).splitlines()[0])}."
    except Exception as e:
        db_status = "error"
        db_message = f"Database health check encountered an exception: {pyhtml.escape(str(e).splitlines()[0])}"
    finally:
        if db_conn:
            db_conn.close()

    overall_status = "ok"
    if db_status == "error" or log_table_config_status == "error":
        overall_status = "error"
    elif db_status == "warning":
        overall_status = "degraded"

    return jsonify({
        "service_status": overall_status,
        "message": "Data Analysis Service Health Status",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dependencies": {
            "database_connection": {"status": db_status, "message": db_message},
            "log_table_configuration": {
                "status": log_table_config_status,
                "name_configured": log_table_name_value,
                "message": log_table_config_message
            },
            "gemini_ai": {
                "status": "configured" if analysis_engine.USE_GEMINI else "not_configured",
                "message": "Gemini AI is " + ("" if analysis_engine.USE_GEMINI else "not ") + "configured."
            }
        }
    })


@app.context_processor
def inject_global_vars():
    return {'SCRIPT_LOAD_TIME': datetime.now(timezone.utc)}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.getenv("PORT", 8000)))