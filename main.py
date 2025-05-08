# main.py

import os
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import analysis_engine  # Import the refactored analysis logic

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'your_very_secret_key_for_flask_sessions'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_files():
    if 'new_file' not in request.files:
        flash('New data file (CSV) is required.')
        return redirect(request.url)

    new_file = request.files['new_file']
    if new_file.filename == '' or not allowed_file(new_file.filename):
        flash('Invalid or no new data file selected. Please upload a CSV.')
        return redirect(url_for('index'))

    new_filename_secure = secure_filename(new_file.filename)
    new_filepath = os.path.join(app.config['UPLOAD_FOLDER'], new_filename_secure)
    new_file.save(new_filepath)

    report_html = "<h3>Analysis could not be run.</h3>"
    new_preview_html = "<p>No new data preview available.</p>"
    try:
        report_html, _, new_preview_html = analysis_engine.run_analysis(new_filepath)
    except Exception as e:
        flash(f'Error during analysis: {e}')
        report_html = f"<h2>Error</h2><p>{e}</p>"
    finally:
        if os.path.exists(new_filepath):
            os.remove(new_filepath)

    return render_template('results.html',
                           report_content=report_html,
                           historic_preview="<p>Historic data is loaded from PostgreSQL database for comparison.</p>",
                           new_preview=new_preview_html,
                           historic_filename="(From Database)",
                           new_filename=new_filename_secure)

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "message": "Service is running"})

if __name__ == '__main__':
    app.run(debug=True)
