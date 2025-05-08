# main.py (or app.py)

import os
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import analysis_engine  # Import the refactored analysis logic

# Configuration
UPLOAD_FOLDER = 'uploads'  # Folder to store uploads temporarily
ALLOWED_EXTENSIONS = {'csv'}  # Allow only CSV files

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'your_very_secret_key_for_flask_sessions'  # Important for flashing messages
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # Limit upload size (e.g., 32MB)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET'])
def index():
    """Renders the upload page."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_files():
    """
    Handles new data file upload, optional historic sample upload,
    runs analysis against historic DB stats, and displays results.
    """
    # New data file is mandatory
    if 'new_file' not in request.files:
        flash('New data file (CSV) is required.')
        return redirect(request.url)

    new_file_from_form = request.files['new_file']
    # Historic sample file is optional
    historic_sample_file_from_form = request.files.get('historic_file')

    # Check if a new file was actually selected
    if new_file_from_form.filename == '':
        flash('No new data file selected. Please select a CSV file.')
        return redirect(url_for('index'))

    new_filepath = None
    historic_sample_filepath = None # Will remain None if no historic sample is provided

    # Default values for template rendering in case of early exit or error
    report_html = "<h3>Analysis could not be run or was not initiated.</h3>"
    historic_preview_html = "<p>No historic sample data provided or processed for preview.</p>"
    new_preview_html = "<p>No new data processed for preview.</p>"
    new_filename_display = "N/A"
    historic_filename_display = "N/A (Not Provided or Invalid)"


    try:
        # Process New Data File (Mandatory)
        if new_file_from_form and allowed_file(new_file_from_form.filename):
            new_filename_secure = secure_filename(new_file_from_form.filename)
            new_filepath = os.path.join(app.config['UPLOAD_FOLDER'], new_filename_secure)
            new_file_from_form.save(new_filepath)
            new_filename_display = new_filename_secure # For display on results page
            print(f"New data file saved: {new_filepath}")
        else:
            flash('Invalid new data file type. Please upload a CSV file.')
            # Clean up potentially saved (but invalid) new file if an error occurs before analysis
            if new_filepath and os.path.exists(new_filepath): os.remove(new_filepath)
            return redirect(url_for('index'))


        # Process Optional Historic Sample File
        if historic_sample_file_from_form and historic_sample_file_from_form.filename != '':
            if allowed_file(historic_sample_file_from_form.filename):
                historic_filename_secure = secure_filename(historic_sample_file_from_form.filename)
                historic_sample_filepath = os.path.join(app.config['UPLOAD_FOLDER'], historic_filename_secure)
                historic_sample_file_from_form.save(historic_sample_filepath)
                historic_filename_display = historic_filename_secure # For display
                print(f"Historic sample file saved: {historic_sample_filepath}")
            else:
                # Historic file provided but invalid type
                flash('Invalid historic sample file type (must be CSV). It will be ignored.')
                historic_filename_display = f"{secure_filename(historic_sample_file_from_form.filename)} (Ignored - Invalid Type)"
                # No need to save historic_sample_filepath if invalid

        # --- Run the Analysis ---
        print("Starting analysis engine (DB historic mode)...")
        # analysis_engine.run_analysis now expects new_filepath and optional historic_sample_filepath
        report_html, historic_preview_html, new_preview_html = analysis_engine.run_analysis(
            new_filepath,
            historic_sample_filepath  # This can be None if no valid historic sample was provided
        )
        print("Analysis engine finished.")

    except Exception as e:
        # Catch errors during file saving or analysis triggering
        print(f"Error during file processing or analysis startup: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for server logs
        flash(f'An error occurred: {e}')
        report_html = f"<h2>Error</h2><p>Could not process files or run analysis: {e}</p>"
        # Previews will use default "No data" messages
    finally:
        # --- Clean up uploaded files ---
        try:
            if new_filepath and os.path.exists(new_filepath):
                os.remove(new_filepath)
                print(f"Removed temporary new data file: {new_filepath}")
            if historic_sample_filepath and os.path.exists(historic_sample_filepath):
                os.remove(historic_sample_filepath)
                print(f"Removed temporary historic sample file: {historic_sample_filepath}")
        except Exception as e_remove:
            print(f"Error removing temporary files: {e_remove}")  # Log error, but continue

    # Render the results page
    return render_template('results.html',
                          report_content=report_html,
                          historic_preview=historic_preview_html, # Preview of the historic *sample*
                          new_preview=new_preview_html,
                          historic_filename=historic_filename_display, # Display name of historic sample
                          new_filename=new_filename_display)


@app.route('/api/health', methods=['GET'])
def health_check():
    """API endpoint for health check."""
    return jsonify({"status": "ok", "message": "Service is running"})

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)