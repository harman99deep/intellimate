# app.py

import os
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import analysis_engine  # Import the refactored analysis logic

# Configuration
UPLOAD_FOLDER = 'uploads'  # Folder to store uploads temporarily
ALLOWED_EXTENSIONS = {'csv'}  # Allow only CSV files

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'your_secret_key_here'  # Important for flashing messages
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
    """Handles file uploads, runs analysis, and displays results."""
    if 'historic_file' not in request.files or 'new_file' not in request.files:
        flash('Both historic and new data files are required.')
        return redirect(request.url)  # Redirect back to upload page

    historic_file = request.files['historic_file']
    new_file = request.files['new_file']

    # Check if files are selected
    if historic_file.filename == '' or new_file.filename == '':
        flash('No selected file. Please select both files.')
        return redirect(url_for('index'))  # Redirect to index page

    # Check file extensions and save securely
    if historic_file and allowed_file(historic_file.filename) and \
       new_file and allowed_file(new_file.filename):

        historic_filename = secure_filename(historic_file.filename)
        new_filename = secure_filename(new_file.filename)

        historic_filepath = os.path.join(app.config['UPLOAD_FOLDER'], historic_filename)
        new_filepath = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)

        try:
            historic_file.save(historic_filepath)
            new_file.save(new_filepath)
            print(f"Files saved: {historic_filepath}, {new_filepath}")

            # --- Run the Enhanced Analysis ---
            print("Starting analysis engine...")
            report_html, historic_preview, new_preview = analysis_engine.run_analysis(historic_filepath, new_filepath)
            print("Analysis engine finished.")

        except Exception as e:
            # Catch errors during file saving or analysis triggering
            print(f"Error during file processing or analysis startup: {e}")
            flash(f'An error occurred: {e}')
            report_html = f"<h2>Error</h2><p>Could not process files or run analysis: {e}</p>"
            plot_data = {}
            historic_preview = ''
            new_preview = ''
        finally:
            # --- Clean up uploaded files ---
            try:
                if os.path.exists(historic_filepath):
                    os.remove(historic_filepath)
                    print(f"Removed temporary file: {historic_filepath}")
                if os.path.exists(new_filepath):
                    os.remove(new_filepath)
                    print(f"Removed temporary file: {new_filepath}")
            except Exception as e:
                print(f"Error removing temporary files: {e}")  # Log error, but continue

        # Render the results page with the HTML report and preview tables
        return render_template('results.html',
                       report_content=report_html,
                       historic_preview=historic_preview,
                       new_preview=new_preview,
                       historic_filename=historic_filename, # Keep these if needed
                       new_filename=new_filename) # Keep these if needed

    else:
        flash('Invalid file type. Please upload CSV files only.')
        return redirect(url_for('index'))

@app.route('/api/health', methods=['GET'])
def health_check():
    """API endpoint for health check."""
    return jsonify({"status": "ok", "message": "Service is running"})

if __name__ == '__main__':
    # Use host='0.0.0.0' to make it accessible on your network (use with caution)
    # Debug=True enables auto-reloading and detailed error pages (DO NOT USE IN PRODUCTION)
    app.run(debug=True, host='127.0.0.1', port=5000)


# # app.py

# import os
# from flask import Flask, render_template, request, redirect, url_for, flash
# from werkzeug.utils import secure_filename
# import analysis_engine # Import the refactored analysis logic

# # Configuration
# UPLOAD_FOLDER = 'uploads' # Folder to store uploads temporarily
# ALLOWED_EXTENSIONS = {'csv'} # Allow only CSV files

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['SECRET_KEY'] = 'your secret key here' # Important for flashing messages
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit upload size (e.g., 16MB)

# def allowed_file(filename):
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# # Ensure the upload folder exists
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# @app.route('/', methods=['GET'])
# def index():
#     """Renders the upload page."""
#     return render_template('index.html')

# @app.route('/analyze', methods=['POST'])
# def analyze_files():
#     """Handles file uploads, runs analysis, and displays results."""
#     if 'historic_file' not in request.files or 'new_file' not in request.files:
#         flash('Both historic and new data files are required.')
#         return redirect(request.url) # Redirect back to upload page

#     historic_file = request.files['historic_file']
#     new_file = request.files['new_file']

#     # Check if files are selected
#     if historic_file.filename == '' or new_file.filename == '':
#         flash('No selected file. Please select both files.')
#         return redirect(url_for('index')) # Redirect to index page

#     # Check file extensions and save securely
#     if historic_file and allowed_file(historic_file.filename) and \
#        new_file and allowed_file(new_file.filename):

#         historic_filename = secure_filename(historic_file.filename)
#         new_filename = secure_filename(new_file.filename)

#         historic_filepath = os.path.join(app.config['UPLOAD_FOLDER'], historic_filename)
#         new_filepath = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)

#         try:
#             historic_file.save(historic_filepath)
#             new_file.save(new_filepath)
#             print(f"Files saved: {historic_filepath}, {new_filepath}")

#             # --- Run the Analysis ---
#             print("Starting analysis engine...")
#             report_html, plot_data, historic_preview, new_preview = analysis_engine.run_analysis(historic_filepath, new_filepath)
#             print("Analysis engine finished.")

#         except Exception as e:
#              # Catch errors during file saving or analysis triggering
#              print(f"Error during file processing or analysis startup: {e}")
#              flash(f'An error occurred: {e}')
#              report_html = f"<h2>Error</h2><p>Could not process files or run analysis: {e}</p>"
#              plot_data = {}
#              historic_preview = ''
#              new_preview = ''
#         finally:
#             # --- Clean up uploaded files ---
#             try:
#                 if os.path.exists(historic_filepath):
#                     os.remove(historic_filepath)
#                     print(f"Removed temporary file: {historic_filepath}")
#                 if os.path.exists(new_filepath):
#                     os.remove(new_filepath)
#                     print(f"Removed temporary file: {new_filepath}")
#             except Exception as e:
#                 print(f"Error removing temporary files: {e}") # Log error, but continue

#         # Render the results page with the HTML report and preview tables
#         return render_template('results.html', report_content=report_html, historic_preview=historic_preview, new_preview=new_preview)

#     else:
#         flash('Invalid file type. Please upload CSV files only.')
#         return redirect(url_for('index'))

# if __name__ == '__main__':
#     # Use host='0.0.0.0' to make it accessible on your network (use with caution)
#     # Debug=True enables auto-reloading and detailed error pages (DO NOT USE IN PRODUCTION)
#     app.run(debug=True, host='127.0.0.1', port=5000)