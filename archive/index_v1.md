<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Upload New Data for Analysis</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='styles.css') }}">
    <style>
        /* Additional styles specific to index.html if needed, or move to styles.css */
        .subtitle {
            text-align: center;
            color: #555;
            margin-bottom: 25px;
            font-size: 0.95em;
        }
        .file-label-main {
            display: block;
            font-size: 1.1em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .file-label-sub {
            display: block;
            font-size: 0.85em;
            color: #666;
        }
        .filename-display {
            margin-top: 10px;
            font-style: italic;
            color: #777;
            font-size: 0.9em;
            min-height: 1.2em; /* Prevent layout shift when text changes */
        }
        footer {
            text-align: center;
            margin-top: 40px;
            font-size: 0.9em;
            color: #777;
        }
        /* Flash message base style */
        .flashes li {
            list-style-type: none;
            padding: 12px 15px;
            margin-bottom: 15px;
            border: 1px solid transparent;
            border-radius: 4px;
            font-size: 0.95em;
        }
        /* Specific flash categories (add these to your styles.css for better organization) */
        .flashes li.error, .flashes li.danger { /* 'danger' is a common Bootstrap category name */
            color: #721c24;
            background-color: #f8d7da;
            border-color: #f5c6cb;
        }
        .flashes li.success {
            color: #155724;
            background-color: #d4edda;
            border-color: #c3e6cb;
        }
        .flashes li.info {
            color: #0c5460;
            background-color: #d1ecf1;
            border-color: #bee5eb;
        }
        .flashes li.warning {
            color: #856404;
            background-color: #fff3cd;
            border-color: #ffeeba;
        }
    </style>
</head>
<body>
    <h1>New Data Analysis Engine</h1>
    <p class="subtitle">Analyze a new dataset (CSV) against historical context from a database. Optionally, provide a historic sample CSV for some advanced checks like Isolation Forest.</p>

    <!-- Flash messages -->
    {% with messages = get_flashed_messages(with_categories=true) %}
      {% if messages %}
        <ul class="flashes">
        {% for category, message in messages %}
          {# Use category as class, default to 'info' if no category #}
          <li class="{{ category if category else 'info' }}">{{ message }}</li>
        {% endfor %}
        </ul>
      {% endif %}
    {% endwith %}

    <form method="post" action="/analyze" enctype="multipart/form-data">
        <div class="upload-area">
            <label for="new_file" class="file-label">
                <span class="file-label-main">Drag & Drop New Data (CSV) or Click to Browse</span>
                <span class="file-label-sub">(Required for analysis)</span>
                <input type="file" name="new_file" id="new_file" required>
            </label>
            <p id="new-filename" class="filename-display">No new file selected</p>
        </div>

        <div class="upload-area">
            <label for="historic_file" class="file-label">
                <span class="file-label-main">Drag & Drop Historic Sample Data (CSV) or Click</span>
                <span class="file-label-sub">(Optional - enables Isolation Forest & richer comparisons)</span>
                <input type="file" name="historic_file" id="historic_file">
            </label>
             <p id="historic-filename" class="filename-display">No historic sample file selected</p>
        </div>

        <input type="submit" value="Analyze New Data" class="submit-button">
    </form>

    <footer>
        <p>This tool analyzes the new dataset against historical patterns and statistics (primarily fetched from a backend database if configured).</p>
    </footer>

    <script>
        function updateFilenameDisplay(inputId, displayId) {
            const fileInput = document.getElementById(inputId);
            const filenameDisplay = document.getElementById(displayId);
            const defaultText = displayId === 'new-filename' ? 'No new file selected' : 'No historic sample file selected';

            if (fileInput.files && fileInput.files.length > 0) {
                filenameDisplay.textContent = 'Selected: ' + fileInput.files[0].name;
                filenameDisplay.style.color = '#333'; // Or a CSS class for selected state
            } else {
                filenameDisplay.textContent = defaultText;
                filenameDisplay.style.color = '#777'; // Or a CSS class for default state
            }
        }

        document.getElementById('new_file').onchange = function () {
            updateFilenameDisplay('new_file', 'new-filename');
        };
        document.getElementById('historic_file').onchange = function () {
             updateFilenameDisplay('historic_file', 'historic-filename');
        };

        // Initialize display on page load in case of browser auto-fill (might not always work due to security)
        // updateFilenameDisplay('new_file', 'new-filename');
        // updateFilenameDisplay('historic_file', 'historic-filename');
    </script>
</body>
</html>