<!-- results.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analysis Results</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='styles.css') }}">
    <style>
        /* Basic styling for the report */
        body { font-family: sans-serif; line-height: 1.6; padding: 20px; background-color: #f4f7f6; }
        h1, h2, h3, h4 { color: #333; }
        h1 { text-align: center; margin-bottom: 20px;}
        hr { border: 0; height: 1px; background: #ccc; margin: 30px 0; }

        /* AI Analysis Sections */
        pre {
            background-color: #e9f5f9; /* Light blue-grey background for preformatted text */
            padding: 15px;
            border: 1px solid #b0c4de; /* Light steel blue border */
            border-radius: 4px;
            white-space: pre-wrap; /* Wrap long lines */
            word-wrap: break-word;
            font-family: 'Courier New', Courier, monospace; /* Use monospace font */
            font-size: 0.95em;
            line-height: 1.5;
        }
        details {
            border: 1px solid #d1d8e0; /* Softer border for details */
            border-radius: 5px;
            margin-bottom: 20px;
            background-color: #ffffff; /* White background for details */
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        summary {
            font-weight: bold;
            padding: 12px 15px;
            background-color: #eaf1f8; /* Lighter blue for summary bar */
            color: #2c3e50;
            cursor: pointer;
            border-bottom: 1px solid #d1d8e0;
            border-radius: 5px 5px 0 0; /* Rounded top corners */
            list-style: none; /* Remove default marker */
            position: relative;
        }
        summary::-webkit-details-marker { display: none; } /* Hide marker for Chrome/Safari */
        summary::before { /* Custom marker */
            content: '►';
            margin-right: 8px;
            font-size: 0.9em;
            color: #007bff;
            display: inline-block;
            transition: transform 0.2s ease-in-out;
        }
        details[open] > summary::before {
            transform: rotate(90deg);
        }
        details[open] > summary {
             background-color: #d6e4f0; /* Darker blue when open */
             border-bottom-left-radius: 0;
             border-bottom-right-radius: 0;
        }
        details > pre, details > p, details > table, details > .preview-table-container, details > div > table {
            padding: 15px;
            border-top: 1px solid #d1d8e0; /* Separator inside for content */
        }

        /* Data Preview Tables */
        .preview-table-container {
            max-width: 100%;
            max-height: 350px;
            overflow: auto;
            border: 1px solid #ced4da; /* Standard border color */
            margin-bottom: 20px;
            background: #fff;
            border-radius: 4px;
        }
        .preview-table {
            border-collapse: collapse;
            width: 100%;
            min-width: 600px;
        }
        .preview-table th, .preview-table td {
            border: 1px solid #dee2e6; /* Lighter border for table cells */
            padding: 6px 10px; /* Increased padding */
            text-align: left;
            font-size: 0.9em; /* Slightly smaller font */
        }
        .preview-table th {
            background: #e9ecef; /* Light grey for headers */
            color: #495057;
            position: sticky;
            top: 0;
            z-index: 2;
        }
        .preview-table tr:nth-child(even) {
            background-color: #f8f9fa; /* Zebra striping for readability */
        }

        /* Result Tables (Drift, Anomalies) */
        .result-table {
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 30px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            border-radius: 4px;
            overflow: hidden; /* For rounded corners on table */
        }
        .result-table th, .result-table td {
            border: 1px solid #dee2e6;
            padding: 8px 12px;
            text-align: left;
        }
        .result-table th {
            background: #007bff; /* Primary blue for main result table headers */
            color: #fff;
        }
        .result-table tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        .result-table tr:hover {
            background-color: #e9ecef;
        }

        /* Highlighting for Drift and Anomalies in tables */
        .highlight-drift { background-color: #fff3cd !important; } /* Light yellow for drift */
        .highlight-anomaly-high { background-color: #f8d7da !important; font-weight: bold; } /* Reddish for high severity */
        .highlight-anomaly-med { background-color: #ffeeba !important; } /* Orangish for medium severity */

        #historic-preview-section, #new-preview-section {
            display: none; /* Hidden by default */
            margin-bottom: 30px;
            background-color: #fff;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .preview-btn {
            background: #007bff;
            color: #fff;
            border: none;
            padding: 10px 18px;
            border-radius: 5px;
            font-size: 1em;
            cursor: pointer;
            margin-bottom: 18px;
            margin-right: 10px;
            transition: background-color 0.2s ease;
        }
        .preview-btn:hover {
            background: #0056b3;
        }
        .action-link {
            display: inline-block;
            margin-top: 20px;
            margin-bottom: 20px;
            padding: 10px 15px;
            background-color: #28a745;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            transition: background-color 0.2s ease;
        }
        .action-link:hover {
            background-color: #218838;
        }

        /* Ensure preview tables inside details work well */
         details .preview-table-container {
            max-width: 100%;
            max-height: 400px; /* Adjust height as needed */
            overflow: auto;
            border: 1px solid #ccc;
            margin-top: 10px; /* Add some space */
         }
         details .preview-table th {
            position: sticky;
            top: 0;
            z-index: 1; /* Lower z-index than main preview */
            background-color: #f2f2f2; /* Lighter header for nested tables */
         }
    </style>
</head>
<body>
    <h1>Data Drift & Anomaly Analysis Results</h1>
    <a href="/" class="action-link">Run New Analysis</a>
    <hr>

    <p>
        <strong>Historic Dataset:</strong> {{ historic_filename | default('N/A') }} <br>
        <strong>New Dataset:</strong> {{ new_filename | default('N/A') }}
    </p>

    <button class="preview-btn" onclick="togglePreview('historic')">Preview: Historic Data (First 10 Rows)</button>
    <button class="preview-btn" onclick="togglePreview('new')">Preview: New Data (First 10 Rows)</button>

    <div id="historic-preview-section">
        <h3>Historic Data Preview (First 10 Rows)</h3>
        <div class="preview-table-container">{{ historic_preview|safe }}</div>
    </div>
    <div id="new-preview-section">
        <h3>New Data Preview (First 10 Rows)</h3>
        <div class="preview-table-container">{{ new_preview|safe }}</div>
    </div>

    <hr>

    <!-- Render the HTML report content passed from Flask -->
    {{ report_content|safe }}

    <hr>
    <a href="/" class="action-link">Run New Analysis</a>

    <script>
    function togglePreview(which) {
        var historicSection = document.getElementById('historic-preview-section');
        var newSection = document.getElementById('new-preview-section');
        if (which === 'historic') {
            historicSection.style.display = (historicSection.style.display === 'none' || historicSection.style.display === '') ? 'block' : 'none';
        } else if (which === 'new') {
            newSection.style.display = (newSection.style.display === 'none' || newSection.style.display === '') ? 'block' : 'none';
        }
    }
    </script>
</body>
</html>