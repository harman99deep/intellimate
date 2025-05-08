<!-- results.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analysis Results</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='styles.css') }}">
    <style>
        /* Add some basic styling for the report */
        body { font-family: sans-serif; line-height: 1.6; padding: 20px; }
        h2, h3, h4 { color: #333; }
        hr { border: 0; height: 1px; background: #ccc; margin: 20px 0; }
        pre { background-color: #96ebfe; padding: 10px; border-radius: 4px; white-space: pre-wrap; word-wrap: break-word; }
        details { border: 1px solid #ccc; border-radius: 4px; margin-bottom: 10px; }
        summary { font-weight: bold; padding: 10px; background-color: #eee; cursor: pointer; }
        details ul { padding: 10px 10px 10px 30px; } /* Indent list inside details */
        li { margin-bottom: 10px; }
        img { margin-top: 10px; border: 1px solid #ddd; }
        .preview-table-container {
            max-width: 100%;
            max-height: 350px;
            overflow: auto;
            border: 1px solid #ccc;
            margin-bottom: 20px;
            background: #fff;
        }
        .preview-table {
            border-collapse: collapse;
            width: 100%;
            min-width: 600px;
        }
        .preview-table th, .preview-table td {
            border: 1px solid #aaa;
            padding: 4px 8px;
            text-align: left;
            font-size: 0.95em;
        }
        .preview-table th {
            background: #e0f7fa;
            position: sticky;
            top: 0;
            z-index: 2;
        }
        .result-table {
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 30px;
        }
        .result-table th, .result-table td {
            border: 1px solid #aaa;
            padding: 6px 10px;
            text-align: left;
        }
        .result-table th {
            background: #b2ebf2;
        }
        .highlight {
            background: #ffe082;
        }
        #historic-preview-section, #new-preview-section {
            display: none;
            margin-bottom: 30px;
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
        }
        .preview-btn:hover {
            background: #0056b3;
        }
    </style>
</head>
<body>
    <h1>Analysis Results</h1>
    <a href="/">Run New Analysis</a>
    <hr>

    <button class="preview-btn" onclick="togglePreview('historic')">Preview Historic Data (First 10 Rows)</button>
    <button class="preview-btn" onclick="togglePreview('new')">Preview New Data (First 10 Rows)</button>
    <div id="historic-preview-section">
        <h3>Historic Data (First 10 Rows)</h3>
        <div class="preview-table-container">{{ historic_preview|safe }}</div>
    </div>
    <div id="new-preview-section">
        <h3>New Data (First 10 Rows)</h3>
        <div class="preview-table-container">{{ new_preview|safe }}</div>
    </div>

    <!-- Render the HTML report content passed from Flask -->
    {{ report_content|safe }}

    <hr>
     <a href="/">Run New Analysis</a>

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