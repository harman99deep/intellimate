<!-- index.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Upload Data for Analysis</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='styles.css') }}">
</head>
<body>
    <h1>Data Drift & Anomaly Detector</h1>

    <!-- Flash messages -->
    {% with messages = get_flashed_messages() %}
      {% if messages %}
        <ul class=flashes>
        {% for message in messages %}
          <li style="color: red;">{{ message }}</li>
        {% endfor %}
        </ul>
      {% endif %}
    {% endwith %}

    <form method="post" action="/analyze" enctype="multipart/form-data">
        <div class="upload-area">
            <label for="historic_file" class="file-label">
                <span>Drag & Drop Historic Data (CSV) or Click to Browse</span>
                <input type="file" name="historic_file" id="historic_file" required>
            </label>
            <p id="historic-filename">No historic file selected</p>
        </div>

        <div class="upload-area">
            <label for="new_file" class="file-label">
                <span>Drag & Drop New Data (CSV) or Click to Browse</span>
                <input type="file" name="new_file" id="new_file" required>
            </label>
             <p id="new-filename">No new file selected</p>
        </div>

        <input type="submit" value="Analyze Data" class="submit-button">
    </form>

    <script>
        // Basic script to show selected filenames
        document.getElementById('historic_file').onchange = function () {
            document.getElementById('historic-filename').textContent = 'Selected: ' + this.files[0].name;
        };
        document.getElementById('new_file').onchange = function () {
             document.getElementById('new-filename').textContent = 'Selected: ' + this.files[0].name;
        };
    </script>
</body>
</html>