# Data Monitoring Tool

A Flask-based web application for monitoring data quality and detecting data drift, volume anomalies, and schema changes by comparing a new dataset against a historic baseline in a Supabase PostgreSQL database.

## Features

- Upload new CSV datasets for analysis.
- Compare new data against a configured historic baseline table in PostgreSQL.
- Detect and report statistical drift, distribution drift, volume anomalies, and schema changes.
- Generate detailed analysis reports.
- View a history of past analysis runs.
- Health check endpoint for monitoring the application and database status.

## Setup

### Prerequisites

- Python 3.8+
- Supabase account with PostgreSQL database
- pip (Python package installer)

### Database Setup (Supabase)

1. Create a project in Supabase and get your database connection details
2. Configure your environment variables in `.env`:
   ```env
   # Supabase Database Configuration
   USER=your_supabase_user
   PASSWORD=your_supabase_password
   HOST=your_supabase_host
   PORT=6543
   DBNAME=postgres
   HISTORIC_TABLE_NAME=your_table_name
   ANALYSIS_LOG_TABLE_NAME=data_analysis_logs

   # Other Configuration
   FLASK_SECRET_KEY=your_secret_key
   GEMINI_API_KEY=your_gemini_api_key  # Optional
   ```

### Application Setup

1. Create a Python virtual environment (recommended):
   ```bash
   python -m venv .venv
   ```
2. Activate the virtual environment:
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```
3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the project root directory based on the `.env.example` (if available, otherwise create one manually) and configure your database connection details and Flask secret key. Example `.env`:
   ```env
   DATABASE_URL=postgresql://user:password@localhost:5432/mydatabase
   FLASK_SECRET_KEY=your_secret_key_here
   UPLOAD_FOLDER=uploads
   MAX_FILE_UPLOAD_MB=32
   HISTORIC_TABLE_NAME=your_baseline_table_name
   ```
   Replace `user`, `password`, `mydatabase`, and `your_baseline_table_name` with your actual database credentials and desired baseline table.

## Running the Application

1. Make sure your virtual environment is activated.
2. Ensure the PostgreSQL database is running.
3. Run the Flask application:
   ```bash
   python main.py
   ```
4. The application will be available at `http://localhost:8000` (or the port specified in your `.env` or default 8000).

## Project Structure

```
.
├── analysis_engine.py         # Core data analysis logic
├── Dockerfile                 # Dockerfile for the Flask app (optional)
├── main.py                    # Flask application entry point and routes
├── requirements.txt           # Python dependencies
├── sample.ipynb               # Example Jupyter Notebook (if applicable)
├── .env.example               # Example environment variables file
├── postgres/                  # PostgreSQL setup (docker-compose, init scripts)
│   ├── docker-compose.yml
│   └── init/
│       └── schema.sql         # Database schema
├── static/                    # Static files (CSS, JS, images)
│   └── styles.css
└── templates/                 # Jinja2 HTML templates
    ├── alerts.html
    ├── analysis_nav.html
    ├── data_drift.html
    ├── index.html
    ├── overview.html
    ├── runs_list.html
    ├── schema_changes.html
    └── volume_anomalies.html
```

## Technologies Used

- Flask
- Pandas
- NumPy
- SciPy
- Scikit-learn
- Matplotlib
- Seaborn
- Google Generative AI (if used for analysis summary)
- Psycopg2 (PostgreSQL adapter)
- Python-dotenv
- Werkzeug
- Tabulate
- Jinja2
- PostgreSQL
- Docker

## Health Check

Access the health check endpoint at `/api/health` to check the status of the application and its dependencies.

## Contributing

(Add contributing guidelines here if applicable)

## License

(Add license information here if applicable)
