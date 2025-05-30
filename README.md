# Data Monitoring Tool

A Flask-based web application for monitoring data quality and detecting data drift, volume anomalies, and schema changes by comparing a new dataset against a historic baseline in a Supabase PostgreSQL database.

## Migration Notice

This tool has been migrated from local PostgreSQL (Docker) to Supabase for improved scalability and easier deployment. The `/postgres` directory and Docker configuration files have been archived.

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

1. Create a new project in Supabase:
   - Go to [Supabase Dashboard](https://app.supabase.com)
   - Click "New Project"
   - Fill in the project details
   - Note down your database password

2. Get your database connection details:
   - In your Supabase project dashboard, go to Settings > Database
   - Find the "Connection Info" section
   - Note down the following:
     - Host: `[project-ref].supabase.co`
     - Port: `6543` (Supabase connection pooler port)
     - Database name: `postgres`
     - User: `postgres`
     - Password: (the one you set during project creation)

3. Configure your environment variables in `.env`:
   ```env
   # Supabase Database Configuration
   USER=postgres
   PASSWORD=your_db_password
   HOST=your-project-ref.supabase.co
   PORT=6543
   DBNAME=postgres
   HISTORIC_TABLE_NAME=data_baseline_table
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

- Flask 3.1.1
- Pandas 2.2.6
- NumPy 1.15.3
- SciPy 1.15.3
- Scikit-learn 1.4.2
- Matplotlib 3.10.3
- Seaborn 0.13.2
- Google Generative AI (optional, for enhanced analysis summaries)
- Psycopg2-binary 2.9.9 (PostgreSQL adapter)
- Python-dotenv 1.0.1
- Werkzeug 3.0.1
- Tabulate 0.9.0
- Jinja2 3.1.3
- Supabase PostgreSQL (managed database)

## Health Check

Access the health check endpoint at `/api/health` to check the status of the application and its dependencies.

## Contributing

(Add contributing guidelines here if applicable)

## License

(Add license information here if applicable)
