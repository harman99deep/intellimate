CREATE TABLE historical_data (
    customer_id VARCHAR PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    age INT,
    email TEXT,
    city TEXT,
    state TEXT);

-- Load data from CSV
COPY historical_data(customer_id, first_name, last_name, age, email, city, state)
FROM 'C:\Users\HarmandeepSingh\data_monitoring_tool\postgres\imports\historical_data.csv'
DELIMITER ','
CSV HEADER;

docker-entrypoint-initdb.d