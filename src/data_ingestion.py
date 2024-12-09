"""

Using query_database to query the agri.db database and list_tables to show the available tables found in agri_db. 
With the table name, we can use sql query to call the correct table.

"""

# Function to query the database
import sqlite3  # For database connection
import pandas as pd  # For data manipulation

def load_data(db_path):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM farm_data"
    data = pd.read_sql_query(query, conn)
    conn.close()
    return data

if __name__ == "__main__":
    db_path = "Data/agri.db"  # Path to your database file
    
    # Step 1: Load data
    agri_data = load_data(db_path)
    print("Data Loaded:")
    print(agri_data.head())
    
    # Step 2: Save the data to a CSV file
    output_path = "Data/agri_data.csv"
    agri_data.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
