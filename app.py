from flask import Flask, jsonify
import sqlite3

app = Flask(__name__)

# Path to your SQLite database
db_path = 'advanced_denormalized_data.sqlite'

def get_column_names():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Dictionary to hold table names and their columns
    database_structure = {}

    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    # For each table, retrieve the column names
    for table in tables:
        table_name = table[0]
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        
        # Store the column names in the dictionary
        column_names = [column[1] for column in columns]
        database_structure[table_name] = column_names

    conn.close()
    return database_structure

@app.route('/api/columns', methods=['GET'])
def columns():
    columns_data = get_column_names()
    return jsonify(columns_data)

if __name__ == '__main__':
    app.run(debug=True)
