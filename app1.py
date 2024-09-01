from flask import Flask, request, jsonify, g
import sqlite3
import pandas as pd
import re

app = Flask(__name__)

# Function to get a database connection
def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect('advanced_denormalized_data.sqlite')
    return g.db

# Function to close the database connection
@app.teardown_appcontext
def close_db(exception):
    db = g.pop('db', None)
    if db is not None:
        db.close()

# Get a list of all table names
def get_table_names():
    conn = get_db()
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
    return tables['name'].tolist()

# Load and prepare data from all tables
def load_data():
    conn = get_db()
    table_names = get_table_names()
    df_combined = pd.DataFrame()

    for table in table_names:
        try:
            df = pd.read_sql_query(f"SELECT * FROM {table};", conn)
            df['Source Table'] = table
            df_combined = pd.concat([df_combined, df], ignore_index=True)
        except Exception as e:
            print(f"Error loading table {table}: {e}")

    df_combined = df_combined.astype(str)
    return df_combined

# Function to initialize the data
def initialize_data():
    global df_combined
    with app.app_context():
        df_combined = load_data()

# Call initialize_data() when the application starts
initialize_data()

# Function to highlight matches in text
def highlight_text(text, terms):
    for term in terms:
        text = re.sub(f"({re.escape(term)})", r"<mark>\1</mark>", text, flags=re.IGNORECASE)
    return text

# Function to parse the search query
def parse_query(query):
    terms = []
    operators = []

    for token in re.findall(r'(".*?"|\S+)', query):
        token = token.strip('"')
        if token.upper() in ["AND", "OR", "NOT"]:
            operators.append(token.upper())
        else:
            terms.append(token)

    return terms, operators

# Function to handle exact phrase matching only
def exact_phrase_match(term, text):
    pattern = r'\b' + re.escape(term) + r'\b'
    return re.search(pattern, text, flags=re.IGNORECASE) is not None

# Function to handle specific column searches with exact phrase matching
def search_data(query, df):
    combined_results = {}
    total_match_count = 0
    terms, operators = parse_query(query)
    search_patterns = {}

    for term in terms:
        if ':' in term:
            column_name, search_term = term.split(':', 1)
            search_term = search_term.strip()
            if column_name not in search_patterns:
                search_patterns[column_name] = []
            search_patterns[column_name].append(search_term)
        else:
            search_patterns['global'] = search_patterns.get('global', []) + [term.strip()]

    for idx, row in df.iterrows():
        match_results = []
        full_row_dict = row.to_dict()

        for column_name, patterns in search_patterns.items():
            if column_name != 'global':
                column_matches = []
                for search_term in patterns:
                    if column_name in row and exact_phrase_match(search_term, row[column_name]):
                        highlighted_value = highlight_text(row[column_name], [search_term])
                        full_row_dict[column_name] = highlighted_value
                        total_match_count += 1
                        column_matches.append(True)
                    else:
                        column_matches.append(False)
                match_results.append(all(column_matches))
            else:
                global_matches = []
                row_text = ' '.join(row.values)
                for search_term in patterns:
                    if exact_phrase_match(search_term, row_text):
                        for column, value in row.items():
                            if exact_phrase_match(search_term, value):
                                highlighted_value = highlight_text(value, [search_term])
                                full_row_dict[column] = highlighted_value
                                total_match_count += 1
                        global_matches.append(True)
                    else:
                        global_matches.append(False)
                match_results.append(any(global_matches))

        if all(match_results):
            combined_results[idx] = {
                'Source Table': row['Source Table'],
                'Row Index': idx,
                'Full Row': full_row_dict,
            }

    return list(combined_results.values()), total_match_count

# API endpoint for searching data
@app.route("/search", methods=["POST"])
def perform_search():
    data = request.get_json()
    query = data.get("query")

    if not query:
        return jsonify({"error": "Query is required"}), 400

    results, match_count = search_data(query, df_combined)

    if not results:
        return jsonify({"error": "No relevant results found."}), 404

    return jsonify({
        "results": results,
        "total_matches": match_count,
    })

if __name__ == "__main__":
    app.run(debug=True)
