from flask import Flask, request, jsonify
import sqlite3
import pandas as pd
import re
import json

app = Flask(__name__)

# Initialize SQLite connection
db_path = 'advanced_denormalized_data.sqlite'
conn = sqlite3.connect(db_path, check_same_thread=False)

# Load all tables into a single DataFrame
def load_combined_data():
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
    table_names = tables['name'].tolist()
    
    df_combined = pd.DataFrame()
    for table in table_names:
        try:
            df = pd.read_sql_query(f"SELECT * FROM {table};", conn)
            df['Source Table'] = table
            if not df.empty:
                df_combined = pd.concat([df_combined, df], ignore_index=True)
        except Exception as e:
            print(f"Error loading table {table}: {e}")
    
    return df_combined.astype(str)

df_combined = load_combined_data()

# Helper function to highlight text
def highlight_text(text, terms):
    for term in terms:
        term_escaped = re.escape(term)
        text = re.sub(f"(?i)(\\b{term_escaped}\\b)", lambda m: f"<highlight>{m.group(0)}</highlight>", text)
    return text

# Function to parse the search query into terms and operators
def parse_query(query):
    terms = []
    operators = []

    # Detect date range pattern
    date_pattern = r'(\w+):\[(.*?)\s*TO\s*(.*?)\]'
    
    tokens = re.split(r'(\band\b|\bor\b|\bnot\b|\(|\)|\+)', query, flags=re.IGNORECASE)
    buffer = []
    in_parentheses = False

    for token in tokens:
        token = token.strip()
        if not token:
            continue
        # Handle date range pattern
        date_match = re.match(date_pattern, token)
        if date_match:
            column_name = date_match.group(1)
            start_date = date_match.group(2).strip()
            end_date = date_match.group(3).strip()
            terms.append(f"{column_name}:[{start_date} TO {end_date}]")
        elif token == "(":
            in_parentheses = True
            buffer.append(token)
        elif token == ")":
            in_parentheses = False
            buffer.append(token)
        elif token.upper() in ["AND", "OR", "NOT"]:
            if in_parentheses:
                buffer.append(token)
            else:
                if buffer:
                    terms.append(" ".join(buffer))
                    buffer = []
                operators.append(token.upper())
        elif token == "+":
            operators.append("OR")
        else:
            buffer.append(token)

    if buffer:
        terms.append(" ".join(buffer))

    return terms, operators

# Function to search the data based on the parsed query
def search_data(query, df):
    exact_matches = []
    terms, operators = parse_query(query)

    for idx, row in df.iterrows():
        match = None

        for i, term in enumerate(terms):
            # Handle date range
            date_pattern = r'(\w+):\[(.*?)\s*TO\s*(.*?)\]'
            date_match = re.match(date_pattern, term)
            if date_match:
                column_name = date_match.group(1)
                start_date = pd.to_datetime(date_match.group(2).strip(), errors='coerce')
                end_date = pd.to_datetime(date_match.group(3).strip(), errors='coerce')

                if column_name in row.index:
                    row_date = pd.to_datetime(row[column_name], errors='coerce')
                    term_found = start_date <= row_date <= end_date
            else:
                if ':' in term:
                    column_name, search_term = term.split(':', 1)
                    column_name = column_name.strip()
                    search_term = search_term.strip()
                    if column_name in row.index:
                        term_found = search_term.lower() in row[column_name].lower()
                else:
                    row_text = ' '.join(row.values).strip()
                    term_pattern = re.compile(f"(?i)\\b{re.escape(term)}\\b")
                    term_found = bool(term_pattern.search(row_text))
            
            if i == 0:
                match = term_found
            else:
                if operators[i - 1] == "AND":
                    match = match and term_found
                elif operators[i - 1] == "OR":
                    match = match or term_found
                elif operators[i - 1] == "NOT":
                    match = match and not term_found

        if match:
            highlighted_row = {k: highlight_text(v, terms) for k, v in row.items()}
            exact_matches.append(highlighted_row)

    return exact_matches

# Function to filter data by date range
def filter_by_date(df, date_column, start_date, end_date):
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    filtered_df = df[(df[date_column] >= start_date) & (df[date_column] <= end_date)]
    return filtered_df

# Function to handle sorting of the results
def sort_results(df, sort_columns):
    sort_ascending = [True if col[0] != '-' else False for col in sort_columns]
    sort_columns = [col.lstrip('-') for col in sort_columns]
    sorted_df = df.sort_values(by=sort_columns, ascending=sort_ascending)
    return sorted_df

# API endpoint to handle search queries
@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query', '')
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 10))
    sort = request.args.get('sort', None)
    
    # Perform the search
    results = search_data(query, df_combined)
    
    # Apply sorting if provided
    if sort:
        sort_columns = [col.strip() for col in sort.split(',')]
        df_sorted = sort_results(pd.DataFrame(results), sort_columns)
        results = df_sorted.to_dict(orient='records')

    # Handle pagination
    total_results = len(results)
    total_pages = (total_results + per_page - 1) // per_page
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page

    paginated_results = results[start_idx:end_idx]

    return jsonify({
        "total_results": total_results,
        "page": page,
        "per_page": per_page,
        "results": paginated_results
    })

# API endpoint to retrieve saved queries
@app.route('/saved_queries', methods=['GET'])
def get_saved_queries():
    saved_queries = load_query_filters()
    return jsonify({"saved_queries": saved_queries})

# Utility function to load saved queries
def load_query_filters():
    try:
        with open('saved_queries.json', 'r') as f:
            saved_queries = [json.loads(line) for line in f]
        return saved_queries
    except FileNotFoundError:
        return []

# API endpoint to save a query filter
@app.route('/save_query', methods=['POST'])
def save_query():
    data = request.json
    query_name = data.get('query_name')
    query = data.get('query')
    
    if query_name and query:
        save_query_filter(query_name, query)
        return jsonify({"message": f"Query filter '{query_name}' saved."})
    else:
        return jsonify({"error": "Missing query_name or query"}), 400

# Utility function to save query filters
def save_query_filter(query_name, query):
    with open('saved_queries.json', 'a') as f:
        json.dump({query_name: query}, f)
        f.write("\n")
    print(f"Query filter '{query_name}' saved.")

if __name__ == '__main__':
    app.run(debug=True)
