# from fastapi import FastAPI, HTTPException, Query, Depends
# from fastapi.responses import JSONResponse
# from typing import List, Dict, Any
# import sqlite3
# import pandas as pd
# import re

# app = FastAPI()

# # Function to highlight matches in text
# def highlight_text(text, terms):
#     for term in terms:
#         text = re.sub(f"({re.escape(term)})", r"<mark>\1</mark>", text, flags=re.IGNORECASE)
#     return text

# # Function to parse the search query
# def parse_query(query):
#     terms = []
#     operators = []

#     for token in re.findall(r'(".*?"|\S+)', query):
#         token = token.strip('"')
#         if token.upper() in ["AND", "OR", "NOT"]:
#             operators.append(token.upper())
#         else:
#             terms.append(token)

#     return terms, operators

# # Function to handle exact phrase matching only
# def exact_phrase_match(term, text):
#     pattern = r'\b' + re.escape(term) + r'\b'
#     return re.search(pattern, text, flags=re.IGNORECASE) is not None

# # Function to search data
# def search_data(query, df):
#     combined_results = {}
#     total_match_count = 0
#     terms, operators = parse_query(query)
#     search_patterns = {}

#     for term in terms:
#         if ':' in term:
#             column_name, search_term = term.split(':', 1)
#             search_term = search_term.strip()
#             if column_name not in search_patterns:
#                 search_patterns[column_name] = []
#             search_patterns[column_name].append(search_term)
#         else:
#             search_patterns['global'] = search_patterns.get('global', []) + [term.strip()]

#     for idx, row in df.iterrows():
#         match_results = []
#         full_row_dict = row.to_dict()

#         for column_name, patterns in search_patterns.items():
#             if column_name != 'global':
#                 column_matches = []
#                 for search_term in patterns:
#                     if column_name in row and exact_phrase_match(search_term, row[column_name]):
#                         highlighted_value = highlight_text(row[column_name], [search_term])
#                         full_row_dict[column_name] = highlighted_value
#                         total_match_count += 1
#                         column_matches.append(True)
#                     else:
#                         column_matches.append(False)
#                 match_results.append(all(column_matches))
#             else:
#                 global_matches = []
#                 row_text = ' '.join(row.values)
#                 for search_term in patterns:
#                     if exact_phrase_match(search_term, row_text):
#                         for column, value in row.items():
#                             if exact_phrase_match(search_term, value):
#                                 highlighted_value = highlight_text(value, [search_term])
#                                 full_row_dict[column] = highlighted_value
#                                 total_match_count += 1
#                         global_matches.append(True)
#                     else:
#                         global_matches.append(False)
#                 match_results.append(any(global_matches))

#         if all(match_results):
#             combined_results[idx] = {
#                 'Source Table': row['Source Table'],
#                 'Row Index': idx,
#                 'Full Row': full_row_dict,
#             }

#     return list(combined_results.values()), total_match_count

# # Dependency to get a new SQLite connection
# def get_db_connection():
#     conn = sqlite3.connect('advanced_denormalized_data.sqlite')
#     conn.row_factory = sqlite3.Row
#     return conn

# # Endpoint to get all table names
# @app.get("/tables", response_model=List[str])
# def get_table_names(conn: sqlite3.Connection = Depends(get_db_connection)):
#     try:
#         tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
#         table_names = tables['name'].tolist()
#         return table_names
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# # Endpoint to perform search
# @app.get("/search")
# def search(query: str, page: int = 1, per_page: int = 10, conn: sqlite3.Connection = Depends(get_db_connection)):
#     try:
#         tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
#         table_names = tables['name'].tolist()
        
#         df_combined = pd.DataFrame()
#         for table in table_names:
#             try:
#                 df = pd.read_sql_query(f"SELECT * FROM {table};", conn)
#                 df['Source Table'] = table
#                 df_combined = pd.concat([df_combined, df], ignore_index=True)
#             except Exception as e:
#                 print(f"Error loading table {table}: {e}")

#         df_combined = df_combined.astype(str)
#         results, match_count = search_data(query, df_combined)

#         total_pages = (len(results) + per_page - 1) // per_page
#         start_idx = (page - 1) * per_page
#         end_idx = start_idx + per_page
#         paginated_results = results[start_idx:end_idx]

#         return JSONResponse(content={
#             "query": query,
#             "total_matches": match_count,
#             "total_pages": total_pages,
#             "current_page": page,
#             "results": paginated_results
#         })
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# # Endpoint to save search results to a file
# @app.post("/save_results")
# def save_search_results(results: List[Dict[str, Any]], filename: str = "search_results.csv"):
#     try:
#         df_results = pd.DataFrame(results)
#         df_results.to_csv(filename, index=False)
#         return {"message": f"Search results saved to {filename}"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))




# final 

# app/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3
import pandas as pd
import re
from typing import Optional

app = FastAPI()

db_path = 'advanced_denormalized_data.sqlite'  # Replace with your SQLite database path
conn = sqlite3.connect(db_path, check_same_thread=False)


class SearchRequest(BaseModel):
    query: str
    sort_column: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    page: Optional[int] = 1
    per_page: Optional[int] = 10


# Helper function to load and prepare data from all tables
def load_data():
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
    
    df_combined = df_combined.astype(str)
    return df_combined


df_combined = load_data()


# Function to highlight exact phrase matches in text
def highlight_text(text, terms):
    for term in terms:
        term_escaped = re.escape(term)
        text = re.sub(f"(?i)(\\b{term_escaped}\\b)", lambda m: f"<highlight>{m.group(0)}</highlight>", text)
    return text


# Function to parse the search query into terms and operators
def parse_query(query):
    tokens = re.split(r'(\band\b|\bor\b|\bnot\b)', query, flags=re.IGNORECASE)
    terms = []
    operators = []

    for token in tokens:
        token = token.strip()
        if token.upper() in ["AND", "OR", "NOT"]:
            operators.append(token.upper())
        elif token:
            terms.append(token.strip('"'))

    return terms, operators


# Function to perform the search
def search_data(query, df):
    exact_matches = []
    terms, operators = parse_query(query)

    for idx, row in df.iterrows():
        row_text = ' '.join(row.values).strip()
        match = None

        for i, term in enumerate(terms):
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
            highlighted_row = {k: highlight_text(v, terms) if any(re.search(f"(?i)\\b{re.escape(term)}\\b", v) for term in terms) else v for k, v in row.items()}
            exact_matches.append(highlighted_row)

    return exact_matches


# Function to sort results based on a specific column within a given range
def sort_results(df, column_name, start_date=None, end_date=None):
    if start_date and end_date:
        df[column_name] = pd.to_datetime(df[column_name], errors='coerce')
        sorted_df = df[(df[column_name] >= start_date) & (df[column_name] <= end_date)]
    else:
        sorted_df = df.sort_values(by=column_name)

    return sorted_df


@app.get("/search")
def search(
    query: str,
    sort_column: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    page: Optional[int] = 1,
    per_page: Optional[int] = 10
):
    exact_matches = search_data(query, df_combined)

    if not exact_matches:
        raise HTTPException(status_code=404, detail="No relevant results found for the query.")

    if sort_column:
        exact_matches = sort_results(pd.DataFrame(exact_matches), sort_column, start_date, end_date).to_dict(orient='records')

    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    paginated_results = exact_matches[start_idx:end_idx]

    return {
        "total_results": len(exact_matches),
        "page": page,
        "per_page": per_page,
        "results": paginated_results
    }
