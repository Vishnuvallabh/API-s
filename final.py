# from fastapi import FastAPI, Query, HTTPException
# from pydantic import BaseModel
# from typing import List, Optional
# import sqlite3
# import pandas as pd
# import re
# from datetime import datetime
# import json

# app = FastAPI()

# # Initialize SQLite connection
# db_path = 'advanced_denormalized_data.sqlite'
# conn = sqlite3.connect(db_path)

# # Load all tables into a single DataFrame
# def load_combined_data():
#     tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
#     table_names = tables['name'].tolist()
    
#     df_combined = pd.DataFrame()
#     for table in table_names:
#         try:
#             df = pd.read_sql_query(f"SELECT * FROM {table};", conn)
#             df['Source Table'] = table
#             if not df.empty:
#                 df_combined = pd.concat([df_combined, df], ignore_index=True)
#         except Exception as e:
#             print(f"Error loading table {table}: {e}")
    
#     return df_combined.astype(str)

# df_combined = load_combined_data()

# # Models for API responses
# class SearchResult(BaseModel):
#     total_results: int
#     page: int
#     per_page: int
#     results: List[dict]

# # Helper function to highlight text
# def highlight_text(text, terms):
#     for term in terms:
#         term_escaped = re.escape(term)
#         text = re.sub(f"(?i)(\\b{term_escaped}\\b)", lambda m: f"<highlight>{m.group(0)}</highlight>", text)
#     return text

# # Function to parse the search query into terms and operators
# def parse_query(query):
#     terms = []
#     operators = []

#     tokens = re.split(r'(\band\b|\bor\b|\bnot\b|\(|\)|\+)', query, flags=re.IGNORECASE)
#     buffer = []
#     in_parentheses = False

#     for token in tokens:
#         token = token.strip()
#         if not token:
#             continue
#         if token == "(":
#             in_parentheses = True
#             buffer.append(token)
#         elif token == ")":
#             in_parentheses = False
#             buffer.append(token)
#         elif token.upper() in ["AND", "OR", "NOT"]:
#             if in_parentheses:
#                 buffer.append(token)
#             else:
#                 if buffer:
#                     terms.append(" ".join(buffer))
#                     buffer = []
#                 operators.append(token.upper())
#         elif token == "+":
#             operators.append("OR")
#         else:
#             buffer.append(token)

#     if buffer:
#         terms.append(" ".join(buffer))

#     return terms, operators

# # Function to search the data based on the parsed query
# def search_data(query, df):
#     exact_matches = []
#     terms, operators = parse_query(query)

#     for idx, row in df.iterrows():
#         match = None

#         for i, term in enumerate(terms):
#             if ':' in term:
#                 column_name, search_term = term.split(':', 1)
#                 column_name = column_name.strip()
#                 search_term = search_term.strip()
#                 if column_name in row.index:
#                     term_found = search_term.lower() in row[column_name].lower()
#             else:
#                 row_text = ' '.join(row.values).strip()
#                 term_pattern = re.compile(f"(?i)\\b{re.escape(term)}\\b")
#                 term_found = bool(term_pattern.search(row_text))
            
#             if i == 0:
#                 match = term_found
#             else:
#                 if operators[i - 1] == "AND":
#                     match = match and term_found
#                 elif operators[i - 1] == "OR":
#                     match = match or term_found
#                 elif operators[i - 1] == "NOT":
#                     match = match and not term_found

#         if match:
#             highlighted_row = {k: highlight_text(v, terms) for k, v in row.items()}
#             exact_matches.append(highlighted_row)

#     return exact_matches

# # Function to filter data by date range
# def filter_by_date(df, date_column, start_date, end_date):
#     df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
#     filtered_df = df[(df[date_column] >= start_date) & (df[date_column] <= end_date)]
#     return filtered_df

# # Function to handle sorting of the results
# def sort_results(df, sort_columns):
#     sort_ascending = [True if col[0] != '-' else False for col in sort_columns]
#     sort_columns = [col.lstrip('-') for col in sort_columns]
#     sorted_df = df.sort_values(by=sort_columns, ascending=sort_ascending)
#     return sorted_df

# # API endpoint to handle search queries
# @app.get("/search", response_model=SearchResult)
# def search(
#     query: str,
#     page: int = 1,
#     per_page: int = 10,
#     sort: Optional[str] = None,
#     start_date: Optional[str] = None,
#     end_date: Optional[str] = None
# ):
#     df_filtered = df_combined

#     if start_date and end_date:
#         df_filtered = filter_by_date(df_filtered, 'DateOfBirth', pd.to_datetime(start_date), pd.to_datetime(end_date))

#     results = search_data(query, df_filtered)
    
#     if sort:
#         sort_columns = [col.strip() for col in sort.split(',')]
#         df_sorted = sort_results(pd.DataFrame(results), sort_columns)
#         results = df_sorted.to_dict(orient='records')

#     total_results = len(results)
#     total_pages = (total_results + per_page - 1) // per_page
#     start_idx = (page - 1) * per_page
#     end_idx = start_idx + per_page

#     paginated_results = results[start_idx:end_idx]

#     return {
#         "total_results": total_results,
#         "page": page,
#         "per_page": per_page,
#         "results": paginated_results
#     }

# # API endpoint to retrieve saved queries
# @app.get("/saved_queries")
# def get_saved_queries():
#     saved_queries = load_query_filters()
#     return {"saved_queries": saved_queries}

# # Utility function to load saved queries
# def load_query_filters():
#     try:
#         with open('saved_queries.json', 'r') as f:
#             saved_queries = [json.loads(line) for line in f]
#         return saved_queries
#     except FileNotFoundError:
#         return []

# # API endpoint to save a query filter
# @app.post("/save_query")
# def save_query(query_name: str, query: str):
#     save_query_filter(query_name, query)
#     return {"message": f"Query filter '{query_name}' saved."}

# # Utility function to save query filters
# def save_query_filter(query_name, query):
#     with open('saved_queries.json', 'a') as f:
#         json.dump({query_name: query}, f)
#         f.write("\n")
#     print(f"Query filter '{query_name}' saved.")






# from fastapi import FastAPI, Query, HTTPException
# from pydantic import BaseModel
# from typing import List, Optional
# import sqlite3
# import pandas as pd
# import re
# from datetime import datetime
# import json

# app = FastAPI()

# # Initialize SQLite connection
# db_path = 'advanced_denormalized_data.sqlite'
# conn = sqlite3.connect(db_path)

# # Load all tables into a single DataFrame
# def load_combined_data():
#     tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
#     table_names = tables['name'].tolist()
    
#     df_combined = pd.DataFrame()
#     for table in table_names:
#         try:
#             df = pd.read_sql_query(f"SELECT * FROM {table};", conn)
#             df['Source Table'] = table
#             if not df.empty:
#                 df_combined = pd.concat([df_combined, df], ignore_index=True)
#         except Exception as e:
#             print(f"Error loading table {table}: {e}")
    
#     return df_combined.astype(str)

# df_combined = load_combined_data()

# # Models for API responses
# class SearchResult(BaseModel):
#     total_results: int
#     page: int
#     per_page: int
#     results: List[dict]

# # Helper function to highlight text
# def highlight_text(text, terms):
#     for term in terms:
#         term_escaped = re.escape(term)
#         text = re.sub(f"(?i)(\\b{term_escaped}\\b)", lambda m: f"<highlight>{m.group(0)}</highlight>", text)
#     return text

# # Function to parse the search query into terms and operators
# def parse_query(query):
#     terms = []
#     operators = []
#     date_ranges = []

#     # Regular expression to match date ranges
#     date_range_pattern = re.compile(r"(\w+):\[(\d{4}-\d{2}-\d{2})\sTO\s(\d{4}-\d{2}-\d{2})\]")

#     # Extract date ranges
#     matches = date_range_pattern.findall(query)
#     for column_name, start_date, end_date in matches:
#         date_ranges.append((column_name, start_date, end_date))
#         query = query.replace(f"{column_name}:[{start_date} TO {end_date}]", "")
    
#     # Proceed with normal parsing
#     tokens = re.split(r'(\band\b|\bor\b|\bnot\b|\(|\)|\+)', query, flags=re.IGNORECASE)
#     buffer = []
#     in_parentheses = False

#     for token in tokens:
#         token = token.strip()
#         if not token:
#             continue
#         if token == "(":
#             in_parentheses = True
#             buffer.append(token)
#         elif token == ")":
#             in_parentheses = False
#             buffer.append(token)
#         elif token.upper() in ["AND", "OR", "NOT"]:
#             if in_parentheses:
#                 buffer.append(token)
#             else:
#                 if buffer:
#                     terms.append(" ".join(buffer))
#                     buffer = []
#                 operators.append(token.upper())
#         elif token == "+":
#             operators.append("OR")
#         else:
#             buffer.append(token)

#     if buffer:
#         terms.append(" ".join(buffer))

#     return terms, operators, date_ranges

# # Function to search the data based on the parsed query
# def search_data(query, df):
#     exact_matches = []
#     terms, operators, date_ranges = parse_query(query)

#     # Apply date range filters
#     for column_name, start_date, end_date in date_ranges:
#         df = filter_by_date(df, column_name, pd.to_datetime(start_date), pd.to_datetime(end_date))

#     for idx, row in df.iterrows():
#         match = None

#         for i, term in enumerate(terms):
#             if ':' in term:
#                 column_name, search_term = term.split(':', 1)
#                 column_name = column_name.strip()
#                 search_term = search_term.strip()
#                 if column_name in row.index:
#                     term_found = search_term.lower() in row[column_name].lower()
#             else:
#                 row_text = ' '.join(row.values).strip()
#                 term_pattern = re.compile(f"(?i)\\b{re.escape(term)}\\b")
#                 term_found = bool(term_pattern.search(row_text))
            
#             if i == 0:
#                 match = term_found
#             else:
#                 if operators[i - 1] == "AND":
#                     match = match and term_found
#                 elif operators[i - 1] == "OR":
#                     match = match or term_found
#                 elif operators[i - 1] == "NOT":
#                     match = match and not term_found

#         if match:
#             highlighted_row = {k: highlight_text(v, terms) for k, v in row.items()}
#             exact_matches.append(highlighted_row)

#     return exact_matches

# # Function to filter data by date range
# def filter_by_date(df, date_column, start_date, end_date):
#     df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
#     filtered_df = df[(df[date_column] >= start_date) & (df[date_column] <= end_date)]
#     return filtered_df

# # Function to handle sorting of the results
# def sort_results(df, sort_columns):
#     sort_ascending = [True if col[0] != '-' else False for col in sort_columns]
#     sort_columns = [col.lstrip('-') for col in sort_columns]
#     sorted_df = df.sort_values(by=sort_columns, ascending=sort_ascending)
#     return sorted_df

# # API endpoint to handle search queries
# @app.get("/search", response_model=SearchResult)
# def search(
#     query: str,
#     page: int = 1,
#     per_page: int = 10,
#     sort: Optional[str] = None,
#     start_date: Optional[str] = None,
#     end_date: Optional[str] = None
# ):
#     df_filtered = df_combined

#     if start_date and end_date:
#         df_filtered = filter_by_date(df_filtered, 'DateOfBirth', pd.to_datetime(start_date), pd.to_datetime(end_date))

#     results = search_data(query, df_filtered)
    
#     if sort:
#         sort_columns = [col.strip() for col in sort.split(',')]
#         df_sorted = sort_results(pd.DataFrame(results), sort_columns)
#         results = df_sorted.to_dict(orient='records')

#     total_results = len(results)
#     total_pages = (total_results + per_page - 1) // per_page
#     start_idx = (page - 1) * per_page
#     end_idx = start_idx + per_page

#     paginated_results = results[start_idx:end_idx]

#     return {
#         "total_results": total_results,
#         "page": page,
#         "per_page": per_page,
#         "results": paginated_results
#     }

# # API endpoint to retrieve saved queries
# @app.get("/saved_queries")
# def get_saved_queries():
#     saved_queries = load_query_filters()
#     return {"saved_queries": saved_queries}

# # Utility function to load saved queries
# def load_query_filters():
#     try:
#         with open('saved_queries.json', 'r') as f:
#             saved_queries = [json.loads(line) for line in f]
#         return saved_queries
#     except FileNotFoundError:
#         return []

# # API endpoint to save a query filter
# @app.post("/save_query")
# def save_query(query_name: str, query: str):
#     save_query_filter(query_name, query)
#     return {"message": f"Query filter '{query_name}' saved."}

# # Utility function to save query filters
# def save_query_filter(query_name, query):
#     with open('saved_queries.json', 'a') as f:
#         json.dump({query_name: query}, f)
#         f.write("\n")
#     print(f"Query filter '{query_name}' saved.")





# main.py
# from fastapi import FastAPI, HTTPException, Query
# from fastapi.responses import FileResponse, JSONResponse
# from pydantic import BaseModel
# import sqlite3
# import pandas as pd
# import re
# from datetime import datetime
# import json
# import os
# from typing import List, Optional, Dict

# app = FastAPI()

# # Global configuration
# DB_PATH = '/content/advanced_denormalized_data.sqlite'  # Replace with your SQLite database path
# EXPORT_FOLDER = '/content/exports'  # Replace with your export folder path

# if not os.path.exists(EXPORT_FOLDER):
#     os.makedirs(EXPORT_FOLDER)

# def get_connection():
#     return sqlite3.connect(DB_PATH)

# def load_data():
#     conn = get_connection()
#     tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
#     table_names = tables['name'].tolist()

#     df_combined = pd.DataFrame()
#     for table in table_names:
#         try:
#             df = pd.read_sql_query(f"SELECT * FROM {table};", conn)
#             df['Source Table'] = table
#             if not df.empty:
#                 df_combined = pd.concat([df_combined, df], ignore_index=True)
#         except Exception as e:
#             print(f"Error loading table {table}: {e}")
   
#     df_combined = df_combined.astype(str)
#     conn.close()
#     return df_combined

# def highlight_text(text, terms):
#     for term in terms:
#         term_escaped = re.escape(term)
#         text = re.sub(f"(?i)(\\b{term_escaped}\\b)", lambda m: f"**{m.group(0)}**", text)
#     return text

# def parse_query(query):
#     terms, operators = [], []
#     tokens = re.split(r'(\band\b|\bor\b|\bnot\b|\(|\)|\+)', query, flags=re.IGNORECASE)
#     buffer, in_parentheses = [], False

#     for token in tokens:
#         token = token.strip()
#         if not token:
#             continue
#         if token == "(":
#             in_parentheses = True
#             buffer.append(token)
#         elif token == ")":
#             in_parentheses = False
#             buffer.append(token)
#         elif token.upper() in ["AND", "OR", "NOT"]:
#             if in_parentheses:
#                 buffer.append(token)
#             else:
#                 if buffer:
#                     terms.append(" ".join(buffer))
#                     buffer = []
#                 operators.append(token.upper())
#         elif token == "+":
#             operators.append("OR")
#         else:
#             buffer.append(token)

#     if buffer:
#         terms.append(" ".join(buffer))
#     return terms, operators

# def search_data(query, df):
#     exact_matches, terms, operators = [], *parse_query(query)
#     total_term_counts, total_count = {term: 0 for term in terms}, 0

#     for idx, row in df.iterrows():
#         match, term_counts = None, {term: 0 for term in terms}
#         for i, term in enumerate(terms):
#             if ':' in term:
#                 column_name, search_term = term.split(':', 1)
#                 if column_name.strip() in row.index:
#                     term_found = search_term.strip().lower() in row[column_name.strip()].lower()
#                     if term_found:
#                         term_counts[term] += 1
#                         total_term_counts[term] += 1
#                         total_count += 1
#             else:
#                 row_text = ' '.join(row.values).strip()
#                 term_pattern = re.compile(f"(?i){re.escape(term)}")
#                 term_found = bool(term_pattern.search(row_text))
#                 if term_found:
#                     term_counts[term] += len(term_pattern.findall(row_text))
#                     total_term_counts[term] += len(term_pattern.findall(row_text))
#                     total_count += len(term_pattern.findall(row_text))

#             match = match and term_found if i > 0 else term_found
#             if i > 0:
#                 if operators[i - 1] == "AND":
#                     match = match and term_found
#                 elif operators[i - 1] == "OR":
#                     match = match or term_found
#                 elif operators[i - 1] == "NOT":
#                     match = match and not term_found

#         if match:
#             highlighted_row = {k: (highlight_text(v, [term.split(':')[1].strip() for term in terms if ':' in term]) if any(re.search(f"(?i)\\b{re.escape(search_term)}\\b", v) for term in terms if ':' in term and term.split(':')[0].strip().upper() == k.upper()) else highlight_text(v, terms)) for k, v in row.items()}
#             highlighted_row['Term Counts'] = term_counts
#             exact_matches.append(highlighted_row)

#     return exact_matches

# def filter_by_date(df, date_column, start_date, end_date):
#     if date_column not in df.columns:
#         return pd.DataFrame()
#     df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
#     return df[(df[date_column] >= start_date) & (df[date_column] <= end_date)]

# def export_to_excel(results, file_path):
#     pd.DataFrame(results).to_excel(file_path, index=False, engine='xlsxwriter')

# # def export_to_pdf(results, file_path):
# #     pdf = FPDF()
# #     pdf.add_page()
# #     pdf.set_font("Arial", size=12)
# #     for result in results:
# #         for key, value in result.items():
# #             pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)
# #         pdf.cell(200, 10, txt="=" * 50, ln=True)
# #     pdf.output(file_path)

# def save_query_filter(query_name, query):
#     with open('saved_queries.json', 'a') as f:
#         json.dump({query_name: query}, f)
#         f.write("\n")

# def load_query_filters():
#     try:
#         with open('saved_queries.json', 'r') as f:
#             return [json.loads(line) for line in f]
#     except FileNotFoundError:
#         return []

# class SearchQuery(BaseModel):
#     query: str

# class DateFilterQuery(BaseModel):
#     date_column: str
#     start_date: str
#     end_date: str
#     query: Optional[str] = ""

# class ExportQuery(BaseModel):
#     query: str
#     file_type: str

# class SaveQueryFilter(BaseModel):
#     query_name: str
#     query: str

# @app.get('/tables')
# def get_tables():
#     conn = get_connection()
#     tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
#     conn.close()
#     return tables['name'].tolist()

# @app.post('/search')
# def search(query: SearchQuery):
#     df_combined = load_data()
#     exact_matches = search_data(query.query, df_combined)
#     return exact_matches

# @app.post('/filter-by-date')
# def filter_date(query: DateFilterQuery):
#     start_date = pd.to_datetime(query.start_date, errors='coerce')
#     end_date = pd.to_datetime(query.end_date, errors='coerce')
   
#     df_combined = load_data()
#     exact_matches = search_data(query.query, df_combined)
#     filtered_matches = filter_by_date(pd.DataFrame(exact_matches), query.date_column, start_date, end_date).to_dict(orient='records')
   
#     return filtered_matches

# @app.post('/export')
# def export(query: ExportQuery):
#     file_name = f"{datetime.now().strftime('%Y%m%d%H%M%S')}.{query.file_type}"
#     file_path = os.path.join(EXPORT_FOLDER, file_name)
   
#     df_combined = load_data()
#     exact_matches = search_data(query.query, df_combined)

#     if query.file_type == 'excel':
#         export_to_excel(exact_matches, file_path)
#     # elif query.file_type == 'pdf':
#     #     export_to_pdf(exact_matches, file_path)
#     else:
#         raise HTTPException(status_code=400, detail="Invalid file type")

#     return FileResponse(file_path, filename=file_name)

# @app.post('/save-query')
# def save_query(query: SaveQueryFilter):
#     save_query_filter(query.query_name, query.query)
#     return {"message": "Query saved successfully"}

# @app.get('/load-queries')
# def load_queries():
#     queries = load_query_filters()
#     return JSONResponse(content=queries)

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)




# from fastapi import FastAPI, Query, HTTPException
# from pydantic import BaseModel
# from typing import List, Optional
# import sqlite3
# import pandas as pd
# import re
# from datetime import datetime
# import json

# app = FastAPI()

# # Initialize SQLite connection
# db_path = 'advanced_denormalized_data.sqlite'
# conn = sqlite3.connect(db_path)

# # Load all tables into a single DataFrame
# def load_combined_data():
#     tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
#     table_names = tables['name'].tolist()
    
#     df_combined = pd.DataFrame()
#     for table in table_names:
#         try:
#             df = pd.read_sql_query(f"SELECT * FROM {table};", conn)
#             df['Source Table'] = table
#             if not df.empty:
#                 df_combined = pd.concat([df_combined, df], ignore_index=True)
#         except Exception as e:
#             print(f"Error loading table {table}: {e}")
    
#     return df_combined.astype(str)

# df_combined = load_combined_data()

# # Models for API responses
# class SearchResult(BaseModel):
#     total_results: int
#     page: int
#     per_page: int
#     results: List[dict]

# # Helper function to highlight text
# def highlight_text(text, terms):
#     for term in terms:
#         term_escaped = re.escape(term)
#         text = re.sub(f"(?i)(\\b{term_escaped}\\b)", lambda m: f"<highlight>{m.group(0)}</highlight>", text)
#     return text

# # Function to parse the search query into terms and operators
# def parse_query(query):
#     terms = []
#     operators = []

#     tokens = re.split(r'(\band\b|\bor\b|\bnot\b|\(|\)|\+)', query, flags=re.IGNORECASE)
#     buffer = []
#     in_parentheses = False

#     for token in tokens:
#         token = token.strip()
#         if not token:
#             continue
#         if token == "(":
#             in_parentheses = True
#             buffer.append(token)
#         elif token == ")":
#             in_parentheses = False
#             buffer.append(token)
#         elif token.upper() in ["AND", "OR", "NOT"]:
#             if in_parentheses:
#                 buffer.append(token)
#             else:
#                 if buffer:
#                     terms.append(" ".join(buffer))
#                     buffer = []
#                 operators.append(token.upper())
#         elif token == "+":
#             operators.append("OR")
#         else:
#             buffer.append(token)

#     if buffer:
#         terms.append(" ".join(buffer))

#     return terms, operators

# # Function to search the data based on the parsed query
# def search_data(query, df):
#     exact_matches = []
#     terms, operators = parse_query(query)

#     for idx, row in df.iterrows():
#         match = None

#         for i, term in enumerate(terms):
#             if ':' in term:
#                 column_name, search_term = term.split(':', 1)
#                 column_name = column_name.strip()
#                 search_term = search_term.strip()
#                 if column_name in row.index:
#                     term_found = search_term.lower() in row[column_name].lower()
#             else:
#                 row_text = ' '.join(row.values).strip()
#                 term_pattern = re.compile(f"(?i)\\b{re.escape(term)}\\b")
#                 term_found = bool(term_pattern.search(row_text))
            
#             if i == 0:
#                 match = term_found
#             else:
#                 if operators[i - 1] == "AND":
#                     match = match and term_found
#                 elif operators[i - 1] == "OR":
#                     match = match or term_found
#                 elif operators[i - 1] == "NOT":
#                     match = match and not term_found

#         if match:
#             highlighted_row = {k: highlight_text(v, terms) for k, v in row.items()}
#             exact_matches.append(highlighted_row)

#     return exact_matches

# # Function to filter data by date range
# def filter_by_date(df, date_column, start_date, end_date):
#     df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
#     filtered_df = df[(df[date_column] >= start_date) & (df[date_column] <= end_date)]
#     return filtered_df

# # Function to handle sorting of the results
# def sort_results(df, sort_columns):
#     sort_ascending = [True if col[0] != '-' else False for col in sort_columns]
#     sort_columns = [col.lstrip('-') for col in sort_columns]
#     sorted_df = df.sort_values(by=sort_columns, ascending=sort_ascending)
#     return sorted_df

# # API endpoint to handle search queries
# @app.get("/search", response_model=SearchResult)
# def search(
#     query: str,
#     page: int = 1,
#     per_page: int = 10,
#     sort: Optional[str] = None,
#     start_date: Optional[str] = None,
#     end_date: Optional[str] = None
# ):
#     df_filtered = df_combined

#     # Apply date range filter if provided
#     if start_date and end_date:
#         df_filtered = filter_by_date(df_filtered, 'DateOfBirth', pd.to_datetime(start_date), pd.to_datetime(end_date))

#     # Perform the search
#     results = search_data(query, df_filtered)
    
#     # Apply sorting if provided
#     if sort:
#         sort_columns = [col.strip() for col in sort.split(',')]
#         df_sorted = sort_results(pd.DataFrame(results), sort_columns)
#         results = df_sorted.to_dict(orient='records')

#     # Handle pagination
#     total_results = len(results)
#     total_pages = (total_results + per_page - 1) // per_page
#     start_idx = (page - 1) * per_page
#     end_idx = start_idx + per_page

#     paginated_results = results[start_idx:end_idx]

#     return {
#         "total_results": total_results,
#         "page": page,
#         "per_page": per_page,
#         "results": paginated_results
#     }

# # API endpoint to retrieve saved queries
# @app.get("/saved_queries")
# def get_saved_queries():
#     saved_queries = load_query_filters()
#     return {"saved_queries": saved_queries}

# # Utility function to load saved queries
# def load_query_filters():
#     try:
#         with open('saved_queries.json', 'r') as f:
#             saved_queries = [json.loads(line) for line in f]
#         return saved_queries
#     except FileNotFoundError:
#         return []

# # API endpoint to save a query filter
# @app.post("/save_query")
# def save_query(query_name: str, query: str):
#     save_query_filter(query_name, query)
#     return {"message": f"Query filter '{query_name}' saved."}

# # Utility function to save query filters
# def save_query_filter(query_name, query):
#     with open('saved_queries.json', 'a') as f:
#         json.dump({query_name: query}, f)
#         f.write("\n")
#     print(f"Query filter '{query_name}' saved.") 

#working but slow
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import sqlite3
import pandas as pd
import re
from datetime import datetime
import json

app = FastAPI()

# Initialize SQLite connection
db_path = 'advanced_denormalized_data.sqlite'
conn = sqlite3.connect(db_path)

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

# Models for API responses
class SearchResult(BaseModel):
    total_results: int
    page: int
    per_page: int
    results: List[dict]

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
@app.get("/search", response_model=SearchResult)
def search(
    query: str,
    page: int = 1,
    per_page: int = 10,
    sort: Optional[str] = None
):
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

    return {
        "total_results": total_results,
        "page": page,
        "per_page": per_page,
        "results": paginated_results
    }

# API endpoint to retrieve saved queries
@app.get("/saved_queries")
def get_saved_queries():
    saved_queries = load_query_filters()
    return {"saved_queries": saved_queries}

# Utility function to load saved queries
def load_query_filters():
    try:
        with open('saved_queries.json', 'r') as f:
            saved_queries = [json.loads(line) for line in f]
        return saved_queries
    except FileNotFoundError:
        return []

# API endpoint to save a query filter
@app.post("/save_query")
def save_query(query_name: str, query: str):
    save_query_filter(query_name, query)
    return {"message": f"Query filter '{query_name}' saved."}

# Utility function to save query filters
def save_query_filter(query_name, query):
    with open('saved_queries.json', 'a') as f:
        json.dump({query_name: query}, f)
        f.write("\n")
    print(f"Query filter '{query_name}' saved.") 




# from fastapi import FastAPI, Query, HTTPException
# from pydantic import BaseModel
# from typing import List, Optional
# import sqlite3
# import pandas as pd
# import re
# from datetime import datetime
# import json

# app = FastAPI()

# # Initialize SQLite connection
# db_path = 'advanced_denormalized_data.sqlite'
# conn = sqlite3.connect(db_path)

# # Load all tables into a single DataFrame
# def load_combined_data():
#     tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
#     table_names = tables['name'].tolist()
    
#     df_combined = pd.DataFrame()
#     for table in table_names:
#         try:
#             df = pd.read_sql_query(f"SELECT * FROM {table};", conn)
#             df['Source Table'] = table
#             if not df.empty:
#                 df_combined = pd.concat([df_combined, df], ignore_index=True)
#         except Exception as e:
#             print(f"Error loading table {table}: {e}")
    
#     return df_combined.astype(str).set_index('id', drop=False)

# df_combined = load_combined_data()

# # Models for API responses
# class SearchResult(BaseModel):
#     total_results: int
#     page: int
#     per_page: int
#     results: List[dict]

# # Helper function to highlight text
# def highlight_text(df, terms):
#     terms_escaped = [re.escape(term) for term in terms]
#     pattern = "|".join([f"(?i)\\b{term}\\b" for term in terms_escaped])
#     df = df.apply(lambda x: x.str.replace(pattern, lambda m: f"<highlight>{m.group(0)}</highlight>", regex=True))
#     return df

# # Function to parse the search query into terms and operators
# def parse_query(query):
#     terms = []
#     operators = []

#     # Detect date range pattern
#     date_pattern = r'(\w+):\[(.*?)\s*TO\s*(.*?)\]'
    
#     tokens = re.split(r'(\band\b|\bor\b|\bnot\b|\(|\)|\+)', query, flags=re.IGNORECASE)
#     buffer = []
#     in_parentheses = False

#     for token in tokens:
#         token = token.strip()
#         if not token:
#             continue
#         # Handle date range pattern
#         date_match = re.match(date_pattern, token)
#         if date_match:
#             column_name = date_match.group(1)
#             start_date = date_match.group(2).strip()
#             end_date = date_match.group(3).strip()
#             terms.append(f"{column_name}:[{start_date} TO {end_date}]")
#         elif token == "(":
#             in_parentheses = True
#             buffer.append(token)
#         elif token == ")":
#             in_parentheses = False
#             buffer.append(token)
#         elif token.upper() in ["AND", "OR", "NOT"]:
#             if in_parentheses:
#                 buffer.append(token)
#             else:
#                 if buffer:
#                     terms.append(" ".join(buffer))
#                     buffer = []
#                 operators.append(token.upper())
#         elif token == "+":
#             operators.append("OR")
#         else:
#             buffer.append(token)

#     if buffer:
#         terms.append(" ".join(buffer))

#     return terms, operators

# # Function to search the data based on the parsed query
# def search_data(query, df):
#     terms, operators = parse_query(query)

#     # Initial filter based on terms
#     for i, term in enumerate(terms):
#         if ':' in term:
#             column_name, search_term = term.split(':', 1)
#             column_name = column_name.strip()
#             search_term = search_term.strip()
            
#             # Date range filtering
#             date_pattern = r'(\w+):\[(.*?)\s*TO\s*(.*?)\]'
#             date_match = re.match(date_pattern, term)
#             if date_match:
#                 start_date = pd.to_datetime(date_match.group(2).strip(), errors='coerce')
#                 end_date = pd.to_datetime(date_match.group(3).strip(), errors='coerce')
#                 df = df[(df[column_name] >= start_date) & (df[column_name] <= end_date)]
#             else:
#                 df = df[df[column_name].str.contains(search_term, case=False, na=False)]
#         else:
#             # General term search across all columns
#             pattern = re.compile(f"(?i)\\b{re.escape(term)}\\b")
#             mask = df.apply(lambda row: bool(
# pattern.search
# (' '.join(row.values))), axis=1)
#             df = df[mask]
    
#     # Highlighting matches
#     df = highlight_text(df, terms)

#     return df

# # Function to handle sorting of the results
# def sort_results(df, sort_columns):
#     sort_ascending = [True if col[0] != '-' else False for col in sort_columns]
#     sort_columns = [col.lstrip('-') for col in sort_columns]
#     sorted_df = df.sort_values(by=sort_columns, ascending=sort_ascending)
#     return sorted_df

# # API endpoint to handle search queries
# @app.get("/search", response_model=SearchResult)
# def search(
#     query: str,
#     page: int = 1,
#     per_page: int = 10,
#     sort: Optional[str] = None
# ):
#     # Perform the search
#     results = search_data(query, df_combined)
    
#     # Apply sorting if provided
#     if sort:
#         sort_columns = [col.strip() for col in sort.split(',')]
#         results = sort_results(results, sort_columns)

#     # Handle pagination
#     total_results = results.shape[0]
#     start_idx = (page - 1) * per_page
#     end_idx = start_idx + per_page

#     paginated_results = results.iloc[start_idx:end_idx].to_dict(orient='records')

#     return {
#         "total_results": total_results,
#         "page": page,
#         "per_page": per_page,
#         "results": paginated_results
#     }

# # API endpoint to retrieve saved queries
# @app.get("/saved_queries")
# def get_saved_queries():
#     saved_queries = load_query_filters()
#     return {"saved_queries": saved_queries}

# # Utility function to load saved queries
# def load_query_filters():
#     try:
#         with open('saved_queries.json', 'r') as f:
#             saved_queries = [json.loads(line) for line in f]
#         return saved_queries
#     except FileNotFoundError:
#         return []

# # API endpoint to save a query filter
# @app.post("/save_query")
# def save_query(query_name: str, query: str):
#     save_query_filter(query_name, query)
#     return {"message": f"Query filter '{query_name}' saved."}

# # Utility function to save query filters
# def save_query_filter(query_name, query):
#     with open('saved_queries.json', 'a') as f:
#         json.dump({query_name: query}, f)
#         f.write("\n")
#     print(f"Query filter '{query_name}' saved.") 



# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import sqlite3
# import pandas as pd
# import re
# from typing import Optional

# app = FastAPI()

# # Initialize SQLite connection
# db_path = 'advanced_denormalized_data.sqlite'  # Replace with your SQLite database path
# conn = sqlite3.connect(db_path, check_same_thread=False)


# class SearchRequest(BaseModel):
#     query: str
#     sort_column: Optional[str] = None
#     start_date: Optional[str] = None
#     end_date: Optional[str] = None
#     page: Optional[int] = 1
#     per_page: Optional[int] = 10


# # Load and prepare data from all tables
# def load_data():
#     tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
#     table_names = tables['name'].tolist()
#     df_combined = pd.DataFrame()

#     for table in table_names:
#         try:
#             df = pd.read_sql_query(f"SELECT * FROM {table};", conn)
#             df['Source Table'] = table
#             if not df.empty:
#                 df_combined = pd.concat([df_combined, df], ignore_index=True)
#         except Exception as e:
#             print(f"Error loading table {table}: {e}")
    
#     # Convert to strings for consistent searching and indexing
#     df_combined = df_combined.astype(str)
#     return df_combined


# df_combined = load_data()


# # Function to highlight exact phrase matches in text
# def highlight_text(text, terms):
#     for term in terms:
#         term_escaped = re.escape(term)
#         text = re.sub(f"(?i)(\\b{term_escaped}\\b)", lambda m: f"<highlight>{m.group(0)}</highlight>", text)
#     return text


# # Function to parse the search query into terms and operators
# def parse_query(query):
#     tokens = re.split(r'(\band\b|\bor\b|\bnot\b)', query, flags=re.IGNORECASE)
#     terms = []
#     operators = []

#     for token in tokens:
#         token = token.strip()
#         if token.upper() in ["AND", "OR", "NOT"]:
#             operators.append(token.upper())
#         elif token:
#             terms.append(token.strip('"'))

#     return terms, operators


# # Function to perform the search
# def search_data(query, df):
#     exact_matches = []
#     terms, operators = parse_query(query)

#     for idx, row in df.iterrows():
#         row_text = ' '.join(row.values).strip()
#         match = None

#         for i, term in enumerate(terms):
#             term_pattern = re.compile(f"(?i)\\b{re.escape(term)}\\b")
#             term_found = bool(term_pattern.search(row_text))

#             if i == 0:
#                 match = term_found
#             else:
#                 if operators[i - 1] == "AND":
#                     match = match and term_found
#                 elif operators[i - 1] == "OR":
#                     match = match or term_found
#                 elif operators[i - 1] == "NOT":
#                     match = match and not term_found

#         if match:
#             highlighted_row = {k: highlight_text(v, terms) if any(
                
# re.search
# (f"(?i)\\b{re.escape(term)}\\b", v) for term in terms) else v for k, v in row.items()}
#             exact_matches.append(highlighted_row)

#     return exact_matches


# # Function to sort results based on a specific column within a given range
# def sort_results(df, column_name, start_date=None, end_date=None):
#     if start_date and end_date:
#         df[column_name] = pd.to_datetime(df[column_name], errors='coerce')
#         sorted_df = df[(df[column_name] >= start_date) & (df[column_name] <= end_date)]
#     else:
#         sorted_df = df.sort_values(by=column_name)

#     return sorted_df


# @app.get("/search")
# def search(
#     query: str,
#     sort_column: Optional[str] = None,
#     start_date: Optional[str] = None,
#     end_date: Optional[str] = None,
#     page: Optional[int] = 1,
#     per_page: Optional[int] = 10
# ):
#     exact_matches = search_data(query, df_combined)

#     if not exact_matches:
#         raise HTTPException(status_code=404, detail="No relevant results found for the query.")

#     if sort_column:
#         exact_matches = sort_results(pd.DataFrame(exact_matches), sort_column, start_date, end_date).to_dict(orient='records')

#     start_idx = (page - 1) * per_page
#     end_idx = start_idx + per_page
#     paginated_results = exact_matches[start_idx:end_idx]

#     return {
#         "total_results": len(exact_matches),
#         "page": page,
#         "per_page": per_page,
#         "results": paginated_results
#     } 



# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from typing import List, Optional
# import sqlite3
# import pandas as pd
# import re
# from datetime import datetime
# import json

# app = FastAPI()

# # Connect to the SQLite database
# db_path = 'advanced_denormalized_data.sqlite'  # Replace with your SQLite database path
# conn = sqlite3.connect(db_path, check_same_thread=False)

# # Load and prepare data from all tables
# def load_combined_data():
#     tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
#     table_names = tables['name'].tolist()

#     df_combined = pd.DataFrame()
#     for table in table_names:
#         try:
#             df = pd.read_sql_query(f"SELECT * FROM {table};", conn)
#             df['Source Table'] = table  # Add a column to track the source table
#             if not df.empty:
#                 df_combined = pd.concat([df_combined, df], ignore_index=True)
#         except Exception as e:
#             print(f"Error loading table {table}: {e}")

#     df_combined = df_combined.astype(str)  # Convert all data to strings for consistent searching
#     return df_combined

# df_combined = load_combined_data()

# # Helper function to highlight text
# def highlight_text(text, terms):
#     for term in terms:
#         term_escaped = re.escape(term)
#         text = re.sub(f"(?i)(\\b{term_escaped}\\b)", lambda m: f"<highlight>{m.group(0)}</highlight>", text)
#     return text

# # Helper function to parse the query
# def parse_query(query):
#     terms = []
#     operators = []

#     tokens = re.split(r'(\band\b|\bor\b|\bnot\b|\(|\)|\+)', query, flags=re.IGNORECASE)
#     buffer = []
#     in_parentheses = False

#     for token in tokens:
#         token = token.strip()
#         if not token:
#             continue
#         if token == "(":
#             in_parentheses = True
#             buffer.append(token)
#         elif token == ")":
#             in_parentheses = False
#             buffer.append(token)
#         elif token.upper() in ["AND", "OR", "NOT"]:
#             if in_parentheses:
#                 buffer.append(token)
#             else:
#                 if buffer:
#                     terms.append(" ".join(buffer))
#                     buffer = []
#                 operators.append(token.upper())
#         elif token == "+":
#             operators.append("OR")
#         else:
#             buffer.append(token)

#     if buffer:
#         terms.append(" ".join(buffer))

#     return terms, operators

# # Helper function to perform the search
# def search_data(query, df):
#     exact_matches = []
#     terms, operators = parse_query(query)

#     total_term_counts = {term: 0 for term in terms}
#     total_count = 0

#     for idx, row in df.iterrows():
#         match = None
#         term_counts = {term: 0 for term in terms}

#         for i, term in enumerate(terms):
#             # Handle date range
#             date_pattern = r'(\w+):\[(.*?)\s*TO\s*(.*?)\]'
#             date_match = re.match(date_pattern, term)
#             if date_match:
#                 column_name = date_match.group(1).strip()
#                 start_date = pd.to_datetime(date_match.group(2).strip(), errors='coerce')
#                 end_date = pd.to_datetime(date_match.group(3).strip(), errors='coerce')
#                 if column_name in row.index:
#                     row_date = pd.to_datetime(row[column_name], errors='coerce')
#                     term_found = start_date <= row_date <= end_date if pd.notna(row_date) else False
#             else:
#                 if ':' in term:
#                     column_name, search_term = term.split(':', 1)
#                     column_name = column_name.strip()
#                     search_term = search_term.strip()
#                     if column_name in row.index:
#                         term_found = search_term.lower() in row[column_name].lower()
#                 else:
#                     row_text = ' '.join(row.values).strip()
#                     term_pattern = re.compile(f"(?i)\\b{re.escape(term)}\\b")
#                     term_found = bool(term_pattern.search(row_text))

#             if term_found:
#                 term_counts[term] += 1
#                 total_term_counts[term] += 1
#                 total_count += 1

#             if i == 0:
#                 match = term_found
#             else:
#                 if operators[i - 1] == "AND":
#                     match = match and term_found
#                 elif operators[i - 1] == "OR":
#                     match = match or term_found
#                 elif operators[i - 1] == "NOT":
#                     match = match and not term_found

#         if match:
#             highlighted_row = {}
#             for k, v in row.items():
#                 if any(re.search(f"(?i)\\b{re.escape(search_term)}\\b", v) for term in terms if ':' in term and term.split(':')[0].strip().upper() == k.upper()):
#                     highlighted_row[k] = highlight_text(v, [term.split(':')[1].strip() for term in terms if ':' in term])
#                 else:
#                     highlighted_row[k] = highlight_text(v, terms) if any(re.search(f"(?i){re.escape(term)}", v) for term in terms) else v
#             highlighted_row['Term Counts'] = term_counts
#             exact_matches.append(highlighted_row)

#     return exact_matches, total_term_counts, total_count

# # API endpoint to handle search queries
# @app.get("/search")
# def search(
#     query: str,
#     sort_columns: Optional[str] = None,
#     start_date: Optional[str] = None,
#     end_date: Optional[str] = None,
#     page: Optional[int] = 1,
#     per_page: Optional[int] = 10
# ):
#     exact_matches, total_term_counts, total_count = search_data(query, df_combined)

#     if sort_columns:
#         sort_columns_list = [col.strip() for col in sort_columns.split(',')]
#         exact_matches = sort_results(pd.DataFrame(exact_matches), sort_columns_list, case_insensitive=True).to_dict(orient='records')

#     total_results = len(exact_matches)
#     start_idx = (page - 1) * per_page
#     end_idx = start_idx + per_page
#     paginated_results = exact_matches[start_idx:end_idx]

#     return {
#         "total_results": total_results,
#         "total_term_counts": total_term_counts,
#         "total_term_occurrences": total_count,
#         "page": page,
#         "per_page": per_page,
#         "results": paginated_results
#     }

# # Helper function to sort results
# def sort_results(df, sort_columns, custom_sort=None, case_insensitive=False):
#     def sort_key(col):
#         if custom_sort:
#             return col.apply(custom_sort)
#         elif case_insensitive:
#             return col.str.lower()
#         return col

#     sort_ascending = [True if col[0] != '-' else False for col in sort_columns]
#     sort_columns = [col.lstrip('-') for col in sort_columns]

#     df.columns = df.columns.str.lower()
#     sort_columns = [col.lower() for col in sort_columns]

#     sorted_df = df.sort_values(by=sort_columns, ascending=sort_ascending, key=sort_key)
#     return sorted_df

# # Helper function to load saved queries
# def load_query_filters():
#     try:
#         with open('saved_queries.json', 'r') as f:
#             saved_queries = [json.loads(line) for line in f]
#         return {"saved_queries": saved_queries}
#     except FileNotFoundError:
#         return {"saved_queries": []}

# # Helper function to save query filters
# def save_query_filter(query_name, query):
#     with open('saved_queries.json', 'a') as f:
#         json.dump({query_name: query}, f)
#         f.write("\n")





# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from typing import List, Optional
# import sqlite3
# import pandas as pd
# import re
# from datetime import datetime
# import json

# app = FastAPI()

# # Connect to the SQLite database
# db_path = 'advanced_denormalized_data.sqlite'  # Replace with your SQLite database path
# conn = sqlite3.connect(db_path, check_same_thread=False)

# # Load and prepare data from all tables
# def load_combined_data():
#     tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
#     table_names = tables['name'].tolist()

#     df_combined = pd.DataFrame()
#     for table in table_names:
#         try:
#             df = pd.read_sql_query(f"SELECT * FROM {table};", conn)
#             df['Source Table'] = table  # Add a column to track the source table
#             if not df.empty:
#                 df_combined = pd.concat([df_combined, df], ignore_index=True)
#         except Exception as e:
#             print(f"Error loading table {table}: {e}")

#     df_combined = df_combined.astype(str)  # Convert all data to strings for consistent searching
#     return df_combined

# df_combined = load_combined_data()

# # Function to highlight exact phrase matches in text
# def highlight_text(text, terms):
#     for term in terms:
#         term_escaped = re.escape(term)
#         # Highlight exact word, date, or paragraph matches
#         text = re.sub(f"(?i)(\\b{term_escaped}\\b)", lambda m: f"<highlight>{m.group(0)}</highlight>", text)
#     return text

# # Function to parse the search query into terms and operators, handling specific column searches
# def parse_query(query):
#     terms = []
#     operators = []

#     # Split terms by AND, OR, NOT operators while preserving the operators in the result list
#     tokens = re.split(r'(\band\b|\bor\b|\bnot\b|\(|\)|\+)', query, flags=re.IGNORECASE)
#     buffer = []
#     in_parentheses = False

#     for token in tokens:
#         token = token.strip()
#         if not token:
#             continue
#         if token == "(":
#             in_parentheses = True
#             buffer.append(token)
#         elif token == ")":
#             in_parentheses = False
#             buffer.append(token)
#         elif token.upper() in ["AND", "OR", "NOT"]:
#             if in_parentheses:
#                 buffer.append(token)
#             else:
#                 if buffer:
#                     terms.append(" ".join(buffer))
#                     buffer = []
#                 operators.append(token.upper())
#         elif token == "+":
#             operators.append("OR")
#         else:
#             buffer.append(token)

#     if buffer:
#         terms.append(" ".join(buffer))

#     return terms, operators

# # Function to perform the search with exact phrase matching and additional features
# def search_data(query, df):
#     exact_matches = []
#     terms, operators = parse_query(query)

#     total_term_counts = {term: 0 for term in terms}
#     total_count = 0

#     # Filter results based on terms and logical operations
#     for idx, row in df.iterrows():
#         match = None
#         term_counts = {term: 0 for term in terms}

#         for i, term in enumerate(terms):
#             if ':' in term:
#                 column_name, search_term = term.split(':', 1)
#                 column_name = column_name.strip()
#                 search_term = search_term.strip()
#                 if column_name in row.index:  # Ensure the column exists
#                     term_found = search_term.lower() in row[column_name].lower()
#                     if term_found:
#                         term_counts[term] += 1
#                         total_term_counts[term] += 1
#                         total_count += 1
#             else:
#                 # General term search in the whole row text
#                 row_text = ' '.join(row.values).strip()
#                 term_pattern = re.compile(f"(?i){re.escape(term)}")
#                 term_found = bool(term_pattern.search(row_text))
#                 if term_found:
#                     term_counts[term] += len(term_pattern.findall(row_text))
#                     total_term_counts[term] += len(term_pattern.findall(row_text))
#                     total_count += len(term_pattern.findall(row_text))

#             if i == 0:
#                 match = term_found
#             else:
#                 if operators[i - 1] == "AND":
#                     match = match and term_found
#                 elif operators[i - 1] == "OR":
#                     match = match or term_found
#                 elif operators[i - 1] == "NOT":
#                     match = match and not term_found

#         if match:
#             # Apply highlighting specifically for column matches as well
#             highlighted_row = {}
#             for k, v in row.items():
#                 # Highlight only the relevant columns
#                 if any(re.search(f"(?i)\\b{re.escape(search_term)}\\b", v) for term in terms if ':' in term and term.split(':')[0].strip().upper() == k.upper()):
#                     highlighted_row[k] = highlight_text(v, [term.split(':')[1].strip() for term in terms if ':' in term])
#                 else:
#                     highlighted_row[k] = highlight_text(v, terms) if any(re.search(f"(?i){re.escape(term)}", v) for term in terms) else v
#             highlighted_row['Term Counts'] = term_counts
#             exact_matches.append(highlighted_row)

#     return exact_matches, total_term_counts, total_count

# # Function to filter data by date range for a specified date column
# def filter_by_date(df, date_column, start_date, end_date):
#     if date_column not in df.columns:
#         print(f"Column '{date_column}' not found in the data.")
#         return pd.DataFrame()

#     df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
#     filtered_df = df[(df[date_column] >= start_date) & (df[date_column] <= end_date)]
#     return filtered_df

# # Function to sort results based on specific column(s) or by custom criteria
# def sort_results(df, sort_columns, custom_sort=None, case_insensitive=False):
#     def sort_key(col):
#         # Handle custom sort key
#         if custom_sort:
#             return col.apply(custom_sort)
#         # Handle case-insensitive sorting
#         elif case_insensitive:
#             return col.str.lower()
#         return col

#     # Apply sort with multiple levels and directions
#     sort_ascending = [True if col[0] != '-' else False for col in sort_columns]
#     sort_columns = [col.lstrip('-') for col in sort_columns]

#     # Convert the DataFrame columns to lower case to avoid KeyError
#     df.columns = df.columns.str.lower()
#     sort_columns = [col.lower() for col in sort_columns]

#     sorted_df = df.sort_values(by=sort_columns, ascending=sort_ascending, key=sort_key)
#     return sorted_df

# # API endpoint to handle search queries
# @app.get("/search")
# def search(
#     query: str,
#     sort_columns: Optional[str] = None,
#     start_date: Optional[str] = None,
#     end_date: Optional[str] = None,
#     page: Optional[int] = 1,
#     per_page: Optional[int] = 10
# ):
#     if start_date and end_date:
#         df_filtered = filter_by_date(df_combined, 'DateOfBirth', pd.to_datetime(start_date), pd.to_datetime(end_date))
#     else:
#         df_filtered = df_combined

#     exact_matches, total_term_counts, total_count = search_data(query, df_filtered)

#     if sort_columns:
#         sort_columns_list = [col.strip() for col in sort_columns.split(',')]
#         exact_matches = sort_results(pd.DataFrame(exact_matches), sort_columns_list, case_insensitive=True).to_dict(orient='records')

#     total_results = len(exact_matches)
#     start_idx = (page - 1) * per_page
#     end_idx = start_idx + per_page
#     paginated_results = exact_matches[start_idx:end_idx]

#     return {
#         "total_results": total_results,
#         "total_term_counts": total_term_counts,
#         "total_term_occurrences": total_count,
#         "page": page,
#         "per_page": per_page,
#         "results": paginated_results
#     }

# # Function to save query filters
# def save_query_filter(query_name, query):
#     with open('saved_queries.json', 'a') as f:
#         json.dump({query_name: query}, f)
#         f.write("\n")
#     print(f"Query filter '{query_name}' saved.")

# # Function to load query filters
# def load_query_filters():
#     try:
#         with open('saved_queries.json', 'r') as f:
#             saved_queries = [json.loads(line) for line in f]
#         return saved_queries
#     except FileNotFoundError:
#         return []

# # API endpoint to retrieve saved queries
# @app.get("/saved_queries")
# def get_saved_queries():
#     return {"saved_queries": load_query_filters()}

# # API endpoint to save a query filter
# @app.post("/save_query")
# def save_query(query_name: str, query: str):
#     save_query_filter(query_name, query)
#     return {"message": f"Query filter '{query_name}' saved."}
