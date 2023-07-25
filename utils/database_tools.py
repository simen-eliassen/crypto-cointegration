import sqlite3
import pandas as pd

def query_active_symbols():
    query = """
    SELECT pk_symbols, symbol FROM DimSymbols 
    WHERE active = 1
    AND quote = 'USDT'
    --AND type = 'future'
    """
    with sqlite3.connect("./data/database.db") as conn:
        df = pd.read_sql_query(query, conn)
        
    return df