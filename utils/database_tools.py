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

def query_topN_symbols(N):
    query = f"""
        SELECT distinct pk_symbols, S.symbol, cmcRank
        FROM DimSymbols S
            inner join (
                select *
                from tblCoinbaseRanking
                where cmcRank <= {N}
            ) R on S.base = R.symbol
        WHERE active = 1
            AND quote = 'USDT'
            AND type = 'spot'
        order by S.symbol
    """
    with sqlite3.connect("./data/database.db") as conn:
        df = pd.read_sql_query(query, conn)
        
    return df

def query_topN_symbols_with_last_date(N):
    query = f"""
        select fk_symbols,
            symbol,
            min([date]) as first_date,
            max([date]) as last_date
        from FactPriceData P
        where P.symbol in (        
            SELECT distinct S.symbol
                FROM DimSymbols S
                    inner join (
                        select *
                        from tblCoinbaseRanking
                        where cmcRank <= {N}
                    ) R on S.base = R.symbol
                WHERE active = 1
                    AND quote = 'USDT'
                    AND type = 'spot'
                order by S.symbol)
        group by symbol
    """
    with sqlite3.connect("./data/database.db") as conn:
        df = pd.read_sql_query(query, conn)
        
    df.first_date = pd.to_datetime(df.first_date) + pd.Timedelta(days=1)
    df.last_date = pd.to_datetime(df.last_date) + pd.Timedelta(days=1)

    return df

def query_topN_price_data(N):
    query = f"""
        select P.date,
            S.id,
            P.symbol,
            P.fk_symbols,
            S.base as symbol_short,
            P.open_price,
            P.high_price,
            P.low_price,
            P.close_price,
            P.volume
        from FactPriceData P
            left join DimSymbols S on P.fk_symbols = S.pk_symbols
        where P.symbol in (
                SELECT distinct S.symbol
                FROM DimSymbols S
                    inner join (
                        select *
                        from tblCoinbaseRanking
                        where cmcRank <= {N}
                    ) R on S.base = R.symbol
                WHERE active = 1
                    AND quote = 'USDT'
                    AND type = 'spot'
                order by S.symbol
            )
    """
    with sqlite3.connect("./data/database.db") as conn:
        df = pd.read_sql_query(query, conn)
        
    df.date = pd.to_datetime(df.date) 
    
    return df


def query_stationary_test():
    query = """
    SELECT *
    FROM tblStationaryTest
    """
    with sqlite3.connect("./data/database.db") as conn:
        df = pd.read_sql_query(query, conn)
        
    return df

