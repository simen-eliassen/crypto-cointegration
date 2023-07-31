delete from FactPriceData
WHERE rowid in (
        select rowid
        from (
                SELECT rowid,
                    ROW_NUMBER() OVER (
                        PARTITION BY symbol,
                        timeframe,
                        fk_symbols,
                        date
                        ORDER BY download_date desc
                    ) AS rn
                FROM FactPriceData
            ) A
        WHERE A.rn > 1
    );