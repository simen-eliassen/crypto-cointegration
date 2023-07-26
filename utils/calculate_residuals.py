
# Import libraries
import pandas as pd
import statsmodels.api as sm


def calc_resid(df):
    symbols = [col for col in df.columns if col != "date"]

    try:
        X = sm.add_constant(df[symbols[:-1]])
        y = df[symbols[-1]]
        model = sm.OLS(y, X).fit()
        df_tmp = pd.DataFrame(model.resid, columns=["_".join(symbols)])
        df_tmp["date"] = df.date
        df_tmp["symbol"] = df_tmp.columns[0]
        df_tmp = df_tmp.rename(columns={df_tmp.columns[0]: "close_price"})
    except Exception:
        df_tmp = pd.DataFrame()
        
    return df_tmp