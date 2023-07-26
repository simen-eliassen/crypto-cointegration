import pandas as pd
import numpy as np
import pandas as pd
import datetime
import statsmodels.api as sm
import numpy as np
import tqdm
import itertools
import requests
import json
import multiprocessing
import warnings


def generate_crypto_pairs(dependent_coin, all_symbols, min_pairs=2, max_pairs=4):
    crypto_pairs = []
    for coin in tqdm.tqdm(all_symbols):
        for i in range(min_pairs-1, max_pairs-1):
            if coin in [dependent_coin]:
                symbols2combine = [s for s in all_symbols if s != coin]
                combs = list(itertools.combinations(symbols2combine, i))
                crypto_pairs = crypto_pairs + [c + (coin,) for c in combs]
                
    return crypto_pairs


def linear_model(df, symbol_list):
    df = df.dropna()

    try:
        X = sm.add_constant(df[symbol_list[:-1]])
        y = df[symbol_list[-1]]
        model = sm.OLS(y, X).fit()
        df_coeff = model.params.reset_index()
        df_coeff.columns = ["symbol", "coeff"]
        df_coeff.coeff = -df_coeff.coeff
        df_coeff.loc[df_coeff.symbol == "const", "coeff"] = 1
        df_coeff.loc[df_coeff.symbol == "const", "symbol"] = symbol_list[-1]
        # orient : str {'dict', 'list', 'series', 'split', 'tight', 'records', 'index'}
        coeff = df_coeff.set_index("symbol").to_dict("dict")
        data = {"model_id": "_".join(symbol_list), "model_coeff": coeff}
        df_out = pd.DataFrame(data)
        df_out["model_date"] = datetime.datetime.now()
    except Exception:
        df_out = pd.DataFrame()

    return df_out
