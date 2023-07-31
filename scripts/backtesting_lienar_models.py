import os
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import numpy as np
import ast
import tqdm
from dotenv import load_dotenv


# Load environment variables
_ = load_dotenv()
working_dir = os.getenv("WORKING_DIR")

# Change to working dir
os.chdir(working_dir)

# Import custom modules
from utils.custom_logger import initiate_logger
from utils import database_tools

class Spread(Strategy):
    def init(self):
        pass
    
    def next(self):
        if crossover(self.data.Close, self.data.Mean - self.data.Std*1.5):
            self.buy(size=1)#, sl=self.data.Close * 0.9)#, tp=self.data.Close * 1.1)
        elif crossover(self.data.Close, self.data.Mean + self.data.Std*1.5):
            self.sell(size=1)#, sl=self.data.Close * 1.1)#, tp=self.data.Close * 0.9)
        elif crossover(self.data.Close, self.data.Mean ):
            self.position.close()



df_models = pd.read_parquet('./data/models.parquet')

df_coins = database_tools.query_topN_price_data(100)
output_list = []

# Create pivot table
values = ['open_price',	'high_price',	'low_price',	'close_price']


#df_models = df_models.iloc[:10]
for _, model in tqdm.tqdm(df_models.iterrows(), total=df_models.shape[0]):
    try:
        model_id = model.model_id
        symbol_list = model_id.split('_')
        model_coeff  = {k: v for k, v in model.model_coeff.items() if v is not None}

        for value in values:
            df_pivot = df_coins.pivot_table(
                values=value,
                index=["date"],
                columns="id",
            ).reset_index()

            
            df = df_pivot[['date'] + symbol_list]
            df = df.dropna()
            df[value] = 0


            for symbol in symbol_list:
                df[value] = df[value] + df[symbol] * model_coeff[symbol]


            if value==values[0]:
                df_input = df[['date', value]]
            else:
                df_input = pd.concat([df_input, df[value]], axis=1)

        df_input.columns = ['date', 'Open','High','Low','Close']
        df_input = df_input.assign(date = pd.to_datetime(df_input.date),Mean = df_input.Close.mean(), Std = df_input.Close.std())
        df_input = df_input.set_index('date')
        df_input.index.name = None

        bt = Backtest(df_input, 
                      Spread,
                      cash=1000000, 
                      commission=0.002,
                      exclusive_orders=False,
                      margin=0.2 # 0.2,#5x leverage
                      )

        output = bt.run()        
        df_output = output.to_frame().T
        df_output['model_id'] = model_id

        output_list.append(df_output)
    except Exception as e:
        print(e)
        continue

df_all_output = pd.concat(output_list)
df_all_output.to_excel('./data/backtesting_spread_15std_5xLeverage.xlsx')

#df_all_output.to_clipboard(index=False)

# output_list =[]
# for symbol in symbol_list:
#     df_input = df_coins[df_coins.symbol==symbol][['open_price', 'high_price', 'low_price', 'close_price',
#         'volume', 'date',]]
#     df_input.columns = ['Open','High','Low','Close','Volume','date']
#     df_input = df[['date','Spread']].merge(df_input, on=['date'], how='left')

#     df_input = df_input.assign(date = pd.to_datetime(df_input.date),Mean = df_input.Spread.mean(), Std = df_input.Spread.std(), sign=model_coeff[symbol] / abs(model_coeff[symbol]))
#     df_input = df_input.set_index('date')
#     df_input.index.name = None
#     df_input = df_input[['Open','High','Low','Close','Volume','Spread','Mean','Std','sign']]


#     bt = Backtest(df_input, Spread,
#                 cash=1000000, commission=.002,
#                 exclusive_orders=False)

#     output = bt.run()
#     #bt.plot()

#     df_output = output.to_frame().T
#     df_output['symbol'] = symbol

#     output_list.append(df_output)
    
# df_all_output = pd.concat(output_list)