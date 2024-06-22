import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from utility import to_pickled_df



if __name__ == '__main__':
    data_directory = '/mnt/data/yoochoose'
    click_df = pd.read_csv(os.path.join(data_directory, 'yoochoose-clicks.dat'), header=None)
    click_df.columns = ['session_id', 'timestamp', 'item_id','category']
    click_df['valid_session'] = click_df.session_id.map(click_df.groupby('session_id')['item_id'].size() > 2)
    click_df = click_df.loc[click_df.valid_session].drop('valid_session', axis=1)
    for i in range(5):
        print(i)
        buy_df = pd.read_csv(os.path.join(data_directory, 'yoochoose-buys.dat'), header=None)
        buy_df.columns = ['session_id', 'timestamp', 'item_id', 'price', 'quantity']

        sampled_session_id = np.random.choice(click_df.session_id.unique(), 200000, replace=False)
        sampled_click_df = click_df.loc[click_df.session_id.isin(sampled_session_id)]

        item_encoder = LabelEncoder()
        sampled_click_df['item_id'] = item_encoder.fit_transform(sampled_click_df.item_id)

        sampled_buy_df = buy_df.loc[buy_df.session_id.isin(sampled_click_df.session_id)]
        sampled_buy_df['item_id'] = item_encoder.transform(sampled_buy_df.item_id)

        file_name_clicks = 'sampled_clicks'
        file_name_buys = 'sampled_buys'
        kwargs_clicks = {file_name_clicks: sampled_click_df}
        kwargs_buys = {file_name_buys: sampled_buy_df}
        to_pickled_df(data_directory, **kwargs_clicks)
        to_pickled_df(data_directory, **kwargs_buys)

