import os
import numpy as np
import pandas as pd
from utility import to_pickled_df
from datetime import datetime


def convert_to_unix_timestamp(timestamp_str):
    dt_object = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S.%fZ')
    return int(dt_object.timestamp())

if __name__ == '__main__':
    data_directory = '/mnt/data/yoochoose'
    # sampled_buys=pd.read_pickle(os.path.join(data_directory, 'sampled_buys.df'))
    #
    # buy_sessions=sampled_buys.session_id.unique()
    for i in range(5):

        sorted_events = pd.read_pickle(os.path.join(data_directory, 'sampled_sessions.df'))
        sorted_events['timestamp'] = sorted_events['timestamp'].apply(convert_to_unix_timestamp)
        print("total", len(sorted_events))
        print(sorted_events.head(10))

        max_timestamps = sorted_events.groupby('session_id')['timestamp'].max()
        sorted_max_timestamps = max_timestamps.sort_values()
        print(sorted_max_timestamps.head(10))
        T = sorted_max_timestamps.quantile(0.9)
        sessions_test = max_timestamps[max_timestamps > T].index
        test_sessions = sorted_events[sorted_events['session_id'].isin(sessions_test)]
        train_val = sorted_events[~sorted_events['session_id'].isin(sessions_test)]
        T_val = sorted_max_timestamps.quantile(0.8)
        sessions_val = max_timestamps[max_timestamps > T_val].index
        val_sessions = train_val[train_val['session_id'].isin(sessions_val)]
        train_sessions = train_val[~train_val['session_id'].isin(sessions_val)]
        print("train:", len(train_sessions))
        print("val:", len(val_sessions))
        print("test:", len(test_sessions))

        file_name_train = 'sampled_train'
        file_name_val = 'sampled_val'
        file_name_test = 'sampled_test'
        kwargs_clicks = {file_name_train: train_sessions}
        kwargs_buys = {file_name_val: val_sessions}
        kwargs_test = {file_name_test: test_sessions}
        to_pickled_df(data_directory, **kwargs_clicks)
        to_pickled_df(data_directory, **kwargs_buys)
        to_pickled_df(data_directory, **kwargs_test)