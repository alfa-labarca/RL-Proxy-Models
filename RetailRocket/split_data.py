import os
import numpy as np
import pandas as pd
from utility import to_pickled_df


if __name__ == '__main__':
    data_directory = '/mnt/data/RetailRocket/data'
    # sampled_buys=pd.read_pickle(os.path.join(data_directory, 'sampled_buys.df'))
    #
    # buy_sessions=sampled_buys.session_id.unique()
    sorted_events = pd.read_pickle(os.path.join(data_directory, 'sorted_events.df'))
    print("total:", len(sorted_events))
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

    to_pickled_df(data_directory, sampled_train=train_sessions)
    to_pickled_df(data_directory, sampled_val=val_sessions)
    to_pickled_df(data_directory,sampled_test=test_sessions)