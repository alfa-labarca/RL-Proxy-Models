import os
import pandas as pd
import tensorflow as tf
from utility import to_pickled_df, pad_history

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('history_length',10,'uniform history length')

def compute_cumulative_discounted_rewards(rewards, gamma):
    n = len(rewards)
    returns = [0] * n
    returns[-1] = rewards[-1]
    for t in reversed(range(n - 1)):
        returns[t] = rewards[t] + gamma * returns[t + 1]
    return returns

def categorize(row):
    if row['is_buy'] == 0 and row['next_inter'] == 0:
        return 0
    elif row['is_buy'] == 0 and row['next_inter'] == 1:
        return 1
    elif row['is_buy'] == 1 and row['next_inter'] == 0:
        return 2
    elif row['is_buy'] == 1 and row['next_inter'] == 1:
        return 3

if __name__ == '__main__':

    data_directory = '/mnt/data/RetailRocket/data'

    length=FLAGS.history_length

    # reply_buffer = pd.DataFrame(columns=['state','action','reward','next_state','is_done'])
    sorted_events=pd.read_pickle(os.path.join(data_directory, 'sorted_events.df'))
    item_ids=sorted_events.item_id.unique()
    pad_item=len(item_ids)

    train_sessions = pd.read_pickle(os.path.join(data_directory, 'sampled_train.df'))
    groups=train_sessions.groupby('session_id')
    ids=train_sessions.session_id.unique()

    state,len_state,action,is_buy,next_state,len_next_state,is_done,fut_reward,next_action,next_inter,category,category2,n_cat,inter2,size,hist,fut = [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
    for id in ids:
        group=groups.get_group(id)
        history=[]
        returns = [5 if row['is_buy'] == 1 else 1 for _, row in group.iterrows()]
        this_reward = compute_cumulative_discounted_rewards(returns, 0.5)
        enter = False
        enter2 = False
        for index, row in group.iterrows():
            if len(group) <= 3:
                size.append(0)
            elif len(group) <= 10:
                size.append(1)
            elif len(group) <= 30:
                size.append(2)
            else:
                size.append(3)
            s=list(history)
            len_state.append(length if len(s)>=length else 1 if len(s)==0 else len(s))
            s=pad_history(s,length,pad_item)
            hist.append(len(s))
            a=row['item_id']
            fut.append(len(group) - len(s) - 1)
            is_b=row['is_buy']
            state.append(s)
            action.append(a)
            if enter2:
                inter2.append(is_b+1)
            if enter:
                next_action.append(a)
                next_inter.append(is_b+1)
                enter2 = True
            is_buy.append(is_b+1)
            history.append(row['item_id'])
            next_s=list(history)
            len_next_state.append(length if len(next_s)>=length else 1 if len(next_s)==0 else len(next_s))
            next_s=pad_history(next_s,length,pad_item)
            next_state.append(next_s)
            is_done.append(False)
            enter = True
        next_action.append(0)
        next_inter.append(0)
        inter2.append(0)
        if enter2:
            inter2.append(0)
        fut_reward.extend(this_reward)
        is_done[-1]=True
    for i in range(len(next_inter)):
        if is_buy[i] == 1 and next_inter[i] == 0 and inter2[i] == 0:
            category2.append(0)
        elif is_buy[i] == 1 and next_inter[i] == 1 and inter2[i] == 0:
            category2.append(1)
        elif is_buy[i] == 1 and next_inter[i] == 1 and inter2[i] == 1:
            category2.append(2)
        elif is_buy[i] == 1 and next_inter[i] == 1 and inter2[i] == 2:
            category2.append(3)
        elif is_buy[i] == 1 and next_inter[i] == 2 and inter2[i] == 0:
            category2.append(4)
        elif is_buy[i] == 1 and next_inter[i] == 2 and inter2[i] == 1:
            category2.append(5)
        elif is_buy[i] == 1 and next_inter[i] == 2 and inter2[i] == 2:
            category2.append(6)
        elif is_buy[i] == 2 and next_inter[i] == 0 and inter2[i] == 0:
            category2.append(7)
        elif is_buy[i] == 2 and next_inter[i] == 1 and inter2[i] == 0:
            category2.append(8)
        elif is_buy[i] == 2 and next_inter[i] == 1 and inter2[i] == 1:
            category2.append(9)
        elif is_buy[i] == 2 and next_inter[i] == 1 and inter2[i] == 2:
            category2.append(10)
        elif is_buy[i] == 2 and next_inter[i] == 2 and inter2[i] == 0:
            category2.append(11)
        elif is_buy[i] == 2 and next_inter[i] == 2 and inter2[i] == 1:
            category2.append(12)
        elif is_buy[i] == 2 and next_inter[i] == 2 and inter2[i] == 2:
            category2.append(13)
        else:
            print("Error", group, id)
    for i in range(len(next_inter)):
        if is_buy[i] == 1 and next_inter[i] == 0:
            category.append(0)
            n_cat.append(0)
        elif is_buy[i] == 1 and next_inter[i] == 1:
            category.append(1)
            n_cat.append(1)
        elif is_buy[i] == 1 and next_inter[i] == 2:
            category.append(2)
            n_cat.append(2)
        elif is_buy[i] == 2 and next_inter[i] == 0:
            category.append(3)
            n_cat.append(3)
        elif is_buy[i] == 2 and next_inter[i] == 1:
            category.append(4)
            n_cat.append(4)
        elif is_buy[i] == 2 and next_inter[i] == 2:
            category.append(5)
            n_cat.append(5)
        else:
            print("Error", group, id)
    n_cat = n_cat[1:]
    n_cat.append(6)
    print("listo, largos: ", len(state), len(action), len(next_inter), len(category), len(n_cat))

    dic={'state':state,'len_state':len_state,'action':action,'is_buy':is_buy,'next_state':next_state,'len_next_states':len_next_state,
        'is_done':is_done,'fut_reward':fut_reward, 'next_action': next_action, 'next_inter': next_inter, 'category': category,
        'category2': category2, 'inter2': inter2, 'n_cat':n_cat, 'size':size, 'hist':hist, 'fut':fut}
    reply_buffer=pd.DataFrame(data=dic)
    reply_buffer.to_pickle(os.path.join(data_directory, 'replay_buffer.df'))

    dic={'state_size':[length],'item_num':[pad_item]}
    data_statis=pd.DataFrame(data=dic)
    to_pickled_df(data_directory,data_statis=data_statis)