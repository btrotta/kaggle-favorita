"""Combine predictions."""

import pandas as pd
import numpy as np
import datetime as dt
import pickle
import gc
import os
import matplotlib.pyplot as plt

test_mode = False

# read data
with open('processed_data.bin', 'rb') as f:
    all_data = pickle.load(f)

model_list = ['public_script', 'public_script_longer_train', 'gbdt_deep_160215_v4', 'gbdt_deep_170216_v4', 'gbdt_deep_170515_v4']
model_weights = {'public_script': 0.2, 'public_script_longer_train': 0.2,
                 'gbdt_deep_170216_v4': 0.2, 'gbdt_deep_160215_v4': 0.2, 'gbdt_deep_170515_v4': 0.2}

def weighted_rmse(true, predicted, weights):
    return np.sqrt(np.sum(np.multiply(weights, (true - predicted) **2)) / np.sum(weights))

if test_mode:
    predictions = all_data[['id', 'unit_sales', 'item_nbr', 'store_nbr', 'date', 'perishable']].copy()
    del all_data
    gc.collect()
    predictions.sort_index(inplace=True)
    dates = (pd.datetime(2017, 8, 1), pd.datetime(2017, 8, 15))
    test_bool = (predictions['date'] >= dates[0]) & (predictions['date'] <= dates[1])
    scores = {}
    scores_first_part = {}
    scores_last_part = {}

    for m in model_list:
        new_values = pd.read_csv(os.path.join('trained_models_7', 'test_output_{}.csv'.format(m)), index_col=0,
                                     parse_dates=['date'])
        new_values['date'] = pd.to_datetime(new_values['date'])
        predictions['date'] = pd.to_datetime(predictions['date'])
        new_values.loc[new_values['predicted_sales'] < 0, 'predicted_sales'] = 0
        new_values.sort_index(inplace=True)
        new_values.rename(columns={'predicted_sales': 'predicted_sales_' + m}, inplace=True)
        predictions = pd.merge(predictions, new_values[['item_nbr', 'store_nbr', 'date', 'predicted_sales_' + m]],
                               'left', ['item_nbr', 'store_nbr', 'date'])
        predictions['predicted_sales_' + m] = predictions['predicted_sales_' + m].fillna(0)
        test_bool = (predictions['date'] >= dates[0]) & (predictions['date'] <= dates[1])
        scores[m] = weighted_rmse(predictions.loc[test_bool, 'unit_sales'].values.flatten(),
                                    predictions.loc[test_bool, 'predicted_sales_' + m].values.flatten(),
                                    (predictions.loc[test_bool, 'perishable'] * 0.25 + 1).values.flatten())
        test_bool_last_part = test_bool & (predictions['date'] >= predictions.loc[test_bool, 'date'].min() + pd.Timedelta(days=5))
        test_bool_first_part = test_bool & (~test_bool_last_part)
        scores_first_part[m] = weighted_rmse(predictions.loc[test_bool_first_part, 'unit_sales'].values.flatten(),
                                            predictions.loc[test_bool_first_part, 'predicted_sales_' + m].values.flatten(),
                                             (predictions.loc[test_bool_first_part, 'perishable'].values.flatten()) * 0.25 + 1)
        scores_last_part[m] = weighted_rmse(predictions.loc[test_bool_last_part, 'unit_sales'].values.flatten(),
                                            predictions.loc[test_bool_last_part, 'predicted_sales_' + m].values.flatten(),
                                             (predictions.loc[test_bool_last_part, 'perishable'].values.flatten()) * 0.25 + 1)
        predictions['err_'+m] = np.nan
        predictions.loc[test_bool, 'err_'+m] \
            = (predictions.loc[test_bool, 'unit_sales'] - predictions['predicted_sales_' + m])


    err_df = predictions.loc[test_bool, ['date'] +['err_' + m for m in model_list]].groupby('date').agg(lambda x: np.sqrt((x**2).mean()))
    plt.plot(err_df.index, err_df.values)
    plt.legend(err_df.columns)

    print('Scores: {}'.format(scores))
    print('Scores first 5 days: {}'.format(scores_first_part))
    print('Scores last 11 days: {}'.format(scores_last_part))

    predictions['blended_prediction'] = 0
    for m in model_weights:
        predictions['blended_prediction'] += model_weights[m] * predictions['predicted_sales_' + m]
    print(weighted_rmse(predictions.loc[test_bool, 'unit_sales'].values.flatten(),
                                    predictions.loc[test_bool, 'blended_prediction'].values.flatten(),
                                    (predictions.loc[test_bool, 'perishable'] * 0.25 + 1).values.flatten()))
    print(weighted_rmse(predictions.loc[test_bool_first_part, 'unit_sales'].values.flatten(),
                                    predictions.loc[test_bool_first_part, 'blended_prediction'].values.flatten(),
                                    (predictions.loc[test_bool_first_part, 'perishable'] * 0.25 + 1).values.flatten()))
    print(weighted_rmse(predictions.loc[test_bool_last_part, 'unit_sales'].values.flatten(),
                                    predictions.loc[test_bool_last_part, 'blended_prediction'].values.flatten(),
                                    (predictions.loc[test_bool_last_part, 'perishable'] * 0.25 + 1).values.flatten()))
else:
    gc.collect()
    predictions = all_data.loc[all_data['date'] >= pd.datetime(2017, 8, 16),
                               ['id', 'item_store_in_train']].copy()
    del all_data
    predictions['id'] = predictions['id'].astype(int)
    predictions['unit_sales'] = 0
    predictions.set_index('id', inplace=True, drop=True)
    for m in model_weights.keys():
        if 'public_script' in m:
            new_values = pd.read_csv(os.path.join('trained_models_7', 'submit_output_{}.csv'.format(m)), usecols=['id', 'unit_sales'])
            new_values.rename(columns={'unit_sales': 'predicted_sales'}, inplace=True)
        else:
            new_values = pd.read_csv(os.path.join('trained_models_7', 'submit_output_{}.csv'.format(m)),
                                     usecols=['id', 'predicted_sales'])
        new_values.loc[new_values['predicted_sales'] < 0, 'predicted_sales'] = 0
        new_values.set_index('id', inplace=True, drop=True)
        predictions['unit_sales'] = predictions['unit_sales'] + model_weights[m] * (np.exp(new_values['predicted_sales'])-1)
        predictions[m] = np.exp(new_values['predicted_sales'])-1
        print(m, predictions.loc[predictions['item_store_in_train'] == 1, m].median())
    predictions.loc[predictions['item_store_in_train'] == 0] = 0
    predictions['unit_sales'].to_csv('Submission_{}.csv'.format(dt.datetime.now().strftime('%y%m%d_%H%M')),
                       index=True, header=True)


# Scores: {'gbdt_deep_170515_v4': 0.59706565354276708, 'public_script': 0.59980719898527846, 'public_script_longer_train': 0.60203304765944743, 'gbdt_deep_160215_v4': 0.59361704981097807, 'gbdt_deep_170216_v4': 0.594668011091532}
# Scores first 5 days: {'gbdt_deep_170515_v4': 0.58669494370483166, 'public_script': 0.5959978705459299, 'public_script_longer_train': 0.59549316300816946, 'gbdt_deep_160215_v4': 0.58224607822696905, 'gbdt_deep_170216_v4': 0.5827553710202702}
# Scores last 11 days: {'gbdt_deep_170515_v4': 0.60217641477322914, 'public_script': 0.60169998936098124, 'public_script_longer_train': 0.60527165508290637, 'gbdt_deep_160215_v4': 0.59921328239078808, 'gbdt_deep_170216_v4': 0.60052699076279459}
# 0.588801452927
# 0.578779260031
# 0.593741746281



