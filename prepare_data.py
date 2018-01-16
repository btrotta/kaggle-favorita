"""Prepare the data and add features."""

import pandas as pd
import numpy as np
from sklearn import preprocessing
import avg_transform2 as avg_transform
import pickle
import gc
import time
import os


def calc_holiday_factors(transactions, stores, holidays):

    # merge
    all_data = pd.merge(transactions, stores, 'left', ['store_nbr'])
    holidays['is_holiday'] = (~holidays['transferred']).astype(int)
    all_data = pd.merge(all_data, holidays[['date', 'is_holiday', 'locale_name', 'description']], 'left',
                        left_on=['city', 'date'], right_on=['locale_name', 'date'])
    all_data.rename(columns={'is_holiday': 'is_holiday_city'}, inplace=True)
    all_data.drop('locale_name', axis=1, inplace=True)
    all_data = pd.merge(all_data, holidays[['date', 'is_holiday', 'locale_name', 'description']], 'left',
                        left_on=['state', 'date'], right_on=['locale_name', 'date'])
    all_data.rename(columns={'is_holiday': 'is_holiday_state'}, inplace=True)
    all_data.drop('locale_name', axis=1, inplace=True)
    all_data = pd.merge(all_data, holidays.loc[holidays['locale'] == 'National', ['date', 'is_holiday', 'description']],
                        'left', left_on='date', right_on='date')
    all_data.rename(columns={'is_holiday': 'is_holiday_national'}, inplace=True)
    all_data['is_holiday'] = all_data[['is_holiday_city', 'is_holiday_state', 'is_holiday_national']].max(axis=1)
    all_data.drop(['is_holiday_city', 'is_holiday_state', 'is_holiday_national'], axis=1, inplace=True)
    all_data['is_holiday'] = all_data['is_holiday'].fillna(0).astype(int)
    all_data[['description', 'description_x', 'description_y']] = all_data[
        ['description', 'description_x', 'description_y']].fillna('')
    all_data['description'] = all_data[['description', 'description_x', 'description_y']].max(axis=1)
    all_data.drop(['description_x', 'description_y'], axis=1, inplace=True)
    all_data['description_raw'] = all_data['description'].str.replace('Traslado ', '')

    # transform to log scale
    all_data['transactions'] = np.log(all_data['transactions'] + 1)
    all_data['transactions'] = all_data['transactions'].fillna(0)

    # calculate holiday factors
    all_data.sort_values(['store_nbr', 'date'], inplace=True)
    for lag in [7, 14, 21]:
        lag_col = 'transactions_lag_' + str(lag)
        all_data[lag_col] = all_data['transactions'].shift(lag)
        store_diff = all_data['store_nbr'].diff(lag)
        all_data.loc[store_diff != 0, lag_col] = np.nan
    lag_cols = ['transactions_lag_' + str(lag) for lag in [7, 14, 21]]
    all_data['reference_trans'] = all_data[lag_cols].mean(axis=1)
    all_data['holiday_factor'] = all_data['transactions'] - all_data['reference_trans']
    holiday_factors = all_data.loc[(all_data['is_holiday'] == 1)].groupby('description_raw')[
        'holiday_factor'].mean().to_frame()
    all_data.drop('holiday_factor', axis=1, inplace=True)
    all_data = pd.merge(all_data, holiday_factors, 'left', left_on='description_raw', right_index=True)
    all_data.loc[all_data['is_holiday'] == 0, 'holiday_factor'] = 0
    all_data['holiday_factor'] = all_data['holiday_factor'].fillna(0)

    return all_data[['date', 'store_nbr', 'holiday_factor']]


def prepare_data(load_from_file=True, include_2016=False):

    if load_from_file:
        with open('processed_data.bin', 'rb') as f:
            all_data = pickle.load(f)
    else:
        # read data
        train = pd.read_csv(os.path.join('data', 'train.csv'), parse_dates=['date'],
                            dtype={'id': int, 'store_nbr': int, 'item_nbr': int, 'unit_sales': float, 'onpromotion': bool})
        test = pd.read_csv(os.path.join('data', 'test.csv'), parse_dates=['date'],
                           dtype={'id': int, 'store_nbr': int, 'item_nbr': int, 'unit_sales': float, 'onpromotion': bool})
        train['onpromotion'] = train['onpromotion'].fillna(False)
        train['onpromotion'] = train['onpromotion'].astype(int)

        # expand to fill missing values
        train['first_product_date'] = train.groupby(['store_nbr', 'item_nbr'])['date'].transform('min')
        train['last_product_date'] = train.groupby(['store_nbr', 'item_nbr'])['date'].transform('max')
        new_index = pd.MultiIndex.from_product([train['store_nbr'].unique(), train['item_nbr'].unique(), train['date'].unique()],
                                               names=['store_nbr', 'item_nbr', 'date'])
        train.set_index(['store_nbr', 'item_nbr', 'date'], inplace=True, drop=True)
        train = train.reindex(new_index)
        train.reset_index(inplace=True, drop=False)
        train['first_product_date'] = train.groupby(['store_nbr', 'item_nbr'])['first_product_date'].transform('min')
        train['last_product_date'] = train.groupby(['store_nbr', 'item_nbr'])['last_product_date'].transform('min')
        train = train.loc[train['date'] >= train['first_product_date']]
        train = train.loc[train['last_product_date'] >= pd.datetime(2017, 1, 15)]
        train['onpromotion'] = train['onpromotion'].fillna(0)  # assume items out of stock are not on promotion


        # calculate item sales by day of month
        train['day_of_month'] = train['date'].dt.day
        sales_by_day = train.groupby(['item_nbr', 'day_of_month'])['unit_sales'].mean().to_frame(
            'sales_by_day_of_month')
        sales_by_day = sales_by_day.groupby(level=[0])['sales_by_day_of_month']\
            .rolling(window=7, center=True, min_periods=1).mean().reset_index(level=[0])
        sales_by_day['item_mean'] = sales_by_day.groupby(level=[0])['sales_by_day_of_month'].transform('mean')
        sales_by_day['sales_by_day_of_month'] = sales_by_day['sales_by_day_of_month'] - sales_by_day['item_mean']

        # identify products present in test but not train
        item_store_in_train = train[['item_nbr', 'store_nbr']].drop_duplicates()
        item_store_in_train['item_store_in_train'] = 1
        item_in_train = train['item_nbr'].drop_duplicates().to_frame()
        item_in_train['item_in_train'] = 1

        # combine test and train
        if include_2016:
            train = train.loc[train['date'] >= pd.datetime(2016, 1, 15)]
        else:
            train = train.loc[train['date'] >= pd.datetime(2017, 1, 15)]
        all_data = pd.concat([train, test], axis=0).reset_index(drop=True)
        all_data = pd.merge(all_data, item_store_in_train, 'left', ['item_nbr', 'store_nbr'])
        del train, test
        gc.collect()

        # merge data
        all_data['item_store_in_train'] = all_data['item_store_in_train'].fillna(0)
        all_data = pd.merge(all_data, item_in_train, 'left', 'item_nbr')
        all_data['item_in_train'] = all_data['item_in_train'].fillna(0)
        del item_in_train
        gc.collect()
        items = pd.read_csv(os.path.join('data', 'items.csv'))
        all_data = pd.merge(all_data, items, 'left', 'item_nbr')
        del items
        gc.collect()
        stores = pd.read_csv(os.path.join('data', 'stores.csv'))
        all_data = pd.merge(all_data, stores, 'left', 'store_nbr')
        gc.collect()
        all_data['day_of_month'] = all_data['date'].dt.day
        all_data = pd.merge(all_data, sales_by_day[['sales_by_day_of_month']], 'left',
                            left_on=['item_nbr', 'day_of_month'], right_index=True)
        all_data['sales_by_day_of_month'] = all_data['sales_by_day_of_month'].fillna(0)
        del sales_by_day
        gc.collect()
        transactions = pd.read_csv(os.path.join('data', 'transactions.csv'), parse_dates=['date'])
        all_data = pd.merge(all_data, transactions, 'left', ['date', 'store_nbr'])
        gc.collect()

        # calculate holidays and holiday factors
        holidays = pd.read_csv(os.path.join('data', 'holidays_events.csv'), parse_dates=['date'])
        holidays['is_holiday'] = (~holidays['transferred']).astype(int)
        all_data = pd.merge(all_data, holidays[['date', 'is_holiday', 'locale_name']], 'left',
                            left_on=['city', 'date'], right_on=['locale_name', 'date'])
        all_data.rename(columns={'is_holiday': 'is_holiday_city'}, inplace=True)
        all_data.drop('locale_name', axis=1, inplace=True)
        all_data = pd.merge(all_data, holidays[['date', 'is_holiday', 'locale_name']], 'left',
                            left_on=['state', 'date'], right_on=['locale_name', 'date'])
        all_data.rename(columns={'is_holiday': 'is_holiday_state'}, inplace=True)
        all_data.drop('locale_name', axis=1, inplace=True)
        all_data = pd.merge(all_data, holidays.loc[holidays['locale'] == 'National', ['date', 'is_holiday']],
                            'left', left_on='date', right_on='date')
        all_data.rename(columns={'is_holiday': 'is_holiday_national'}, inplace=True)
        all_data['is_holiday'] = all_data[['is_holiday_city', 'is_holiday_state', 'is_holiday_national']].max(axis=1)
        all_data.drop(['is_holiday_city', 'is_holiday_state', 'is_holiday_national'], axis=1, inplace=True)
        all_data['is_holiday'] = all_data['is_holiday'].fillna(0).astype(int)
        holiday_factors = calc_holiday_factors(transactions, stores, holidays)
        all_data = pd.merge(all_data, holiday_factors, 'left', ['store_nbr', 'date'])
        all_data['holiday_factor'] = all_data['holiday_factor'].fillna(0)
        del holiday_factors, transactions, stores
        gc.collect()

        # encode non-numeric columns
        enc = preprocessing.LabelEncoder()
        all_data['family'] = enc.fit_transform(all_data['family'])
        all_data['city'] = enc.fit_transform(all_data['city'])
        all_data['state'] = enc.fit_transform(all_data['state'])
        all_data['type'] = enc.fit_transform(all_data['type'])
        all_data['onpromotion'] = all_data['onpromotion'].astype(float)

        # transform target
        all_data['unit_sales'] = all_data['unit_sales'].fillna(0)
        all_data.loc[all_data['unit_sales'] < 0, 'unit_sales'] = 0
        all_data['unit_sales'] = np.log(all_data['unit_sales'] + 1)
        all_data['transactions'] = np.log(all_data['transactions'] + 1)

        # date features
        all_data['month'] = all_data['date'].dt.month
        all_data['day_of_week'] = all_data['date'].dt.dayofweek

    filename = 'processed_data_2016.bin' if include_2016 else 'processed_data.bin'
    with open(filename, 'wb') as f:
        pickle.dump(all_data, f, pickle.HIGHEST_PROTOCOL)

    return all_data


def project_ts(all_data, test_bool, ts_cols):
    # for test period, project forward the values of the time series features at the start of the period
    all_data['index'] = all_data.index
    start_date = all_data.loc[test_bool, 'date'].min()
    values_at_start = all_data.loc[all_data['date'] == start_date, ['item_nbr', 'store_nbr'] + ts_cols]
    projected_values = pd.merge(all_data.loc[test_bool, ['index', 'item_nbr', 'store_nbr']], values_at_start,
                                'left', on=['item_nbr', 'store_nbr'])
    projected_values.set_index('index', drop=True, inplace=True)
    all_data.loc[test_bool, ts_cols] = projected_values.loc[test_bool, ts_cols]
    del values_at_start, projected_values
    for col_name in ts_cols:
        all_data.loc[all_data[col_name].isnull(), col_name] \
            = all_data.groupby(['item_nbr', 'store_nbr'])['unit_sales_norm'].transform('mean').loc[
            all_data[col_name].isnull()]
        all_data.loc[all_data[col_name].isnull(), col_name] \
            = all_data.groupby(['item_nbr'])['unit_sales_norm'].transform('mean').loc[all_data[col_name].isnull()]
        all_data[col_name] = all_data[col_name].fillna(all_data['unit_sales_norm'].mean())
    all_data.drop('index', axis=1, inplace=True)
    gc.collect()


def get_features(load_from_file, test_mode, train_start, input_data_file=None):
    mode_string = 'test' if test_mode else 'submit'
    if load_from_file:
        with open('feature_data_{}_train_start_{}.bin'.format(mode_string, train_start.strftime('%y%m%d')), 'rb') as f:
            all_data = pickle.load(f)
        return all_data
    else:
        with open(input_data_file, 'rb') as f:
            all_data = pickle.load(f)
        if test_mode:
            train_end = pd.datetime(2017, 7, 31)
        else:
            train_end = pd.datetime(2017, 8, 15)

        exclude_earthquake = pd.date_range(pd.datetime(2016, 4, 1), pd.datetime(2016, 6, 1), freq='D')
        exclude_christmas = pd.date_range(pd.datetime(2016, 12, 15), pd.datetime(2017, 1, 15), freq='D')

        train_bool = (all_data['date'] >= train_start) & (all_data['date'] <= train_end) \
                     & (~all_data['date'].isin(exclude_earthquake)) & (~all_data['date'].isin(exclude_christmas))
        train_bool_short = train_bool & (all_data['onpromotion'] == 0) & (all_data['is_holiday'] == 0)

        # days since promotion, days to next promotion
        t = time.time()
        all_data['day_int'] = all_data['date'].dt.dayofyear
        all_data.loc[all_data['date'].dt.year == 2017, 'day_int'] \
            = all_data.loc[all_data['date'].dt.year == 2017, 'day_int'] + 365
        all_data['promotion_dates'] = all_data['day_int'].copy()
        all_data.loc[all_data['onpromotion'] == 0, 'promotion_dates'] = np.nan
        all_data.sort_values(['store_nbr', 'item_nbr', 'date'], inplace=True)
        all_data['promotion_dates_ffill'] = all_data['promotion_dates'].fillna(method='ffill').shift(1)
        item_store_diff = (all_data['item_nbr'].diff() != 0) | (all_data['store_nbr'].diff() != 0)
        all_data.loc[item_store_diff, 'promotion_dates_ffill'] = np.nan
        all_data['promotion_dates_bfill'] = all_data['promotion_dates'].fillna(method='bfill').shift(-1)
        item_store_diff = (all_data['item_nbr'].diff(-1) != 0) | (all_data['store_nbr'].diff(-1) != 0)
        all_data.loc[item_store_diff, 'promotion_dates_bfill'] = np.nan
        all_data['days_since_promotion'] = all_data['day_int'] - all_data['promotion_dates_ffill']
        all_data['days_to_promotion'] = all_data['promotion_dates_bfill'] - all_data['day_int']
        for c in ['days_since_promotion', 'days_to_promotion']:
            c_is_null = all_data[c].isnull()
            days_to_end = (all_data['date'].max() - all_data.loc[c_is_null, 'date']).dt.days + 1
            fill1 = all_data.groupby(['item_nbr', 'store_nbr'])[c].transform('mean').loc[c_is_null]
            all_data.loc[c_is_null, c] = np.maximum(fill1, days_to_end)
            fill2 = all_data.groupby('family')[c].transform('mean').loc[c_is_null]
            all_data.loc[c_is_null, c] = np.maximum(fill2, days_to_end)
        all_data.drop(['promotion_dates', 'promotion_dates_ffill', 'promotion_dates_bfill', 'day_int'], axis=1, inplace=True)
        gc.collect()
        print('days since promotion calculation time: ', time.time() - t)

        t = time.time()
        # average features
        all_data['avg_sales_by_item_store'] \
            = avg_transform.bayesian_group_estimate(all_data, ['item_nbr', 'store_nbr'], 'unit_sales', 'normal',
                                                    train_bool_short, exclude_current_row=True, prior_group=['store_nbr', 'family'])
        # Use average by item, store, and day of week, as done in the following public kernel:
        # https://www.kaggle.com/tarobxl/ma-for-each-day-lb-0-537/
        all_data['avg_sales_by_item_store_day'] \
            = avg_transform.bayesian_group_estimate(all_data, ['item_nbr', 'store_nbr', 'day_of_week'], 'unit_sales',
                                                    'normal', train_bool_short, exclude_current_row=True, prior_group=['item_nbr', 'store_nbr'])
        all_data['avg_sales_by_item_store_day'] = all_data['avg_sales_by_item_store_day'] - all_data['avg_sales_by_item_store']
        all_data['avg_sales_by_class_store'] \
            = avg_transform.bayesian_group_estimate(all_data, ['class', 'store_nbr'], 'unit_sales', 'normal',
                                                    train_bool_short,  exclude_current_row=True)
        all_data['avg_sales_by_item_day'] \
            = avg_transform.bayesian_group_estimate(all_data, ['item_nbr', 'day_of_week'], 'unit_sales', 'normal',
                                                    train_bool_short,  exclude_current_row=True)
        all_data['avg_sales_by_item'] \
            = avg_transform.bayesian_group_estimate(all_data, ['item_nbr'], 'unit_sales', 'normal', train_bool_short)
        all_data['avg_sales_by_item_day'] = all_data['avg_sales_by_item_day'] - all_data['avg_sales_by_item']

        all_data['trans_norm'] = all_data['transactions'] - all_data['holiday_factor']
        all_data['trans_by_store_day'] \
            = avg_transform.bayesian_group_estimate(all_data, ['store_nbr', 'day_of_week'], 'transactions', 'normal', train_bool)
        all_data['trans_by_store'] \
            = avg_transform.bayesian_group_estimate(all_data, ['store_nbr'], 'transactions', 'normal',  train_bool)
        all_data['trans_by_store_day'] = all_data['trans_by_store_day'] - all_data['trans_by_store']
        gc.collect()
        print('bayesian calculation time: ', time.time() - t)

        # calculate promotion factor and use it to normalise sales
        t = time.time()
        all_data['prediction_err'] = all_data['unit_sales'] \
                                     - (all_data['avg_sales_by_item_store_day'] + all_data['avg_sales_by_item_store'])
        promo_train = (all_data['onpromotion'] == 1) & train_bool
        all_data['promotion_factor'] \
            = avg_transform.bayesian_group_estimate(all_data,  ['item_nbr', 'store_nbr'], 'prediction_err', 'normal',
                                                    promo_train, exclude_current_row=True, prior_group=['item_nbr'])
        all_data.loc[all_data['onpromotion'] == 0, 'promotion_factor'] = 0
        all_data.drop(['prediction_err'], axis=1, inplace=True)
        all_data['unit_sales_norm'] = all_data['unit_sales'] - all_data['promotion_factor'] - all_data['holiday_factor']
        print('promotion normalisation time: ', time.time() - t)

        # time series features
        t = time.time()
        all_data.sort_values(['store_nbr', 'item_nbr', 'date'], inplace=True)
        for p in [7, 14, 28]:
            col_name = 'last_{}_days_item_store_avg'.format(p)
            all_data[col_name] = all_data['unit_sales_norm'].rolling(window=p, min_periods=1).mean().shift(1)
            item_store_diff = (all_data['item_nbr'].diff(p+1) != 0) | (all_data['store_nbr'].diff(p+1) != 0)
            all_data.loc[item_store_diff, col_name] = np.nan
            all_data.loc[all_data[col_name].isnull(), col_name] \
                = all_data.loc[all_data[col_name].isnull(), 'avg_sales_by_item']
            all_data.loc[all_data[col_name].isnull(), col_name] = all_data.loc[train_bool, 'avg_sales_by_item'].mean()
            del item_store_diff
        gc.collect()
        print('time series features time: ', time.time() - t)
        ts_cols = ['last_{}_days_item_store_avg'.format(p) for p in [7, 14, 28]]
        test_bool = all_data['date'] > all_data.loc[train_bool, 'date'].max()
        project_ts(all_data, test_bool, ts_cols)
        gc.collect()

        # std feature
        all_data['item_store_std'] \
            = avg_transform.group_std(all_data, ['item_nbr', 'store_nbr'], 'last_7_days_item_store_avg', train_bool)

        with open('feature_data_{}_train_start_{}.bin'.format(mode_string, train_start.strftime('%y%m%d')), 'wb') as f:
            pickle.dump(all_data, f, pickle.HIGHEST_PROTOCOL)

