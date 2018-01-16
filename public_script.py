"""
This code has been modified by me, Belinda Trotta, from public scripts.
I originally took a copy of the following script by Lingzhi (which is itself based on other public work):
https://www.kaggle.com/vrtjso/lgbm-one-step-ahead?scriptVersionId=1965435
I later incorporated some of the modifications made in the following script by Bojan Tunguz:
https://www.kaggle.com/tunguz/lgbm-one-step-ahead?scriptVersionId=1993971
Licence: http://www.apache.org/licenses/LICENSE-2.0
"""

# Below comment is by the author of https://www.kaggle.com/vrtjso/lgbm-one-step-ahead
"""
This is an upgraded version of Ceshine's LGBM starter script, simply adding more
average features and weekly average features on it.
"""
from datetime import date, timedelta

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

def run_public_script(test_mode=True, longer_training=False):

    df_train = pd.read_csv(
        'data//train.csv', usecols=[0, 1, 2, 3, 4, 5],
        dtype={'onpromotion': bool},
        converters={'unit_sales': lambda u: np.log1p(
            float(u)) if float(u) > 0 else 0},
        parse_dates=["date"],
        skiprows=range(1, 66458909)  # 2016-01-01
    )

    if test_mode:
        # make sure we don't leak any data when measuring performance on validation set
        df_train.loc[df_train['date'] >= pd.datetime(2017, 8, 1), 'unit_sales'] = np.nan


    df_test = pd.read_csv(
        "data//test.csv", usecols=[0, 1, 2, 3, 4],
        dtype={'onpromotion': bool},
        parse_dates=["date"]  # , date_parser=parser
    ).set_index(
        ['store_nbr', 'item_nbr', 'date']
    )

    items = pd.read_csv(
        "data//items.csv",
    ).set_index("item_nbr")

    if longer_training:
        df_2017 = df_train
    else:
        df_2017 = df_train.loc[df_train.date>=pd.datetime(2017,1,1)]
    del df_train


    promo_2017_train = df_2017.set_index(
        ["store_nbr", "item_nbr", "date"])[["onpromotion"]].unstack(
            level=-1).fillna(False)
    promo_2017_train.columns = promo_2017_train.columns.get_level_values(1)
    promo_2017_test = df_test[["onpromotion"]].unstack(level=-1).fillna(False)
    promo_2017_test.columns = promo_2017_test.columns.get_level_values(1)
    promo_2017_test = promo_2017_test.reindex(promo_2017_train.index).fillna(False)
    promo_2017 = pd.concat([promo_2017_train, promo_2017_test], axis=1)
    del promo_2017_test, promo_2017_train

    df_2017 = df_2017.set_index(
        ["store_nbr", "item_nbr", "date"])[["unit_sales"]].unstack(
            level=-1).fillna(0)
    df_2017.columns = df_2017.columns.get_level_values(1)
    df_2017[pd.Timestamp(2017,8,16)] = 0

    items = items.reindex(df_2017.index.get_level_values(1))

    def get_timespan(df, dt, minus, periods, freq='D'):
        return df[pd.date_range(dt - timedelta(days=minus), periods=periods, freq=freq)]

    if longer_training:
        def prepare_dataset(t2017, num_days, is_train=True):
            X = pd.DataFrame({
                "day_1_2017": get_timespan(df_2017, t2017, 1, 1).values.ravel(),
                "mean_3_2017": get_timespan(df_2017, t2017, 3, 3).mean(axis=1).values,
                "mean_7_2017": get_timespan(df_2017, t2017, 7, 7).mean(axis=1).values,
                "mean_14_2017": get_timespan(df_2017, t2017, 14, 14).mean(axis=1).values,
                "mean_30_2017": get_timespan(df_2017, t2017, 30, 30).mean(axis=1).values,
                "mean_49_2017": get_timespan(df_2017, t2017, 49, 49).mean(axis=1).values,
                "promo_14_2017": get_timespan(promo_2017, t2017, 14, 14).sum(axis=1).values,
                "promo_49_2017": get_timespan(promo_2017, t2017, 49, 49).sum(axis=1).values,
            })
            for i in range(7):
                X['mean_4_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 28 - i, 4, freq='7D').mean(axis=1).values
                X['mean_7_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 49 - i, 7, freq='7D').mean(axis=1).values
            for i in range(num_days):
                X["promo_{}".format(i)] = promo_2017[
                    t2017 + timedelta(days=i)].values.astype(np.uint8)
            if is_train:
                y = df_2017[
                    pd.date_range(t2017, periods=num_days)
                ].values
                return X, y
            return X
    else:
        def prepare_dataset(t2017, num_days, is_train=True):
            X = pd.DataFrame({
                "day_1_2017": get_timespan(df_2017, t2017, 1, 1).values.ravel(),
                "mean_3_2017": get_timespan(df_2017, t2017, 3, 3).mean(axis=1).values,
                "mean_7_2017": get_timespan(df_2017, t2017, 7, 7).mean(axis=1).values,
                "mean_14_2017": get_timespan(df_2017, t2017, 14, 14).mean(axis=1).values,
                "mean_30_2017": get_timespan(df_2017, t2017, 30, 30).mean(axis=1).values,
                "mean_60_2017": get_timespan(df_2017, t2017, 60, 60).mean(axis=1).values,
                "mean_140_2017": get_timespan(df_2017, t2017, 140, 140).mean(axis=1).values,
                "promo_14_2017": get_timespan(promo_2017, t2017, 14, 14).sum(axis=1).values,
                "promo_60_2017": get_timespan(promo_2017, t2017, 60, 60).sum(axis=1).values,
                "promo_140_2017": get_timespan(promo_2017, t2017, 140, 140).sum(axis=1).values
            })
            for i in range(7):
                X['mean_4_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 28-i, 4, freq='7D').mean(axis=1).values
                X['mean_20_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 140-i, 20, freq='7D').mean(axis=1).values
            for i in range(num_days):
                X["promo_{}".format(i)] = promo_2017[
                    t2017 + timedelta(days=i)].values.astype(np.uint8)
            if is_train:
                y = df_2017[
                    pd.date_range(t2017, periods=num_days)
                ].values
                return X, y
            return X

    print("Preparing dataset...")
    if longer_training:
        if test_mode:
            t2017 = date(2017, 3, 7)
        else:
            t2017 = date(2017, 3, 1)
        num_weeks = 19 if test_mode else 22
    else:
        if test_mode:
            t2017 = date(2017, 6, 6)
        else:
            t2017 = date(2017, 5, 31)
        num_weeks = 6 if test_mode else 9
    X_l, y_l = [], []
    for i in range(num_weeks):
        delta = timedelta(days=7 * i)
        X_tmp, y_tmp = prepare_dataset(
            t2017 + delta, 16
        )
        X_l.append(X_tmp)
        y_l.append(y_tmp)
    X_train = pd.concat(X_l, axis=0)
    y_train = np.concatenate(y_l, axis=0)
    del X_l, y_l
    X_val, y_val = prepare_dataset(date(2017, 8, 1), 16)
    X_test = prepare_dataset(date(2017, 8, 16), 16, is_train=False)

    print("Training and predicting models...")
    params = {
        'num_leaves': 33,
        'objective': 'regression',
        'min_data_in_leaf': 250,
        'learning_rate': 0.02,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 2,
        'metric': 'l2',
        'num_threads': 4,
        'seed': 0
    }

    MAX_ROUNDS = 3000
    val_pred = []
    test_pred = []
    cate_vars = []
    for i in range(16):
        print("=" * 50)
        print("Step %d" % (i+1))
        print("=" * 50)
        dtrain = lgb.Dataset(
            X_train, label=y_train[:, i],
            categorical_feature=cate_vars,
            weight=pd.concat([items["perishable"]] * num_weeks) * 0.25 + 1
        )
        dval = lgb.Dataset(
            X_val, label=y_val[:, i], reference=dtrain,
            weight=items["perishable"] * 0.25 + 1,
            categorical_feature=cate_vars)
        bst = lgb.train(
            params, dtrain, num_boost_round=MAX_ROUNDS,
            valid_sets=[dtrain, dval], verbose_eval=100
        )
        print("\n".join(("%s: %.2f" % x) for x in sorted(
            zip(X_train.columns, bst.feature_importance("gain")),
            key=lambda x: x[1], reverse=True
        )))
        val_pred.append(bst.predict(
            X_val))
        test_pred.append(bst.predict(
            X_test))


    def weighted_rmse(true, predicted, weights):
        return np.sqrt(np.sum(np.multiply(weights, (true - predicted) **2)) / np.sum(weights))

    print("Validation mse:", weighted_rmse(
        y_val[:, :15].flatten(), np.array(val_pred).transpose()[:, :15].flatten(),
        np.repeat(items["perishable"].values.flatten() * 0.25 + 1, 15)))

    print("Validation mse, last 11 days:", weighted_rmse(
        y_val[:, 5:15].flatten(), np.array(val_pred).transpose()[:, 5:15].flatten(),
        np.repeat(items["perishable"].values.flatten() * 0.25 + 1, 10)))

    print("Making submission...")
    y_test = np.array(test_pred).transpose()
    df_preds = pd.DataFrame(
        y_test, index=df_2017.index,
        columns=pd.date_range("2017-08-16", periods=16)
    ).stack().to_frame("unit_sales")
    df_preds.index.set_names(["store_nbr", "item_nbr", "date"], inplace=True)


    if test_mode:
        # write test output
        df_val = pd.DataFrame(
            np.array(val_pred[:15]).transpose(), index=df_2017.index,
            columns=pd.date_range("2017-08-1", periods=15)
        ).stack().to_frame("predicted_sales")
        df_val.index.set_names(["store_nbr", "item_nbr", "date"], inplace=True)
        if longer_training:
            df_val.reset_index().to_csv('test_output_public_script_longer_train.csv')
        else:
            df_val.reset_index().to_csv('test_output_public_script.csv')

    else:
        submission = df_test[["id"]].join(df_preds, how="left").fillna(0)
        if longer_training:
            submission.to_csv('submit_output_public_script_longer_train.csv', float_format='%.4f', index=None)
        else:
            submission.to_csv('submit_output_public_script.csv', float_format='%.4f', index=None)
