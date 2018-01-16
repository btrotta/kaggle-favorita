import pandas as pd
import numpy as np
import gc
import lightgbm as lgb


def weighted_rmse(true, predicted, weights):
    return np.sqrt(np.sum(np.multiply(weights, (true - predicted) **2)) / np.sum(weights))


def run_full_model(test_mode, all_data, train_bool, test_bool):
    train_start = all_data.loc[train_bool, 'date'].min()
    version = 4
    output_name = 'gbdt_deep_{}_v{}'.format(train_start.strftime('%y%m%d'), version)

    params = {'application': 'regression_l2', 'boosting': 'gbdt', 'subsample': 0.8, 'subsample_freq': 1,
              'lambda_l2': 1, 'lambda_l1': 0,  'min_gain_to_split': 0.0, 'metric': 'mse', 'learning_rate': 0.05,
              'num_leaves': 25, 'seed': 0, 'verbosity': 0, 'cat_smooth': 1, 'cat_l2': 1}
    num_rounds = 3000
    train_cols = ['holiday_factor', 'day_of_week',  'month', 'sales_by_day_of_month', 'day_of_month',
                  'promotion_factor', 'avg_sales_by_item_store', 'avg_sales_by_item_store_day', 'avg_sales_by_item_day',
                  'avg_sales_by_class_store', 'avg_sales_by_item', 'last_14_days_item_store_avg','last_28_days_item_store_avg',
                  'last_7_days_item_store_avg', 'trans_by_store', 'trans_by_store_day', 'family', 'class', 'store_nbr',
                  'days_since_promotion', 'days_to_promotion', 'item_store_std']
    cat_cols = ['family', 'class', 'store_nbr']

    np.random.seed(0)
    if test_mode:
        # train and predict
        train_data = lgb.Dataset(data=all_data.loc[train_bool, train_cols], label=all_data.loc[train_bool, 'unit_sales'],
                                 weight=(all_data.loc[train_bool, 'perishable'] * 0.25 + 1).values)
        test_data = lgb.Dataset(data=all_data.loc[test_bool, train_cols], label=all_data.loc[test_bool, 'unit_sales'],
                                weight=(all_data.loc[test_bool, 'perishable'] * 0.25 + 1).values)
        valid_sets = [train_data, test_data]
        valid_names = ['train', 'test']
        gc.collect()
        bst = lgb.train(params, train_set=train_data, valid_sets=valid_sets, valid_names=valid_names, verbose_eval=10,
                        num_boost_round=num_rounds, categorical_feature=cat_cols)
        importance_arr = bst.feature_importance()
        importance_dict = {c: importance_arr[i] for i, c in enumerate(train_cols)}
        all_data['predicted_sales'] = np.nan
        all_data.loc[test_bool, 'predicted_sales'] = bst.predict(all_data.loc[test_bool, train_cols])
        all_data.loc[all_data['predicted_sales'] < 0, 'predicted_sales'] = 0

        # evaluate
        test_score = weighted_rmse(all_data.loc[test_bool, 'unit_sales'].values.flatten(),
                                             all_data.loc[test_bool, 'predicted_sales'].values.flatten(),
                                             (all_data.loc[test_bool, 'perishable'].values.flatten()) * 0.25 + 1)
        test_bool_last_part = test_bool & (all_data['date'] >= all_data.loc[test_bool, 'date'].min() + pd.Timedelta(days=5))
        test_score_last_part = weighted_rmse(all_data.loc[test_bool_last_part, 'unit_sales'].values.flatten(),
                                             all_data.loc[test_bool_last_part, 'predicted_sales'].values.flatten(),
                                             (all_data.loc[test_bool_last_part, 'perishable'].values.flatten()) * 0.25 + 1)
        print(test_score, test_score_last_part)

        # write output and log
        all_data.loc[test_bool, ['item_nbr', 'store_nbr', 'date', 'predicted_sales']].to_csv('test_output_{}.csv'.format(output_name))
        with open('log_{}.txt'.format(output_name), 'w') as f:
            f.write('\nGBDT scores:\ntest: {}\ntest last part:{}'.format(test_score, test_score_last_part))

    else:
        train_data = lgb.Dataset(data=all_data.loc[train_bool, train_cols], label=all_data.loc[train_bool, 'unit_sales'],
                                 weight=(all_data.loc[train_bool, 'perishable'] * 0.25 + 1).values)
        valid_sets = [train_data]
        valid_names = ['train']
        gc.collect()
        bst = lgb.train(params, train_set=train_data, valid_sets=valid_sets, valid_names=valid_names, verbose_eval=10,
                        num_boost_round=num_rounds, categorical_feature=cat_cols)
        all_data['predicted_sales'] = np.nan
        all_data.loc[test_bool, 'predicted_sales'] = bst.predict(all_data.loc[test_bool, train_cols])
        all_data.loc[test_bool, ['id', 'predicted_sales']].to_csv('submit_output_{}.csv'.format(output_name),
                                                                 header=True)