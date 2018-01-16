import pandas as pd
import prepare_data
import gbdt_models_v9_deep
import public_script
import gc

prepare_data.prepare_data(load_from_file=False, include_2016=False)
prepare_data.prepare_data(load_from_file=False, include_2016=True)

prepare_data.get_features(False, True, pd.datetime(2016, 1, 15), 'processed_data_2016.bin')
prepare_data.get_features(False, False, pd.datetime(2016, 1, 15), 'processed_data_2016.bin')
prepare_data.get_features(False, True, pd.datetime(2017, 1, 15), 'processed_data.bin')
prepare_data.get_features(False, False, pd.datetime(2017, 1, 15), 'processed_data.bin')
prepare_data.get_features(False, True, pd.datetime(2017, 4, 15), 'processed_data.bin')
prepare_data.get_features(False, False, pd.datetime(2017, 4, 15), 'processed_data.bin')

for test_mode in [True, False]:
    public_script.run_public_script(test_mode, True)
    public_script.run_public_script(test_mode, False)
    for train_start in [pd.datetime(2017, 5, 15), pd.datetime(2017, 2, 15), pd.datetime(2016, 2, 15)]:
        if train_start == pd.datetime(2016, 2, 15):
            all_data = prepare_data.get_features(True, test_mode, pd.datetime(2016, 1, 15), None)
        elif train_start == pd.datetime(2017, 2, 15):
            all_data = prepare_data.get_features(True, test_mode, pd.datetime(2017, 1, 15), None)
        else:
            all_data = prepare_data.get_features(True, test_mode, pd.datetime(2017, 4, 15), None)
        if test_mode:
            train_end = pd.datetime(2017, 7, 31)
            test_end = pd.datetime(2017, 8, 15)
        else:
            train_end = pd.datetime(2017, 8, 15)
            test_end = pd.datetime(2017, 9, 1)
        train_bool = (all_data['date'] >= train_start) & (all_data['date'] <= train_end)
        exclude_earthquake = pd.date_range(pd.datetime(2016, 4, 1), pd.datetime(2016, 7, 1), freq='D')
        exclude_christmas = pd.date_range(pd.datetime(2016, 12, 15), pd.datetime(2017, 2, 15), freq='D')
        train_bool = train_bool & (~all_data['date'].isin(exclude_earthquake)) \
                     & (~all_data['date'].isin(exclude_christmas))
        test_bool = (all_data['date'] > train_end) & (all_data['date'] <= test_end)
        gbdt_models_v9_deep.run_full_model(test_mode, all_data, train_bool, test_bool)
        gc.collect()
