from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

class DataPreprocessor:
    def __init__(self, data_storage):
        self.data_storage = data_storage

    def add_month_and_days(self, train: pd.DataFrame, month_column: str = 'date_block_num') -> pd.DataFrame:
        train['month'] = train[month_column] % 12 + 1
        days_in_month_dict = {
            1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30,12: 31
        }
        train['days_in_month'] = train['month'].map(days_in_month_dict)
        return train
    

    def group_by_month(self, train: pd.DataFrame) -> pd.DataFrame:
        index_cols = ['date_block_num', 'shop_id', 'item_id']
        monthly_sales = train.groupby(index_cols).agg(
        item_cnt_month_=('item_cnt_day', 'sum'),
            purch_cnt_month=('item_cnt_day', 'count')
        ).reset_index()
        month_sales = monthly_sales

        month_sales.fillna(0, inplace=True)


        return month_sales
    
    def add_cat_features(self, train: pd.DataFrame) -> pd.DataFrame:
        item_categories = self.data_storage['item_categories']
        
        items = self.data_storage['items']
        shops = self.data_storage['shops']
        
        train = train.merge(items, how='left', on='item_id')
        train = train.merge(item_categories, how ='left', on='item_category_id')
        train = train.merge(shops, how = 'left', on='shop_id')
        return train


    def encode_categorical_features(self, train: pd.DataFrame, cat_features: list) -> pd.DataFrame:
        encode = lambda df, feature: LabelEncoder().fit_transform(df[feature])

        for feature in cat_features:
            train[feature] = encode(train, feature)

        
        return train
    

    def add_lag_features(self, df: pd.DataFrame, col_to_agg: str, group_levels: list, n_lags: list, aggfunc: str = 'sum') -> pd.DataFrame:

        pivot_df = pd.pivot_table(df, values=col_to_agg, index=group_levels, aggfunc=aggfunc).reset_index()
        pivot_df.rename(columns={col_to_agg: f'{col_to_agg}_{aggfunc}'}, inplace=True)
        
        df = df.merge(pivot_df, on=group_levels, how='left')

        for lag in n_lags:
            lag_col_name = f'item_cnt_month_lag_{lag}'
            lag_df = pivot_df.copy()
            lag_df['date_block_num'] += lag
            lag_df.rename(columns={f'{col_to_agg}_{aggfunc}': lag_col_name}, inplace=True)
            
            df = df.merge(lag_df, on=group_levels, how='left')
            df[lag_col_name] = df[lag_col_name].fillna(0).astype('float32')

        df.drop(columns=[f'{col_to_agg}_{aggfunc}'], inplace=True)
        
        return df
    

    def add_item_price_to_test(self, train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:

        last_known_prices = train[train['date_block_num'] == 33].groupby('item_id')['item_price'].mean().reset_index()
        test_df = test.merge(last_known_prices, on='item_id', how='left')
        test_df["item_price"].fillna(test_df["item_price"].mean(), inplace=True)
        return test_df
    
