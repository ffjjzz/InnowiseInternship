import pandas as pd
from sklearn.metrics import root_mean_squared_error

class TimeSeriesRollingValidator:
    def __init__(self, df: pd.DataFrame, time_col: str, train_window: int, test_window: int):
        self.df = df.copy()
        self.time_col = time_col
        self.df = self.df.sort_values(by=time_col)
        self.train_window = train_window
        self.test_window = test_window

    def split_data_rolling(self):
        all_splits = []
        unique_months = self.df[self.time_col].unique()
        unique_months.sort()
        
        min_months_for_split = self.train_window + self.test_window * 2
        

        for i in range(len(unique_months) - min_months_for_split + 1):
            start_index = i
            
            train_months = unique_months[start_index : start_index + self.train_window]
            val_months = unique_months[start_index + self.train_window : start_index + self.train_window + self.test_window]
            test_months = unique_months[start_index + self.train_window + self.test_window : start_index + self.train_window + self.test_window * 2]

            train_set = self.df[self.df[self.time_col].isin(train_months)]
            val_set = self.df[self.df[self.time_col].isin(val_months)]
            test_set = self.df[self.df[self.time_col].isin(test_months)]
            
            all_splits.append((train_set, val_set, test_set))
            
        return all_splits
    

    def validate(self, models: list, splits: list, target_col: str = 'target'):
        results = []    
        for model, split in zip(models, splits):
            train_set, val_set, test_set = split

            X_test = test_set.drop(columns=[target_col])
            y_test = test_set[target_col]

            predictions = model.predict(X_test)
            
            rmse = root_mean_squared_error(y_test, predictions)
            results.append({
                'model': model.__class__.__name__,
                'rmse': rmse,
                'train_months': len(train_set),
                'val_months': len(val_set),
                'test_months': len(test_set)
            })
        return pd.DataFrame(results)
    