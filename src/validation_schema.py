import pandas as pd
from sklearn.dummy import DummyRegressor
import matplotlib.pyplot as plt
import numpy as np


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
    

    def validate(self, model_learning_function, metrics: dict, splits: list, target_col: str = 'target', dummy_model=DummyRegressor):
        results = []    
        for split in splits:
            train_set, val_set, test_set = split
            model = model_learning_function(train_set, val_set, test_set)



            statistics = {
                'model': model.__class__.__name__,
                'train_months': len(train_set),
                'val_months': len(val_set),
                'test_months': len(test_set), 

            }

            dummy = dummy_model()
            dummy.fit(train_set.drop(columns= [target_col]), train_set[target_col])

            dummy_test_out = dummy.predict(test_set.drop(columns=[target_col]))
            dummy_val_out = dummy.predict(val_set.drop(columns=[target_col]))
            dummy_train_out = dummy.predict(train_set.drop(columns=[target_col]))

            statistics.update({f'dummy_{name}_test' : metrics[name](dummy_test_out, test_set[target_col]) for name in metrics})
            statistics.update({f'dummy_{name}_val' : metrics[name](dummy_val_out, val_set[target_col]) for name in metrics})
            statistics.update({f'dummy_{name}_train' : metrics[name](dummy_train_out, train_set[target_col]) for name in metrics})

            model_test_out = model.predict(test_set.drop(columns=[target_col]))
            model_val_out = model.predict(val_set.drop(columns=[target_col]))
            model_train_out = model.predict(train_set.drop(columns=[target_col]))

            statistics.update({f'model_{name}_test' : metrics[name](model_test_out, test_set[target_col]) for name in metrics})
            statistics.update({f'model_{name}_val' : metrics[name](model_val_out, val_set[target_col]) for name in metrics})
            statistics.update({f'model_{name}_train' : metrics[name](model_train_out, train_set[target_col]) for name in metrics})


            results.append(statistics)

        return pd.DataFrame(results)
    


    def visualize_validation_results_with_linear_trend(self, validation_results, metrics: str = 'RMSE'):
        plt.figure(figsize=(12, 7))
        plt.title(f'Metric {metrics} dynamics during validation with linear trend')
        plt.xlabel('Validation Iteration')
        plt.ylabel(f'Metric {metrics} value')

        train_data = np.array(validation_results[f"model_{metrics}_train"])
        val_data = np.array(validation_results[f"model_{metrics}_val"])
        test_data = np.array(validation_results[f"model_{metrics}_test"])
        
        iterations = np.arange(len(train_data))

        def plot_linear_trend(data, label, color):
            import numpy as np
            coefficients = np.polyfit(iterations, data, 1)
            trend_line = np.poly1d(coefficients)(iterations)
            

            plt.plot(iterations, data, label=label, color=color, alpha=0.3)
            plt.plot(iterations, trend_line, linestyle='--', color=color, linewidth=2, label=f'Linear Trend ({label})')

        plot_linear_trend(train_data, 'Training Data', 'C0')
        plot_linear_trend(val_data, 'Validation Data', 'C1')
        plot_linear_trend(test_data, 'Testing Data', 'C2')
        
        plt.legend()
        plt.grid(True)
        plt.show()