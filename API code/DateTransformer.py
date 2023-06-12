from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class DateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['Date'] = pd.to_datetime(X['Date'])
        X['Hour'] = X['Date'].dt.hour
        X['Month'] = X['Date'].dt.month
        X['Day'] = X['Date'].dt.day
        X['DayOfWeek'] = X['Date'].dt.weekday
        X=X.drop(columns = "Date", axis=1)
        
        return X[['Hour','Month', 'Day','DayOfWeek']]

    def get_feature_names_out(self):
        return [('Date', 'Hour'), ('Date', 'Month'), ('Date', 'Day'),('Date', 'DayOfWeek')]