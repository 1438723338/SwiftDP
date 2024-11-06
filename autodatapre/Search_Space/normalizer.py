import time
import pandas as pd
from sklearn.preprocessing import (MinMaxScaler, quantile_transform)

pd.options.mode.chained_assignment = None


class Normalizer():
    def __init__(self, dataset, strategy='ZS', threshold=None):
        self.dataset = dataset
        self.strategy = strategy
        self.threshold = threshold

    def get_params(self):
        return {'strategy': self.strategy,
                'threshold': self.threshold}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)

    def ZS_normalization(self, dataset):
        d = dataset
        if (type(dataset) != pd.core.series.Series):
            X = dataset.select_dtypes(['number'])
            Y = dataset.select_dtypes(['object'])
            Z = dataset.select_dtypes(['datetime64'])
            for column in X.columns:
                if X[column].std() == 0:
                    X[column] = 1
                else:
                    X[column] -= X[column].mean()
                    X[column] /= X[column].std()
            df = X.join(Y)
            df = df.join(Z)
        else:
            X = dataset
            X -= X.mean()
            X /= X.std()
            df = X
        return df.sort_index()

    def MM_normalization(self, dataset):
        d = dataset
        if (type(dataset) != pd.core.series.Series):
            Xf = dataset.select_dtypes(['number'])
            X = Xf.dropna()
            X_na = Xf[Xf.isnull().any(axis=1)]
            Y = dataset.select_dtypes(['object'])
            Z = dataset.select_dtypes(['datetime64'])
            Scaler = MinMaxScaler()
            scaled_values = Scaler.fit_transform(X)
            scaled_X = pd.DataFrame(
                scaled_values, index=X.index, columns=X.columns)
            scaled_Xf = pd.concat(
                [scaled_X, X_na], ignore_index=False, sort=True).sort_index()
            df = scaled_Xf.join(Y)
            df = df.join(Z)
        else:
            X = dataset.dropna()
            X_na = dataset[dataset.isna()]
            Scaler = MinMaxScaler()
            scaled_X = Scaler.fit_transform(X.values)
            scaled_Xf = pd.concat(
                [scaled_X, X_na], ignore_index=False, sort=True).sort_index()
            df = pd.Series(scaled_Xf, index=X.index, columns=X.columns)
        return df.sort_index()

    def DS_normalization(self, dataset):
        d = dataset
        if (type(dataset) != pd.core.series.Series):
            Xf = dataset.select_dtypes(['number'])
            X = Xf.dropna()
            X_na = Xf[Xf.isnull().any(axis=1)]
            Y = dataset.select_dtypes(['object'])
            Z = dataset.select_dtypes(['datetime64'])
            scaled_values = quantile_transform(
                X, n_quantiles=10, random_state=0)
            scaled_X = pd.DataFrame(
                scaled_values, index=X.index, columns=X.columns)
            scaled_Xf = pd.concat([scaled_X, X_na])
            df = scaled_Xf.join(Y)
            df = df.join(Z)
        else:
            X = dataset.dropna()
            X_na = dataset[dataset.isna()]
            scaled_X = X.quantile(q=0.1, interpolation='linear')
            scaled_Xf = pd.concat(
                [scaled_X, X_na], ignore_index=False, sort=True).sort_index()
            df = pd.Series(scaled_Xf, index=X.index, columns=X.columns)
        return df


    def transform(self):
        normd = self.dataset
        start_time = time.time()
        for key in ['train', 'test']:
            if (not isinstance(self.dataset[key], dict)):
                d = self.dataset[key]
                if (self.strategy == "DS"):
                    dn = self.DS_normalization(d)
                elif (self.strategy == "ZS"):
                    dn = self.ZS_normalization(d)
                elif (self.strategy == "MM"):
                    dn = self.MM_normalization(d)
                normd[key] = dn
            else:
                normd[key] = self.dataset[key]
        return normd
