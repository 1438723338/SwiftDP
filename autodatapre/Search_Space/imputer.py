import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype


class Imputer():
    def __init__(self, dataset, strategy='DROP', threshold=None):
        self.dataset = dataset
        self.strategy = strategy
        self.threshold = threshold

    def get_params(self, deep=True):
        return {'strategy': self.strategy,
                'threshold': self.threshold}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)

    def mean_imputation(self, dataset):
        df = dataset
        if dataset.select_dtypes(['number']).isnull().sum().sum() > 0:
            X = dataset.select_dtypes(['number'])
            for i in X.columns:
                X[i] = X[i].fillna(int(X[i].mean()))
            Z = dataset.select_dtypes(exclude=['number'])
            df = pd.DataFrame.from_records(
                X, index=dataset.index, columns=dataset.select_dtypes(['number']).columns)
            df = pd.concat([X, Z], axis=1)
        else:
            pass
        return df

    def median_imputation(self, dataset):
        df = dataset
        if dataset.select_dtypes(['number']).isnull().sum().sum() > 0:
            X = dataset.select_dtypes(['number'])
            for i in X.columns:
                X[i] = X[i].fillna(int(X[i].median()))
            Z = dataset.select_dtypes(include=['object'])
            df = pd.DataFrame.from_records(
                X, index=dataset.index, columns=dataset.select_dtypes(['number']).columns)
            df = df.join(Z)
        else:
            pass
        return df

    def NaN_drop(self, dataset):
        return dataset.dropna()

    def MF_most_frequent_imputation(self, dataset):
        for i in dataset.columns:
            mfv = dataset[i].value_counts()
            try:
                mfv = mfv.idxmax()
            except Exception as e:
                mfv = 0
            dataset[i] = dataset[i].replace(np.nan, mfv)
        return dataset

    def NaN_random_replace(self, dataset):
        M = len(dataset.index)
        N = len(dataset.columns)
        ran = pd.DataFrame(np.random.randn(
            M, N), columns=dataset.columns, index=dataset.index)
        dataset.update(ran)
        return dataset

    def KNN_imputation(self, dataset, k=4):
        from impyute.imputation.cs import fast_knn
        df = dataset
        if dataset.select_dtypes(['number']).isnull().sum().sum() > 0:
            X = dataset.select_dtypes(['number']).values
            X = fast_knn(X, 4)
            Z = dataset.select_dtypes(include=['object'])
            df = pd.DataFrame.from_records(
                X, index=dataset.index, columns=dataset.select_dtypes(['number']).columns)
            df = df.join(Z)
        else:
            pass
        return df

    def MICE_imputation(self, dataset):
        from impyute.imputation.cs import mice
        df = dataset
        if dataset.select_dtypes(['number']).isnull().sum().sum() > 0:
            X = dataset.select_dtypes(['number']).values
            X = mice(X)
            Z = dataset.select_dtypes(include=['object'])
            df = pd.DataFrame.from_records(
                X, index=dataset.index, columns=dataset.select_dtypes(['number']).columns)
            df = df.join(Z)
        else:
            pass
        return df

    def EM_imputation(self, dataset):
        import impyute as imp
        df = dataset
        if dataset.select_dtypes(['number']).isnull().sum().sum() > 0:
            X = imp.em(dataset.select_dtypes(['number']).iloc[:, :].values)
            Z = dataset.select_dtypes(include=['object'])
            df = pd.DataFrame.from_records(
                X, index=dataset.index, columns=dataset.select_dtypes(['number']).columns)
            df = df.join(Z)
        else:
            pass
        return df

    def Fill_zero(self, dataset):
        for i in dataset.columns:
            if is_string_dtype(dataset[i]):
                dataset[i] = dataset[i].replace(np.nan, 'Null')
            else:
                dataset[i] = dataset[i].replace(np.nan, 0)
        return dataset

    def transform(self):
        impd = self.dataset
        for key in ['train', 'test']:
            if (not isinstance(self.dataset[key], dict)):
                d = self.dataset[key].copy()
                if (self.strategy == "EM"):
                    dn = self.EM_imputation(d)
                elif (self.strategy == "MICE"):
                    dn = self.MICE_imputation(d)
                elif (self.strategy == "KNN"):
                    dn = self.KNN_imputation(d)
                elif (self.strategy == "RAND"):
                    dn = self.NaN_random_replace(d)
                elif (self.strategy == "MF"):
                    dn = self.MF_most_frequent_imputation(d)
                elif (self.strategy == "MEAN"):
                    dn = self.mean_imputation(d)
                elif (self.strategy == "MEDIAN"):
                    dn = self.median_imputation(d)
                elif (self.strategy == "DROP"):
                    dn = self.NaN_drop(d)
                elif (self.strategy == "FillZero"):
                    dn = self.Fill_zero(d)
                impd[key] = dn
        return impd
