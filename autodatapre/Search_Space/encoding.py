# import pandas as pd
# from sklearn.preprocessing import OrdinalEncoder

# pd.options.mode.chained_assignment = None


# class Encoding():
#     def __init__(self, dataset, strategy='OE',threshold=None):
#         self.dataset = dataset
#         self.strategy = strategy
#         self.threshold = threshold

#     def get_params(self):
#         return {'strategy': self.strategy,
#                 'threshold': self.threshold}

#     def set_params(self, **params):
#         for k, v in params.items():
#             setattr(self, k, v)

#     def ordinal_encoding(self, dataset):
#         X = dataset.select_dtypes(['object'])
#         if X.empty:
#             return dataset
#         else:
#             Y = dataset.select_dtypes(['number'])
#             Z = dataset.select_dtypes(['datetime64'])
#             encoding_values = OrdinalEncoder().fit_transform(X)
#             encoding_X = pd.DataFrame(
#                 encoding_values, index=X.index, columns=X.columns)
#             df = encoding_X.join(Y)
#             df = df.join(Z)
#             return df

#     def binary_encoding(self, dataset):
#         import category_encoders as ce
#         X = dataset.select_dtypes(['object'])
#         if X.empty:
#             return dataset
#         else:
#             Y = dataset.select_dtypes(['number'])
#             Z = dataset.select_dtypes(['datetime64'])
#             encoding_values = ce.BinaryEncoder().fit_transform(X)
#             encoding_X = pd.DataFrame(
#                 encoding_values, index=X.index, columns=encoding_values.columns)
#             df = encoding_X.join(Y)
#             df = df.join(Z)
#             return df

#     def frequency_encoding(self, dataset):
#         X = dataset.select_dtypes(['object'])
#         if X.empty:
#             return dataset
#         else:
#             Y = dataset.select_dtypes(['number'])
#             Z = dataset.select_dtypes(['datetime64'])
#             for x in X:
#                 fe = X.groupby(x).size() / len(X)
#                 X.loc[:, x] = X[x].map(fe)
#             df = X.join(Y)
#             df = df.join(Z)
#             return df

#     def CatBoost_encoding(self, d_train, d_target):
#         import category_encoders as encoders
#         X = d_train.select_dtypes(['object'])
#         if X.empty:
#             return d_train
#         else:
#             Y = d_train.select_dtypes(['number'])
#             Z = d_train.select_dtypes(['datetime64'])
#             target = d_target
#             enc = encoders.CatBoostEncoder()
#             obtained = enc.fit_transform(X, target)
#             obtained = obtained.join(Y)
#             obtained = obtained.join(Z)
#             return obtained

#     def transform(self):
#         normd = self.dataset
#         X = pd.concat([self.dataset['train'], self.dataset['test']], axis=0)
#         trainlen = len(self.dataset['train'])
#         totallen = len(X)
#         if (self.strategy == "OE"):
#             dn = self.ordinal_encoding(X)
#         elif (self.strategy == "BE"):
#             dn = self.binary_encoding(X)
#         elif (self.strategy == "FE"):
#             dn = self.frequency_encoding(X)
#         elif (self.strategy == "CBE"):
#             target = pd.concat([self.dataset['target'], self.dataset['target_test']], axis=0)
#             dn = self.CatBoost_encoding(X, target)
#         normd['train'] = dn.head(trainlen)
#         normd['test'] = dn.tail(totallen - trainlen)
#         return normd



import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

pd.options.mode.chained_assignment = None


class Encoding():
    def __init__(self, dataset, strategy='OE', threshold=None):
        self.dataset = dataset
        self.strategy = strategy
        self.threshold = threshold

    def get_params(self):
        return {'strategy': self.strategy,
                'threshold': self.threshold}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)

    def ordinal_encoding(self, dataset):
        X = dataset.select_dtypes(['object'])
        if X.empty:
            return dataset
        else:
            Y = dataset.select_dtypes(['number'])
            Z = dataset.select_dtypes(['datetime64'])
            encoding_values = OrdinalEncoder().fit_transform(X)
            encoding_X = pd.DataFrame(
                encoding_values, index=X.index, columns=X.columns)
            df = encoding_X.join(Y)
            df = df.join(Z)
            return df

    def binary_encoding(self, dataset):
        import category_encoders as ce
        X = dataset.select_dtypes(['object'])
        if X.empty:
            return dataset
        else:
            Y = dataset.select_dtypes(['number'])
            Z = dataset.select_dtypes(['datetime64'])
            encoding_values = ce.BinaryEncoder().fit_transform(X)
            encoding_X = pd.DataFrame(
                encoding_values, index=X.index, columns=encoding_values.columns)
            df = encoding_X.join(Y)
            df = df.join(Z)
            return df

    def frequency_encoding(self, dataset):
        X = dataset.select_dtypes(['object'])
        if X.empty:
            return dataset
        else:
            Y = dataset.select_dtypes(['number'])
            Z = dataset.select_dtypes(['datetime64'])
            for x in X:
                fe = X.groupby(x).size() / len(X)
                X.loc[:, x] = X[x].map(fe)
            df = X.join(Y)
            df = df.join(Z)
            return df

    def CatBoost_encoding(self, d_train, d_target):
        import category_encoders as encoders
        X = d_train.select_dtypes(['object'])
        if X.empty:
            return d_train
        else:
            Y = d_train.select_dtypes(['number'])
            Z = d_train.select_dtypes(['datetime64'])
            target = d_target
            enc = encoders.CatBoostEncoder()
            obtained = enc.fit_transform(X, target)
            obtained = obtained.join(Y)
            obtained = obtained.join(Z)
            return obtained

    def label_encoding(self, dataset):
        X = dataset.select_dtypes(['object'])
        if X.empty:
            return dataset
        else:
            Y = dataset.select_dtypes(['number'])
            Z = dataset.select_dtypes(['datetime64'])
            label_enc = LabelEncoder()
            for column in X.columns:
                X[column] = label_enc.fit_transform(X[column])
            df = X.join(Y)
            df = df.join(Z)
            return df

    def transform(self):
        normd = self.dataset
        X = pd.concat([self.dataset['train'], self.dataset['test']], axis=0)
        trainlen = len(self.dataset['train'])
        totallen = len(X)
        if (self.strategy == "OE"):
            dn = self.ordinal_encoding(X)
        elif (self.strategy == "BE"):
            dn = self.binary_encoding(X)
        elif (self.strategy == "FE"):
            dn = self.frequency_encoding(X)
        elif (self.strategy == "CBE"):
            target = pd.concat([self.dataset['target'], self.dataset['target_test']], axis=0)
            dn = self.CatBoost_encoding(X, target)
        elif (self.strategy == "LE"):
            dn = self.label_encoding(X)
        normd['train'] = dn.head(trainlen)
        normd['test'] = dn.tail(totallen - trainlen)
        return normd

