import time
import numpy as np
import pandas as pd


class Feature_selector():
    def __init__(self, dataset, strategy='LC', threshold=0.3,):
        self.dataset = dataset
        self.strategy = strategy
        self.threshold = threshold

    def get_params(self):
        return {'strategy': self.strategy,
                'threshold': self.threshold
                }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)

    def FS_MR_missing_ratio(self, dataset, missing_threshold=.2):
        missing_series = dataset.isnull().sum() / dataset.shape[0]
        record_missing = pd.DataFrame(missing_series[missing_series >
                                                     missing_threshold]).reset_index(). \
            rename(columns={'index': 'feature', 0: 'missing_fraction'})
        to_drop = list(record_missing['feature'])
        to_keep = set(dataset.columns) - set(to_drop)
        return dataset[list(to_keep)]

    def FS_LC_identify_collinear(self, dataset, correlation_threshold=0.8):
        corr_matrix = dataset.corr()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        to_drop = [column for column in upper.columns if any(
            upper[column].abs() > correlation_threshold)]
        record_collinear = pd.DataFrame(
            columns=['drop_feature', 'corr_feature', 'corr_value'])
        for column in to_drop:
            corr_features = list(
                upper.index[upper[column].abs() > correlation_threshold])
            corr_values = list(
                upper[column][upper[column].abs() > correlation_threshold])
            drop_features = [column for _ in range(len(corr_features))]
            temp_df = pd.DataFrame.from_dict({'drop_feature': drop_features,
                                              'corr_feature': corr_features,
                                              'corr_value': corr_values})
            record_collinear = record_collinear.append(
                temp_df, ignore_index=True)
        to_keep = set(dataset.columns) - set(to_drop)
        return dataset[list(to_keep)]

    def FS_WR_identify_best_subset(self, df_train, df_target, k=10):
        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import chi2
        if df_train.isnull().sum().sum() > 0:
            df_train = df_train.dropna()
            print('WR requires no missing values, so '
                  'missing values have been removed applying '
                  'DROP on the train dataset.')
        X = df_train.select_dtypes(['number'])
        Y = df_target
        if len(df_train.columns) < 1 or len(df_train) < 1:
            df = df_train
        else:
            selector = SelectKBest(score_func=chi2, k=k)
            lsv = list(X.lt(0).sum().values)
            lis = list(X.lt(0).sum().index)
            for i in range(0, len(lsv) - 1):
                if lsv[i] > 0:
                    del lis[i]
            if len(lis) == 0:
                df = df_train
            else:
                X = X[lis]
                selector.fit(X, Y)
                Best_Flist = X.columns[selector.get_support(
                    indices=True)].tolist()
                df = X[Best_Flist]
        return df


    def FS_Tree_based(self, df_train, df_target):
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.feature_selection import SelectFromModel
        if df_train.isnull().sum().sum() > 0:
            df_train = df_train.dropna()
        if len(df_train.columns) < 1 or len(df_train) < 1:
            df = df_train
        else:
            X = df_train.select_dtypes(['number'])
            Y = df_target
            clf = ExtraTreesClassifier(n_estimators=10)
            clf = clf.fit(X, Y.values.ravel())
            model = SelectFromModel(clf, prefit=True)
            Best_Flist = X.columns[model.get_support(indices=True)].tolist()
            df = X[Best_Flist]
        return df

    def transform(self):
        df = self.dataset['train'].copy()
        fsd = self.dataset
        start_time = time.time()
        to_keep = []
        if (self.strategy == "MR"):
            dn = self.FS_MR_missing_ratio(
                df, missing_threshold=self.threshold)
        elif (self.strategy == "LC"):
            d = df.select_dtypes(['number'])
            do = df.select_dtypes(exclude=['number'])
            dn = self.FS_LC_identify_collinear(
                d, correlation_threshold=self.threshold)
            dn = dn.join(do)
        elif (not isinstance(self.dataset['target'], dict)):
            dn = df.select_dtypes(['number'])
            if dn.isnull().sum().sum() > 0:
                dn = dn.dropna()
                dt = self.dataset['target'].loc[dn.index]
            else:
                dt = self.dataset['target'].loc[dn.index]
            if (self.strategy == "TB"):
                dn = self.FS_Tree_based(dn, dt)
            elif (self.strategy == "WR"):
                if len(self.dataset['train'].columns) > 10:
                    dn = self.FS_WR_identify_best_subset(dn, dt)
                else:
                    dn = self.dataset['train'].copy()
            else:
                dn = self.dataset['train'].copy()
        else:
            dn = self.dataset['train'].copy()
        to_keep = [column for column in dn.columns]
        fsd['train'] = dn[to_keep]
        if (not isinstance(self.dataset['test'], dict)):
            df_test = self.dataset['test']
            fsd['test'] = df_test[to_keep]
        return fsd
