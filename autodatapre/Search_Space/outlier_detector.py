import numpy as np
from sklearn.neighbors import LocalOutlierFactor


class Outlier_detector():
    def __init__(self, dataset, strategy='ZSB', threshold=0.3,):
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

    def IQR_outlier_detection(self, dataset, threshold):
        X = dataset.select_dtypes(['number'])
        Y = dataset.select_dtypes(['object'])
        if len(X.columns) < 1:
            print(
                "Error: Need at least one numeric variable for LOF"
                "outlier detection\n Dataset inchanged")
        Q1 = X.quantile(0.25)
        Q3 = X.quantile(0.75)
        IQR = Q3 - Q1
        outliers = X[((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR)))]
        to_drop = X[outliers.sum(axis=1) / outliers.shape[1] >
                    threshold].index
        to_keep = set(X.index) - set(to_drop)
        if (threshold == -1):
            X = X[~((X < (Q1 - 1.5 * IQR)) |
                    (X > (Q3 + 1.5 * IQR))).any(axis=1)]
        else:
            X = X.loc[list(to_keep)]
        df = X.join(Y)
        return df

    def ZSB_outlier_detection(self, dataset, threshold):
        X = dataset.select_dtypes(['number'])
        Y = dataset.select_dtypes(['object'])
        if (len(X.columns) < 1):
            df = dataset
        else:
            median = X.apply(np.median, axis=0)
            median_absolute_deviation = 1.4296 * \
                                        np.abs(X - median).apply(np.median, axis=0)
            modified_z_scores = (X - median) / median_absolute_deviation
            outliers = X[np.abs(modified_z_scores) > 1.6]
            to_drop = outliers[(outliers.count(axis=1) /
                                outliers.shape[1]) > threshold].index
            to_keep = [i for i in X.index if i not in to_drop]
            if (threshold == -1):
                X = X[~(np.abs(modified_z_scores) > 1.6).any(axis=1)]
            else:
                X = X.loc[list(to_keep)]
            df = X.join(Y)
        return df

    def LOF_outlier_detection(self, dataset, threshold):
        if dataset.isnull().sum().sum() > 0:
            dataset = dataset.dropna()
        X = dataset.select_dtypes(['number'])
        Y = dataset.select_dtypes(['object'])
        k = int(threshold * 100)
        if len(X.columns) < 1 or len(X) < 1:
            df = dataset
        else:
            clf = LocalOutlierFactor(n_neighbors=4, contamination=0.1)
            clf.fit_predict(X)
            LOF_scores = clf.negative_outlier_factor_
            top_k_idx = np.argsort(LOF_scores)[-k:]
            top_k_values = [LOF_scores[i] for i in top_k_idx]
            data = X[LOF_scores < top_k_values[0]]
            to_drop = X[~(LOF_scores < top_k_values[0])].index
            df = data.join(Y)
        return df

    def transform(self):
        osd = self.dataset
        for key in ['train', 'test']:
            if (not isinstance(self.dataset[key], dict)):
                if not self.dataset[key].empty:
                    d = self.dataset[key]
                    if (self.strategy == "ZSB"):
                        dn = self.ZSB_outlier_detection(d, self.threshold)
                    elif (self.strategy == 'IQR'):
                        dn = self.IQR_outlier_detection(d, self.threshold)
                    elif (self.strategy == "LOF"):
                        dn = self.LOF_outlier_detection(d, self.threshold)
                    osd[key] = dn
        return osd
