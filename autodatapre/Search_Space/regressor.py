import time
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import skew
import statsmodels.api as sm
import numpy as np

np.seterr(divide='ignore', invalid='ignore')


def LT_log_transform_skew_features(dataset):
    numeric_feats = dataset.dtypes[dataset.dtypes != "object"].index
    Y = dataset.select_dtypes(['object'])
    skewed_feats = dataset[numeric_feats].apply(
        lambda x: skew(x.dropna()))
    skewed_feats = skewed_feats[skewed_feats >= 0.75]
    skewed_feats = skewed_feats.index
    dataset[skewed_feats] = np.log1p(dataset[skewed_feats])
    return dataset[skewed_feats].join(Y)


class Regressor():
    def __init__(self, dataset, target, strategy='LASSO',
                 k_folds=10,):
        self.dataset = dataset
        self.target = target
        self.strategy = strategy
        self.k_folds = k_folds

    def get_params(self):
        return {'strategy': self.strategy,
                'target': self.target,
                'k_folds': self.k_folds}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)

    def OLS_regression(self, dataset, target):
        k = self.k_folds
        X_train = dataset['train'].select_dtypes(['number']).dropna()
        X_test = dataset['test'].select_dtypes(['number']).dropna()
        y_train = dataset['target'].loc[X_train.index]
        y_test = dataset['target_test']
        if (len(X_train.columns) < 1) or (len(X_train) < k):
            mse = None
        else:
            X1Train = sm.add_constant(X_train, has_constant='add')
            reg = sm.OLS(y_train, X1Train)
            resReg = reg.fit()
            X1Test = sm.add_constant(X_test, has_constant='add')
            ypReg = reg.predict(resReg.params, X1Test)
            if len(y_test) == 0:
                mse = resReg.mse_total
            else:
                y_test = dataset['target_test'].loc[X_test.index]
                mse = mean_squared_error(y_test, ypReg)
        return mse

    def LASSO_regression(self, dataset, target):
        k = self.k_folds
        X_train = dataset['train'].select_dtypes(['number']).dropna()
        X_test = dataset['test'].select_dtypes(['number']).dropna()
        y_test = dataset['target_test']
        y_train = dataset['target'].loc[X_train.index]
        if (len(X_train.columns) < 1) or (len(X_train) < k):
            mse = None
        else:
            my_alphas = np.array(
                [0.001, 0.01, 0.02, 0.025, 0.05, 0.1,
                 0.25, 0.5, 0.8, 1.0, 1.2])
            try:
                lcv = LassoCV(alphas=my_alphas,
                              fit_intercept=False, random_state=0,
                              cv=k, tol=0.0001)
            except Exception as e:
                print(e)
            lcv.fit(X_train, y_train)
            avg_mse = np.mean(lcv.mse_path_, axis=1)
            if len(y_test) == 0:
                mse = min(avg_mse)
            else:
                y_test = dataset['target_test'].loc[X_test.index]
                ypLasso = lcv.predict(X_test)
                mse = mean_squared_error(y_test, ypLasso)
        return mse

    def RF_regression(self, dataset, target):
        k = self.k_folds
        X_train = dataset['train'].select_dtypes(['number']).dropna()
        X_test = dataset['test'].select_dtypes(['number']).dropna()
        y_test = dataset['target_test']
        y_train = dataset['target'].loc[X_train.index]
        if (len(X_train.columns) < 1) or (len(X_train) < k):
            mse = None
        else:
            clf = RandomForestRegressor()
            rf = clf.fit(X_train, y_train.values.ravel())
            y_pred = rf.predict(X_test)
            y_test = dataset['target_test'].loc[X_test.index]
            mse = mean_squared_error(y_test, y_pred)
        return mse

    def transform(self):
        start_time = time.time()
        d = self.dataset
        if (self.strategy == "OLS"):
            dn = self.OLS_regression(d, self.target)
        elif (self.strategy == "LASSO"):
            dn = self.LASSO_regression(d, self.target)
        elif (self.strategy == "RF"):
            dn = self.RF_regression(d, self.target)
        return {'quality_metric': dn}
