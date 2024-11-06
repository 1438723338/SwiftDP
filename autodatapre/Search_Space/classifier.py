# from sklearn.model_selection import GridSearchCV, StratifiedKFold
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.naive_bayes import GaussianNB
# from sklearn.ensemble import RandomForestClassifier
# import numpy as np

# np.seterr(divide='ignore', invalid='ignore')

# class Classifier():
#     def __init__(self, dataset, target, strategy='NB', k_folds=10):
#         self.dataset = dataset
#         self.target = target
#         self.strategy = strategy
#         self.k_folds = k_folds

#     def get_params(self):
#         return {'strategy': self.strategy,
#                 'target': self.target,
#                 'k_folds': self.k_folds}

#     def set_params(self, **params):
#         for k, v in params.items():
#             setattr(self, k, v)

#     def LDA_classification(self, dataset, target):
#         k = self.k_folds
#         X_train = dataset['train'].select_dtypes(['number']).dropna()
#         if (len(X_train.columns) < 1) or (len(X_train) < k):
#             accuracy = None
#         else:
#             y_train = dataset['target'].loc[X_train.index]
#             X_test = dataset['test'].select_dtypes(['number']).dropna()
#             if (isinstance(self.dataset['target_test'], dict)):
#                 y_test = dataset['target'].loc[X_test.index]
#             else:
#                 y_test = dataset['target_test']
#             if target in X_train.columns.values:
#                 X_train = X_train.drop([target], 1)
#             if target in X_test.columns.values:
#                 X_test = X_test.drop([target], 1)
#             if ((dataset['target'].nunique().values[0]) < k):
#                 k = dataset['target'].nunique().values[0]
#             skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=1)
#             params = {}
#             model = LinearDiscriminantAnalysis(n_components=1)
#             gs = GridSearchCV(
#                 model, cv=skf, param_grid=params, scoring='accuracy')
#             gs.fit(X_train, y_train.values.ravel())
#             results = gs.cv_results_
#             clf = gs.best_estimator_
#             best_index = np.nonzero(results['rank_test_score'] == 1)[0][0]
#             accuracy = results['mean_test_score'][best_index]
#             if target in X_test.columns.values:
#                 accuracy = clf.score(X_test, y_test)
#         return accuracy


#     def NB_classification(self, dataset, target):
#         k = self.k_folds
#         X_train = dataset['train'].select_dtypes(['number']).dropna()
#         if (len(X_train.columns) < 1) or (len(X_train) < k):
#             accuracy = None
#         else:
#             y_train = dataset['target'].loc[X_train.index]
#             X_test = dataset['test'].select_dtypes(['number']).dropna()
#             y_test = dataset['target_test'].loc[X_test.index]
#             if target in X_train.columns.values:
#                 X_train = X_train.drop([target], 1)
#             if target in X_test.columns.values:
#                 X_test = X_test.drop([target], 1)
#             skf = StratifiedKFold(n_splits=k)
#             model = GaussianNB().fit(X_train, y_train.values.ravel())
#             accuracy = model.score(X_test, y_test)
#         return accuracy

#     def RF_classification(self, dataset, target):
#         NUM_TRIALS = 10
#         k = self.k_folds
#         X_train = dataset['train'].select_dtypes(['number']).dropna()
#         if (len(X_train.columns) < 1) or (len(X_train) < k):
#             accuracy = None
#         else:
#             y_train = dataset['target'].loc[X_train.index]
#             X_test = dataset['test'].select_dtypes(['number']).dropna()
#             y_test = dataset['target_test'].loc[X_test.index]
#             if target in X_train.columns.values:
#                 X_train = X_train.drop([target], 1)
#             if target in X_test.columns.values:
#                 X_test = X_test.drop([target], 1)
#             model = RandomForestClassifier()
#             model.fit(X_train, y_train.values.ravel())
#             accuracy = model.score(X_test, y_test)
#         return accuracy

#     def transform(self):
#         d = self.dataset
#         if (self.strategy == "LDA"):
#             dn = self.LDA_classification(dataset=d, target=self.target)
#         elif (self.strategy == "NB"):
#             dn = self.NB_classification(dataset=d, target=self.target)
#         elif (self.strategy == "RF"):
#             dn = self.RF_classification(dataset=d, target=self.target)
#         return {'quality_metric': dn}



from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

np.seterr(divide='ignore', invalid='ignore')

class Classifier():
    def __init__(self, dataset, target, strategy='NB', k_folds=10):
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

    def LDA_classification(self, dataset, target):
        k = self.k_folds
        X_train = dataset['train'].select_dtypes(['number']).dropna()
        if (len(X_train.columns) < 1) or (len(X_train) < k):
            accuracy = None
        else:
            y_train = dataset['target'].loc[X_train.index]
            X_test = dataset['test'].select_dtypes(['number']).dropna()
            if (isinstance(self.dataset['target_test'], dict)):
                y_test = dataset['target'].loc[X_test.index]
            else:
                y_test = dataset['target_test']
            if target in X_train.columns.values:
                X_train = X_train.drop([target], 1)
            if target in X_test.columns.values:
                X_test = X_test.drop([target], 1)
            if ((dataset['target'].nunique().values[0]) < k):
                k = dataset['target'].nunique().values[0]
            skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=1)
            params = {}
            model = LinearDiscriminantAnalysis(n_components=1)
            gs = GridSearchCV(
                model, cv=skf, param_grid=params, scoring='accuracy')
            gs.fit(X_train, y_train.values.ravel())
            results = gs.cv_results_
            clf = gs.best_estimator_
            best_index = np.nonzero(results['rank_test_score'] == 1)[0][0]
            accuracy = results['mean_test_score'][best_index]
            if target in X_test.columns.values:
                accuracy = clf.score(X_test, y_test)
        return accuracy


    def NB_classification(self, dataset, target):
        k = self.k_folds
        X_train = dataset['train'].select_dtypes(['number']).dropna()
        if (len(X_train.columns) < 1) or (len(X_train) < k):
            accuracy = None
        else:
            y_train = dataset['target'].loc[X_train.index]
            X_test = dataset['test'].select_dtypes(['number']).dropna()
            y_test = dataset['target_test'].loc[X_test.index]
            if target in X_train.columns.values:
                X_train = X_train.drop([target], 1)
            if target in X_test.columns.values:
                X_test = X_test.drop([target], 1)
            skf = StratifiedKFold(n_splits=k)
            model = GaussianNB().fit(X_train, y_train.values.ravel())
            accuracy = model.score(X_test, y_test)
        return accuracy

    def RF_classification(self, dataset, target):
        NUM_TRIALS = 10
        k = self.k_folds
        X_train = dataset['train'].select_dtypes(['number']).dropna()
        if (len(X_train.columns) < 1) or (len(X_train) < k):
            accuracy = None
        else:
            y_train = dataset['target'].loc[X_train.index]
            X_test = dataset['test'].select_dtypes(['number']).dropna()
            y_test = dataset['target_test'].loc[X_test.index]
            if target in X_train.columns.values:
                X_train = X_train.drop([target], 1)
            if target in X_test.columns.values:
                X_test = X_test.drop([target], 1)
            model = RandomForestClassifier()
            model.fit(X_train, y_train.values.ravel())
            accuracy = model.score(X_test, y_test)
        return accuracy

    def LR_classification(self, dataset, target):
        k = self.k_folds
        X_train = dataset['train'].select_dtypes(['number']).dropna()
        if (len(X_train.columns) < 1) or (len(X_train) < k):
            accuracy = None
        else:
            y_train = dataset['target'].loc[X_train.index]
            X_test = dataset['test'].select_dtypes(['number']).dropna()
            y_test = dataset['target_test'].loc[X_test.index]
            if target in X_train.columns.values:
                X_train = X_train.drop([target], 1)
            if target in X_test.columns.values:
                X_test = X_test.drop([target], 1)
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train.values.ravel())
            accuracy = model.score(X_test, y_test)
        return accuracy

    def transform(self):
        d = self.dataset
        if (self.strategy == "LDA"):
            dn = self.LDA_classification(dataset=d, target=self.target)
        elif (self.strategy == "NB"):
            dn = self.NB_classification(dataset=d, target=self.target)
        elif (self.strategy == "RF"):
            dn = self.RF_classification(dataset=d, target=self.target)
        elif (self.strategy == "LR"):
            dn = self.LR_classification(dataset=d, target=self.target)
        return {'quality_metric': dn}
