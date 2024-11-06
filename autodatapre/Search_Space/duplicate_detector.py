import time
import numpy as np
import jellyfish as jf
import py_stringsimjoin as ssj
import py_stringmatching as sm
import pandas as pd

pd.options.mode.chained_assignment = None

def add_key_reindex(dataset, rand=False):
    if rand:
        dataset = dataset.reindex(np.random.permutation(dataset.index))
    dataset['New_ID'] = range(1, 1 + len(dataset))
    dataset['New_ID'].apply(str)
    return (dataset)


class Duplicate_detector():
    def __init__(self, dataset, strategy='ED', threshold=0.6,
                 metric='DL'):
        self.dataset = dataset
        self.strategy = strategy
        self.threshold = threshold
        self.metric = metric

    def get_params(self):
        return {'strategy': self.strategy,
                'metric': self.metric,
                'threshold': self.threshold,
                }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)

    def ED_Exact_duplicate_removal(self, dataset):
        if not dataset.empty:
            df = dataset.drop_duplicates()
        return df

    def AD_Approx_string_duplicate_removal(self, dataset,
                                           threshold, metric="DL"):
        dataset = add_key_reindex(dataset, rand=True)
        data = dataset.applymap(str)
        data = data.apply(lambda x: '*'.join(x.values.tolist()),
                          axis=1)
        data = data.astype(str)
        data = data.str.replace(" ", "")
        for row in data.index:
            data[row] = data[row].lower()
        out = pd.DataFrame(columns=["Dup_ID1", "Dup_ID2", "Dup_1", "Dup_2"])
        if metric == "DL":
            res = {_d: [] for _d in data}
            for _d in res.keys():
                for row in data.index:
                    if _d != data[row] \
                            and jf.damerau_levenshtein_distance(_d, data[row]) < \
                            ((len(_d) + len(data[row]) / 2) * threshold):
                        res[_d].append(data[row])
                        out.loc[len(out)] = (
                            _d.split("*")[-1], row, _d, data[row])
        elif metric == "LM":
            res = {_d: [] for _d in data}
            for _d in res.keys():
                for row in data.index:
                    if _d != data[row] \
                            and jf.levenshtein_distance(_d, data[row]) < \
                            ((len(_d) + len(data[row]) / 2) * threshold):
                        res[_d].append(data[row])
                        out.loc[len(out)] = (
                            _d.split("*")[-1], row, _d, data[row])
        elif metric == "JW":
            res = {_d: [] for _d in data}
            for _d in res.keys():
                for row in data.index:
                    if _d != data[row] and jf.jaro_winkler(_d, data[row]) > \
                            ((len(_d) + len(data[row]) / 2) * threshold):
                        res[_d].append(data[row])
                        out.loc[len(out)] = (
                            _d.split("*")[-1], row, _d, data[row])
        filtered = {k: v for k, v in res.items() if len(v) > 0}
        out = out[~out[["Dup_ID1", "Dup_ID2"]].apply(
            frozenset, axis=1).duplicated()]
        out.reset_index(drop=True, inplace=True)
        df = dataset[~dataset['New_ID'].isin(out['Dup_ID2'])]
        return df

    def jaccard_similarity(self, dataset, threshold):
        df = add_key_reindex(dataset)
        A = dataset.applymap(str)
        A = A.apply(lambda x: '*'.join(x.values.tolist()), axis=1)
        A = A.astype(str)
        A = A.str.replace(" ", "")
        df['row'] = A
        ssj.profile_table_for_join(df)
        ws = sm.WhitespaceTokenizer(return_set=True)
        try:
            output_pairs = ssj.jaccard_join(df, df, 'New_ID',
                                            'New_ID', 'row', 'row', ws,
                                            threshold, l_out_attrs=['row'],
                                            r_out_attrs=['row'], n_jobs=-1, show_progress=False)
        except Exception as e:
            print(e)
        dup = output_pairs[output_pairs['l_New_ID']
                           != output_pairs['r_New_ID']]
        dataset = df[~df['New_ID'].isin(dup['r_New_ID'])]
        dataset.drop(["New_ID", "row"], axis=1, inplace=True)
        return dataset

    def transform(self):
        dedup = self.dataset
        start_time = time.time()
        for key in ['train', 'test']:
            if (not isinstance(self.dataset[key], dict)):
                if not self.dataset[key].empty:
                    if (self.strategy == "ED"):
                        dn = self.ED_Exact_duplicate_removal(self.dataset[key])
                    elif (self.strategy == "AD"):
                        dn = self.AD_Approx_string_duplicate_removal(
                            self.dataset[key],
                            metric=self.metric,
                            threshold=self.threshold)
                    dedup[key] = dn
        return dedup
