import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import pickle
import os
from . import MetaFeature
class_mapping = {'RAND':0, 'MF':1, 'MICE':2, 'KNN':3, 'EM':4,'MEDIAN':5,'MEAN':6,'DROP':7,
                 'OE':8,'BE':9,'FE':10,'CBE':11,
                 'ZS':12, 'DS':13,'MM':14,
                 'MR':15, 'WR':16, 'LC':17, 'TB':18,
                 'ED':19, 'AD':20,
                 'ZSB':21, 'IQR':22, 'LOF':23,
                 'NB':24, 'LDA':25,'CART':26,'RF':27,'OLS':28,'LASSO':29,'RF_REG':30,'LR':31,'LE':32}

class MultiHeadAttention(nn.Module):
    def __init__(self, query_dim, key_dim, num_units, num_heads):
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
    def forward(self, query, key, mask=None):
        querys = self.W_query(query)
        keys = self.W_key(key)
        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=1), dim=0)
        keys = torch.stack(torch.split(keys, split_size, dim=1), dim=0)
        scores = torch.matmul(querys, keys.transpose(1, 2))
        scores = scores / (self.key_dim ** 0.5)
        scores = F.softmax(scores, dim=1)
        scores1 = scores.transpose(1, 2)
        scores1 = scores1.squeeze(0)
        out1=torch.matmul(scores1,querys)
        out1 = torch.cat(torch.split(out1, 1, dim=0), dim=2).squeeze(0)
        out1=torch.cat((out1,keys.squeeze(0)),1)
        return out1, scores1
def get_Estimate(dataset,choice,taskType):
    if taskType=="REG":
        choice = ['RF_REG' if item == 'RF' else item for item in choice]
    choice=pd.Series(data=choice)
    choice=choice.map(class_mapping)
    matrix = MetaFeature.getfeature(dataset)
    matrix = F.softmax(matrix, dim=0)
    choices = torch.Tensor(choice).unsqueeze(0)
    if taskType=="CLA":
        current_dir = os.path.dirname(__file__)
        csv_path = os.path.join(current_dir, '..',  'datasets', 'dataset','Metafeature.csv')
        csv_path = os.path.normpath(csv_path)
        # meta = pd.read_csv('../datasets/dataset/Metafeature.csv', sep=',', encoding='ISO-8859-1')
        meta = pd.read_csv(csv_path, sep=',', encoding='ISO-8859-1')
    else:
        current_dir = os.path.dirname(__file__)
        csv_path = os.path.join(current_dir, '..',  'datasets', 'dataset','MetafeatureREG.csv')
        csv_path = os.path.normpath(csv_path)
        # meta = pd.read_csv('../datasets/dataset/MetafeatureREG.csv', sep=',', encoding='ISO-8859-1')
        meta = pd.read_csv(csv_path, sep=',', encoding='ISO-8859-1')
    metalen = meta.shape[1]
    attention = MultiHeadAttention(metalen, choices.size(1), 7, 1)
    out1, scores1 = attention(matrix, choices)
    out1=out1.detach().numpy()
    df1=torch.tensor(out1)
    if taskType == "CLA":
        current_dir = os.path.dirname(__file__)
        path = os.path.join(current_dir,  'Estimation_Model', 'model_CLA.pickle')
        path = os.path.normpath(path)
        f = open(path, 'rb')
        rfc1 = pickle.load(f)
        f.close()
        result = rfc1.predict(df1)
    else:
        current_dir = os.path.dirname(__file__)
        path = os.path.join(current_dir,  'Estimation_Model', 'model_REG.pickle')
        path = os.path.normpath(path)
        f = open(path, 'rb')
        rfc1 = pickle.load(f)
        f.close()
        result = 1/rfc1.predict(df1)
    return result

