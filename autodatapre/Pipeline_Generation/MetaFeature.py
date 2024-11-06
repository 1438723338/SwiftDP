import pandas as pd
import torch
from pandas.api.types import is_string_dtype
from sklearn import preprocessing

class MetaFeature(object):
    def __init__(self,dataset):
        self.dataset=dataset
        self.matrix = torch.zeros(dataset.shape[1], 7)
    def setMetaFeature(self):
        NumberOfInstances = []
        NumberOfClasses = []
        Isnumeric = []
        Isunique = []
        InstancesMissingRate = []
        Skewnesses = []
        Kurtosis = []
        for i in range(self.dataset.shape[1]):
            data=self.dataset.iloc[:, i]
            NumberOfInstances.append(self.dataset.shape[0])
            temp= list(data.unique())
            NumberOfClasses.append(len(temp))
            Isnumeric.append(is_string_dtype(data))
            Isunique.append(data.is_unique)
            if is_string_dtype(data) or data[0].dtype == bool:
                enc=preprocessing.LabelEncoder()
                data=enc.fit_transform(data)
                data=pd.Series(data)
            InstancesMissingRate.append(data.isnull().sum() / len(data))
            Skewnesses.append(data.skew())
            Kurtosis.append(data.kurt())
        self.matrix[:, 0] = torch.Tensor(NumberOfInstances)
        self.matrix[:, 1] = torch.Tensor(NumberOfClasses)
        self.matrix[:, 2] = torch.Tensor(Isnumeric)
        self.matrix[:, 3] = torch.Tensor(Isunique)
        self.matrix[:, 4] = torch.Tensor(InstancesMissingRate)
        self.matrix[:, 5] = torch.Tensor(Skewnesses)
        self.matrix[:, 6] = torch.Tensor(Kurtosis)

# 计算数据集元特征矩阵（n*7）     
def getfeature(dataset):
    A=MetaFeature(dataset)
    A.setMetaFeature()
    matrix=A.matrix
    return matrix

