from sklearn.model_selection import train_test_split
from ..Search_Space import classifier, encoding, imputer, feature_selector, regressor, duplicate_detector, \
    outlier_detector, normalizer
from sklearn.preprocessing import LabelEncoder
list1 = ['imp_null', 'RAND','MF', 'MICE', 'KNN', 'EM','MEDIAN','MEAN','DROP']
list2 = ['enc_null','OE','BE','FE','CBE', 'LE']
list3 = ['nor_null', 'ZS', 'DS','MM']
list4 = ['fea_null', 'MR', 'WR', 'LC', 'TB']
list5 = ['dup_null', 'ED', 'AD']
list6 = ['out_null', 'ZSB', 'IQR', 'LOF']
def read_dataset(df,datasettarget):
    df[datasettarget]=LabelEncoder().fit_transform(df[datasettarget])
    X=df.drop([datasettarget],axis=1)
    Y=df[[datasettarget]]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    return {"train": x_train,
            "test": x_test,
            "target": y_train,
            "target_test": y_test}
def choose_normalizer(dataset,id):
    nor = normalizer.Normalizer(dataset, strategy=id,).transform()
def choose_encoding(dataset,id):
    enc = encoding.Encoding(dataset, strategy=id).transform()
def choose_feature(dataset,id):
    fea = feature_selector.Feature_selector(dataset, strategy=id).transform()
def choose_imputer(dataset,id):
    imp = imputer.Imputer(dataset, strategy=id).transform()
def choose_duolicate(dataset,id):
    dup = duplicate_detector.Duplicate_detector(dataset, strategy=id).transform()
def choose_outlier(dataset,id):
    out = outlier_detector.Outlier_detector(dataset, strategy=id, threshold=0.5).transform()
def choose_classifier(dataset,id,target):
    CLA = classifier.Classifier(dataset, target=target, k_folds=10, strategy=id).transform()
    return CLA
def choose_regressor(dataset,id,target):
    reg = regressor.Regressor(dataset, target=target, k_folds=10, strategy=id).transform()
    return reg
def getAcc(dataset,order,target):
    for i in range(1,len(order)):
        if order[i] in list1:
            choose_imputer(dataset,order[i])
        elif order[i] in list2:
            choose_encoding(dataset,order[i])
        elif order[i] in list3:
            choose_normalizer(dataset,order[i])
        elif order[i] in list4:
            choose_feature(dataset,order[i])
        elif order[i] in list5:
            choose_duolicate(dataset,order[i])
        elif order[i] in list6:
            choose_outlier(dataset,order[i])
    CLA = choose_classifier(dataset,order[0],target)
    return CLA.get('quality_metric')
def getMse(dataset,order,target):
    try:
        for i in range(1, len(order)):
            if order[i] in list1:
                choose_imputer(dataset, order[i])
            elif order[i] in list2:
                choose_encoding(dataset, order[i])
            elif order[i] in list3:
                choose_normalizer(dataset, order[i])
            elif order[i] in list4:
                choose_feature(dataset, order[i])
            elif order[i] in list5:
                choose_duolicate(dataset, order[i])
            elif order[i] in list6:
                choose_outlier(dataset, order[i])
        reg = choose_regressor(dataset, order[0], target)
        Mse=reg.get('quality_metric')
    except Exception as e:
        import sys
        import traceback
        print(e)
        print(sys.exc_info())
        Mse=-1
    return 1/Mse
def getdataset(dataset,order,target):
    for i in range(1,len(order)):
        if order[i] in list1:
            choose_imputer(dataset,order[i])
        elif order[i] in list2:
            choose_encoding(dataset,order[i])
        elif order[i] in list3:
            choose_normalizer(dataset,order[i])
        elif order[i] in list4:
            choose_feature(dataset,order[i])
        elif order[i] in list5:
            choose_duolicate(dataset,order[i])
        elif order[i] in list6:
            choose_outlier(dataset,order[i])
    choose_classifier(dataset,order[0],target)
    return dataset

