import os
import pandas as pd
import torch
from . import MetaFeature  

# 定义默认 pipeline 任务类型顺序
default_pipeline = {
    "Imputation": 'imp_null',
    "Encoding": 'enc_null',
    "Normalization": 'nor_null',
    "Feature Selection": 'fea_null',
    "Duplication": 'dup_null',
    "Outlier Exclusion": 'out_null',
    "Classification": 'cla_null'
}

# 定义每个任务的算法列表
list1 = ['RAND', 'MF', 'MICE', 'KNN', 'EM', 'MEDIAN', 'MEAN', 'DROP']   # Imputation
list2 = ['OE', 'BE', 'FE', 'CBE', 'LE']                                 # Encoding
list3 = ['ZS', 'DS', 'MM']                                              # Normalization
list4 = ['MR', 'WR', 'LC', 'TB']                                        # Feature Selection
list5 = ['ED', 'AD']                                                    # Duplication
list6 = ['ZSB', 'IQR', 'LOF']                                           # Outlier Exclusion
list7 = ['NB', 'LDA', 'RF', 'LR']                                       # Classification

# 将原始 pipeline 转换为完整的 pipeline
def complete_pipeline(raw_pipeline):
    # 创建一个包含默认值的 pipeline
    complete_pipeline = default_pipeline.copy()
    
    # 遍历每个给定的原始步骤
    for step in raw_pipeline:
        # 根据步骤所在的列表来确定它属于哪一类
        if step in list1:
            complete_pipeline["Imputation"] = step
        elif step in list2:
            complete_pipeline["Encoding"] = step
        elif step in list3:
            complete_pipeline["Normalization"] = step
        elif step in list4:
            complete_pipeline["Feature Selection"] = step
        elif step in list5:
            complete_pipeline["Duplication"] = step
        elif step in list6:
            complete_pipeline["Outlier Exclusion"] = step
        elif step in list7:
            complete_pipeline["Classification"] = step
    
    # 返回按顺序的完整 pipeline
    return [
        complete_pipeline["Imputation"],
        complete_pipeline["Encoding"],
        complete_pipeline["Normalization"],
        complete_pipeline["Feature Selection"],
        complete_pipeline["Duplication"],
        complete_pipeline["Outlier Exclusion"],
        complete_pipeline["Classification"]
    ]


def UpdateMetaData(df, detail_result, metafeature_path, label_path):
    # 从 detail_result 中提取所需的信息
    times = detail_result[0]
    accuracies = detail_result[1]
    dataset_name = detail_result[2]
    target = detail_result[3]
    raw_pipeline = detail_result[4]
    pipeline = complete_pipeline(raw_pipeline)

    # 计算当前数据集的元特征矩阵
    Matrix = torch.zeros((7, 1), dtype=torch.float64)
    matrix = MetaFeature.getfeature(df)  
    matrix = matrix.T
    for i in range(7):
        Matrix[i] = torch.mean(matrix[i])
    Matrix = Matrix.T

    # 将当前数据集的元特征添加到 Metafeature.csv 文件
    metafeature_df = pd.read_csv(metafeature_path, sep=',', encoding='ISO-8859-1')
    new_metafeature = pd.DataFrame(Matrix.numpy(), columns=metafeature_df.columns)
    metafeature_df = pd.concat([metafeature_df, new_metafeature], ignore_index=True)
    metafeature_df.to_csv(metafeature_path, sep=',', encoding='ISO-8859-1', index=False)

    # 更新 label.csv 文件
    label_df = pd.read_csv(label_path, sep=',', encoding='ISO-8859-1')
    new_id = label_df['Id'].max() + 1 if not label_df.empty else 1  # 为新记录生成 ID
    new_label = {
        'Id': new_id,
        'DatasetName': dataset_name,                # 数据集名称
        'Target': target,                           # 目标变量的名称
        'Pipeline': ','.join(pipeline),             # 完整 pipeline
        'EvaluationMetric': accuracies[-1],         # 使用 accuracies 列表的最后一个值作为最终准确率
        'Time': times[-1],                          # 使用 times 列表的最后一个值作为最终运行时间
        'Size': f"({df.shape[0]}, {df.shape[1]})",  # 数据集大小
        'Website': 'https'                          # 若有实际链接可更新
    }
    label_df = label_df.append(new_label, ignore_index=True)
    
    # 保存更新后的 label.csv 文件
    label_df.to_csv(label_path, sep=',', encoding='ISO-8859-1', index=False)

    print("Incremental update is complete")
