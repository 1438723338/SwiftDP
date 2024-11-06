import os
import sys
import math
import random
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import traceback  
import ipywidgets as widgets
from IPython.display import display, HTML
from . import MCTS_DATA as mctsdata
from . import Estimate_after_profit
from . import MetaFeature

import networkx as nx
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

Mingap = 0.001
Weight = 0.3
Is_BatchTraining = True

list1 = ['RAND', 'MF', 'MICE', 'KNN', 'EM', 'MEDIAN', 'MEAN', 'DROP']
list2 = ['OE', 'BE', 'FE', 'CBE', 'LE']
list3 = ['ZS', 'DS', 'MM']
list4 = ['MR', 'WR', 'LC', 'TB']
list5 = ['ED', 'AD']
list6 = ['ZSB', 'IQR', 'LOF']
list7 = ['NB', 'LDA', 'RF', 'LR']
list8 = ['OLS', 'LASSO', 'RF']

# def get_REG_meta_task_order(df):
#     Matrix = np.zeros((7, 1), dtype=np.float64)
#     matrix = MetaFeature.getfeature(df)
#     matrix = np.transpose(matrix)
#     for i in range(0, 7):
#         Matrix[i] = np.average(matrix[i])
#     Matrix = np.transpose(Matrix)
#     Matrix = torch.Tensor(Matrix)
#     current_dir = os.path.dirname(__file__)
#     csv_path = os.path.join(current_dir, '..',  'datasets', 'dataset','MetafeatureREG.csv')
#     csv_path = os.path.normpath(csv_path)
#     dataset = pd.read_csv(csv_path, sep=',', encoding='ISO-8859-1')
#     min = sys.maxsize
#     minid = 0
#     for index, row in dataset.iterrows():
#         row = torch.Tensor(row)
#         from scipy.spatial.distance import cdist
#         result = cdist(Matrix.numpy(), [row.numpy()], metric='seuclidean')
#         if min > result:
#             min = result
#             minid = index
#     current_dir = os.path.dirname(__file__)
#     csv_path = os.path.join(current_dir, '..',  'datasets', 'dataset','label_reg.csv')
#     csv_path = os.path.normpath(csv_path)
#     df = pd.read_csv(csv_path, sep=',', encoding='ISO-8859-1')
#     data = df.iloc[10 * minid:10 * minid + 10][['Pipeline', 'EvaluationMetric']]
#     MaxPipeline = data.loc[data['EvaluationMetric'].idxmin()]['Pipeline']
#     pipeline = MaxPipeline.split(',')
#     List = [list8]
#     for i in pipeline:
#         if i in list1:
#             List.append(list1)
#         if i in list2:
#             List.append(list2)
#         if i in list3:
#             List.append(list3)
#         if i in list4:
#             List.append(list4)
#         if i in list5:
#             List.append(list5)
#         if i in list6:
#             List.append(list6)
#     return List


def get_REG_meta_task_order(df):
    #print("Step 1: Compute the current dataset's feature matrix")
    # Step 1: Compute the current dataset's feature matrix
    Matrix = np.zeros((7, 1), dtype=np.float64)
    matrix = MetaFeature.getfeature(df)
    matrix = np.transpose(matrix)
    for i in range(0, 7):
        Matrix[i] = np.average(matrix[i])
    Matrix = np.transpose(Matrix)
    Matrix = torch.Tensor(Matrix)
    
    current_dir = os.path.dirname(__file__)
    csv_path = os.path.join(current_dir, '..', 'datasets', 'dataset', 'MetafeatureREG.csv')
    csv_path = os.path.normpath(csv_path)
    dataset = pd.read_csv(csv_path, sep=',', encoding='ISO-8859-1')
    
    #print("Step 2: Calculate distances to all datasets in MetafeatureREG.csv")
    # Convert dataset to tensor
    dataset_tensor = torch.Tensor(dataset.values)
    
    # Step 2: Build graph and use KNN to find nearest neighbors
    G = nx.Graph()
    for i in range(len(dataset)):
        G.add_node(i, features=dataset_tensor[i])
    
    # Adding edges based on KNN
    nbrs = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(dataset)
    distances, indices = nbrs.kneighbors(dataset)
    
    for i, node in enumerate(G.nodes):
        for j_idx in range(1, len(indices[i])):  # Skip the first element as it is the node itself
            j = indices[i][j_idx]
            G.add_edge(i, j, weight=distances[i][j_idx])
    
    #print("Step 3: Find the indices of the three closest datasets")
    # Step 3: Find the indices of the three closest datasets to the target dataset
    distances = []
    for index, row in dataset.iterrows():
        row = torch.Tensor(row)
        temp = np.linalg.norm(torch.squeeze(Matrix) - row)
        distances.append((temp, index))
    
    distances.sort()
    closest_indices = [index for _, index in distances[:5]]
    
    csv_path = os.path.join(current_dir, '..', 'datasets', 'dataset', 'label_reg.csv')
    csv_path = os.path.normpath(csv_path)
    df_label = pd.read_csv(csv_path, sep=',', encoding='ISO-8859-1')
    
    #print("Step 4: Collect the pipelines for the three closest datasets", closest_indices)
    # Step 4: Collect the pipelines for the three closest datasets
    pipelines = []
    for index in closest_indices:
        data = df_label.iloc[10 * index:10 * index + 10][['Pipeline', 'EvaluationMetric']]
        MaxPipeline = data.loc[data['EvaluationMetric'].idxmin()]['Pipeline']
        pipeline = MaxPipeline.split(',')
        pipelines.append(pipeline)
    
    #print("Step 5: Compute the average pipeline", pipelines)
    # Step 5: Compute the average pipeline
    pipeline_counts = {}
    for pipeline in pipelines:
        for step in pipeline:
            if step in pipeline_counts:
                pipeline_counts[step] += 1
            else:
                pipeline_counts[step] = 1
    
    average_pipeline = [step for step, count in pipeline_counts.items() if count > 1]  # Steps that appear in at least 2 pipelines
    
    #print("Step 6: Determine the list based on the average pipeline", average_pipeline)
    # Step 6: Determine the list based on the average pipeline
    List = [list8]
    seen = set()  # Set to keep track of seen preprocessing types
    for i in average_pipeline:
        if i in list1 and "list1" not in seen:
            List.append(list1)
            seen.add("list1")
        if i in list2 and "list2" not in seen:
            List.append(list2)
            seen.add("list2")
        if i in list3 and "list3" not in seen:
            List.append(list3)
            seen.add("list3")
        if i in list4 and "list4" not in seen:
            List.append(list4)
            seen.add("list4")
        if i in list5 and "list5" not in seen:
            List.append(list5)
            seen.add("list5")
        if i in list6 and "list6" not in seen:
            List.append(list6)
            seen.add("list6")
    
    #print("Step 7: Visualizing the graph in 3D")
    # Step 7: Visualizing the graph in 3D
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(dataset)
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(scaled_data)
    
    # Randomly select 50 other datasets
    other_indices = list(set(range(len(dataset))) - set(closest_indices))
    random_indices = random.sample(other_indices, 50)
    selected_indices = closest_indices + random_indices
    
    # Visualization function
    # def visualize_3d_graph(pca_result, G, target_pca, selected_indices):
    #     fig = plt.figure(figsize=(10, 9))  # Adjust the size as needed
    #     ax = fig.add_subplot(111, projection='3d')
        
    #     # Plot edges only between red node and blue nodes
    #     for i in selected_indices:
    #         x0, y0, z0 = target_pca[0]
    #         x1, y1, z1 = pca_result[i]
    #         ax.plot([x0, x1], [y0, y1], [z0, z1], 'k-', linewidth=0.5)
        
    #     # Plot nodes
    #     selected_pca_result = pca_result[selected_indices]
    #     ax.scatter(selected_pca_result[:, 0], selected_pca_result[:, 1], selected_pca_result[:, 2], c='skyblue', s=50)
        
    #     # Highlight the target dataset
    #     ax.scatter(target_pca[:, 0], target_pca[:, 1], target_pca[:, 2], c='red', s=100, label='Target')
        
    #     ax.set_title('3D Visualization of Datasets with K-Nearest Neighbors')
    #     plt.show()

    # Highlight the target dataset
    #target_pca = pca.transform(scaler.transform([Matrix.numpy().flatten()]))
    #visualize_3d_graph(pca_result, G, target_pca, selected_indices)
    
    #print("Visualization created")
    
    return List



# get_CLA_meta_task_order 函数第四次修改版本：优化元特征匹配最佳管道策略（建模为图结构，使用 KNN 确定搜索空间）
def get_CLA_meta_task_order(df):
    #print("Step 1: Compute the current dataset's feature matrix")
    # Step 1: Compute the current dataset's feature matrix
    Matrix = np.zeros((7, 1), dtype=np.float64)
    matrix = MetaFeature.getfeature(df)
    matrix = np.transpose(matrix)
    for i in range(0, 7):
        Matrix[i] = np.average(matrix[i])
    Matrix = np.transpose(Matrix)
    Matrix = torch.Tensor(Matrix)
    
    current_dir = os.path.dirname(__file__)
    csv_path = os.path.join(current_dir, '..', 'datasets', 'dataset', 'Metafeature.csv')
    csv_path = os.path.normpath(csv_path)
    dataset = pd.read_csv(csv_path, sep=',', encoding='ISO-8859-1')
    
    #print("Step 2: Calculate distances to all datasets in Metafeature.csv")
    # Convert dataset to tensor
    dataset_tensor = torch.Tensor(dataset.values)
    
    # Step 2: Build graph and use KNN to find nearest neighbors
    G = nx.Graph()
    for i in range(len(dataset)):
        G.add_node(i, features=dataset_tensor[i])
    
    # Adding edges based on KNN
    nbrs = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(dataset)
    distances, indices = nbrs.kneighbors(dataset)
    
    for i, node in enumerate(G.nodes):
        for j_idx in range(1, len(indices[i])):  # Skip the first element as it is the node itself
            j = indices[i][j_idx]
            G.add_edge(i, j, weight=distances[i][j_idx])
    
    #print("Step 3: Find the indices of the three closest datasets")
    # Step 3: Find the indices of the three closest datasets to the target dataset
    distances = []
    for index, row in dataset.iterrows():
        row = torch.Tensor(row)
        temp = np.linalg.norm(torch.squeeze(Matrix) - row)
        distances.append((temp, index))
    
    distances.sort()
    closest_indices = [index for _, index in distances[:5]]
    
    csv_path = os.path.join(current_dir, '..', 'datasets', 'dataset', 'label.csv')
    csv_path = os.path.normpath(csv_path)
    df_label = pd.read_csv(csv_path, sep=',', encoding='ISO-8859-1')
    
    #print("Step 4: Collect the pipelines for the three closest datasets", closest_indices)
    # Step 4: Collect the pipelines for the three closest datasets
    pipelines = []
    for index in closest_indices:
        data = df_label.iloc[10 * index:10 * index + 10][['Pipeline', 'EvaluationMetric']]
        MaxPipeline = data.loc[data['EvaluationMetric'].idxmax()]['Pipeline']
        pipeline = MaxPipeline.split(',')
        pipelines.append(pipeline)
    
    #print("Step 5: Compute the average pipeline", pipelines)
    # Step 5: Compute the average pipeline
    pipeline_counts = {}
    for pipeline in pipelines:
        for step in pipeline:
            if step in pipeline_counts:
                pipeline_counts[step] += 1
            else:
                pipeline_counts[step] = 1
    
    average_pipeline = [step for step, count in pipeline_counts.items() if count > 1]  # Steps that appear in at least 2 pipelines
    
    #print("Step 6: Determine the list based on the average pipeline", average_pipeline)
    # Step 6: Determine the list based on the average pipeline
    List = [list7]
    seen = set()  # Set to keep track of seen preprocessing types
    for i in average_pipeline:
        if i in list1 and "list1" not in seen:
            List.append(list1)
            seen.add("list1")
        if i in list2 and "list2" not in seen:
            List.append(list2)
            seen.add("list2")
        if i in list3 and "list3" not in seen:
            List.append(list3)
            seen.add("list3")
        if i in list4 and "list4" not in seen:
            List.append(list4)
            seen.add("list4")
        if i in list5 and "list5" not in seen:
            List.append(list5)
            seen.add("list5")
        if i in list6 and "list6" not in seen:
            List.append(list6)
            seen.add("list6")
    
    #print("Step 7: Visualizing the graph in 3D")
    # Step 7: Visualizing the graph in 3D
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(dataset)
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(scaled_data)
    
    # Randomly select 50 other datasets
    other_indices = list(set(range(len(dataset))) - set(closest_indices))
    random_indices = random.sample(other_indices, 50)
    selected_indices = closest_indices + random_indices
    
    # Visualization function
    # def visualize_3d_graph(pca_result, G, target_pca, selected_indices):
    #     fig = plt.figure(figsize=(10, 9))  # Adjust the size as needed
    #     ax = fig.add_subplot(111, projection='3d')
        
    #     # Plot edges only between red node and blue nodes
    #     for i in selected_indices:
    #         x0, y0, z0 = target_pca[0]
    #         x1, y1, z1 = pca_result[i]
    #         ax.plot([x0, x1], [y0, y1], [z0, z1], 'k-', linewidth=0.5)
        
    #     # Plot nodes
    #     selected_pca_result = pca_result[selected_indices]
    #     ax.scatter(selected_pca_result[:, 0], selected_pca_result[:, 1], selected_pca_result[:, 2], c='skyblue', s=50)
        
    #     # Highlight the target dataset
    #     ax.scatter(target_pca[:, 0], target_pca[:, 1], target_pca[:, 2], c='red', s=100, label='Target')
        
    #     ax.set_title('3D Visualization of Datasets with K-Nearest Neighbors')
    #     plt.show()

    # # Highlight the target dataset
    # target_pca = pca.transform(scaler.transform([Matrix.numpy().flatten()]))
    # visualize_3d_graph(pca_result, G, target_pca, selected_indices)
    
    # print("Visualization created")
    
    return List



# # get_CLA_meta_task_order 函数第三次修改版本：优化元特征匹配最佳管道策略（求出最接近的三个数据集，简单取平均）
# def get_CLA_meta_task_order(df):
#     # Step 1: Compute the current dataset's feature matrix
#     Matrix = np.zeros((7, 1), dtype=np.float64)
#     matrix = MetaFeature.getfeature(df)
#     matrix = np.transpose(matrix)
#     for i in range(0, 7):
#         Matrix[i] = np.average(matrix[i])
#     Matrix = np.transpose(Matrix)
#     Matrix = torch.Tensor(Matrix)
    
#     current_dir = os.path.dirname(__file__)
#     csv_path = os.path.join(current_dir, '..',  'datasets', 'dataset', 'Metafeature.csv')
#     csv_path = os.path.normpath(csv_path)
#     dataset = pd.read_csv(csv_path, sep=',', encoding='ISO-8859-1')
    
#     # Step 2: Calculate distances to all datasets in Metafeature.csv
#     distances = []
#     for index, row in dataset.iterrows():
#         row = torch.Tensor(row)
#         temp = np.linalg.norm(torch.squeeze(Matrix) - row)
#         std1 = torch.std(torch.squeeze(Matrix))
#         std2 = torch.std(row)
#         temp = temp / np.sqrt(std1 ** 2 + std2 ** 2)
#         distances.append((temp, index))
    
#     # Step 3: Find the indices of the three closest datasets
#     distances.sort()
#     closest_indices = [index for _, index in distances[:3]]
    
#     csv_path = os.path.join(current_dir, '..',  'datasets', 'dataset', 'label.csv')
#     csv_path = os.path.normpath(csv_path)
#     df = pd.read_csv(csv_path, sep=',', encoding='ISO-8859-1')
    
#     # Step 4: Collect the pipelines for the three closest datasets
#     pipelines = []
#     for index in closest_indices:
#         data = df.iloc[10 * index:10 * index + 10][['Pipeline', 'EvaluationMetric']]
#         MaxPipeline = data.loc[data['EvaluationMetric'].idxmax()]['Pipeline']
#         pipeline = MaxPipeline.split(',')
#         pipelines.append(pipeline)
    
#     # Step 5: Compute the average pipeline
#     pipeline_counts = {}
#     for pipeline in pipelines:
#         for step in pipeline:
#             if step in pipeline_counts:
#                 pipeline_counts[step] += 1
#             else:
#                 pipeline_counts[step] = 1
    
#     average_pipeline = [step for step, count in pipeline_counts.items() if count > 1]  # Steps that appear in at least 2 pipelines
    
#     # Step 6: Determine the list based on the average pipeline
#     List = [list7]
#     seen = set()  # Set to keep track of seen preprocessing types
#     for i in average_pipeline:
#         if i in list1 and "list1" not in seen:
#             List.append(list1)
#             seen.add("list1")
#         if i in list2 and "list2" not in seen:
#             List.append(list2)
#             seen.add("list2")
#         if i in list3 and "list3" not in seen:
#             List.append(list3)
#             seen.add("list3")
#         if i in list4 and "list4" not in seen:
#             List.append(list4)
#             seen.add("list4")
#         if i in list5 and "list5" not in seen:
#             List.append(list5)
#             seen.add("list5")
#         if i in list6 and "list6" not in seen:
#             List.append(list6)
#             seen.add("list6")
    
#     return List




# # get_CLA_meta_task_order 函数第二次修改版本：优化元特征匹配最佳管道策略（求出最接近的 N 个数据集，随机选择一个作为最佳管道）
# def get_CLA_meta_task_order(df):
#     Matrix = np.zeros((7, 1), dtype=np.float64)
#     matrix = MetaFeature.getfeature(df)
#     matrix = np.transpose(matrix)
#     for i in range(0, 7):
#         Matrix[i] = np.average(matrix[i])
#     Matrix = np.transpose(Matrix)
#     Matrix = torch.Tensor(Matrix)
    
#     current_dir = os.path.dirname(__file__)
#     csv_path = os.path.join(current_dir, '..',  'datasets', 'dataset','Metafeature.csv')
#     csv_path = os.path.normpath(csv_path)
#     dataset = pd.read_csv(csv_path, sep=',', encoding='ISO-8859-1')
    
#     distances = []
#     for index, row in dataset.iterrows():
#         row = torch.Tensor(row)
#         temp = np.linalg.norm(torch.squeeze(Matrix) - row)
#         std1 = torch.std(torch.squeeze(Matrix))
#         std2 = torch.std(row)
#         temp = temp / np.sqrt(std1 ** 2 + std2 ** 2)
#         distances.append((temp, index))
    
#     # 找到距离最小的 N 个数据集索引
#     distances.sort()
#     closest_indices = [index for _, index in distances[:3]]
    
#     # 从最接近的 N 个数据集中随机选择一个
#     minid = random.choice(closest_indices)
    
#     csv_path = os.path.join(current_dir, '..',  'datasets', 'dataset','label.csv')
#     csv_path = os.path.normpath(csv_path)
#     df = pd.read_csv(csv_path, sep=',', encoding='ISO-8859-1')
#     data = df.iloc[10 * minid:10 * minid + 10][['Pipeline', 'EvaluationMetric']]
#     MaxPipeline = data.loc[data['EvaluationMetric'].idxmax()]['Pipeline']
#     pipeline = MaxPipeline.split(',')
    
#     List = [list7]
#     seen = set()  # Set to keep track of seen preprocessing types
#     for i in pipeline:
#         if i in list1 and "list1" not in seen:
#             List.append(list1)
#             seen.add("list1")
#         if i in list2 and "list2" not in seen:
#             List.append(list2)
#             seen.add("list2")
#         if i in list3 and "list3" not in seen:
#             List.append(list3)
#             seen.add("list3")
#         if i in list4 and "list4" not in seen:
#             List.append(list4)
#             seen.add("list4")
#         if i in list5 and "list5" not in seen:
#             List.append(list5)
#             seen.add("list5")
#         if i in list6 and "list6" not in seen:
#             List.append(list6)
#             seen.add("list6")
    
#     return List


# # get_CLA_meta_task_order 函数第一次修改版本：去除重复搜索空间
# def get_CLA_meta_task_order(df):
#     Matrix = np.zeros((7, 1), dtype=np.float64)
#     matrix = MetaFeature.getfeature(df)
#     matrix = np.transpose(matrix)
#     for i in range(0, 7):
#         Matrix[i] = np.average(matrix[i])
#     Matrix = np.transpose(Matrix)
#     Matrix = torch.Tensor(Matrix)
    
#     current_dir = os.path.dirname(__file__)
#     csv_path = os.path.join(current_dir, '..',  'datasets', 'dataset','Metafeature.csv')
#     csv_path = os.path.normpath(csv_path)
#     dataset = pd.read_csv(csv_path, sep=',', encoding='ISO-8859-1')
    
#     min = sys.maxsize
#     minid = 0
#     for index, row in dataset.iterrows():
#         row = torch.Tensor(row)
#         temp = np.linalg.norm(torch.squeeze(Matrix) - row)
#         std1 = torch.std(torch.squeeze(Matrix))
#         std2 = torch.std(row)
#         temp = temp / np.sqrt(std1 ** 2 + std2 ** 2)
#         if min > temp:
#             min = temp
#             minid = index
    
#     csv_path = os.path.join(current_dir, '..',  'datasets', 'dataset','label.csv')
#     csv_path = os.path.normpath(csv_path)
#     df = pd.read_csv(csv_path, sep=',', encoding='ISO-8859-1')
#     data = df.iloc[10 * minid:10 * minid + 10][['Pipeline', 'EvaluationMetric']]
#     MaxPipeline = data.loc[data['EvaluationMetric'].idxmax()]['Pipeline']
#     pipeline = MaxPipeline.split(',')
    
#     List = [list7]
#     seen = set()  # Set to keep track of seen preprocessing types
#     for i in pipeline:
#         if i in list1 and "list1" not in seen:
#             List.append(list1)
#             seen.add("list1")
#         if i in list2 and "list2" not in seen:
#             List.append(list2)
#             seen.add("list2")
#         if i in list3 and "list3" not in seen:
#             List.append(list3)
#             seen.add("list3")
#         if i in list4 and "list4" not in seen:
#             List.append(list4)
#             seen.add("list4")
#         if i in list5 and "list5" not in seen:
#             List.append(list5)
#             seen.add("list5")
#         if i in list6 and "list6" not in seen:
#             List.append(list6)
#             seen.add("list6")
    
#     return List


# # get_CLA_meta_task_order 函数原代码
# def get_CLA_meta_task_order(df):
#     Matrix = np.zeros((7, 1), dtype=np.float64)
#     matrix = MetaFeature.getfeature(df)
#     matrix = np.transpose(matrix)
#     for i in range(0, 7):
#         Matrix[i] = np.average(matrix[i])
#     Matrix = np.transpose(Matrix)
#     Matrix = torch.Tensor(Matrix)
#     current_dir = os.path.dirname(__file__)
#     csv_path = os.path.join(current_dir, '..',  'datasets', 'dataset','Metafeature.csv')
#     csv_path = os.path.normpath(csv_path)
#     dataset = pd.read_csv(csv_path, sep=',', encoding='ISO-8859-1')
#     min = sys.maxsize
#     minid = 0
#     for index, row in dataset.iterrows():
#         row = torch.Tensor(row)
#         temp = np.linalg.norm(torch.squeeze(Matrix) - row)
#         std1 = torch.std(torch.squeeze(Matrix))
#         std2 = torch.std(row)
#         temp = temp / np.sqrt(std1 ** 2 + std2 ** 2)
#         if min > temp:
#             min = temp
#             minid = index
#     current_dir = os.path.dirname(__file__)
#     csv_path = os.path.join(current_dir, '..',  'datasets', 'dataset','label.csv')
#     csv_path = os.path.normpath(csv_path)
#     df = pd.read_csv(csv_path, sep=',', encoding='ISO-8859-1')
#     data = df.iloc[10 * minid:10 * minid + 10][['Pipeline', 'EvaluationMetric']]
#     MaxPipeline = data.loc[data['EvaluationMetric'].idxmax()]['Pipeline']
#     pipeline = MaxPipeline.split(',')
#     List = [list7]
#     for i in pipeline:
#         if i in list1:
#             List.append(list1)
#         if i in list2:
#             List.append(list2)
#         if i in list3:
#             List.append(list3)
#         if i in list4:
#             List.append(list4)
#         if i in list5:
#             List.append(list5)
#         if i in list6:
#             List.append(list6)
#     return List


class State(object):
    def __init__(self):
        self.current_value = 'null'
        self.current_round_index = 0
        self.cumulative_choices = []
        self.current_depth = -1

    def get_current_value(self):
        return self.current_value

    def set_current_value(self, value):
        self.current_value = value

    def get_current_round_index(self):
        return self.current_round_index

    def set_current_round_index(self, turn):
        self.current_round_index = turn

    def get_current_depth(self):
        return self.current_depth

    def set_current_depth(self, turn):
        self.current_depth = turn

    def get_cumulative_choices(self):
        return self.cumulative_choices

    def set_cumulative_choices(self, choices):
        self.cumulative_choices = choices

    def is_terminal(self):
        return self.current_depth == MAX_DEPTH

    def compute_reward(self):
        return -abs(1 - self.current_value)

    def get_next_state_with_random_choice(self):
        random_choice = random.choice([choice for choice in List[self.current_depth + 1]])
        next_state = State()
        next_state.set_current_value(random_choice)
        next_state.set_current_round_index(self.current_round_index + 1)
        next_state.set_cumulative_choices(self.cumulative_choices +
                                          [random_choice])
        return next_state

    def __repr__(self):
        return " value: {}, round: {}, choices: {}".format(
            self.current_value, self.current_round_index,
            self.cumulative_choices)


class Node(object):
    def __init__(self):
        self.parent = None
        self.children = []
        self.is_promising = []
        self.visit_times = 0
        self.pre_profit = 0.0
        self.after_profit = 0.0
        self.state = None

    def set_state(self, state):
        self.state = state

    def get_state(self):
        return self.state

    def get_parent(self):
        return self.parent

    def set_parent(self, parent):
        self.parent = parent

    def get_children(self):
        return self.children

    def get_visit_times(self):
        return self.visit_times

    def set_visit_times(self, times):
        self.visit_times = times

    def visit_times_add_one(self):
        self.visit_times += 1

    def get_profit_value(self):
        return self.pre_profit + self.after_profit

    def get_pre_profit(self):
        return self.pre_profit

    def get_after_profit(self):
        return self.after_profit

    def set_pre_profit(self, value):
        self.pre_profit = value

    def set_after_profit(self, value):
        self.after_profit = value

    def is_all_expand(self):
        return self.children != []

    def add_child(self, sub_node):
        sub_node.set_parent(self)
        self.children.append(sub_node)
        self.is_promising.append(True)

    def __repr__(self):
        return "P/A/N: {}/{}/{}, state: {}".format(
            self.pre_profit, self.after_profit, self.visit_times, self.state)


def get_part_dataset(dataset, current_depth, Is_BatchTraining):
    if Is_BatchTraining:
        rate = (current_depth + 1) / (MAX_DEPTH + 1)
        for key, value in dataset.items():
            if key == 'train' or key == 'target':
                dataset[key] = value.sample(frac=rate, random_state=42)
    return dataset


def get_profit(node, dataset, taskType, datasetTarget):
    try:
        part_dataset = get_part_dataset(dataset.copy(), node.get_state().get_current_depth(), Is_BatchTraining)
    except Exception as e:
        print("Zyy")
        print(e)
    if taskType == "CLA":
        profit = mctsdata.getAcc(part_dataset, node.get_state().cumulative_choices, datasetTarget)
    else:
        profit = mctsdata.getMse(part_dataset, node.get_state().cumulative_choices, datasetTarget)
    node.set_pre_profit(profit)


# def tree_policy(node, dataset, datasetTarget, taskType):
#     if True not in node.is_promising:
#         return node
#     sub_node = best_child(node)
#     if sub_node.get_state().is_terminal() == False:
#         if sub_node.is_all_expand():
#             sub_node = tree_policy(sub_node, dataset, datasetTarget, taskType)
#         else:
#             for i in range(len(List[sub_node.state.get_current_depth() + 1])):
#                 datasetcopy = dataset.copy()
#                 expand(sub_node, datasetcopy, datasetTarget, taskType)
#         return sub_node
#     else:
#         return sub_node


# def default_policy(node, dataset,taskType):
#     df = merge_datasets(dataset)
#     after_profit = Estimate_after_profit.get_Estimate(df, node.state.cumulative_choices,taskType)
#     return after_profit


def expand(node, dataset, datasetTarget, taskType):
    tried_sub_node_states = [
        sub_node.get_state().get_cumulative_choices() for sub_node in node.get_children()
    ]
    new_state = node.get_state().get_next_state_with_random_choice()
    while new_state.get_cumulative_choices() in tried_sub_node_states:
        new_state = node.get_state().get_next_state_with_random_choice()
    sub_node = Node()
    sub_node.set_state(new_state)
    node.add_child(sub_node)
    sub_node.state.set_current_depth(node.state.current_depth + 1)
    get_profit(sub_node, dataset, taskType, datasetTarget)
    return sub_node


def best_child(node):
    best_score = -sys.maxsize
    best_sub_node = None
    for i in range(len(node.children)):
        if node.is_promising[i] == True:
            sub_node = node.children[i]
            profit = sub_node.get_profit_value()
            temp = math.log(node.get_visit_times() + 1) / (sub_node.get_visit_times() + 1)
            right = 15 * math.floor(math.log(0.001, 10)) * math.sqrt(temp)
            score = profit + right
            if score > best_score:
                best_sub_node = sub_node
                best_score = score
    return best_sub_node


# def backup(node, reward):
#     while node != None:
#         node.visit_times_add_one()
#         if reward < node.after_profit:
#             reward = node.after_profit
#         node.set_after_profit(reward)
#         node = node.parent





def monte_carlo_tree_search(node, dataset, datasetTarget, taskType):
    expand_node = tree_policy(node, dataset, datasetTarget, taskType)
    reward = default_policy(expand_node, dataset,taskType)
    backup(expand_node, reward)
    drop_unpromising(expand_node, Weight)
    return expand_node





def tree_policy(node, dataset, datasetTarget, taskType):
    #print(f"Tree policy start: Node state: {node.state}")
    if True not in node.is_promising:
        #print("No promising nodes left.")
        return node
    sub_node = best_child(node)
    #print(f"Best child selected: {sub_node.state}")
    if sub_node.get_state().is_terminal() == False:
        if sub_node.is_all_expand():
            #print("Sub node is fully expanded, calling tree_policy recursively.")
            sub_node = tree_policy(sub_node, dataset, datasetTarget, taskType)
        else:
            #print("Expanding sub node.")
            for i in range(len(List[sub_node.state.get_current_depth() + 1])):
                datasetcopy = dataset.copy()
                expand(sub_node, datasetcopy, datasetTarget, taskType)
        return sub_node
    else:
        #print("Sub node is terminal.")
        return sub_node

def default_policy(node, dataset, taskType):
    #print(f"Default policy start: Node state: {node.state}")
    df = merge_datasets(dataset)
    after_profit = Estimate_after_profit.get_Estimate(df, node.state.cumulative_choices, taskType)
    #print(f"Estimated profit: {after_profit}")
    return after_profit

def backup(node, reward):
    #print(f"Backup start: Initial reward: {reward}")
    while node is not None:
        node.visit_times_add_one()
        if reward < node.after_profit:
            reward = node.after_profit
        node.set_after_profit(reward)
        #print(f"Backup: Node state: {node.state}, reward: {reward}")
        node = node.parent

def drop_unpromising(node, Weight):
    #print(f"Drop unpromising start: Node state: {node.state}, Weight: {Weight}")
    keep_number = math.ceil(Weight * len(node.children))
    profit = []
    if node.children:
        for sub in node.children:
            profit.append(sub.pre_profit)
        profit.sort(reverse=True)
        target = profit[keep_number - 1]
        for i in range(len(node.children)):
            sub_node = node.children[i]
            if sub_node.pre_profit < target:
                node.is_promising[i] = False
                #print(f"Dropped unpromising node: {sub_node.state}")





# def drop_unpromising(node, Weight):
#     keep_number = math.ceil(Weight * len(node.children))
#     profit = []
#     if node.children:
#         for sub in node.children:
#             profit.append(sub.pre_profit)
#         profit.sort(reverse=True)
#         target = profit[keep_number - 1]
#         for i in range(len(node.children)):
#             sub_node = node.children[i]
#             if sub_node.pre_profit < target:
#                 node.is_promising[i] = False


def CLA_With_TimeBudget(df, dataset, time_budget, datasetTarget):
    global MAX_DEPTH
    global List
    List = get_CLA_meta_task_order(df)
    methods = List
    tasks = map_list_to_task(methods, taskType="CLA")
    tasks = move_first_to_last(tasks)
    methods = move_first_to_last(methods)
    create_task_flowchart(tasks, methods, 4, "        Search space:")
    MAX_DEPTH = len(List) - 1
    init_state = State()
    init_node = Node()
    init_node.set_state(init_state)
    progress_label = widgets.HTML(
        value='<span style="font-family: Times New Roman; font-size: 14pt;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Progress:&nbsp&nbsp</span>')
    progress_bar = widgets.FloatProgress(min=0, max=100, layout=widgets.Layout(width='708px'))
    progress_bar.style.bar_color = 'rgb(168, 209, 141)'
    box = widgets.HBox([progress_label, progress_bar])
    display(box)
    start_time = time.time()
    for i in range(len(List[init_node.state.get_current_depth() + 1])):
        expand(init_node, dataset, datasetTarget, taskType="CLA")
    best_node = init_node
    accuracies = []
    times = []
    while True:
        if time.time() - start_time > time_budget:
            break
        progress_bar.value = (time.time() - start_time) / time_budget * 100
        init_dataset = dataset.copy()
        try:
            times.append(time.time() - start_time)
            temp_node = monte_carlo_tree_search(init_node, init_dataset, datasetTarget, taskType="CLA")
            accuracies.append(best_node.get_pre_profit())
            gap = temp_node.get_pre_profit() - best_node.get_pre_profit()
            if gap > 0:
                best_node = temp_node
        except:
            times.pop()
    progress_bar.value = 100
    method = best_node.state.cumulative_choices
    methods = [[m] for m in method]
    bestpipeline = method
    methods = move_first_to_last(methods)
    method = [item for sublist in methods for item in sublist]
    tasks = map_list_to_task(method, taskType="CLA")
    create_task_flowchart(tasks, methods, 4, "Optimal Pipeline:")
    return times, accuracies, bestpipeline


# def CLA_With_TimeBudget(df, dataset, time_budget, datasetTarget):
#     global MAX_DEPTH
#     global List
#     List = get_CLA_meta_task_order(df)
#     methods = List
#     tasks = map_list_to_task(methods, taskType="CLA")
#     tasks = move_first_to_last(tasks)
#     methods = move_first_to_last(methods)
#     create_task_flowchart(tasks, methods, 4, "        Search space:")
#     MAX_DEPTH = len(List) - 1
#     init_state = State()
#     init_node = Node()
#     init_node.set_state(init_state)
    
#     # Debug: 初始节点状态
#     print("Initial Node State:", init_node.state)
    
#     progress_label = widgets.HTML(
#         value='<span style="font-family: Times New Roman; font-size: 14pt;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Progress:&nbsp&nbsp</span>')
#     progress_bar = widgets.FloatProgress(min=0, max=100, layout=widgets.Layout(width='708px'))
#     progress_bar.style.bar_color = 'rgb(168, 209, 141)'
#     box = widgets.HBox([progress_label, progress_bar])
#     display(box)
#     start_time = time.time()
    
#     for i in range(len(List[init_node.state.get_current_depth() + 1])):
#         expand(init_node, dataset, datasetTarget, taskType="CLA")
    
#     # Debug: 初始节点扩展后的状态
#     print("After Expansion, Node State:", init_node.state)
    
#     best_node = init_node
#     accuracies = []
#     times = []
    
#     while True:
#         if time.time() - start_time > time_budget:
#             break
#         progress_bar.value = (time.time() - start_time) / time_budget * 100
#         init_dataset = dataset.copy()
#         try:
#             times.append(time.time() - start_time)
#             temp_node = monte_carlo_tree_search(init_node, init_dataset, datasetTarget, taskType="CLA")
            
#             # Debug: 每次搜索后的临时节点和最佳节点状态
#             print("Temp Node State:", temp_node.state)
#             print("Best Node State before update:", best_node.state)
            
#             accuracies.append(best_node.get_pre_profit())
#             gap = temp_node.get_pre_profit() - best_node.get_pre_profit()
#             if gap > 0:
#                 best_node = temp_node
                
#                 # Debug: 更新后的最佳节点状态
#                 print("Best Node State updated:", best_node.state)
#         except Exception as e:
#             print("Exception during MCTS:", e)
#             times.pop()
    
#     progress_bar.value = 100
    
#     method = best_node.state.cumulative_choices
    
#     # Debug: 最佳管道的选择
#     print("Best Pipeline Choices:", method)
    
#     methods = [[m] for m in method]
#     bestpipeline = method
#     methods = move_first_to_last(methods)
#     method = [item for sublist in methods for item in sublist]
#     tasks = map_list_to_task(method, taskType="CLA")
#     create_task_flowchart(tasks, methods, 4, "Optimal Pipeline:")
    
#     return times, accuracies, bestpipeline



def CLA_Without_TimeBudget(df, dataset, datasetTarget):
    global MAX_DEPTH
    global List
    gapcount=0
    List = get_CLA_meta_task_order(df)
    methods = List
    tasks = map_list_to_task(methods, taskType="CLA")
    tasks = move_first_to_last(tasks)
    methods = move_first_to_last(methods)
    create_task_flowchart(tasks, methods, 4, "        Search space:")
    MAX_DEPTH = len(List) - 1
    init_state = State()
    init_node = Node()
    init_node.set_state(init_state)
    start_time = time.time()
    for i in range(len(List[init_node.state.get_current_depth() + 1])):
        expand(init_node, dataset, datasetTarget, taskType="CLA")
    best_node = init_node
    accuracies = []
    times = []
    while True:
        if gapcount==20:
            break
        init_dataset = dataset.copy()
        try:
            elapsed_time = time.time() - start_time
            times.append(elapsed_time)
            temp_node = monte_carlo_tree_search(init_node, init_dataset, datasetTarget, taskType="CLA")
            accuracies.append(best_node.get_pre_profit())
            gap = temp_node.get_pre_profit() - best_node.get_pre_profit()
            if gap > 0:
                best_node = temp_node
            if gap<Mingap:
                gapcount=gapcount+1
            else:
                gapcount=0
        except:
            times.pop()
    display(HTML(
        f"<div style='font-size: 20px; font-family: Times New Roman;'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Run Time:&nbsp; {elapsed_time:.2f}s</div>"))
    method = best_node.state.cumulative_choices
    methods = [[m] for m in method]
    bestpipeline = method
    methods = move_first_to_last(methods)
    method = [item for sublist in methods for item in sublist]
    tasks = map_list_to_task(method, taskType="CLA")
    create_task_flowchart(tasks, methods, 4, "Optimal Pipeline:")
    return times, accuracies, bestpipeline

def REG_Without_TimeBudget(df, dataset, datasetTarget):
    global MAX_DEPTH
    global List
    List = get_REG_meta_task_order(df)
    methods = List
    tasks = map_list_to_task(methods, taskType="REG")
    tasks = move_first_to_last(tasks)
    methods = move_first_to_last(methods)
    create_task_flowchart(tasks, methods, 4, "        Search space:")
    MAX_DEPTH = len(List) - 1
    global running
    gapcount = 0
    init_state = State()
    init_node = Node()
    init_node.set_state(init_state)
    start_time = time.time()
    for i in range(len(List[init_node.state.get_current_depth() + 1])):
        expand(init_node, dataset, datasetTarget, taskType="REG")
    best_node = init_node
    mse = []
    times = []
    while True:
        if gapcount == 20:
            running=False
            break
        init_dataset = dataset.copy()
        try:
            elapsed_time = time.time() - start_time
            times.append(elapsed_time)
            temp_node = monte_carlo_tree_search(init_node, init_dataset, datasetTarget, taskType="REG")
            gap = temp_node.get_pre_profit() - best_node.get_pre_profit()
            last_best_profit = best_node.get_pre_profit()
            if gap > 0:
                best_node = temp_node
            if best_node.get_pre_profit() - last_best_profit < Mingap:
                gapcount = gapcount + 1
            else:
                gapcount = 0
            if best_node.get_pre_profit() != 0:
                mse.append(1 / best_node.get_pre_profit())
        except:
            times.pop()
    display(HTML(
        f"<div style='font-size: 20px; font-family: Times New Roman;'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Run Time:&nbsp; {elapsed_time:.2f}s</div>"))
    method = best_node.state.cumulative_choices
    methods = [[m] for m in method]
    bestpipeline = method
    methods = move_first_to_last(methods)
    method = [item for sublist in methods for item in sublist]
    tasks = map_list_to_task(method, taskType="REG")
    create_task_flowchart(tasks, methods, 4, "Optimal Pipeline:")
    return times, mse, bestpipeline

def REG_With_TimeBudget(df, dataset, time_budget, datasetTarget):
    global MAX_DEPTH
    global List
    List = get_REG_meta_task_order(df)
    methods = List
    tasks = map_list_to_task(methods, taskType="REG")
    tasks = move_first_to_last(tasks)
    methods = move_first_to_last(methods)
    create_task_flowchart(tasks, methods, 4, "        Search space:")
    MAX_DEPTH = len(List) - 1
    init_state = State()
    init_node = Node()
    init_node.set_state(init_state)
    progress_label = widgets.HTML(
        value='<span style="font-family: Times New Roman; font-size: 14pt;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Progress:&nbsp&nbsp</span>')
    progress_bar = widgets.FloatProgress(min=0, max=100, layout=widgets.Layout(width='708px'))
    progress_bar.style.bar_color = 'rgb(168, 209, 141)'
    box = widgets.HBox([progress_label, progress_bar])
    display(box)
    start_time = time.time()
    for i in range(len(List[init_node.state.get_current_depth() + 1])):
        expand(init_node, dataset, datasetTarget, taskType="REG")
    best_node = init_node
    mse = []
    times = []
    while True:
        if time.time() - start_time > time_budget:
            break
        progress_bar.value = (time.time() - start_time) / time_budget * 100
        init_dataset = dataset.copy()
        try:
            elapsed_time = time.time() - start_time
            times.append(elapsed_time)
            temp_node = monte_carlo_tree_search(init_node, init_dataset, datasetTarget, taskType="REG")
            gap = temp_node.get_pre_profit() - best_node.get_pre_profit()
            if gap > 0:
                best_node = temp_node
            if best_node.get_pre_profit() != 0:
                mse.append(1 / best_node.get_pre_profit())
        except:
            times.pop()
    progress_bar.value = 100
    method = best_node.state.cumulative_choices
    methods = [[m] for m in method]
    bestpipeline = method
    methods = move_first_to_last(methods)
    method = [item for sublist in methods for item in sublist]
    tasks = map_list_to_task(method, taskType="REG")
    create_task_flowchart(tasks, methods, 4, "Optimal Pipeline:")
    return times, mse, bestpipeline

def create_task_flowchart(tasks, methods, max_methods, text):
    color = (168 / 255, 209 / 255, 141 / 255)
    fig, ax = plt.subplots(figsize=(10.5, 3))
    method_width = 1.5
    method_height = 0.8
    method_gap = method_width + 0.5
    task_width = max_methods * method_gap
    task_gap = task_width + 1
    task_height = 1.8
    for i, task in enumerate(tasks):
        if i >= len(methods):
            break
        num_methods = len(methods[i])
        start_x = i * task_gap
        ax.add_patch(
            plt.Rectangle((start_x, -task_height / 2), task_width, task_height, fill=False, edgecolor=color,
                          linestyle='dashed'))
        ax.text(i * task_gap + task_width / 2, -task_height + 0.2, task, ha='center',
                va='bottom')
        newMethod_gap = (task_width - num_methods * method_width) / num_methods
        for j, method in enumerate(methods[i]):
            x = start_x + j * (method_width + newMethod_gap) + newMethod_gap / 2
            y = -method_height / 2
            ax.add_patch(plt.Rectangle((x, y), method_width, method_height, fill=True, edgecolor=color, facecolor=color,
                                       alpha=0.5))
            ax.text(x + method_width / 2, y + method_height / 2, method, ha='center', va='center')
        if i < len(methods) - 1:
            ax.annotate("", xy=(start_x + task_width, 0), xytext=(start_x + task_gap, 0),
                        arrowprops=dict(arrowstyle="<-"))
    ax.set_xlim(-3, len(tasks) * task_gap)
    ax.set_ylim(-task_height - 0.2, task_height + 0.2)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.figtext(0.13, 0.5, text, fontsize=12, ha='center', va='center', rotation='horizontal',
                fontname='Times New Roman')
    plt.show()


def move_first_to_last(arr):
    if len(arr) < 2:
        return arr
    return arr[1:] + [arr[0]]


def map_list_to_task(lst, taskType):
    task = []
    for sublist in lst:
        if sublist == list1 or sublist in list1:
            task.append('Imputation')
        elif sublist == list2 or sublist in list2:
            task.append('Encoding')
        elif sublist == list3 or sublist in list3:
            task.append('Normalization')
        elif sublist == list4 or sublist in list4:
            task.append('Feature Selection')
        elif sublist == list5 or sublist in list5:
            task.append('Duplication')
        elif sublist == list6 or sublist in list6:
            task.append('Outlier Exclusion')
        elif taskType == "CLA" and (sublist == list7 or sublist in list7):
            task.append('Classification')
        elif taskType == "REG" and (sublist == list8 or sublist in list8):
            task.append('Regression')
    return task


def merge_datasets(datasets):
    train_data = pd.concat([datasets['train'], datasets['target']], axis=1)
    test_data = pd.concat([datasets['test'], datasets['target_test']], axis=1)
    full_data = pd.concat([train_data, test_data], axis=0)
    full_data_sorted = full_data.sort_index()
    return full_data_sorted


def store(datasetName, preparedDataset):
    current_dir = os.getcwd()
    base_filename = os.path.basename(datasetName)
    filename_without_extension = os.path.splitext(base_filename)[0]
    save_filename = filename_without_extension + "_Prepared.csv"
    save_path = os.path.join(current_dir, '../datasets', save_filename)
    preparedDataset.to_csv(save_path, index=False)
    print_path = os.path.join('../datasets', save_filename)
    print(f"The dataset processed by AutoDP is saved to: {print_path}")

