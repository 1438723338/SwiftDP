from .MCTS import merge_datasets
from .MCTS import CLA_With_TimeBudget,CLA_Without_TimeBudget,REG_With_TimeBudget,REG_Without_TimeBudget
import os
import traceback  
import random
import time
import numpy as np
from . import MCTS_DATA as mctsdata
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

list1 = ['RAND', 'MF', 'MICE', 'KNN', 'EM', 'MEDIAN', 'MEAN', 'DROP']   # Imputation
list2 = ['OE', 'BE', 'FE', 'CBE', 'LE']                                 # Encoding
list3 = ['ZS', 'DS', 'MM']                                              # Normalization
list4 = ['MR', 'WR', 'LC', 'TB']                                        # Feature Selection
list5 = ['ED', 'AD']                                                    # Duplication
list6 = ['ZSB', 'IQR', 'LOF']                                           # Outlier Exclusion
list7 = ['NB', 'LDA', 'RF', 'LR']                                       # Classification
list8 = ['OLS', 'LASSO', 'RF']      

# def Classifier(df, datasetName, datasetTarget, runTime=None):
#     preparedDataset = df
#     base_name = os.path.basename(datasetName)
#     datasetname, _ = os.path.splitext(base_name)
#     detailResult = []
#     try:
#         dataset = mctsdata.read_dataset(df, datasetTarget)
#         if runTime is not None:
#             times, accuracies, bestpipeline = CLA_With_TimeBudget(df, dataset, runTime, datasetTarget)
#         else:
#             times, accuracies, bestpipeline = CLA_Without_TimeBudget(df, dataset, datasetTarget)
#         detailResult = []
#         detailResult.append(times)
#         detailResult.append(accuracies)
#         detailResult.append(datasetname)
#         detailResult.append(datasetTarget)
#         profit = mctsdata.getAcc(dataset, bestpipeline, datasetTarget)
#         preparedDataset = merge_datasets(dataset)
#         preparedDataset = preparedDataset.dropna(how='all', subset=preparedDataset.columns[:-1])
#     except:
#         list = []
#     return detailResult, preparedDataset


def Classifier(df, datasetName, datasetTarget, runTime=None):
    preparedDataset = df
    base_name = os.path.basename(datasetName)
    datasetname, _ = os.path.splitext(base_name)
    detailResult = []

    try:
        #print("Step: Read Dataset")
        dataset = mctsdata.read_dataset(df, datasetTarget)
        
        #print("Dataset read successfully")

        if runTime is not None:
            #print("Step: With Time Budget")
            times, accuracies, bestpipeline = CLA_With_TimeBudget(df, dataset, runTime, datasetTarget)
        else:
            #print("Step: Without Time Budget")
            times, accuracies, bestpipeline = CLA_Without_TimeBudget(df, dataset, datasetTarget)
        
        # 确保 bestpipeline 被正确生成
        if not bestpipeline:
            raise ValueError("Best pipeline was not generated correctly.")
        
        #print(f"Times: {times}")
        #print(f"Accuracies: {accuracies}")
        #print(f"Best Pipeline: {bestpipeline}")

        detailResult.append(times)
        detailResult.append(accuracies)
        detailResult.append(datasetname)
        detailResult.append(datasetTarget)
        detailResult.append(bestpipeline)

        #print("Step: Calculate Accuracy")
        profit = mctsdata.getAcc(dataset, bestpipeline, datasetTarget)

        #print("Step: Merge Datasets")
        preparedDataset = merge_datasets(dataset)
        preparedDataset = preparedDataset.dropna(how='all', subset=preparedDataset.columns[:-1])

    except Exception as e:
        #print(f"Error: {e}")
        #print(traceback.format_exc())
        detailResult = []

    return detailResult, preparedDataset


def Regressor(df, datasetName, datasetTarget,runTime=None):
    preparedDataset = df
    base_name = os.path.basename(datasetName)
    datasetname, _ = os.path.splitext(base_name)
    detailResult = []
    try:
        dataset = mctsdata.read_dataset(df, datasetTarget)
        if runTime is not None:
            times, mse, bestpipeline = REG_With_TimeBudget(df, dataset, runTime, datasetTarget)
        else:
            times, mse, bestpipeline = REG_Without_TimeBudget(df, dataset, datasetTarget)
        detailResult.append(times)
        detailResult.append(mse)
        detailResult.append(datasetname)
        detailResult.append(datasetTarget)
        detailResult.append(bestpipeline)
        
        profit = mctsdata.getMse(dataset, bestpipeline, datasetTarget)
        preparedDataset = merge_datasets(dataset)
        preparedDataset = preparedDataset.dropna(how='all', subset=preparedDataset.columns[:-1])
    except:
        list = []
    return detailResult, preparedDataset

def EnhancedFunction(df, preparedDataset, detailResult, taskType):
    if taskType == "CLA":
        drew1_CLA(df, preparedDataset, detailResult)
    else:
        drew1_REG(df, preparedDataset, detailResult)
    drew2(df, preparedDataset, detailResult)

def get_random_task_order():
    mylen = random.randint(1, 6)
    mylist = random.sample(range(1, 7), mylen)
    List = [list7]
    for i in range(mylen):
        if mylist[i] == 1:
            List.append(list1)
        if mylist[i] == 2:
            List.append(list2)
        if mylist[i] == 3:
            List.append(list3)
        if mylist[i] == 4:
            List.append(list4)
        if mylist[i] == 5:
            List.append(list5)
        if mylist[i] == 6:
            List.append(list6)
    return List


def get_random_task_order_MSE():
    mylen = random.randint(1, 6)
    mylist = random.sample(range(1, 7), mylen)
    List = [list8]
    for i in range(mylen):
        if mylist[i] == 1:
            List.append(list1)
        if mylist[i] == 2:
            List.append(list2)
        if mylist[i] == 3:
            List.append(list3)
        if mylist[i] == 4:
            List.append(list4)
        if mylist[i] == 5:
            List.append(list5)
        if mylist[i] == 6:
            List.append(list6)
    return List

def randomDPwithTimeBudget(dataset, datasetTarget):
    init_dataset = dataset.copy()
    List = get_random_task_order()
    order = []
    start_time = time.time()
    try:
        for i in range(len(List)):
            mylen = random.randint(0, len(List[i]) - 1)
            order.append(List[i][mylen])
        acc = mctsdata.getAcc(init_dataset, order, datasetTarget)
    except:
        acc = 0
    return acc, time.time() - start_time


def randomDPwithTimeBudget_MSE(dataset, datasetTarget):
    init_dataset = dataset.copy()
    List = get_random_task_order_MSE()
    order = []
    start_time = time.time()
    try:
        for i in range(len(List)):
            mylen = random.randint(0, len(List[i]) - 1)
            order.append(List[i][mylen])
        mse = 1 / mctsdata.getMse(init_dataset, order, datasetTarget)
    except:
        mse = -1
        print("There is an issue with the Random pipeline and the results cannot be obtained")
    return mse, time.time() - start_time


def noDPwithTimeBudget(dataset, datasetTarget):
    init_dataset = dataset.copy()
    start_time = time.time()
    selected_i = random.choice(list7)
    try:
        CLA = mctsdata.choose_classifier(init_dataset, selected_i, datasetTarget)
        max = CLA.get('quality_metric')
    except:
        max = 0
    return max, time.time() - start_time


def noDPwithTimeBudget_MSE(dataset, datasetTarget):
    init_dataset = dataset.copy()
    start_time = time.time()
    selected_i = random.choice(list8)
    try:
        REG = mctsdata.choose_regressor(init_dataset, selected_i, datasetTarget)
        mse = REG.get('quality_metric')
    except:
        mse = -1
    return mse, time.time() - start_time

# def drew1_CLA(df, preparedDataset, detailResult):
#     dataset = mctsdata.read_dataset(df, detailResult[3])
#     randomacc, randomTime = randomDPwithTimeBudget(dataset, detailResult[3])
#     noacc, noTime = noDPwithTimeBudget(dataset, detailResult[3])
#     fig, axs = plt.subplots(1, 2, figsize=(8, 3.5))
#     ax = axs[0]
#     ax.plot(detailResult[0], detailResult[1], color='tab:orange', linewidth=2.5, label='AutoDP')
#     ax.plot([randomTime, max(detailResult[0])], [randomacc, randomacc], color='tab:blue', linewidth=2.5,
#             label='RandomDP')
#     ax.plot([0, max(detailResult[0])], [noacc, noacc], color='gray', linestyle='--', linewidth=2.5, label='NoDP')
#     ax.set_xlabel('Time')
#     ax.set_ylabel('Accuracy')
#     ax.set_title('Trend of Accuracy over Run Time')
#     ax.legend()
#     numeric_df = df.select_dtypes(include=[np.number])
#     original_missing_values = df.isnull().sum().sum()
#     original_outliers = ((numeric_df - numeric_df.mean()).abs() > 3 * numeric_df.std()).sum().sum()
#     original_duplicates = numeric_df.duplicated().sum()
#     cleaned_missing_values = preparedDataset.isnull().sum().sum()
#     cleaned_outliers = ((
#                                 preparedDataset - preparedDataset.mean()).abs() > 3 * preparedDataset.std()).sum().sum()
#     cleaned_duplicates = preparedDataset.duplicated().sum()
#     categories = ['Missing Values', 'Outliers', 'Duplicates']
#     original_values = [original_missing_values, original_outliers, original_duplicates]
#     cleaned_values = [cleaned_missing_values, cleaned_outliers, cleaned_duplicates]
#     ax = axs[1]
#     x = range(len(categories))
#     width = 0.35
#     rects1 = ax.bar(x, original_values, width, label="D" + detailResult[2])
#     rects2 = ax.bar([i + width for i in x], cleaned_values, width, label='prepared_D' + detailResult[2])
#     ax.set_ylabel('Count')
#     ax.set_xlabel('Categories')
#     ax.set_title('Comparison of ' + 'D' + detailResult[2] + ' and ' + 'prepared_D' + detailResult[2])
#     ax.set_xticks([i + width / 2 for i in x])
#     ax.set_xticklabels(categories)
#     ax.legend()
#     plt.tight_layout()
#     plt.show()

def drew1_CLA(df, preparedDataset, detailResult):
    dataset = mctsdata.read_dataset(df, detailResult[3])
    randomacc, randomTime = randomDPwithTimeBudget(dataset, detailResult[3])
    noacc, noTime = noDPwithTimeBudget(dataset, detailResult[3])
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.5))
    
    ax = axs[0]
    ax.plot(detailResult[0], detailResult[1], color='tab:orange', linewidth=2.5, label='SwiftDP')
    ax.plot([randomTime, max(detailResult[0])], [randomacc, randomacc], color='tab:blue', linewidth=2.5, label='RandomDP')
    ax.plot([0, max(detailResult[0])], [noacc, noacc], color='gray', linestyle='--', linewidth=2.5, label='NoDP')
    ax.set_xlabel('Time')
    ax.set_ylabel('Accuracy')
    ax.set_title('Trend of Accuracy over Run Time')
    ax.legend()

    numeric_df = df.select_dtypes(include=[np.number])
    original_missing_values = df.isnull().sum().sum()
    original_outliers = ((numeric_df - numeric_df.mean()).abs() > 3 * numeric_df.std()).sum().sum()
    original_duplicates = numeric_df.duplicated().sum()
    cleaned_missing_values = preparedDataset.isnull().sum().sum()
    cleaned_outliers = ((preparedDataset - preparedDataset.mean()).abs() > 3 * preparedDataset.std()).sum().sum()
    cleaned_duplicates = preparedDataset.duplicated().sum()
    
    categories = ['Missing Values', 'Outliers', 'Duplicates']
    original_values = [original_missing_values, original_outliers, original_duplicates]
    cleaned_values = [cleaned_missing_values, cleaned_outliers, cleaned_duplicates]
    
    ax = axs[1]
    x = range(len(categories))
    width = 0.35
    rects1 = ax.bar(x, original_values, width, label="D" + detailResult[2], color='#f08080', edgecolor='white')  
    rects2 = ax.bar([i + width for i in x], cleaned_values, width, label='prepared_D' + detailResult[2], color='#7fc4da', edgecolor='white')  
    
    ax.set_ylabel('Count')
    ax.set_xlabel('Categories')
    ax.set_title('Comparison of ' + 'D' + detailResult[2] + ' and ' + 'prepared_D' + detailResult[2])
    ax.set_xticks([i + width / 2 for i in x])
    ax.set_xticklabels(categories)
    ax.legend()
    
    plt.tight_layout()
    plt.show()



def drew1_REG(df, preparedDataset, detailResult):
    dataset = mctsdata.read_dataset(df, detailResult[3])
    randommse, randomTime = randomDPwithTimeBudget_MSE(dataset, detailResult[3])
    nomse, noTime = noDPwithTimeBudget_MSE(dataset,detailResult[3])
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.5))
    ax = axs[0]
    ax.plot(detailResult[0], detailResult[1], color='tab:orange', linewidth=2.5, label='SwiftDP')
    if randommse != -1:
        ax.plot([randomTime, max(detailResult[0])], [randommse, randommse], color='tab:blue', linewidth=2.5,
            label='RandomDP')
    ax.plot([0, max(detailResult[0])], [nomse, nomse], color='gray', linestyle='--', linewidth=2.5, label='NoDP')
    ax.set_xlabel('Time')
    ax.set_ylabel('MSE')
    ax.set_title('Trend of MSE over Run Time')
    ax.legend()
    original_missing_values = df.isnull().sum().sum()
    original_outliers = ((df - df.mean()).abs() > 3 * df.std()).sum().sum()
    original_duplicates = df.duplicated().sum()
    cleaned_missing_values = preparedDataset.isnull().sum().sum()
    cleaned_outliers = ((preparedDataset - preparedDataset.mean()).abs() > 3 * preparedDataset.std()).sum().sum()
    cleaned_duplicates = preparedDataset.duplicated().sum()
    categories = ['Missing Values', 'Outliers', 'Duplicates']
    original_values = [original_missing_values, original_outliers, original_duplicates]
    cleaned_values = [cleaned_missing_values, cleaned_outliers, cleaned_duplicates]
    ax = axs[1]
    x = range(len(categories))
    width = 0.35
    rects1 = ax.bar(x, original_values, width, label='D' + detailResult[2], color='#f08080', edgecolor='white')
    rects2 = ax.bar([i + width for i in x], cleaned_values, width, label='prepared_D' + detailResult[2], color='#7fc4da', edgecolor='white')
    ax.set_ylabel('Count')
    ax.set_xlabel('Categories')
    ax.set_title('Comparison of ' + 'D' + detailResult[2] + ' and ' + 'prepared_D' + detailResult[2])
    ax.set_xticks([i + width / 2 for i in x])
    ax.set_xticklabels(categories)
    ax.legend()
    plt.tight_layout()
    plt.show()


def drew2(df, preparedDataset, detailResult):
    import seaborn as sns
    corr_df = df.corr()
    corr_df1 = preparedDataset.corr()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    sns.heatmap(corr_df, annot=False, cmap='Oranges', square=True, ax=axes[0])
    axes[0].set_title('Column Correlation of D' + detailResult[2])
    sns.heatmap(corr_df1, annot=False, cmap='Oranges', square=True, ax=axes[1])
    axes[1].set_title('Column Correlation of prepared_D' + detailResult[2])
    plt.tight_layout()
    plt.show()


def store(datasetName, preparedDataset):
    current_dir = os.getcwd()
    base_filename = os.path.basename(datasetName)
    filename_without_extension = os.path.splitext(base_filename)[0]
    save_filename = filename_without_extension + "_Prepared.csv"
    save_path = os.path.join(current_dir, '../datasets', save_filename)
    preparedDataset.to_csv(save_path, index=False)
    print_path = os.path.join('../datasets', save_filename)
    print(f"The dataset processed by AutoDP is saved to: {print_path}")

import pandas as pd

def testFunction():
    current_dir = os.path.dirname(__file__)
    csv_path = os.path.join(current_dir, '..',  'datasets', '42493.csv')
    csv_path = os.path.normpath(csv_path)
    datasetName=csv_path
    datasetTarget = 'Delay'
    runTime = 10
    df = pd.read_csv(datasetName, sep=',', encoding='ISO-8859-1')
    detailResult, preparedDataset = Classifier(df, datasetName, datasetTarget, runTime)
    EnhancedFunction(df, preparedDataset, detailResult, taskType="CLA")

    current_dir = os.path.dirname(__file__)
    csv_path = os.path.join(current_dir, '..',  'datasets', '573.csv')
    csv_path = os.path.normpath(csv_path)
    datasetName=csv_path
    datasetTarget = 'usr'
    df = pd.read_csv(datasetName, sep=',', encoding='ISO-8859-1')
    detailResult, preparedDataset = Regressor(df, datasetName, datasetTarget)
    EnhancedFunction(df, preparedDataset, detailResult, taskType="REG")
