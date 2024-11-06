# 

SwiftDP: An Efficient Framework for Automated Data Preparation Pipeline Generation


#### Quick Start

1.  Before running the code, please make sure your Python version is 3.8.18. 
2.  pip install autodatapre

#### Run Example

1. demo.ipynb provides two quick examples that correspond to the Demonstration Scenarios in the paper.

2. We support classification and regression tasks.

3. Taking classification as an example：

   import autodatapre as SwiftDP

   datasetName = your_dataset_path # e.g. "autodatapre/datasets/42493.csv"

   datasetTarget = the_target_column_name # e.g. 'Delay'

   runTime = 10

   df = pd.read_csv(datasetName, sep = ',', encoding = 'ISO-8859-1')

   detailResult, preparedDataset = SwiftDP.Classifier(df, datasetName, datasetTarget, runTime)

   SwiftDP.EnhancedFunction(df, preparedDataset, detailResult, taskType = "CLA")

4. If runTime is not specified in the Classifier function, run until convergence.

5. Regressor has the same settings.

   

   
