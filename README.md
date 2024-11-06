# 

Automated System for Efficient Generation of Data Preparation Pipeline


#### Quick Start

1.  Before running the code, please make sure your Python version is 3.7.16. 
2.  pip install autodatapre

#### Run Example

1. AutoDP.testFunction() provide two fast examples, Showcased examples from our paper.

2. We support classification and regression tasks, with specified runtime and default runtime until convergence.

3. Taking classification as an example：

   import autodatapre as AutoDP

   datasetName = your_dataset_path # e.g. "E:/1.csv"

   datasetTarget = the_target_column_name # e.g. 'delay'

   runTime = 10

   df = pd.read_csv(datasetName, sep=',', encoding='ISO-8859-1')

   detailResult, preparedDataset = AutoDP.Classifier(df, datasetName, datasetTarget, runTime)

   AutoDP.EnhancedFunction(df, preparedDataset, detailResult, taskType="CLA")

4. If runTime is not specified in the Classifier function, run until convergence.

5. Regressor has the same settings.

   

   
