# SUN_TORY_ML_Shelf_Allocation_Project
### Analysis and Algorithms:EXPERIMENT
#### PCA, Correlation, Extract Trees Classifier 
#### Linear/Non-Linear Algorithms: Logistic Regression, Linear Discriminant Analysis, Gaussian Naive Bayes. Support Vector Machines, K-Nearest Neighbors,. Classification and Regression Trees.

### Python 3.6 64bit, Worked well/checked on Dell Inspiron i5577 for a large dataset (and Google Colab)
### The algorithm comes with a sample dataset 27(column)x9215(rows) which is extendable (I assigned random/sample names to each column, you may revise them with a seed number for all models) 
#### When you run the algorithm, a popup asks you the path to the dataset directory (you may revise it)
### the algorithm generates/saves (in the same directory as the main dataset) a few *.csv files (statistics, correlation,factors) and *.sav models


## Apiriori Algorithm 
### Run "ApirioriMappingSuntory.py" it will ask to get the path to dataset (where you have saved it on your machine) and the following question will be the dataset name (here, our sample is called "TransActionVector.xlsx", feel free to download it)
### The algorithm engages *a lot of memories*, please consider it.

### This experiment setup runs over a hypothetical data sample of sixteen beverages from a vending machine and mapping seven hundred fifty sales-transaction over the period of one month. 
### Sales data was arranged in Excel by using a macro code. An Apiriori Algorithm was developed to capture beverages-VM-sales relationship (costumer purchase behaviour) over the period of one month.
### Association rules and conviction rates were obtained from Apriori Algorithm. 
### The conviction values that the Apriori Algorithm results are used for Multi-Dimensional Scaling Analysis. These values were analysed to determine the shelf layout of products the Monte Carlo algorithm. Data are considered as proximity coefficient and one source matrix for Multi-Dimensional Scaling Analysis model.  The model parameters; the shape is the full matrix, proximity transformations are the interval, the dimension is 2 and proximities are similarities. The product points in two dimensions 
### The normalized raw value was found to be appropriate in an acceptable level of 0.08309. The result of the study, available shelf layout is designed to be side by side the products sold together 
