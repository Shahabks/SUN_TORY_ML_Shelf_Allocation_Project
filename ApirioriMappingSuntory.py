import numpy as np 
import pandas as pd 
from mlxtend.frequent_patterns import apriori, association_rules 

pathy=input("path to dataset:   ")
name=input("dataset name:     ")
pathyy=pathy+"/"+name+".xlsx"

# Loading the Data 
data = pd.read_excel(pathyy) 
data.head() 

# Exploring the columns of the data 
data.columns 

# Exploring the different regions of transactions 
data.VendingMachine.unique() 

# Stripping extra spaces in the description 
data['Description'] = data['Description'].str.strip() 

# Dropping the rows without any invoice number 
data.dropna(axis = 0, subset =['InvoiceNo'], inplace = True) 
data['InvoiceNo'] = data['InvoiceNo'].astype('str') 

# Dropping all transactions which were done on credit 
data = data[~data['InvoiceNo'].str.contains('C')] 

# Transactions done at VM1 
LYO_VM1 = (data[data['VendingMachine'] =="VM1"] 
		.groupby(['InvoiceNo', 'Description'])['Quantity'] 
		.sum().unstack().reset_index().fillna(0) 
		.set_index('InvoiceNo')) 

# Transactions done at VM2 
LYO_VM2 = (data[data['VendingMachine'] =="VM2"] 
		.groupby(['InvoiceNo', 'Description'])['Quantity'] 
		.sum().unstack().reset_index().fillna(0) 
		.set_index('InvoiceNo')) 

# Transactions done at VM3 
LYO_VM3 = (data[data['VendingMachine'] =="VM3"] 
		.groupby(['InvoiceNo', 'Description'])['Quantity'] 
		.sum().unstack().reset_index().fillna(0) 
		.set_index('InvoiceNo')) 

# Transactions done at VM4 
LYO_VM4 = (data[data['VendingMachine'] =="VM4"] 
		.groupby(['InvoiceNo', 'Description'])['Quantity'] 
		.sum().unstack().reset_index().fillna(0) 
		.set_index('InvoiceNo')) 

# Defining the hot encoding function to make the data suitable 
# for the concerned libraries 
def hot_encode(x): 
	if(x<= 0): 
		return 0
	if(x>= 1): 
		return 1

# Encoding the datasets 
basket_encoded = LYO_VM1.applymap(hot_encode) 
LYO_VM1 = basket_encoded 

basket_encoded = LYO_VM2.applymap(hot_encode) 
LYO_VM2 = basket_encoded 

basket_encoded = LYO_VM3.applymap(hot_encode) 
LYO_VM3 = basket_encoded 

basket_encoded = LYO_VM4.applymap(hot_encode) 
LYO_VM4 = basket_encoded 

# Building the model 
frq_items = apriori(LYO_VM1, min_support = 0.05, use_colnames = True) 

# Collecting the inferred rules in a dataframe 
rules = association_rules(frq_items, metric ="lift", min_threshold = 1) 
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False]) 
print(rules.head()) 

frq_items = apriori(LYO_VM2, min_support = 0.01, use_colnames = True) 
rules = association_rules(frq_items, metric ="lift", min_threshold = 1) 
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False]) 
print(rules.head()) 

frq_items = apriori(LYO_VM3, min_support = 0.05, use_colnames = True) 
rules = association_rules(frq_items, metric ="lift", min_threshold = 1) 
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False]) 
print(rules.head()) 

frq_items = apriori(LYO_VM4, min_support = 0.05, use_colnames = True) 
rules = association_rules(frq_items, metric ="lift", min_threshold = 1) 
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False]) 
print(rules.head()) 

