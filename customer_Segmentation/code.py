# --------------
# import packages

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 



# Load Offers
offers = pd.read_excel(path,sheet_name =0)
transactions  = pd.read_excel(path,sheet_name =1)
transactions['n'] = 1
print(offers.shape)
print(transactions.shape)
df = pd.merge(offers, transactions, on=['Offer #', 'Offer #'])
print(df.head())
print(df.shape)
# Load Transactions


# Merge dataframes


# Look at the first 5 rows



# --------------
# Code starts here

# create pivot table
matrix = pd.pivot_table(df,index='Customer Last Name', columns='Offer #',values='n')

# replace missing values with 0
matrix.fillna(0,inplace = True)

# reindex pivot table
matrix.reset_index(inplace=True)

# display first 5 rows
matrix.head()

# Code ends here


# --------------
# import packages
from sklearn.cluster import KMeans

# Code starts here

# initialize KMeans object
cluster = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10,random_state =0)

# create 'cluster' column
matrix['cluster']=cluster.fit_predict(matrix[matrix.columns[1:]])

# Code ends here


# --------------
# import packages
from sklearn.decomposition import PCA

# Code starts here

# initialize pca object with 2 components
pca = PCA(n_components=2,random_state=0)

# create 'x' and 'y' columns donoting observation locations in decomposed form
matrix['x'] = pca.fit_transform(matrix[matrix.columns[1:]])[:,0]
matrix['y'] = pca.fit_transform(matrix[matrix.columns[1:]])[:,1]

# dataframe to visualize clusters by customer names
clusters = matrix.iloc[:,[0, 33, 34,35]]

# visualize clusters
clusters.plot.scatter(x='x', y='y', c='cluster', colormap='viridis')

# Code ends here


# --------------
# Code starts here

# merge 'clusters' and 'transactions'
data = pd.merge(clusters,transactions)

# merge `data` and `offers`
data = pd.merge(offers,data)
# initialzie empty dictionary


# iterate over every cluster

    # observation falls in that cluster

    # sort cluster according to type of 'Varietal'

    # check if 'Champagne' is ordered mostly

        # add it to 'champagne'


# get cluster with maximum orders of 'Champagne' 
champagne = {0:18,1:30,2:47,3:13,4:1}

# print out cluster number
cluster_champagne = 2





# --------------
# Code starts here

# empty dictionary
discount = {}
for i in range(0,5):
    new_df = data[data['cluster'] == i]
    counts = new_df['Discount (%)'].sum()/new_df.shape[0]
    discount[i] = counts
cluster_discount = max(discount, key = discount.get)
print(cluster_discount)

# iterate over cluster numbers

    # dataframe for every cluster

    # average discount for cluster

    # adding cluster number as key and average discount as value 


# cluster with maximum average discount


# Code ends here



