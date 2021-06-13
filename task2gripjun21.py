#!/usr/bin/env python
# coding: utf-8

# ## THE SPARKS FOUNDATION
# 

#  PREDICTION USING UNSUPERVISED ML
#  
#  TASK 2 - to predict the otimum numnber of clusters and to represent it visually from the given set of 'Iris' dataset.
#  
#  Dataset - htttps://bit.ly/3kXTdoc

# BY: HITHA M GOWDA

# In[6]:


#IMPORTING REQUIRED LIBS
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
get_ipython().run_line_magic('matplotlib', 'inline')


# IMPORTING AND EXPLORING DATA

# In[34]:


import pandas as pd
from sklearn import datasets
iris=datasets.load_iris()
iris_df=pd.DataFrame(iris.data, columns= iris.feature_names)
iris_df.head()


# DATASET EXPLORATION

# In[6]:


#cheching data shape
iris_df.shape


# In[8]:


#checking for any duplicate values
iris_df.duplicated().sum()


# In[9]:


#checking for missing values
iris_df.isnull().sum()


# In[10]:


#description of dataset
iris_df.describe()


# #FINDING THE OPTIMUM NUMBER OF CLUSTERS FOR k-means CLASSIFICATION

# Determining the value of k using Elbow Method

# In[37]:


x = iris_df.iloc[:, [0,1,2,3]].values

from sklearn.cluster import KMeans
wcss = []

for i in range(1, 11):
    Kmeans = KMeans(n_clusters = i, init = 'k-means++',
                    max_iter = 300, n_init=10, random_state = 0)
    Kmeans.fit(x)
    wcss.append(Kmeans.inertia_)
    
#plotting   
import matplotlib.pyplot as plt    
plt.plot(range(1, 11),wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('wcss') #witin cluster sum of squares
plt.show()
    
    
import warnings 
warnings.filterwarnings(action="ignore")
    


# From the above graph, the elbow occurs at where there are  optimum clusters. This takes place when WCSS ( within cluster sum of squares)
# does not decrease notably with every iteration
# 
# 
# Now taking number of clusters = 3

# In[39]:


#creating K-mean clusters
kmeans = KMeans(n_clusters = 3, init = 'k-means++',
                max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)
                
               


# VISUALISING CLUSTERS

# In[32]:


plt.scatter(x[y_kmeans == 0,0], x[y_kmeans == 0,1],
           s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1,0], x[y_kmeans == 1, 1],
           s = 100, c ='black', label = 'Iris-versicolor')
plt.scatter(x[y_kmeans == 2,0], x[y_kmeans == 2,1],
           s = 100, c = 'green', label = 'Iris-virfinica')

#plotting centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],
           s = 100, c = 'blue', label = 'centroids')

plt.legend()


# In[ ]:




