#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

df = pd.read_csv('C:\\Users\\wrey\\OneDrive - sram.com\\Documents\\GCU\\DSC-570\\ai4i2020.csv')
warnings.filterwarnings('ignore')


# In[ ]:


#Confirmation of imported data
df.head()


# In[ ]:


#Check for missing values and data type
df.info()


# In[ ]:


#drop UDI and product ID - unnecessary to project
df = df.drop(['Product ID', 'UDI', 'Type', 'TWF','HDF','PWF','OSF','RNF'], axis = 1)


# In[ ]:


#Confirm 
df.head()


# In[ ]:


df.info()


# In[ ]:


#adapted from https://www.geeksforgeeks.org/change-data-type-for-one-or-more-columns-in-pandas-dataframe/#

df = df.astype(float)
print(df.dtypes)


# In[ ]:


#descriptive stats
df.describe(include='all')


# In[ ]:


df.head(5)


# In[ ]:


#Separate independent variables for data exploration 
df2 = df[['Air temperature [K]', 'Process temperature [K]',
       'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']]


# In[ ]:


#Histogram shows data distributions
#Adapted from https://www.statology.org/pandas-histogram-for-each-column/
df2.hist(layout=(2,3))
plt.subplots_adjust(bottom=0.01, right=1.5, top=2)
plt.show()


# In[ ]:


#check for data balancing of dependent variable

sns.countplot('Machine failure', data=df)
plt.title('Machine failure observations', fontsize = 14)


# In[ ]:


print(df['Machine failure'].value_counts())


# In[ ]:


print(df.columns)


# In[ ]:


#visualize outliers
#Adapted from : https://towardsdatascience.com/how-to-detect-handle-and-visualize-outliers-ad0b74af4af7
for column in df2:
    sns.boxplot(data=df, y=column)
    plt.title(f'Boxplot of {column}')
    plt.show()


# In[ ]:


#This visualization does not work well when data is not transformed, as it does not
#provide high-resolution of viewing variables with tightly grouped data.
#Adapted from: https://www.geeksforgeeks.org/box-plot-in-python-using-matplotlib/#
plot_columns = [col for col in df2]
df[plot_columns].plot(kind='box', figsize=(12, 6), title='Box and Whisker Plots', ylabel='Value', grid=True)


# In[ ]:


#outlier management
#adapted from: https://www.geeksforgeeks.org/interquartile-range-to-detect-outliers-in-data/
#Timming method used adatped from https://www.analyticsvidhya.com/blog/2022/09/dealing-with-outliers-using-the-iqr-method/
binary_variables = ['Machine failure','TWF','HDF','PWF','OSF','RNF']
df_IQR = df
for col in df_IQR.columns:
    if col not in binary_variables:
        # calculate the IQR (interquartile range)
        Q1 = df_IQR[col].quantile(0.25)
        Q3 = df_IQR[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df_IQR[(df_IQR[col] <= (Q1 - 1.5 * IQR)) | (df_IQR[col] >= (Q3 + 1.5 * IQR))]
        if not outliers.empty:
            df_IQR.drop(outliers.index, inplace=True)


# In[ ]:


#Trimming removed 22.4% of observed machine failures. 
#This significantly alters the integrity of the dataset.
#Because outliers may have a direct impact on machine failure, outlier management will not be preformed. 
print(df_IQR['Machine failure'].value_counts())


# In[ ]:


#visualize outliers
#Adapted from : https://towardsdatascience.com/how-to-detect-handle-and-visualize-outliers-ad0b74af4af7
for column in df2:
    sns.boxplot(data=df, y=column)
    plt.title(f'Boxplot of {column}')
    plt.show()


# In[ ]:


#Data transformation 
#normalize data using z-score
#Adapted from https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.zscore.html
from scipy.stats import zscore
binary_variables = ['Machine failure','TWF','HDF','PWF','OSF','RNF']
for col in df.columns:
      if col not in binary_variables:
        df[col] = zscore(df[col])


# In[ ]:


#Regraph boxplot post-transformation for better visualization
#Adapted from https://www.geeksforgeeks.org/box-plot-in-python-using-matplotlib/#
plot_columns = [col for col in df2]
df[plot_columns].plot(kind='box', figsize=(12, 6), title='Box and Whisker Plots', ylabel='Value', grid=True)


# In[ ]:


#Updated histograms post cleaning and transformation
#Adapted from https://www.statology.org/pandas-histogram-for-each-column/
df.hist(layout=(2,3))
plt.subplots_adjust(bottom=0.01, right=1.5, top=2)
plt.show()


# In[ ]:


#Geeks for Geeks, 2020
#Adapted from: https://www.statology.org/correlation-in-python/#:~:text=This%20tutorial%20explains%20how%20to%20calculate%20the%20correlation,we%20can%20use%20the%20Numpy%20corrcoef%20%28%29%20function.
correlation_matrix = df.corr()

correlation_with_failure = correlation_matrix['Machine failure']
sorted_correlation = correlation_with_failure.sort_values(ascending=False)

print(sorted_correlation)


# In[ ]:


#visualization
#Adapted from:https://datagy.io/python-correlation-matrix/
map_matrix = df.corr().round(2)
mask = np.triu(np.ones_like(map_matrix, dtype=bool))
sns.heatmap(map_matrix, annot=True, vmax=1, vmin=-1, center=0, cmap='vlag', mask=mask)
plt.show()


# In[ ]:


#Data split using 70/30 ratio
#adapted from https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
from sklearn.model_selection import train_test_split

X = df.drop(['Machine failure'], axis=1)
y = df["Machine failure"]

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.3, 
                                                    random_state = 0,
                                                    stratify = y)


# In[ ]:


#Oversample to combat unbalanced data (Bronwlee, 2020)
#Adapted from https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SVMSMOTE.html
from imblearn.over_sampling import SVMSMOTE

X_train, y_train = SVMSMOTE(random_state = 50).fit_resample(X_train, y_train)


# In[ ]:


X_test.head()


# In[ ]:


#Check if over sampling worked
plt.hist(y_train)


# In[ ]:


from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report, confusion_matrix,ConfusionMatrixDisplay
import time
from sklearn.naive_bayes import GaussianNB


# In[ ]:


#Build Model
#Use time(time) to help track efficency of model
#adapted from: https://www.geeksforgeeks.org/python-time-time-method/

start = time.time()
NBmodel = GaussianNB().fit(X_train, y_train)
stop = time.time()
y_pred = NBmodel.predict(X_test)


# In[ ]:


#evaluate
#adapted from: https://www.datacamp.com/tutorial/naive-bayes-scikit-learn
accuracy_nb = accuracy_score(y_pred,y_test)
f1_nb = f1_score(y_pred,y_test,average="weighted")
time_nb = stop - start
print("Naive Bayes Accuracy:", accuracy_nb)
print("Naive Bayes F1 Score:", f1_nb)
print('Naive Bayes Train Time: ', time_nb) 


# In[ ]:


#adapted from: https://www.datacamp.com/tutorial/naive-bayes-scikit-learn
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot();


# In[ ]:


#Random forest
#Parameter Tuning Resource: https://www.analyticsvidhya.com/blog/2015/06/tuning-random-forest-model/#:~:text=When%20tuning%20a%20random%20forest%2C%20key%20parameters%20to,of%20a%20split%20%28e.g.%2C%20Gini%20impurity%20or%20entropy%29.
#Adapted from: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
from sklearn.ensemble import RandomForestClassifier

start = time.time()
RFmodel = RandomForestClassifier(n_estimators=25, n_jobs=-1,
                               random_state=50).fit(X_train, y_train)
stop = time.time()
y_pred = RFmodel.predict(X_test)


# In[ ]:


accuracy_rf = accuracy_score(y_pred,y_test)
f1_rf = f1_score(y_pred,y_test,average="weighted")
time_rf = stop - start
print("Random Forest Accuracy:", accuracy_rf)
print("Random Forest F1 Score:", f1_rf)
print('Random Forest Train Time: ', time_rf) 


# In[ ]:


rf = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=rf)
disp.plot();


# In[ ]:


from sklearn import tree
from sklearn.tree import plot_tree
#Adapted from https://mljar.com/blog/visualize-tree-from-random-forest/

tree_to_visualize = RFmodel.estimators_[0]

feature_names = ['Air temperature [K]', 'Process temperature [K]',
       'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
# Plot the first decision tree
plt.figure(figsize=(20, 10))  # You can adjust the figure size as needed
plot_tree(tree_to_visualize, filled=True, feature_names=feature_names, class_names=np.unique(y).astype(str))
plt.show()


# In[ ]:


#Adapted from https://mljar.com/blog/visualize-tree-from-random-forest/
tree_graph = RFmodel.estimators_[0]
# Set the maximum depth for better readability
plt.figure(figsize=(30,30))  
plot_tree(tree_graph, filled=True, feature_names=feature_names, class_names=np.unique(y).astype(str), max_depth=2)
plt.show()


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
#Adapted from https://dev.to/balapriya/cross-validation-and-hyperparameter-search-in-scikit-learn-a-complete-guide-5ed8
start = time.time()
knn_class = KNeighborsClassifier()

#Use grid search function for hyperparamter tuning
hyperparameter_range = {'n_neighbors': range(2, 20)}
find_hyperpar = GridSearchCV(knn_class, hyperparameter_range, cv=5, scoring='accuracy')
find_hyperpar.fit(X_train, y_train)
print(find_hyperpar.best_params_)


# In[ ]:


KNNmodel = KNeighborsClassifier(n_neighbors=2).fit(X_train, y_train)

stop = time.time()
y_pred = KNNmodel.predict(X_test)


# In[ ]:


accuracy_knn = accuracy_score(y_pred,y_test)
f1_knn = f1_score(y_pred,y_test,average="weighted")
time_knn = stop - start
print("KNN Accuracy:", accuracy_knn)
print("KNN F1 Score:", f1_knn)
print("KNN Train Time: ", time_knn) 


# In[ ]:


knn = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=knn)
disp.plot();


# In[ ]:


#comparison table
#Adapted from https://learnpython.com/blog/print-table-in-python/
from tabulate import tabulate
table = [['model', 'Accuracy', 'F1', 'Train Time'], ['Naive Bayes', accuracy_nb, f1_nb, time_nb], ['Random Forest', accuracy_rf, f1_rf, time_rf], ['KNN', accuracy_knn, f1_knn, time_knn]]
print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))

