#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
from matplotlib import gridspec
import matplotlib.pyplot as plt
import seaborn as sns

# pandas options
pd.set_option('display.max_colwidth', 1000, 'display.max_rows', None, 'display.max_columns', None)
#plotting options
get_ipython().run_line_magic('matplotlib', 'inline')
mpl.style.use('ggplot')
sns.set(style='whitegrid')


# In[2]:


transactions = pd.read_csv('creditcard.csv')
transactions.shape


# In[3]:


transactions.isnull().any().any()


# In[4]:


transactions['Class'].value_counts()


# In[5]:


transactions['Class'].value_counts(normalize=True)


# In[6]:


x = transactions.drop(labels='Class', axis=1) #Features
y = transactions.loc[:,'Class']               # response
del transactions                              # delete the original data


# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1, stratify=y)
del x, y
x_train.shape


# In[9]:


x_test.shape


# In[10]:


x_train.is_copy = False
x_test.is_copy = False


# In[11]:


x_train['Time'].describe()


# In[12]:


x_train['Time'].max() / 24


# In[13]:


x_train['Amount'].describe()


# In[14]:


plt.figure(figsize=(12,4), dpi=200)
sns.displot(x_train['Amount'], bins = 300, kde = False)
plt.ylabel('Count')
plt.title('Transaction Amounts')


# In[15]:


x_train.loc[:,'Amount'] = x_train['Amount'] + 1e-9 
x_train.loc[:,'Amount'], maxlog, (min_ci, max_ci) = sp.stats.boxcox(x_train['Amount'], alpha=0.01)
maxlog


# In[16]:


(min_ci, max_ci)


# In[17]:


pca_vars = ['V%i' % k for k in range(1,29)]
x_train[pca_vars].describe()


# In[18]:


plt.figure(figsize=(12,4), dpi=80)
sns.barplot(x=pca_vars, y=x_train[pca_vars].mean(), color='darkblue')
plt.xlabel('Column')
plt.ylabel('Mean')
plt.title('V1-V28 Means')


# In[19]:


plt.figure(figsize=(12,4), dpi=80)
sns.barplot(x=pca_vars, y=x_train[pca_vars].std(), color='darkred')
plt.xlabel('Column')
plt.ylabel('Standard Deviation')
plt.title('V1-V28 Standard Deviations')


# In[20]:


plt.figure(figsize=(12,4), dpi=80)
sns.barplot(x=pca_vars, y=x_train[pca_vars].skew(), color="darkgreen")
plt.xlabel("column")
plt.ylabel("Skewness")
plt.title("V1-V28 skewness")


# In[21]:


plt.figure(figsize=(12,4), dpi=80)
plt.yscale('log')
sns.barplot(x=pca_vars, y=x_train[pca_vars].kurtosis(), color="darkorange")
plt.xlabel("column")
plt.ylabel("Kurtosis")
plt.title("V1-V28 kustosis")


# In[22]:


plt.figure(figsize=(12,4), dpi=80)
sns.barplot(x=pca_vars, y=x_train[pca_vars].median(), color="darkblue")
plt.xlabel("column")
plt.ylabel("Median")
plt.title("V1-V28 Median")


# In[19]:


plt.figure(figsize=(12,4), dpi=80)
sns.barplot(x=pca_vars, y=x_train[pca_vars].quantile(0.75), color="darkred")
plt.xlabel("column")
plt.ylabel("IQR")
plt.title("V1-V28 IQR's")


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec


# In[30]:


data = pd.read_csv("creditcard.csv")
data.head()


# In[31]:


print(data.shape)
print(data.describe())


# In[32]:


fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]
outlierFraction = len(fraud)/float(len(valid))
print(outlierFraction)
print('Fraud Cases: {}'.format(len(data[data['Class'] == 1])))
print('Valid Transactions: {}'.format(len(data[data['Class'] == 0])))


# In[33]:


#amount details of the fradulent transaction
fraud.Amount.describe()


# In[34]:


# Details of valid transaction
valid.Amount.describe()


# In[35]:


corrmat = data.corr()
fig = plt.figure(figsize = (12, 9))
sns.heatmap(corrmat, vmax = .8, square = True)
plt.show()


# In[36]:


# dividing the X and the Y from the dataset
X = data.drop(['Class'], axis = 1)
Y = data["Class"]
print(X.shape)
print(Y.shape)
# getting just the values for the sake of processing
# (its a numpy array with no columns)
xData = X.values
yData = Y.values


# In[38]:


# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
xTrain, xTest, yTrain, yTest = train_test_split(
        xData, yData, test_size = 0.2, random_state = 42)
# Building the Random Forest Classifier (RANDOM FOREST)
from sklearn.ensemble import RandomForestClassifier
# random forest model creation
rfc = RandomForestClassifier()
rfc.fit(xTrain, yTrain)
# predictions
yPred = rfc.predict(xTest)


# In[39]:


# Evaluating the classifier
# printing every score of the classifier
# scoring in anything
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix
 
n_outliers = len(fraud)
n_errors = (yPred != yTest).sum()
print("The model used is Random Forest classifier")
 
acc = accuracy_score(yTest, yPred)
print("The accuracy is {}".format(acc))
 
prec = precision_score(yTest, yPred)
print("The precision is {}".format(prec))
 
rec = recall_score(yTest, yPred)
print("The recall is {}".format(rec))
 
f1 = f1_score(yTest, yPred)
print("The F1-Score is {}".format(f1))
 
MCC = matthews_corrcoef(yTest, yPred)
print("The Matthews correlation coefficient is{}".format(MCC))


# In[40]:


# printing the confusion matrix
LABELS = ['Normal', 'Fraud']
conf_matrix = confusion_matrix(yTest, yPred)
plt.figure(figsize =(12, 12))
sns.heatmap(conf_matrix, xticklabels = LABELS,
            yticklabels = LABELS, annot = True, fmt ="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()