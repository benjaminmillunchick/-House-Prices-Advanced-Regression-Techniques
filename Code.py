#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn')
import seaborn as sns
from scipy.stats import norm, skew

# importing models
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import cross_val_score


# In[2]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print("Train set size:", train.shape)
print("Test set size:", test.shape)


# # Data Processing
# ### (These are CRUCIAL , non optional steps . Real world data is even more noisy)
# ### Not all data is 'Clean' enough to be used directly for mathematical models. We need to 'Clean' it first !
# Remove Outliers
# Fill up NULL and missing data . Most models can't handle them.

# In[3]:


## saving the ID's for the test data for the very end 
test_id = test['Id']

## taking out the ID column because its not usefull
train.drop(['Id'], axis=1, inplace=True)
test.drop(['Id'], axis=1, inplace=True)


# In[4]:


df = pd.concat([train.SalePrice, np.log(train.SalePrice + 1).rename('LogSalePrice')], axis=1, names=['SalePrice', 'LogSalePrice'])
df.head()


# In[5]:


plt.subplot(1, 2, 1)
sns.distplot(train.SalePrice, kde=False, fit = norm)

plt.subplot(1, 2, 2)
sns.distplot(np.log(train.SalePrice + 1), kde=False, fit = norm)
plt.xlabel('Log SalePrice')


# In[6]:


train_features = train.drop(['SalePrice'], axis=1)
test_features = test
features = pd.concat([train_features, test_features]).reset_index(drop=True)
print("Train set size:", train.shape)
print("Test set size:", test.shape)
print("Features set size:", features.shape)


# In[7]:


categorical_values = ['Exterior1st', 'Exterior2nd', 'KitchenQual', 'SaleType', 
                    'Utilities', 'Functional', 'MSZoning', 'MasVnrType', 'BsmtFinType1',
                    'BsmtFinType2', 'BsmtQual', 'BsmtExposure', 'BsmtCond', 'GarageType',
                    'GarageFinish', 'GarageQual', 'GarageCond', 'FireplaceQu', 'Fence', 
                    'Alley', 'MiscFeature', 'PoolQC', 'MSSubClass', 'LotShape',
                    'LandContour', 'LotConfig', 'LandSlope', 'Neighborhood','Condition1', 
                    'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond',
                    'RoofStyle', 'RoofMatl', 'ExterQual', 'ExterCond', 'Foundation', 'Heating',
                    'HeatingQC', 'CentralAir', 'Electrical', 'PavedDrive', 'SaleCondition', 'Street',] 


# In[8]:


## allows us to call null_count over and over again to see how we are doing on cleaning our data
def null_count():
    nulls = np.sum(features.isnull())
    nullcols = nulls.loc[(nulls != 0)]
    dtypes = features.dtypes
    dtypes2 = dtypes.loc[(nulls != 0)]
    info = pd.concat([nullcols, dtypes2], axis=1).sort_values(by=0, ascending=False)
    print(info)
    print("There are", len(nullcols), "columns with missing values")


# In[9]:


null_count()


# In[10]:


# Since these column are actually a category , using a numerical number will lead the model to assume
# that it is numerical , so we convert to string .
features['MSSubClass'] = features['MSSubClass'].apply(str)
features['YrSold'] = features['YrSold'].astype(str)
features['MoSold'] = features['MoSold'].astype(str)


# In[11]:


## Filling these with the most frequent value in these columns .
features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0]) 
features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])
features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])


# In[12]:


null_count()


# In[13]:


## Filling these columns With most logical value for these columns 
features['Electrical'] = features['Electrical'].fillna("SBrkr")
features['KitchenQual'] = features['KitchenQual'].fillna("TA")
features['Functional'] = features['Functional'].fillna('Typ')   
features["PoolQC"] = features["PoolQC"].fillna("None")


# In[14]:


null_count()


# In[15]:


### Missing data in GarageYrBit most probably means missing Garage , 
### so replace NaN with none and convert to string because year is a category . 
features['GarageYrBlt'] = features['GarageYrBlt'].fillna('None')
features['GarageYrBlt'] = features['GarageYrBlt'].astype(str)
for col in ('GarageArea', 'GarageCars'):
    features[col] = features[col].fillna(0)
for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    features[col] = features[col].fillna('None')
    
### Same with basement
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    features[col] = features[col].fillna('None')


# In[16]:


null_count()


# In[17]:


features['BsmtFullBath'].index[features['BsmtFullBath'].apply(np.isnan)]


# In[18]:


features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))


# In[19]:


null_count()


# In[20]:


features['MiscFeature'].head()


# In[21]:


# all of these had a none option, and now we just fill with that
objects = []
for i in features.columns:
    if features[i].dtype == object:
        objects.append(i)
features.update(features[objects].fillna('None'))


# In[22]:


null_count()


# In[23]:


# if a MasVnrArea value is missing it would make sense that the house just doesn't have any stone work
features['MasVnrArea'] = features['MasVnrArea'].fillna(0)
# same for full and half baths in the bacement. there may just be no bathroom so fill with 0.
features['BsmtFullBath'] = features['BsmtFullBath'].fillna(0)
features['BsmtHalfBath'] = features['BsmtHalfBath'].fillna(0)
# it is also safe to assume that if there is a missing vlaue for BsmtFinSF1 thene the is no finished bacement
features['BsmtFinSF1'] = features['BsmtFinSF1'].fillna(0)
features['BsmtFinSF2'] = features['BsmtFinSF2'].fillna(0)
features['BsmtUnfSF'] = features['BsmtUnfSF'].fillna(0)
features['TotalBsmtSF'] = features['TotalBsmtSF'].fillna(0)


# In[24]:


null_count()


# In[25]:


features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))


# In[26]:


null_count()


# In[27]:


## For ex, if PoolArea = 0 , Then HasPool = 0 too
features['haspool'] = features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
features['has2ndfloor'] = features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
features['hasgarage'] = features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
features['hasbsmt'] = features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
features['hasfireplace'] = features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)


# In[28]:


features.shape


# In[29]:


features.head()


# In[30]:


from sklearn.preprocessing import LabelEncoder
for i in features.columns:
    le_c = LabelEncoder()
    if i in categorical_values:
        features[i] = le_c.fit_transform(features[str(i)])


# In[31]:


## this is here because for some reason it didn't do a good job encoding this column
le_g = LabelEncoder()
features['GarageYrBlt'] = le_c.fit_transform(features['GarageYrBlt'])


# In[32]:


features.head()


# In[33]:


X = features.iloc[:len(train), :]
X_sub = features.iloc[len(train):, :]
X.shape, test.shape, X_sub.shape


# In[34]:


X.head()


# In[35]:


X_sub.head()


# In[36]:


missing_values_dict = {}
for column in features.columns:
    missing_values_dict[column] = features[column].isnull().sum()
no_missing_values = []
missing_values = []
for i in missing_values_dict:
    if missing_values_dict[i] == 0:
        no_missing_values.append(i)
    else:
        missing_values.append(i)
        

print('number of missing values: ' + str(len(missing_values)))
print('number of values not missing: ' + str(len(no_missing_values)))


# In[37]:


score_lr = cross_val_score(linear_model.LinearRegression(), X, train['SalePrice'], cv=8)
np.average(score_lr)


# In[38]:


score_dt = cross_val_score(DecisionTreeRegressor(), X,  train['SalePrice'], cv=8)
np.average(score_dt)


# In[39]:


score_rf = cross_val_score(RandomForestRegressor(n_estimators = 1000), X, train['SalePrice'], cv=8)
np.average(score_rf)


# In[40]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,train['SalePrice'], test_size=0)
# Fitting the Regression model to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 1000, random_state = 0)
regressor.fit(x_train,y_train)


# In[41]:


y_pred = regressor.predict(X_sub)


# In[42]:


sub = pd.DataFrame()
sub['Id'] = test_id
sub['SalePrice'] = y_pred
sub.head()


# In[43]:


sub.to_csv('submission.csv',index=False)


# In[44]:


s = pd.read_csv('submission.csv')


# In[45]:


s.shape


# In[ ]:




