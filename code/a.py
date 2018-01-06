
# coding: utf-8

# In[1]:


import os
import gc
import pickle
import time
import datetime
import multiprocessing
import numpy as np
import pandas as pd


# In[2]:


DATA_DIR = '/nfs/science/tesco_uk/abhiawa/fgsf/'
CODE_DIR = '/nfs/science/shared/ipythonNotebooks/abhiawa/DS/fgsf/code/'
SUB_DIR = '/nfs/science/shared/ipythonNotebooks/abhiawa/DS/fgsf/submissions/'

num_cores = multiprocessing.cpu_count()
cores_to_use = int(num_cores/2)
print('Total cpus in the machine:', num_cores)
print('cpus to use:', cores_to_use)


# In[3]:


# A naive time function
def time_elapsed(t0, str_result=True):
    t = np.round((time.time()-t0)/60, 3)
    if str_result==True:
        return str(t)+' minutes elapsed!'
    else:
        return t


# In[4]:


test = pd.read_csv(os.path.join(DATA_DIR, 'test.zip'), parse_dates=['date'], infer_datetime_format=True, 
                   dtype={'onpromotion': bool})
print(test.shape)
sample_sub = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.zip'))
print(sample_sub.shape)
stores = pd.read_csv(os.path.join(DATA_DIR, 'stores.zip'))
print(stores.shape)
transactions = pd.read_csv(os.path.join(DATA_DIR, 'transactions.zip'))
print(transactions.shape)
oil = pd.read_csv(os.path.join(DATA_DIR, 'oil.zip'))
print(oil.shape)
items = pd.read_csv(os.path.join(DATA_DIR, 'items.zip'))
print(items.shape)
holidays = pd.read_csv(os.path.join(DATA_DIR, 'holidays_events.zip'))
print(holidays.shape)


# In[5]:


print(test.head(2),'\n')
print(test.dtypes)
print(test.date.value_counts())


# So we just have to predict sales for 16 days starting from 16th August 2017 to 31st August 2017.

# In training we are reading data from 1st Jan 2016 (although we have data from 2013). 

# In[6]:


# Date starting from 2016-01-01
t0 = time.time()
train = pd.read_csv(os.path.join(DATA_DIR, 'train.zip'), usecols=[1,2,3,4,5], dtype={'onpromotion': bool}, 
                    parse_dates=['date'], infer_datetime_format=True, skiprows=range(1,66458909))
print(train.shape)
print(train.head())
print(time_elapsed(t0))


# In[7]:


# Setting negative sales to zeros
train.loc[train.unit_sales<0, 'unit_sales'] = 0


# In[8]:


# Creating some time features from date
train['day'] = train.date.dt.day
train['month'] = train.date.dt.month
train['year'] = train.date.dt.year
train['dayofweek'] = train.date.dt.dayofweek
train['weekofyear'] = train.date.dt.weekofyear

test['day'] = test.date.dt.day
test['month'] = test.date.dt.month
test['year'] = test.date.dt.year
test['dayofweek'] = test.date.dt.dayofweek
test['weekofyear'] = test.date.dt.weekofyear

# Creating feature with number of days from initial date which here is 2016-01-01
train['initial_date'] = datetime.datetime.strptime('2016-01-01', '%Y-%m-%d')
train['t'] = (train.date.values.astype('datetime64[D]')-train.initial_date.values.astype('datetime64[D]')).astype(int)
train.drop(['date','initial_date'], axis=1, inplace=True)

test['initial_date'] = datetime.datetime.strptime('2016-01-01', '%Y-%m-%d')
test['t'] = (test.date.values.astype('datetime64[D]')-test.initial_date.values.astype('datetime64[D]')).astype(int)
test.drop(['date','initial_date'], axis=1, inplace=True)


# In[9]:


test.head()


# In[10]:


train.head()


# In[12]:


train.groupby(['store_nbr','item_nbr','dayofweek'])['unit_sales'].mean()

