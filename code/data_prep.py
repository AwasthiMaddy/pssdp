
# coding: utf-8

# In[1]:


import os
import time
import itertools
from collections import Counter
import scipy.stats as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, mean_absolute_error, accuracy_score
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb

import utils


# In[2]:


BASE_DIR = "/nfs/science/shared/ipythonNotebooks/abhiawa/DS/pssdp"


# ### Reading Data & combined_df

# In[3]:


train = pd.read_csv(os.path.join(BASE_DIR, 'data', 'train.zip'))
test = pd.read_csv(os.path.join(BASE_DIR, 'data', 'test.zip'))
sample_submission = pd.read_csv(os.path.join(BASE_DIR, 'data', 'sample_submission.zip'))
print('Train Shape:', train.shape)
print('Test Shape:', test.shape)
print('Sample Submission Shape:', sample_submission.shape)


# In[4]:


# combined_df
combined_df = utils.create_combined_df(train, test)
print('combined_df Shape:', combined_df.shape)
# Sort combined_df on the basis of id
combined_df = combined_df.sort_values(by='id', axis=0, ascending=True).reset_index(drop=True)
# Changing columns order
cols = ['id','train_or_test','target']
for c in combined_df.columns:
    if c not in cols:
        cols.append(c)
combined_df = combined_df[cols]
combined_df.head(2)


# In[5]:


Counter(combined_df.dtypes.values)


# ## Data Exploration & Feature Engineering

# In[6]:


# Features
feats = []
for c in combined_df.columns:
    if c not in ['id','train_or_test','target']:
        feats.append(c)
print(len(feats))


# In[7]:


get_ipython().run_cell_magic('time', '', "def count_minus1(row):\n    count = 0\n    for r in row:\n        if r==-1:\n            count+=1\n    return count\n\n# Counting NaNs\ncombined_df['num_nans_rowwise'] = combined_df[feats].apply(count_minus1, axis=1, raw=True)")


# The features with suffix `_cat` are categorical features. We will OHE them.

# In[8]:


cat_feats = [col for col in combined_df.columns if '_cat' in col]
print(cat_feats)
combined_df = pd.get_dummies(combined_df, columns=cat_feats)
combined_df.head(2)


# ### Adding Feature Interactions got by XGBFI

# In[9]:


def create_interaction_feats_from_xgbfi(df, xgbfi_fpath, sheetname='Interaction Depth 1', imp_type='Gain', 
                                        num_top_interaction_topick=5):
    '''
    Creates Interaction Features on the basis of XGBFI's XgbFeatureInteractions.xlsx
    
    Params
    ------
    df: Dataframe which has features and where interaction features would be added
    xgbfi_path: XGBFI excel file path (XgbFeatureInteractions.xlsx file path)
    sheetname: sheetname in the excel file to be read (can be 'Interaction Depth 1'/'Interaction Depth 2')
    imp_type: Feature Importance type to consider (can be 'Gain'/'FScore'/'wFScore')
    num_top_interaction_topick: Number of top interactions to pick
    '''
    # Interaction depth 1
    feats_interaction_df = pd.read_excel(xgbfi_fpath, sheetname)
    feats_interaction_df.sort_values(by=imp_type, axis=0, inplace=True, ascending=False)
    feats_interactions = feats_interaction_df.iloc[:num_top_interaction_topick]['Interaction'].values.tolist()
    inter_feats = []
    for f_int in feats_interactions:
        feats = f_int.split('|')
        feats_arrays_dict = {f: df[f].values.reshape((-1,)) for f in feats}
        print('Features arrays: ', feats_arrays_dict)
        inter_feats_df = pd.DataFrame(create_2way_interactions(feats_arrays_dict))
        df = pd.concat([df, inter_feats_df], axis=1)
        inter_feats += inter_feats_df.columns.tolist()
    return df, inter_feats

def give_feats_type(feats_arrays_dict):
    feats_type = {}
    for k,v in feats_arrays_dict.items():
        unique_vals = len(np.unique(v))
        if 'float' in str(v.dtype):
            feats_type[k] = 'numeric'
        elif ('int' in str(v.dtype)) & (unique_vals==2):
            feats_type[k] = 'binary'
        elif ('int' in str(v.dtype)) & (2<unique_vals<100):
            feats_type[k] = 'categorical'
        elif ('int'in str(v.dtype)) & (unique_vals>100):
            feats_type[k] = 'numeric'
    return feats_type

def both_numeric_feats_interactions(feats_arrays_dict):
    '''
    Works for (num,num).
    '''
    feats_names_list = list(feats_arrays_dict.keys())
    arrays_list = list(feats_arrays_dict.values())
    arr1, arr2  = arrays_list[0], arrays_list[1]
    n1, n2 = feats_names_list[0], feats_names_list[1]
    
    inter_feats_dict = {}
    inter_feats_dict[n1+'_add_'+n2] = arr1+arr2
    inter_feats_dict[n1+'_sub_'+n2] = arr1-arr2
    inter_feats_dict[n1+'_mul_'+n2] = arr1*arr2
    inter_feats_dict[n1+'_div_'+n2] = arr1/arr2
    print('Interaction features created: {} \n'.format(inter_feats_dict))
    return inter_feats_dict

def both_categorical_feats_interaction(feats_arrays_dict):
    '''
    Works for (cat,cat), (cat,bin), (bin,bin)
    '''
    feats_names_list = list(feats_arrays_dict.keys())
    arrays_list = list(feats_arrays_dict.values())
    arr1, arr2  = arrays_list[0], arrays_list[1]
    n1, n2 = feats_names_list[0], feats_names_list[1]
    
    inter_feats_dict = {}
    inter_feats_dict[n1+'_inter_'+n2] = np.char.add(arr1.astype(str), arr2.astype(str))
    print('Interaction feature created: {} \n'.format(inter_feats_dict))
    return inter_feats_dict
    
def numeric_categorical_feats_interaction(feats_arrays_dict, feats_type, q=10):
    '''
    Works for (num,cat), (num,bin)
    '''
    for k,v in feats_type.items():
        if v=='numeric':
            feats_arrays_dict[k] = pd.qcut(feats_arrays_dict[k], q=q, labels=False, duplicates='drop').reshape((-1,))
    return both_categorical_feats_interaction(feats_arrays_dict)

def create_2way_interactions(feats_arrays_dict):
    '''
    Features are in the form of np arrays and in a dict. So dict can have any size from 2 to whatever!
    Restricted to 2 features only for time being
    '''
    feats_type = give_feats_type(feats_arrays_dict)
    print('Features type: ', feats_type)
    if len(feats_type)==2:
        f_type = set(feats_type.values())
        if f_type=={'numeric'}:
            inter_feats_dict = both_numeric_feats_interactions(feats_arrays_dict)
        elif (f_type=={'binary','numeric'}) | (f_type=={'categorical','numeric'}):
            inter_feats_dict = numeric_categorical_feats_interaction(feats_arrays_dict, feats_type, q=10)
        else:
            inter_feats_dict = both_categorical_feats_interaction(feats_arrays_dict)
        return inter_feats_dict
    else:
        print('This function works only for 2 features. But you have passed {}'.format(feats_type))


# In[10]:


combined_df, new_feats =         create_interaction_feats_from_xgbfi(combined_df, os.path.join(BASE_DIR, 'models', 'XgbFeatureInteractions.xlsx'),                                             num_top_interaction_topick=5)
print('New interaction features created {}'.format(new_feats))


# In[11]:


get_ipython().run_cell_magic('time', '', "# Saving combined_df\n#combined_df.to_csv(os.path.join(BASE_DIR,'data','combined_df.csv'), index=False)\n# Saving 2-way interaction features only\ncombined_df[new_feats].to_csv(os.path.join(BASE_DIR,'features','2way_interaction_features_top5.csv'), index=False)")


# ## Model Building

# In[6]:


# Columns
cols_to_remove = ['id','target','train_or_test']
cols = [col for col in combined_df.columns.tolist() if col not in cols_to_remove]
print(len(cols))
print(cols)


# In[7]:


X_train = combined_df.loc[combined_df.train_or_test=='train', cols].values
y_train = combined_df.loc[combined_df.train_or_test=='train', 'target'].values.reshape((-1,))
X_test = combined_df.loc[combined_df.train_or_test=='test', cols].values
#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.35, random_state = 42, stratify=y_train)
print(X_train.shape, X_test.shape, y_train.shape)


# ### Stacking

# In[8]:


def transformer(y, func=None):
    """Transforms target variable and prediction"""
    if func is None:
        return y
    else:
        return func(y)

def stacking(models, X_train, y_train, X_test, regression=True,
             transform_target=None, transform_pred=None,
             metric=None, n_folds=4, stratified=False,
             shuffle=False, random_state=0, verbose=0):
   
    # Print type of task
    if regression and verbose > 0:
        print('task:   [regression]')
    elif not regression and verbose > 0:
        print('task:   [classification]')

    # Specify default metric for cross-validation
    if metric is None and regression:
        metric = mean_absolute_error
    elif metric is None and not regression:
        metric = accuracy_score
        
    # Print metric
    if verbose > 0:
        print('metric: [%s]\n' % metric.__name__)
        
    # Split indices to get folds (stratified can be used only for classification)
    if stratified and not regression:
        kf = StratifiedKFold(y_train, n_folds, shuffle = shuffle, random_state = random_state)
    else:
        kf = KFold(len(y_train), n_folds, shuffle = shuffle, random_state = random_state)

    # Create empty numpy arrays for stacking features
    S_train = np.zeros((X_train.shape[0], len(models)))
    S_test = np.zeros((X_test.shape[0], len(models)))
    
    # Loop across models
    for model_counter, model in enumerate(models):
        if verbose > 0:
            print('model %d: [%s]' % (model_counter, model.__class__.__name__))
            
        # Create empty numpy array, which will contain temporary predictions for test set made in each fold
        S_test_temp = np.zeros((X_test.shape[0], len(kf)))
        
        # Loop across folds
        for fold_counter, (tr_index, te_index) in enumerate(kf):
            X_tr = X_train[tr_index]
            y_tr = y_train[tr_index]
            X_te = X_train[te_index]
            y_te = y_train[te_index]
            
            # Fit 1-st level model
            model = model.fit(X_tr, transformer(y_tr, func = transform_target))
            # Predict out-of-fold part of train set
            S_train[te_index, model_counter] = transformer(model.predict_proba(X_te)[:,1], func = transform_pred)
            # Predict full test set
            S_test_temp[:, fold_counter] = transformer(model.predict_proba(X_test)[:,1], func = transform_pred)
            
            if verbose > 1:
                print('    fold %d: [%.8f]' % (fold_counter, metric(y_te, S_train[te_index, model_counter])))
                
        # Compute mean or mode of predictions for test set
        if regression:
            S_test[:, model_counter] = np.mean(S_test_temp, axis = 1)
        else:
            S_test[:, model_counter] = np.mean(S_test_temp, axis = 1)#[0].ravel()
            
        if verbose > 0:
            print('    ----')
            print('    MEAN:   [%.8f]\n' % (metric(y_train, S_train[:, model_counter])))

    return (S_train, S_test)


# In[34]:


# Older Gini Implementation
def gini(y_true, y_pred):
    g = np.asarray(np.c_[y_true, y_pred, np.arange(len(y_true)) ], dtype=np.float)
    g = g[np.lexsort((g[:,2], -1*g[:,1]))]
    gs = g[:,0].cumsum().sum() / g[:,0].sum()
    gs -= (len(y_true) + 1) / 2.
    return gs / len(y_true)

def normalized_gini(y_true, y_pred):
    return gini(y_true, y_pred) / gini(y_true, y_true)


# In[26]:


# Faster Gini Implementation
def ginic(actual, pred):
    actual = np.asarray(actual) #In case, someone passes Series or list
    n = len(actual)
    a_s = actual[np.argsort(pred)]
    a_c = a_s.cumsum()
    giniSum = a_c.sum() / a_s.sum() - (n + 1) / 2.0
    return giniSum / n
 
def gini_normalizedc(a, p):
    return ginic(a, p) / ginic(a, a)


# **1-st level models**

# In[40]:


# Creating models list
models = []
for tup in itertools.product([3,4,5],[0.8,0.9],[0.6,0.8,0.95]):
    # In tuple first is max_depth, second is subsample, third is colsample_bytree
    models.append(xgb.XGBClassifier(random_state=42, n_jobs=-1, learning_rate=0.1, n_estimators=200, missing=-1,
                                    max_depth=tup[0], subsample=tup[1], colsample_bytree=tup[2]))


# In[25]:


'''models = [#ExtraTreesClassifier(random_state=42, n_jobs=-1, n_estimators = 150, max_depth = 20),

          #RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators = 150, max_depth = 20),

          lgb.LGBMClassifier(random_state=42, n_jobs=-1, learning_rate = 0.02, n_estimators = 800, max_depth=6, 
                             num_leaves=150, max_bin=300),
          
          xgb.XGBClassifier(random_state=42, n_jobs=-1, learning_rate=0.02, n_estimators=850, max_depth=4, subsample=0.9, 
                            colsample_bytree=0.75),
          
          LogisticRegression(penalty='l2', solver='liblinear', fit_intercept=True)]'''


# In[41]:


get_ipython().run_cell_magic('time', '', 'S_train, S_test = stacking(models, X_train, y_train, X_test, regression=False, metric=gini_normalizedc, stratified=True,\n                           n_folds=5, verbose=2, shuffle=True)')


# ** 2nd level models **

# In[46]:


S_train = np.hstack((X_train, S_train))
S_test = np.hstack((X_test, S_test))
print(S_train.shape, X_train.shape)
print(S_test.shape, X_test.shape)


# In[49]:


#model = lgb.LGBMClassifier(seed=42, nthread=-1, learning_rate = 0.02, n_estimators = 100, max_depth=6, num_leaves=50)
model = xgb.XGBClassifier(random_state=42, n_jobs=-1, learning_rate=0.02, n_estimators=100, max_depth=4, subsample=0.9, 
                         colsample_bytree=0.8)
skf = StratifiedKFold(y_train, n_folds=5, random_state=42)

pred_val = []
i=0
for train_index, val_index in skf:
    print('Fold:',i)
    new_S_train, new_y_train = S_train[train_index], y_train[train_index]
    new_S_val, new_y_val = S_train[val_index], y_train[val_index]
    #print(new_S_train.shape, new_y_train.shape)
    #print(new_S_val.shape, new_y_val.shape)
    new_model = model.fit(new_S_train, new_y_train)
    # Predict
    y_pred = new_model.predict_proba(new_S_val)[:,1]
    norm_gini = gini_normalizedc(new_y_val, y_pred)
    pred_val.append(norm_gini)
    print('Normalized Gini Coefficient is:', norm_gini)
    i+=1
print('Mean is:', np.mean(pred_val))


# In[50]:


# Train on full data & predict & generate submission
model = model.fit(S_train, y_train)
# Predict
y_pred = model.predict_proba(S_test)[:,1]
print(y_pred.shape)


# In[56]:


sub = combined_df.loc[combined_df.train_or_test=='test', ['id']]
sub['target'] = y_pred
print(sub.shape)
sub.head()


# In[ ]:


sub.to_csv(os.path.join(BASE_DIR,'submissions','ensembled_xgbs'), index=False, compression='gzip')

