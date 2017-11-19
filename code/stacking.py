
# coding: utf-8

# In[17]:


from __future__ import print_function, division
import os
import gc
import time
import multiprocessing
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, mean_absolute_error, log_loss
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb
import xgboost as xgb

import utils


# In[3]:


BASE_DIR = "/nfs/science/shared/ipythonNotebooks/abhiawa/DS/pssdp"
num_cores = multiprocessing.cpu_count()
cores_to_use = int(num_cores/2)
print('Total cpus in the machine:', num_cores)
print('cpus to use:', cores_to_use)


# In[4]:


# A naive time function
def time_elapsed(t0, str_result=False):
    t = np.round((time.time()-t0)/60, 3)
    if str_result==True:
        return str(t)+' minutes elapsed!'
    else:
        return t


# In[ ]:


start_time = time.time()


# In[5]:


# Load combined_df
combined_df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'combined_df.gz'))
print(combined_df.shape)


# In[6]:


# Remove inf values 
combined_df = combined_df.replace([np.inf, -np.inf], -1)


# In[7]:


# Columns
cols_to_remove = ['id','target','train_or_test'] #
cols = [col for col in combined_df.columns.tolist() if col not in cols_to_remove]
print(len(cols))


# In[8]:


trainX = combined_df.loc[combined_df.train_or_test=='train', cols].values
trainY = combined_df.loc[combined_df.train_or_test=='train', 'target'].values.reshape((-1,))
testX = combined_df.loc[combined_df.train_or_test=='test', cols].values
#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.35, random_state = 42, stratify=y_train)
print(trainX.shape, testX.shape, trainY.shape)
#del combined_df
gc.collect()


# In[9]:


# Checks
print(np.any(np.isnan(trainX)))   # Should be False
print(np.all(np.isfinite(trainX))) # Should be True


# In[15]:


def gini_xgb(pred, y):
    y = y.get_label()
    g = 2 * roc_auc_score(y, pred) - 1
    return 'gini', g

def gini_for_stacking(y, pred):
    g = 2 * roc_auc_score(y, pred) - 1
    return g


# In[11]:


# Single Lasso Model
t0 = time.time()

scaler = StandardScaler()
clf = LogisticRegression(penalty='l1', C=0.009, random_state=42, solver='saga', n_jobs=cores_to_use, max_iter=200)
clf.fit(scaler.fit_transform(trainX), trainY)
imp_feats_ind = np.nonzero(clf.coef_[0])[0]
print('Number of features left:', len(imp_feats_ind))
cols = np.array(cols)[imp_feats_ind]

trainX = trainX[:,imp_feats_ind]
testX = testX[:,imp_feats_ind]

print(time_elapsed(t0, str_result=True))


# In[21]:


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


# **1-st level models**

# In[22]:


models = [KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski', n_jobs=cores_to_use),
          
          KNeighborsClassifier(n_neighbors=3, p=1, metric='minkowski', n_jobs=cores_to_use),
    
          lgb.LGBMClassifier(random_state=42, n_jobs=cores_to_use, learning_rate=0.05, n_estimators=400, max_depth=5, 
                             num_leaves=150, max_bin=300),
          
          xgb.XGBClassifier(random_state=42, n_jobs=cores_to_use, learning_rate=0.05, n_estimators=400, max_depth=4, 
                            subsample=0.9, colsample_bytree=0.75),
          
          LogisticRegression(penalty='l2', solver='liblinear', C=0.01, fit_intercept=True),
          
          ExtraTreesClassifier(random_state=42, n_jobs=cores_to_use, n_estimators=150, max_depth=15),

          RandomForestClassifier(random_state=42, n_jobs=cores_to_use, n_estimators=150, max_depth=15),
         
          lgb.LGBMClassifier(random_state=42, n_jobs=cores_to_use, learning_rate=0.05, n_estimators=300, max_depth=4, 
                             num_leaves=120, max_bin=255),
          
          xgb.XGBClassifier(random_state=42, n_jobs=cores_to_use, learning_rate=0.05, n_estimators=400, max_depth=4, 
                            subsample=0.95, colsample_bytree=0.5, gamma=2, min_child_weight=60, alpha=0.7, scale_pos_weight=1.15),
          
          LogisticRegression(penalty='l1', solver='liblinear', C=0.01, fit_intercept=True),
         
          LogisticRegression(penalty='l2', solver='liblinear', C=0.1, fit_intercept=True),
          
          LogisticRegression(penalty='l2', solver='liblinear', C=2, fit_intercept=True)]


# In[ ]:


t0 = time.time()
S_train, S_test = stacking(models, trainX, trainY, testX, regression=False, metric=gini_for_stacking, stratified=True,
                           n_folds=10, verbose=2, shuffle=True)
print(time_elapsed(t0, str_result=True))


# ** 2nd level models **

# In[1]:


# S_train = np.hstack((X_train, S_train))
# S_test = np.hstack((X_test, S_test))
# print(S_train.shape, X_train.shape)
# print(S_test.shape, X_test.shape)


# In[ ]:


#model = lgb.LGBMClassifier(seed=42, nthread=-1, learning_rate = 0.02, n_estimators = 100, max_depth=6, num_leaves=50)
model = xgb.XGBClassifier(random_state=42, n_jobs=cores_to_use, learning_rate=0.01, n_estimators=200, max_depth=4)

skf = StratifiedKFold(trainY, n_folds=10, random_state=42)

pred_val = []
i=0
for train_index, val_index in skf:
    print('Fold:',i)
    new_S_train, new_y_train = S_train[train_index], trainY[train_index]
    new_S_val, new_y_val = S_train[val_index], trainY[val_index]
    #print(new_S_train.shape, new_y_train.shape)
    #print(new_S_val.shape, new_y_val.shape)
    new_model = model.fit(new_S_train, new_y_train)
    # Predict
    y_pred = new_model.predict_proba(new_S_val)[:,1]
    norm_gini = gini_for_stacking(new_y_val, y_pred)
    pred_val.append(norm_gini)
    print('Normalized Gini Coefficient is:', norm_gini)
    i+=1
print('Mean is:', np.mean(pred_val))


# In[ ]:


# # Train on full data & predict & generate submission
# model = model.fit(S_train, y_train)
# # Predict
# y_pred = model.predict_proba(S_test)[:,1]
# print(y_pred.shape)


# In[ ]:


print(time_elapsed(start_time, str_result=True))

