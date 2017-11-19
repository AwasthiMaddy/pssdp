
# coding: utf-8

# In[1]:


from __future__ import print_function, division
import os
import gc
import time
import multiprocessing
import numpy as np
import pandas as pd
from boruta import BorutaPy
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, mean_absolute_error, log_loss
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, space_eval 
from hyperopt.pyll.base import scope

import utils


# In[36]:


BASE_DIR = "/nfs/science/shared/ipythonNotebooks/abhiawa/DS/pssdp"
num_cores = multiprocessing.cpu_count()
cores_to_use = int(num_cores/2)
print('Total cpus in the machine:', num_cores)
print('cpus to use:', cores_to_use)


# In[3]:


# A naive time function
def time_elapsed(t0, str_result=False):
    t = np.round((time.time()-t0)/60, 3)
    if str_result==True:
        return str(t)+' minutes elapsed!'
    else:
        return t


# In[4]:


# Load combined_df
t0 = time.time()
combined_df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'combined_df.gz'))
print(combined_df.shape)
print(time_elapsed(t0, str_result=True))


# In[5]:


# Columns
cols_to_remove = ['id','target','train_or_test']
cols = [col for col in combined_df.columns.tolist() if col not in cols_to_remove]
print(len(cols))


# In[6]:


trainX = combined_df.loc[combined_df.train_or_test=='train', cols].values
trainY = combined_df.loc[combined_df.train_or_test=='train', 'target'].values.reshape((-1,))
testX = combined_df.loc[combined_df.train_or_test=='test', cols].values
#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.35, random_state = 42, stratify=y_train)
print(trainX.shape, testX.shape, trainY.shape)


# In[7]:


def gini_xgb(pred, y):
    y = y.get_label()
    g = 2 * roc_auc_score(y, pred) - 1
    return 'gini', g

# Finding value for scale_pos_weight
demo_scale_pos_weight = len(combined_df[combined_df.target==0])/len(combined_df[combined_df.target==1])
print(demo_scale_pos_weight)

# ### Hyperopt Helper Functions

# In[20]:


def get_space(clf_choice):
    if clf_choice=='LGB':
        lgb_space ={'num_leaves': scope.int(hp.quniform('num_leaves', 70, 200, 1)),
                    'learning_rate': hp.uniform('learning_rate', 0.03, 0.05),
                    'max_bin': scope.int(hp.quniform('max_bin', 200, 500, 1)),
                    'num_boost_round': scope.int(hp.quniform('num_boost_round', 75, 200, 1))}
        return lgb_space
    elif clf_choice=='Lasso':
        lasso_space = {'C': hp.uniform('learning_rate', 0.001, 0.01)}
        return lasso_space
    elif clf_choice=='XGB':
        xgb_space = {'max_depth': scope.int(hp.quniform('max_depth', 3, 5, 1)),
                     'subsample': hp.uniform('subsample', 0.6, 1.0),
                     'colsample_bytree': hp.uniform('colsample_bytree', 0.2, 1.0),
                     'gamma': hp.uniform('gamma', 2, 10),
                     'min_child_weight': scope.int(hp.quniform('min_child_weight', 10, 100, 1)),
                     'alpha': hp.uniform('alpha', 0.1, 1.0),
                     'lambda': hp.uniform('lambda', 0.1, 1.0),
                     'scale_pos_weight': hp.uniform('scale_pos_weight', 1.0, demo_scale_pos_weight)}
        return xgb_space

def model_metrics(y_test, preds_prob, scores):
    auc = roc_auc_score(y_test, preds_prob)
    scores['auc'].append(auc)
    scores['norm_gini'].append(2*auc-1)
    scores['logloss'].append(log_loss(y_test, preds_prob))
    return scores    

def model_train(clf_choice, space, X_train, y_train, X_test):
    if clf_choice=='LGB':
        lgb_params ={'task':'train', 'boosting_type':'gbdt', 'objective':'binary', 'metric': {'auc', 'binary_logloss'},
                     'num_leaves': space['num_leaves'], 'learning_rate': space['learning_rate'], 'max_bin': space['max_bin'], 
                     'nthread':1, 'verbose': 0}
        lgbtrain = lgb.Dataset(X_train, label=y_train)
        lgbtrain.construct()
        model = lgb.train(lgb_params, lgbtrain, num_boost_round=space['num_boost_round'])
        preds_prob = model.predict(X_test, num_iteration=space['num_boost_round'])
    elif clf_choice=='Lasso':
        scaler = StandardScaler()
        model = LogisticRegression(penalty='l1', C=space['C'], random_state=42, solver='saga', n_jobs=cores_to_use, 
                                   max_iter=200)
        model.fit(scaler.fit_transform(X_train), y_train)
        preds_prob = model.predict_proba(scaler.transform(X_test))[:,1]
    elif clf_choice=='XGB':
        xgb_params = {'max_depth': space['max_depth'], 'learning_rate':0.1, 'subsample': space['subsample'], 
                      'colsample_bytree': space['colsample_bytree'], 'eval_metric':'auc', 'objective':'binary:logistic',  
                      'seed':42, 'nthread':cores_to_use, 'gamma': space['gamma'], 'alpha': space['alpha'],
                      'lambda': space['lambda'], 'min_child_weight': space['min_child_weight'], 
                      'scale_pos_weight': space['scale_pos_weight']}
        dtrain = xgb.DMatrix(X_train, label=y_train, missing=-1) 
        dtest = xgb.DMatrix(X_test, missing=-1)
        model = xgb.train(xgb_params, dtrain, num_boost_round=200, feval=gini_xgb, maximize=True)
        preds_prob = model.predict(dtest)
    return model, preds_prob


def hyperopt_param_tuning(space, skf, clf_choice, trainX, trainY, max_evals):
    def objective(space):
        global i
        print('\nIteration:', i)
        scores = {'auc':[], 'norm_gini':[], 'logloss':[]}
        for train_index, test_index in skf.split(trainX, trainY):
            X_train, X_test, y_train, y_test = trainX[train_index], trainX[test_index], trainY[train_index], trainY[test_index]
            model, preds_prob = model_train(clf_choice, space, X_train, y_train, X_test)
            scores = model_metrics(y_test, preds_prob, scores)
            print(scores)
        print('Space is:', space)
        for k in scores:
            print(k, 'is:', np.array(scores[k]).mean())
        i+=1
        return{'loss':1-np.array(scores['norm_gini']).mean(), 'status': STATUS_OK, 'scores':scores}
    
    trials = Trials()
    # Run the hyperparameter search
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    # Get the values of the optimal parameters
    best_params = space_eval(space, best)
    return best_params, trials


# ## Feature Selection

# ### Boruta & Lasso

# In[18]:


'''t0 = time.time()
# Boruta
rf = RandomForestClassifier(n_jobs=cores_to_use, max_depth=5)

# define Boruta feature selection method
feat_selector = BorutaPy(rf, n_estimators=50, verbose=2, random_state=42)
feat_selector.fit(X_train, y_train)

# Selecting cols
cols = np.array(cols)[feat_selector.support_]
print(cols)

# Filtering train & test arrays
X_train = feat_selector.transform(X_train)
X_test = feat_selector.transform(X_test)

print(time_elapsed(t0, str_result=True))'''


# In[ ]:


'''# Lasso Parameter tuning with Hyperopt
t0 = time.time()

i=0
skf = StratifiedKFold(n_splits=3, random_state=42)
clf_choice = 'Lasso'
space = get_space(clf_choice)
tuning_iter=10
best_params, trials = hyperopt_param_tuning(space, skf, clf_choice, trainX, trainY, tuning_iter)
print('Best Lasso Params:', best_params)

print(time_elapsed(t0, str_result=True))'''


# In[10]:


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


# In[11]:


'''def classifiers(trainX, valX, trainY, valY, clf_choice='LGB', n_splits=2, tuning_iter=15):
    skf = StratifiedKFold(n_splits=n_splits, random_state=42)
    if clf_choice=='LGB':
        lgb_space = get_space(clf_choice)
        # Hyperparameter tuning
        lgb_best_params, trials = hyperopt_param_tuning(lgb_space, skf, clf_choice, trainX, trainY, tuning_iter)
        # Train on whole data using best params
        lgb_model, preds_prob = lgb_train(lgb_best_params, trainX, trainY, valX)
        # Evaluating trained model performance
        val_scores = {'auc':[], 'accuracy':[], 'logloss':[]}
        val_scores = model_metrics(valY, preds_prob, val_scores)
        return lgb_model, val_scores, lgb_best_params, preds_prob'''


# ## Single Model

# In[ ]:

'''
# XGB Parameter tuning with Hyperopt
t0 = time.time()

i=0
skf = StratifiedKFold(n_splits=5, random_state=42)
clf_choice = 'XGB'
space = get_space(clf_choice)
tuning_iter=50
best_params, trials = hyperopt_param_tuning(space, skf, clf_choice, trainX, trainY, tuning_iter)
print('Best XGB Params:', best_params)

print(time_elapsed(t0, str_result=True))
'''

# In[24]:


dtrain = xgb.DMatrix(trainX, label=trainY, missing=-1, feature_names=cols)
print(dtrain.num_row(), dtrain.num_col())
dtest = xgb.DMatrix(testX, missing=-1, feature_names=cols)
print(dtest.num_row(), dtest.num_col())


# In[ ]:


del trainX, testX, trainY, combined_df
gc.collect()


# In[49]:


# XGB Params
#params = {'max_depth':4, 'learning_rate':0.05, 'subsample':0.9, 'colsample_bytree':0.9, 'eval_metric':'auc', 
#          'objective':'binary:logistic', 'seed':42, 'nthread':cores_to_use}


# In[27]:

best_params = {'colsample_bytree': 0.47961272611045386, 'scale_pos_weight': 1.147217357619608, 'min_child_weight': 69, 
               'subsample': 0.9798190976022971, 'alpha': 0.6821234362177525, 'max_depth': 4, 'gamma': 2.2260539902559975, 
               'lambda': 0.10281645475554066}

# Best Params from hyperopt
best_params['eta'] = 0.001
best_params['eval_metric'] = 'auc'
best_params['objective'] = 'binary:logistic'
best_params['seed'] = 42
best_params['nthread'] = cores_to_use
print(best_params)


# In[ ]:


t0 = time.time()

# XGBoost CV
xgb_cv = xgb.cv(best_params, dtrain, num_boost_round=50000, nfold=5, stratified=True, early_stopping_rounds=250, verbose_eval=500, 
                seed=42, feval=gini_xgb, maximize=True)
print(xgb_cv.tail(3))
print(time_elapsed(t0, str_result=True))


# In[33]:


#xgb_cv


# In[52]:


# XGBoost Model on whole data
t0 = time.time()
xgb_model = xgb.train(best_params, dtrain, num_boost_round=xgb_cv.shape[0], feval=gini_xgb, maximize=True)
print(time_elapsed(t0, str_result=True))


# In[53]:


# Prediction on dtest
t0 = time.time()

preds_prob = xgb_model.predict(dtest)
print(preds_prob.shape)
# Load sample submission
sub = pd.read_csv(os.path.join(BASE_DIR, 'data', 'sample_submission.zip'))
sub['target'] = preds_prob
print(sub.shape)
print(sub.head(3))

print(time_elapsed(t0, str_result=True))


# In[54]:


# Save submission
t0 = time.time()
sub.to_csv(os.path.join(BASE_DIR, 'submissions', 'single_xgb_hyperopt_tuning_scale_pos_weight_0001.csv'), index=False)
print(time_elapsed(t0, str_result=True))


# In[55]:


# Feature Importance ('weight', 'gain', 'cover')
print(sorted(xgb_model.get_score(importance_type='gain').items(), key=lambda x: x[1], reverse=True))  


# ## Stacking

# In[9]:


'''def transformer(y, func=None):
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

    return (S_train, S_test)'''


# In[ ]:


'''models = [#ExtraTreesClassifier(random_state=42, n_jobs=-1, n_estimators = 150, max_depth = 20),

          #RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators = 150, max_depth = 20),

          lgb.LGBMClassifier(random_state=42, n_jobs=-1, learning_rate = 0.02, n_estimators = 800, max_depth=6, 
                             num_leaves=150, max_bin=300),
          
          xgb.XGBClassifier(random_state=42, n_jobs=-1, learning_rate=0.02, n_estimators=850, max_depth=4, subsample=0.9, 
                            colsample_bytree=0.75),
          
          LogisticRegression(penalty='l2', solver='liblinear', fit_intercept=True)]'''

