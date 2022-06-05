#%%
import pandas as pd
import numpy as np
import time
import sklearn
import matplotlib.pyplot as plt


from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, auc
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
pd.set_option('display.max_columns', None)
#%%
'''Import cleaned dataset'''
data = pd.read_csv('cleaned_data.csv')
#%%
'''Check missing values'''
# # data = data.dropna()
# data
na_summary = data.isna().sum()
#%%
'''Train set and test set split'''
train_set, test_set = \
    train_test_split(data, test_size= 0.2,random_state=208)

#%%
'''Reindex rows'''
train_set = train_set.reset_index(drop = True)
test_set = test_set.reset_index(drop = True)
variable_dropped = ['Unnamed: 0']
train_set = train_set.drop(variable_dropped, axis =  1)
test_set = test_set.drop(variable_dropped, axis = 1) 
# %%
# pca = PCA(n_components=3)
# pca.fit(train_set)
# %%
'''Random Forest Model'''
train_features = train_set.drop(columns = ['is_canceled'])
train_labels = train_set['is_canceled']
test_features = test_set.drop(columns = ['is_canceled'])
test_labels = test_set['is_canceled']
#%%
'''Randon Forest Fitting'''
start_time = time.time()
# rf = RandomForestRegressor(n_estimators = 1500,max_depth=15,min_samples_leaf=4,\
#     min_samples_split=25,max_features=0.9, random_state = 208)

rf = RandomForestRegressor(n_estimators = 1500, max_depth = None, 
min_samples_split = 20,\
    min_samples_leaf = 5,random_state= 208)
rf.fit(train_features, train_labels)

predictions = rf.predict(test_features)
end_time = time.time()
predictions = pd.DataFrame(predictions)
print('Time used by Random Forest is \
    {:.4f}'.format(end_time - start_time))
#%%
predictions_df = predictions.copy()
predictions_df[predictions_df<0.5] = 0
predictions_df[predictions_df>=0.5] = 1


cm_rf = confusion_matrix(test_labels,predictions_df)
cm_rf = pd.DataFrame(cm_rf)

accuracy_rf = (cm_rf.iloc[0,0]+cm_rf.iloc[1,1])/(cm_rf.sum().sum())*100
print('The TPR is ',accuracy_rf)
# importance_rf = pd.DataFrame(rf.feature_importances_)
#%%
'''ROC Curve'''
def ROC(y_test, y_prob):
    
    false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, y_prob)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    
    plt.figure(figsize = (10,10))
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate,\
         color = 'red', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1], linestyle = '--')
    plt.axis('tight')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

ROC(test_labels,predictions)


#%%
# '''Hyperparameters' tuning'''
# start_time = time.time()

# n_estimators = \
#     [int(x) for x in np.linspace(start = 1000, stop = 1500, num = 5)]
# max_features = [x for x in np.linspace(0, 1, num = 5)]
# # max_depth = [int(x) for x in np.linspace(1, 18, num = 18)]
# max_depth = [None]
# min_samples_split = range(20,30)
# min_samples_leaf = range(1,5)
# # # bootstrap = [True, False]

# param_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split':min_samples_split,
#                'min_samples_leaf': min_samples_leaf,}
# # # # %%
# rf = RandomForestRegressor(random_state= 208)
# grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
#                           cv = 3)
# # # # CV: Number of folds
# # # # n_iter: Number of iteration for trial
# # # #%%
# # # '''Fit the model on the trianing set to get best combination'''
# rf_gs = grid_search.fit(train_features, train_labels)


# # # # %%
# # end_time = time.time()
# # print('Time consumed for parameters tuning is {:.4f}'.format(end_time - start_time))
# # #%%
# # rf_gs.best_params_

# #%%
# rf_gs.best_params_

# #%%
# rf_gs.score(train_features,train_labels)
# rf_gs.score(test_features,test_labels)
