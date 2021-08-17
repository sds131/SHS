import easygraph as eg 
import time
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from matplotlib import  pyplot
from imblearn.over_sampling import SMOTE


G2=eg.Graph()
G2.add_edges_from_file("/users/sds/downloads/dataset_WWW2019/dataset_WWW_friendship_new.txt")
SHS=eg.constraint(G2)
SHS=sorted(SHS.items(), key=lambda d: d[1])


SHS2=[]
nonSHS2=[]

k=0
for i in SHS:
    if k<1000:
        SHS2.append(i[0])
        k+=1
    else:
        break
        
random.seed(10)
while len(nonSHS2)<1000:
    t=random.sample(SHS, 1)
    if t[0][0] not in SHS2 and t[0][0] not in nonSHS2:
        nonSHS2.append(t[0][0])
        

venue={}
with open("/users/sds/downloads/dataset_WWW2019/dataset_WWW_Checkins_anonymized.txt","r") as f:
    list1 = f.readlines()
    for i in range(0, len(list1)):
        list1[i] = list1[i].strip('\n')
        list1[i] =list1[i].split( )
    for i in list1:
        if i[0] in SHS2 or i[0] in nonSHS2:
            if i[1] not in venue.keys():
                venue[i[1]]=1
            else:
                venue[i[1]]+=1
            
            
with open("/users/sds/downloads/venue.txt","w") as f:
    for i in venue:
        f.write(i)
        f.write(' ')
        f.write(str(venue[i]))
        f.write('\n')
        
            
with open("/users/sds/downloads/dataset_WWW2019/raw_POIs.txt","r") as f:
    list2 = f.readlines()
    for i in range(0, len(list2)):
        list2[i] = list2[i].strip('\n')
        list2[i] =list2[i].split( )
        


checklist=[]
for i in list1:
    if i[0] in SHS2 or i[0] in nonSHS2:
        s=i[2]+' '+i[3]+' '+i[4]+' '+i[5]+' '+i[7]
        t=time.mktime(time.strptime(s,"%a %b %d %H:%M:%S %Y"))
        if i[8][0]=='-':
            t-=int(i[8][1:])*60
        else:
            t+=int(i[8])*60
        ss=time.ctime(t) 
        s1=i[0]
        s2=i[1]
        s3=ss[0:3]
        s4=ss[4:7]
        ti=int(int(ss[-13:-11])/6)
        s5=str(ti)
        s6=str(t)
        k=[]
        k.append(s1)#user
        k.append(s2)#venue
        k.append(s3)#week
        k.append(s4)#month
        k.append(s5)#timeslot
        k.append(s6)#time
        checklist.append(k)
        

with open("/users/sds/downloads/checklist.txt","w") as f:
    for i in checklist:
        for j in range(0, len(i)):
            f.write(str(i[j]))
            f.write(' ')  
        f.write('\n')
        

venuelist=[]
for i in list2:
    if i[0] in venue.keys():
        category=''
        for k in range(3,len(i)-1):
            category+=i[k]
            if k!=len(i)-2:
                category +=' '
        k=[]
        k.append(i[0])#venue ID
        k.append(category)#category
        k.append(i[-1])#country
        k.append(i[1])#Latitude
        k.append(i[2])#Longitude
        venuelist.append(k)
        
with open("/users/sds/downloads/venuelist.txt","w") as f:
    for i in venuelist:
        for j in range(0, len(i)):
            f.write(str(i[j]))
            f.write(' ')  
        f.write('\n')
        
userlist={}
for i in checklist:   
    for j in venuelist:
        if i[1]==j[0]:           
            if i[0] not in userlist:
                userlist[i[0]]={}
                userlist[i[0]]['checkin']=[]
                userlist[i[0]]['checkin'].append(float(i[5]))
                userlist[i[0]]['country']={}
                userlist[i[0]]['country'][j[2]]=1
                userlist[i[0]]['poi']={}
                userlist[i[0]]['poi'][i[1]]=1
                userlist[i[0]]['category']=[]
                userlist[i[0]]['category'].append(j[1])
                userlist[i[0]]['hour']=[]
                userlist[i[0]]['hour'].append(i[4])
                userlist[i[0]]['day']=[]
                userlist[i[0]]['day'].append(i[2])
                userlist[i[0]]['month']=[]
                userlist[i[0]]['month'].append(i[3])
                userlist[i[0]]['lalong']=[]
                userlist[i[0]]['lalong'].append([float(j[3]),float(j[4])])
            else:
                userlist[i[0]]['checkin'].append(float(i[5]))
                if j[2] not in userlist[i[0]]['country']:
                    userlist[i[0]]['country'][j[2]]=1
                else:
                    userlist[i[0]]['country'][j[2]]+=1
                if i[1] not in userlist[i[0]]['poi']:
                    userlist[i[0]]['poi'][i[1]]=1
                else:
                    userlist[i[0]]['poi'][i[1]]+=1
                userlist[i[0]]['category'].append(j[1])
                userlist[i[0]]['hour'].append(i[4])
                userlist[i[0]]['day'].append(i[2])
                userlist[i[0]]['month'].append(i[3])
                userlist[i[0]]['lalong'].append([float(j[3]),float(j[4])])

#计算信息熵的方法
def calc_ent(x):
    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]
        logp = np.log2(p)
        ent -= p * logp
    return ent


def ave_diff(list1):
    diff_result=[]
    for i in range(0,len(list1)-1):
        temp=abs(list1[i+1]-list1[i])
        diff_result.append(temp)
    return sum(diff_result)/len(diff_result)
def var_diff(list1):
    diff_result=[]
    for i in range(0,len(list1)-1):
        temp=abs(list1[i+1]-list1[i])
        diff_result.append(temp)
    return np.var(diff_result)

import easygraph as eg
import networkx as nx

G1=eg.Graph()
G1.add_edges_from_file("/users/sds/downloads/dataset_WWW2019/dataset_WWW_friendship_old.txt")

shs_effective_size=eg.effective_size(G1)
shs_efficiency=eg.efficiency(G1)
shs_constraint=eg.constraint(G1)
shs_hierarchy=eg.hierarchy(G1)
shs_degree=G1.degree()

Gnx=nx.Graph()
for i in G1.edges:
    (u,v,t)=i
    Gnx.add_edge(u,v)

shs_betweenness_centrality=nx.betweenness_centrality(Gnx, k=1000)

data = {'num_checkin':[],
        'ave_checkin':[],
        'var_checkin':[],
        'ave_checkin_interval':[],
        'var_checkin_interval':[],
        'entropy_hour':[],
        'entropy_day':[],
        'entropy_month':[],
        'entropy_category':[],
        'main_country':[],
        'num_poi_visit':[],
        'ave_latitude':[],
        'ave_longitude':[],
        'var_latitude':[],
        'var_longitude':[],
        'constraint':[],
        'effective_size':[],
        'efficiency':[],
        'hierarchy':[],
        'betweenness_centrality':[],
        'degree':[],
        'label':[]
       }


for i in userlist:     
    data1=np.array(userlist[i]['hour'])
    data2=np.array(userlist[i]['day'])
    data3=np.array(userlist[i]['month'])
    data4=np.array(userlist[i]['category'])
    data['num_checkin'].append(len(userlist[i]['checkin']))
    data['ave_checkin'].append(ave_diff(userlist[i]['checkin']))
    data['var_checkin'].append(np.var(userlist[i]['checkin']))  
    data['ave_checkin_interval'].append(ave_diff(userlist[i]['checkin']))
    data['var_checkin_interval'].append(var_diff(userlist[i]['checkin']))
    data['entropy_hour'].append(calc_ent(data1)) 
    data['entropy_day'].append(calc_ent(data2)) 
    data['entropy_month'].append(calc_ent(data3)) 
    data['entropy_category'].append(calc_ent(data4)) 
    for key,value in userlist[i]['country'].items():
        if(value == max(userlist[i]['country'].values())):
            data['main_country'].append(key)
            break
    data['num_poi_visit'].append(len(userlist[i]['poi']))
    a = np.mat(userlist[i]['lalong'])
    temp=np.mean(a,axis=0)
    data['ave_latitude'].append(temp[0,0])
    data['ave_longitude'].append(temp[0,1])
    temp=np.var(a,axis=0)
    data['var_latitude'].append(temp[0,0])
    data['var_longitude'].append(temp[0,1])
    data['constraint'].append(float(shs_constraint[i]))
    data['effective_size'].append(float(shs_effective_size[i]))
    data['efficiency'].append(float(shs_efficiency[i]))
    data['hierarchy'].append(float(shs_hierarchy[i]))
    data['betweenness_centrality'].append(shs_betweenness_centrality[i])
    data['degree'].append(shs_degree[i])
    if i in SHS2:
        data['label'].append(1)
    elif i in nonSHS2 :
        data['label'].append(0)

data=pd.DataFrame(data)
data = pd.get_dummies(data)
data_x = data.drop('label',axis = 1)
data_y = data['label']
smo = SMOTE(random_state=38)
data_x, data_y = smo.fit_resample(data_x, data_y)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(data_x,data_y,test_size=0.3,random_state=0)

#性能对比实验增加对比的classifier：Catboost、Random Forest、Logistic Regression、CART、LGBT
#XGBoost
from xgboost import XGBClassifier
parameters_grid_XGB = {
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [10, 50, 100], 
    'max_depth': [4,6,8], 
    'min_child_weight':[1,3,5],
    'subsample': [0.6,0.8,1],
    'colsample_bytree':[0.6,0.8,1], 
    'gamma':[0,2,4],
    
    'booster': ['gbtree'],
    'num_class':[2],
    'eval_metric':["auc"],
    'objective':['multi:softprob'],
    'seed':[0],
    'use_label_encoder':['False']
}

XGB = XGBClassifier()
grid = GridSearchCV(XGB, parameters_grid_XGB, cv=5, scoring='precision')
grid.fit(Xtrain, Ytrain)

print("Best parameters set found on development set:")
print()
print(grid.best_params_)
print()
print("Grid scores on development set:")
print()
means = grid.cv_results_['mean_test_score']
stds = grid.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, grid.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()
print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
bclf = grid.best_estimator_
bclf.fit(Xtrain, Ytrain)
y_true = Ytest
y_pred = bclf.predict(Xtest)
y_pred_xgb = y_pred
y_pred_pro = bclf.predict_proba(Xtest)
y_scores = pd.DataFrame(y_pred_pro, columns=bclf.classes_.tolist())[1].values
print(classification_report(y_true, y_pred, digits=16))
auc_value = roc_auc_score(y_true, y_scores)
print("auc_value:")
print(auc_value)

#Catboost
from catboost import CatBoostClassifier

parameters_grid_Cat={
    'learning_rate': [0.01,0.05,0.1],
    'depth': [6,8,10],
    'iterations':[100,500,1000],
    'l2_leaf_reg':[6,8,10],
    'border_count':[100,150,200],
    'one_hot_max_size':[3,5,7],
    
    'eval_metric':['AUC'],
    'custom_loss':['AUC'],
    'bagging_temperature':[0.83],
    'od_type':['Iter'],
    'rsm': [0.78],
    'od_wait':[150],
    'metric_period': [400],
    'thread_count': [20],
    'random_seed': [38]
}

Cat = CatBoostClassifier()
grid = GridSearchCV(Cat, parameters_grid_Cat, cv=5, scoring='precision')
grid.fit(Xtrain, Ytrain)

print("Best parameters set found on development set:")
print()
print(grid.best_params_)
print()
print("Grid scores on development set:")
print()
means = grid.cv_results_['mean_test_score']
stds = grid.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, grid.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()
print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
bclf = grid.best_estimator_
bclf.fit(Xtrain, Ytrain)
y_true = Ytest
y_pred = bclf.predict(Xtest)
y_pred_cat=y_pred
y_pred_pro = bclf.predict_proba(Xtest)
y_scores = pd.DataFrame(y_pred_pro, columns=bclf.classes_.tolist())[1].values
print(classification_report(y_true, y_pred, digits=16))
auc_value = roc_auc_score(y_true, y_scores)
print("auc_value:")
print(auc_value)


#Random Forest
from sklearn.ensemble import RandomForestClassifier
parameters_grid_RF={
    'max_depth': [4,6,8], 
    'n_estimators': [10, 50, 100], 
    'max_features':['auto', 'sqrt', 'log2'],
    'min_samples_split':[2,4,6],
    'min_samples_leaf':[6,8,10],    
}

RF = RandomForestClassifier()
grid = GridSearchCV(RF, parameters_grid_RF, cv=5, scoring='precision')
grid.fit(Xtrain, Ytrain)

print("Best parameters set found on development set:")
print()
print(grid.best_params_)
print()
print("Grid scores on development set:")
print()
means = grid.cv_results_['mean_test_score']
stds = grid.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, grid.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()
print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
bclf = grid.best_estimator_
bclf.fit(Xtrain, Ytrain)
y_true = Ytest
y_pred = bclf.predict(Xtest)
y_pred_rf=y_pred
y_pred_pro = bclf.predict_proba(Xtest)
y_scores = pd.DataFrame(y_pred_pro, columns=bclf.classes_.tolist())[1].values
print(classification_report(y_true, y_pred, digits=16))
auc_value = roc_auc_score(y_true, y_scores)
print("auc_value:")
print(auc_value)


#CART Decision Tree
from sklearn.tree import DecisionTreeClassifier
parameters_grid_DT = {
    'max_depth': [2, 4, 6, 8, 10], 
    'max_features': ['auto', 'sqrt', 'log2', None], 
    'splitter':['best', 'random']
}

DT = DecisionTreeClassifier()
grid = GridSearchCV(DT, parameters_grid_DT, cv=5, scoring='precision')
grid.fit(Xtrain, Ytrain)

print("Best parameters set found on development set:")
print()
print(grid.best_params_)
print()
print("Grid scores on development set:")
print()
means = grid.cv_results_['mean_test_score']
stds = grid.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, grid.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()
print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
bclf = grid.best_estimator_
bclf.fit(Xtrain, Ytrain)
y_true = Ytest
y_pred = bclf.predict(Xtest)
y_pred_dt=y_pred
y_pred_pro = bclf.predict_proba(Xtest)
y_scores = pd.DataFrame(y_pred_pro, columns=bclf.classes_.tolist())[1].values
print(classification_report(y_true, y_pred, digits=16))
auc_value = roc_auc_score(y_true, y_scores)
print("auc_value:")
print(auc_value)




#Logistic Regression
from sklearn.linear_model import LogisticRegression
parameters_grid_LR={
    'tol': [1e-6,1e-4,1e-2], 
    'penalty': [ 'l1', 'l2', 'elasticnet', 'none'], 
    'C':[1,4,16,64],
        
    #'max_features':['l1', 'l2', 'elasticnet', 'none'],
    #'fit_intercep':['True'], 
    #'normalize':['False'], 
    #'copy_X':['True'], 
    #'n_jobs':['None']
}

LR = LogisticRegression()
grid = GridSearchCV(LR, parameters_grid_LR, cv=5, scoring='precision')
grid.fit(Xtrain, Ytrain)

print("Best parameters set found on development set:")
print()
print(grid.best_params_)
print()
print("Grid scores on development set:")
print()
means = grid.cv_results_['mean_test_score']
stds = grid.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, grid.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()
print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
bclf = grid.best_estimator_
bclf.fit(Xtrain, Ytrain)
y_true = Ytest
y_pred = bclf.predict(Xtest)
y_pred_lr=y_pred
y_pred_pro = bclf.predict_proba(Xtest)
y_scores = pd.DataFrame(y_pred_pro, columns=bclf.classes_.tolist())[1].values
print(classification_report(y_true, y_pred, digits=16))
auc_value = roc_auc_score(y_true, y_scores)
print("auc_value:")
print(auc_value)

#LightGBM
from lightgbm import LGBMClassifier
parameters_grid_LightGBM = {
    'max_depth': [5,6,7,8,9,10], 
    'learning_rate': [0.01, 0.1, 0.2, 0.3], 
    'n_estimators': [10, 20, 50, 100, 200, 500], 
    'num_leaves':[20, 30, 40, 50], 
    'verbose':[-1]
}

LGBM = LGBMClassifier()
grid = GridSearchCV(LGBM, parameters_grid_LightGBM, cv=5, scoring='precision')
grid.fit(Xtrain, Ytrain)

print("Best parameters set found on development set:")
print()
print(grid.best_params_)
print()
print("Grid scores on development set:")
print()
means = grid.cv_results_['mean_test_score']
stds = grid.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, grid.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()
print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
bclf = grid.best_estimator_
bclf.fit(Xtrain, Ytrain)
y_true = Ytest
y_pred = bclf.predict(Xtest)
y_pred_lgbm=y_pred
y_pred_pro = bclf.predict_proba(Xtest)
y_scores = pd.DataFrame(y_pred_pro, columns=bclf.classes_.tolist())[1].values
print(classification_report(y_true, y_pred,digits=16))
auc_value = roc_auc_score(y_true, y_scores)
print("auc_value:")
print(auc_value)

A=0
B=0
C=0
D=0
y_temp1=y_pred_xgb#xgb cat rf dt lr linr lgbm
y_temp2=y_pred_rf
a=0
for i in range(0,600):
    if y_temp1[i]==1 and y_temp2[i]==1:
        A+=1
    elif y_temp1[i]==1 and y_temp2[i]==0:
        B+=1
    elif y_temp1[i]==0 and y_temp2[i]==1:
        C+=1
    elif y_temp1[i]==0 and y_temp2[i]==0:
        D+=1
    a+=1

#McNemar’s test
from statsmodels.sandbox.stats.runs import mcnemar

obs=[[A,B],[C,D]]
(statistic, pVal) = mcnemar(obs)
print('statistic = ',statistic)
print('p = ',pVal)