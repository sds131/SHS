import numpy as np
import pandas as pd
import time
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import minmax_scale
import warnings
import random
        

SHS2=[]
nonSHS2=[]
with open("/users/sds/downloads/SHS2_top1000.txt","r") as f:
    list3=f.readlines()
    for i in range(0, len(list3)):
        list3[i] = list3[i].strip('\n')
        SHS2.append(list3[i])
        
with open("/users/sds/downloads/nonSHS2_random1000.txt","r") as f:
    list3=f.readlines()
    for i in range(0, len(list3)):
        list3[i] = list3[i].strip('\n')
        nonSHS2.append(list3[i])        
        
        
#SHS2=random.sample(SHS2, 1000)
#nonSHS2=random.sample(nonSHS2, 1000)


venue1={}
venue2={}
with open("/users/sds/downloads/dataset_WWW2019/dataset_WWW_Checkins_anonymized.txt","r") as f:
    list1 = f.readlines()
    for i in range(0, len(list1)):
        list1[i] = list1[i].strip('\n')
        list1[i] =list1[i].split( )
    for i in list1:
        if i[0] in SHS2:
            if i[1] not in venue1.keys():
                venue1[i[1]]=1
            else:
                venue1[i[1]]+=1
        elif i[0] in nonSHS2:
            if i[1] not in venue2.keys():
                venue2[i[1]]=1
            else:
                venue2[i[1]]+=1

            
            
with open("/users/sds/downloads/venue1.txt","w") as f:
    for i in venue1:
        f.write(i)
        f.write(' ')
        f.write(str(venue1[i]))
        f.write('\n')
        
with open("/users/sds/downloads/venue2.txt","w") as f:
    for i in venue2:
        f.write(i)
        f.write(' ')
        f.write(str(venue2[i]))
        f.write('\n')
        
            
with open("/users/sds/downloads/dataset_WWW2019/raw_POIs.txt","r") as f:
    list2 = f.readlines()
    for i in range(0, len(list2)):
        list2[i] = list2[i].strip('\n')
        list2[i] =list2[i].split( )
        

###
checklist1=[]
for i in list1:
    if i[0] in SHS2:
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
        checklist1.append(k)
        
checklist2=[]
for i in list1:
    if i[0] in nonSHS2:
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
        checklist2.append(k)
        
with open("/users/sds/downloads/checklist_SHS2.txt","w") as f:
    for i in checklist1:
        for j in range(0, len(i)):
            f.write(str(i[j]))
            f.write(' ')  
        f.write('\n')
        
with open("/users/sds/downloads/checklist_nonSHS2.txt","w") as f:
    for i in checklist2:
        for j in range(0, len(i)):
            f.write(str(i[j]))
            f.write(' ')  
        f.write('\n')
        
        
        
venuelist1=[]
for i in list2:
    if i[0] in venue1.keys():
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
        venuelist1.append(k)
        
venuelist2=[]
for i in list2:
    if i[0] in venue2.keys():
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
        venuelist2.append(k)
        
with open("/users/sds/downloads/venuelist_SHS2.txt","w") as f:
    for i in venuelist1:
        for j in range(0, len(i)):
            f.write(str(i[j]))
            f.write(' ')  
        f.write('\n')
        
with open("/users/sds/downloads/venuelist_nonSHS2.txt","w") as f:
    for i in venuelist2:
        for j in range(0, len(i)):
            f.write(str(i[j]))
            f.write(' ')  
        f.write('\n')
        
###        
userlist1={}
for i in checklist1:   
    for j in venuelist1:
        if i[1]==j[0]:           
            if i[0] not in userlist1:
                userlist1[i[0]]={}
                userlist1[i[0]]['checkin']=[]
                userlist1[i[0]]['checkin'].append(float(i[5]))
                userlist1[i[0]]['country']={}
                userlist1[i[0]]['country'][j[2]]=1
                userlist1[i[0]]['poi']={}
                userlist1[i[0]]['poi'][i[1]]=1
                userlist1[i[0]]['category']=[]
                userlist1[i[0]]['category'].append(j[1])
                userlist1[i[0]]['hour']=[]
                userlist1[i[0]]['hour'].append(i[4])
                userlist1[i[0]]['day']=[]
                userlist1[i[0]]['day'].append(i[2])
                userlist1[i[0]]['month']=[]
                userlist1[i[0]]['month'].append(i[3])
                userlist1[i[0]]['lalong']=[]
                userlist1[i[0]]['lalong'].append([float(j[3]),float(j[4])])
            else:
                userlist1[i[0]]['checkin'].append(float(i[5]))
                if j[2] not in userlist1[i[0]]['country']:
                    userlist1[i[0]]['country'][j[2]]=1
                else:
                    userlist1[i[0]]['country'][j[2]]+=1
                if i[1] not in userlist1[i[0]]['poi']:
                    userlist1[i[0]]['poi'][i[1]]=1
                else:
                    userlist1[i[0]]['poi'][i[1]]+=1
                userlist1[i[0]]['category'].append(j[1])
                userlist1[i[0]]['hour'].append(i[4])
                userlist1[i[0]]['day'].append(i[2])
                userlist1[i[0]]['month'].append(i[3])
                userlist1[i[0]]['lalong'].append([float(j[3]),float(j[4])])
                

userlist2={}
for i in checklist2:   
    for j in venuelist2:
        if i[1]==j[0]:           
            if i[0] not in userlist2:
                userlist2[i[0]]={}
                userlist2[i[0]]['checkin']=[]
                userlist2[i[0]]['checkin'].append(float(i[5]))
                userlist2[i[0]]['country']={}
                userlist2[i[0]]['country'][j[2]]=1
                userlist2[i[0]]['poi']={}
                userlist2[i[0]]['poi'][i[1]]=1
                userlist2[i[0]]['category']=[]
                userlist2[i[0]]['category'].append(j[1])
                userlist2[i[0]]['hour']=[]
                userlist2[i[0]]['hour'].append(i[4])
                userlist2[i[0]]['day']=[]
                userlist2[i[0]]['day'].append(i[2])
                userlist2[i[0]]['month']=[]
                userlist2[i[0]]['month'].append(i[3])
                userlist2[i[0]]['lalong']=[]
                userlist2[i[0]]['lalong'].append([float(j[3]),float(j[4])])
            else:
                userlist2[i[0]]['checkin'].append(float(i[5]))
                if j[2] not in userlist2[i[0]]['country']:
                    userlist2[i[0]]['country'][j[2]]=1
                else:
                    userlist2[i[0]]['country'][j[2]]+=1
                if i[1] not in userlist2[i[0]]['poi']:
                    userlist2[i[0]]['poi'][i[1]]=1
                else:
                    userlist2[i[0]]['poi'][i[1]]+=1
                userlist2[i[0]]['category'].append(j[1])
                userlist2[i[0]]['hour'].append(i[4])
                userlist2[i[0]]['day'].append(i[2])
                userlist2[i[0]]['month'].append(i[3])
                userlist2[i[0]]['lalong'].append([float(j[3]),float(j[4])])


#计算信息熵的方法
def calc_ent(x):
    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]
        logp = np.log2(p)
        ent -= p * logp
    return ent


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

sdata = {'num_checkin':[],
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
       }


for i in userlist1:     
    data1=np.array(userlist1[i]['hour'])
    data2=np.array(userlist1[i]['day'])
    data3=np.array(userlist1[i]['month'])
    data4=np.array(userlist1[i]['category'])
    sdata['num_checkin'].append(len(userlist1[i]['checkin']))
    sdata['ave_checkin'].append(ave_diff(userlist1[i]['checkin']))
    sdata['var_checkin'].append(np.var(userlist1[i]['checkin']))  
    sdata['ave_checkin_interval'].append(ave_diff(userlist1[i]['checkin']))
    sdata['var_checkin_interval'].append(var_diff(userlist1[i]['checkin']))
    sdata['entropy_hour'].append(calc_ent(data1)) 
    sdata['entropy_day'].append(calc_ent(data2)) 
    sdata['entropy_month'].append(calc_ent(data3)) 
    sdata['entropy_category'].append(calc_ent(data4)) 
    for key,value in userlist1[i]['country'].items():
        if(value == max(userlist1[i]['country'].values())):
            sdata['main_country'].append(key)
            break
    sdata['num_poi_visit'].append(len(userlist1[i]['poi']))
    a = np.mat(userlist1[i]['lalong'])
    temp=np.mean(a,axis=0)
    sdata['ave_latitude'].append(temp[0,0])
    sdata['ave_longitude'].append(temp[0,1])
    temp=np.var(a,axis=0)
    sdata['var_latitude'].append(temp[0,0])
    sdata['var_longitude'].append(temp[0,1])
    sdata['constraint'].append(float(shs_constraint[i]))
    sdata['effective_size'].append(float(shs_effective_size[i]))
    sdata['efficiency'].append(float(shs_efficiency[i]))
    sdata['hierarchy'].append(float(shs_hierarchy[i]))
    sdata['betweenness_centrality'].append(shs_betweenness_centrality[i])
    sdata['degree'].append(shs_degree[i])
    
test=pd.DataFrame(sdata)
test.to_csv('/users/sds/downloads/SHS2_top1000.csv')

ndata = {'num_checkin':[],
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
       }


for i in userlist2:     
    data1=np.array(userlist2[i]['hour'])
    data2=np.array(userlist2[i]['day'])
    data3=np.array(userlist2[i]['month'])
    data4=np.array(userlist2[i]['category'])
    ndata['num_checkin'].append(len(userlist2[i]['checkin']))
    ndata['ave_checkin'].append(np.mean(userlist2[i]['checkin']))
    ndata['var_checkin'].append(np.var(userlist2[i]['checkin'])) 
    ndata['ave_checkin_interval'].append(ave_diff(userlist2[i]['checkin']))
    ndata['var_checkin_interval'].append(var_diff(userlist2[i]['checkin']))
    ndata['entropy_hour'].append(calc_ent(data1)) 
    ndata['entropy_day'].append(calc_ent(data2)) 
    ndata['entropy_month'].append(calc_ent(data3)) 
    ndata['entropy_category'].append(calc_ent(data4)) 
    for key,value in userlist2[i]['country'].items():
        if(value == max(userlist2[i]['country'].values())):
            ndata['main_country'].append(key)
            break
    ndata['num_poi_visit'].append(len(userlist2[i]['poi']))
    a = np.mat(userlist2[i]['lalong'])
    temp=np.mean(a,axis=0)
    ndata['ave_latitude'].append(temp[0,0])
    ndata['ave_longitude'].append(temp[0,1])
    temp=np.var(a,axis=0)
    ndata['var_latitude'].append(temp[0,0])
    ndata['var_longitude'].append(temp[0,1])
    ndata['constraint'].append(float(shs_constraint[i]))
    ndata['effective_size'].append(float(shs_effective_size[i]))
    ndata['efficiency'].append(float(shs_efficiency[i]))
    ndata['hierarchy'].append(float(shs_hierarchy[i]))
    ndata['betweenness_centrality'].append(shs_betweenness_centrality[i])
    ndata['degree'].append(shs_degree[i])
    
test=pd.DataFrame(ndata)
test.to_csv('/users/sds/downloads/nonSHS2_top1000.csv')



with open("/Users/sds/Downloads/nonSHS2.csv","r") as f:
    list1 = f.readlines()
    for i in range(0, len(list1)):
        list1[i] = list1[i].strip('\n')
        list1[i] = list1[i].split( )
        
with open("/Users/sds/Downloads/SHS2.csv","r") as f:
    list2 = f.readlines()
    for i in range(0, len(list2)):
        list2[i] = list2[i].strip('\n')
        list2[i] = list2[i].split( )
        
dict1={}
dict2={}
for i in sdata['main_country']:
    if i not in dict1:
        dict1[i]=1
    else:
        dict1[i]+=1
for i in ndata['main_country']:
    if i not in dict2:
        dict2[i]=1
    else:
        dict2[i]+=1    

dict1=sorted(dict1.items(),key=lambda d: d[1],reverse=True)
dict2=sorted(dict2.items(),key=lambda d: d[1],reverse=True)

print(dict1)
print(dict2)

import scipy
x,y=scipy.stats.ttest_ind(ndata['degree'], sdata['degree'], axis=0, equal_var=False)

print("statistic:",x,"pvalue:",y)
