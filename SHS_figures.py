import easygraph as eg
import networkx as nx
import pandas as pd

G1=eg.Graph()
G1.add_edges_from_file("/users/sds/downloads/dataset_WWW2019/dataset_WWW_friendship_old.txt")
G2=eg.Graph()
G2.add_edges_from_file("/users/sds/downloads/dataset_WWW2019/dataset_WWW_friendship_new.txt")

Gn1=nx.Graph()
for i in G1.edges:
    (u,v,t)=i
    Gn1.add_edge(u,v)

Gn2=nx.Graph()
for i in G2.edges:
    (u,v,t)=i
    Gn2.add_edge(u,v)



#degree
D1=G1.degree()
D2=G2.degree()

data=[]
for i in D1:
    data.append(D1[i])
for i in G1.nodes:
    if i not in D1:
        data.append(0)
test=pd.DataFrame(data)
test.to_csv('/users/sds/downloads/degree1.csv')

data=[]
for i in D2:
    data.append(D2[i])
for i in G2.nodes:
    if i not in D2:
        data.append(0)
test=pd.DataFrame(data)
test.to_csv('/users/sds/downloads/degree2.csv')



#clustering_coefficient
C1=[]
for i in Gn1.nodes:
    k=nx.clustering(Gn1,i)
    C1.append(k)
test=pd.DataFrame(C1)
test.to_csv('/users/sds/downloads/clustering_coefficient1.csv')

C2=[]
for i in Gn2.nodes:
    k=nx.clustering(Gn2,i)
    C2.append(k)  
test=pd.DataFrame(C2)
test.to_csv('/users/sds/downloads/clustering_coefficient2.csv')



#pagerank
P1=nx.pagerank(Gn1,alpha=0.85)
data=[]
for i in P1:
    data.append(P1[i])
test=pd.DataFrame(data)
test.to_csv('/users/sds/downloads/pagerank1.csv')

P2=nx.pagerank(Gn2,alpha=0.85)
data=[]
for i in P2:
    data.append(P2[i])
test=pd.DataFrame(data)
test.to_csv('/users/sds/downloads/pagerank2.csv')    



#connected_components
CC1=[len(c) for c in sorted(nx.connected_components(Gn1), key=len, reverse=True)]
cc1={
    'size_of_cc':[]
}
for i in CC1:
    cc1['size_of_cc'].append(i)
test=pd.DataFrame(cc1)
test.to_csv('/users/sds/downloads/connected_components1.csv')

CC2=[len(c) for c in sorted(nx.connected_components(Gn2), key=len, reverse=True)]
cc2={
    'size_of_cc':[]
}
for i in CC2:
    cc2['size_of_cc'].append(i)
test=pd.DataFrame(cc2)
test.to_csv('/users/sds/downloads/connected_components2.csv')



#path_length
LCC1=nx.Graph()
c=max(nx.connected_components(Gn1),key=len)
c=list(c)
LCC1=Gn1.subgraph(c).copy()

pl1={}
for i in LCC1.nodes:
    for j in LCC1.nodes:
        t=nx.shortest_path_length(LCC1,i,j)
        if t!=0:
            if t not in pl1:
                pl1[t]=1
            else:
                pl1[t]+=1

LCC2=nx.Graph()
c=max(nx.connected_components(Gn2),key=len)
c=list(c)
LCC2=Gn2.subgraph(c).copy()

pl2={}
for i in LCC2.nodes:
    for j in LCC2.nodes:
        t=nx.shortest_path_length(LCC2,i,j)
        if t!=0:
            if t not in pl2:
                pl2[t]=1
            else:
                pl2[t]+=1

PL1={
    'path_length':[],
    'num_of_path_length':[]
}
for i in pl1:
    PL1['path_length'].append(i)
    PL1['num_of_path_length'].append(pl1[i])
test=pd.DataFrame(PL1)
test.to_csv('/users/sds/downloads/path_length1.csv')

PL2={
    'path_length':[],
    'num_of_path_length':[]
}
for i in pl2:
    PL2['path_length'].append(i)
    PL2['num_of_path_length'].append(pl2[i])
test=pd.DataFrame(PL2)
test.to_csv('/users/sds/downloads/path_length2.csv')


#robustness
L1=len(Gn1)
print("size of Gn1:",L1)
sorted_degree1=sorted(D1.items(), key=lambda d: d[1],reverse=True)

#0.01%
rate=0.0001
l=int(rate*len(Gn1))
s=[]
for i in sorted_degree1:
    if len(s)<l:
        s.append(i[0])
Gr=nx.Graph()
Gr=Gn1.copy()
for i in s:
    Gr.remove_node(i)
c=max(nx.connected_components(Gr),key=len)
len_lcc=len(c)
len_mid=0
len_single=0
for c in sorted(nx.connected_components(Gr), key=len, reverse=True):
    if len(c)!=1 and len(c)!=len_lcc:
        len_mid+=len(c)
len_single=L1-len_lcc-len_mid
print("size of lcc in 0.01%:",len_lcc)
print("size of mid in 0.01%:",len_mid)
print("size of single in 0.01%:",len_single)

#0.1%
rate=0.001
l=int(rate*len(Gn1))
s=[]
for i in sorted_degree1:
    if len(s)<l:
        s.append(i[0])
Gr=nx.Graph()
Gr=Gn1.copy()
for i in s:
    Gr.remove_node(i)
c=max(nx.connected_components(Gr),key=len)
len_lcc=len(c)
len_mid=0
len_single=0
for c in sorted(nx.connected_components(Gr), key=len, reverse=True):
    if len(c)!=1 and len(c)!=len_lcc:
        len_mid+=len(c)
len_single=L1-len_lcc-len_mid
print("size of lcc in 0.1%:",len_lcc)
print("size of mid in 0.1%:",len_mid)
print("size of single in 0.1%:",len_single)

#1%
rate=0.01
l=int(rate*len(Gn1))
s=[]
for i in sorted_degree1:
    if len(s)<l:
        s.append(i[0])
Gr=nx.Graph()
Gr=Gn1.copy()
for i in s:
    Gr.remove_node(i)
c=max(nx.connected_components(Gr),key=len)
len_lcc=len(c)
len_mid=0
len_single=0
for c in sorted(nx.connected_components(Gr), key=len, reverse=True):
    if len(c)!=1 and len(c)!=len_lcc:
        len_mid+=len(c)
len_single=L1-len_lcc-len_mid
print("size of lcc in 1%:",len_lcc)
print("size of mid in 1%:",len_mid)
print("size of single in 1%:",len_single)

#5%
rate=0.05
l=int(rate*len(Gn1))
s=[]
for i in sorted_degree1:
    if len(s)<l:
        s.append(i[0])
Gr=nx.Graph()
Gr=Gn1.copy()
for i in s:
    Gr.remove_node(i)
c=max(nx.connected_components(Gr),key=len)
len_lcc=len(c)
len_mid=0
len_single=0
for c in sorted(nx.connected_components(Gr), key=len, reverse=True):
    if len(c)!=1 and len(c)!=len_lcc:
        len_mid+=len(c)
len_single=L1-len_lcc-len_mid
print("size of lcc in 5%:",len_lcc)
print("size of mid in 5%:",len_mid)
print("size of single in 5%:",len_single)

#10%
rate=0.1
l=int(rate*len(Gn1))
s=[]
for i in sorted_degree1:
    if len(s)<l:
        s.append(i[0])
Gr=nx.Graph()
Gr=Gn1.copy()
for i in s:
    Gr.remove_node(i)
c=max(nx.connected_components(Gr),key=len)
len_lcc=len(c)
len_mid=0
len_single=0
for c in sorted(nx.connected_components(Gr), key=len, reverse=True):
    if len(c)!=1 and len(c)!=len_lcc:
        len_mid+=len(c)
len_single=L1-len_lcc-len_mid
print("size of lcc in 10%:",len_lcc)
print("size of mid in 10%:",len_mid)
print("size of single in 10%:",len_single)

#20%
rate=0.2
l=int(rate*len(Gn1))
s=[]
for i in sorted_degree1:
    if len(s)<l:
        s.append(i[0])
Gr=nx.Graph()
Gr=Gn1.copy()
for i in s:
    Gr.remove_node(i)
c=max(nx.connected_components(Gr),key=len)
len_lcc=len(c)
len_mid=0
len_single=0
for c in sorted(nx.connected_components(Gr), key=len, reverse=True):
    if len(c)!=1 and len(c)!=len_lcc:
        len_mid+=len(c)
len_single=L1-len_lcc-len_mid
print("size of lcc in 20%:",len_lcc)
print("size of mid in 20%:",len_mid)
print("size of single in 20%:",len_single)

#30%
rate=0.3
l=int(rate*len(Gn1))
s=[]
for i in sorted_degree1:
    if len(s)<l:
        s.append(i[0])
Gr=nx.Graph()
Gr=Gn1.copy()
for i in s:
    Gr.remove_node(i)
c=max(nx.connected_components(Gr),key=len)
len_lcc=len(c)
len_mid=0
len_single=0
for c in sorted(nx.connected_components(Gr), key=len, reverse=True):
    if len(c)!=1 and len(c)!=len_lcc:
        len_mid+=len(c)
len_single=L1-len_lcc-len_mid
print("size of lcc in 30%:",len_lcc)
print("size of mid in 30%:",len_mid)
print("size of single in 30%:",len_single)



L2=len(Gn2)
print("size of Gn2:",L2)
sorted_degree2=sorted(D2.items(), key=lambda d: d[1],reverse=True)

#0.01%
rate=0.0001
l=int(rate*len(Gn2))
s=[]
for i in sorted_degree2:
    if len(s)<l:
        s.append(i[0])
Gr=nx.Graph()
Gr=Gn2.copy()
for i in s:
    Gr.remove_node(i)
c=max(nx.connected_components(Gr),key=len)
len_lcc=len(c)
len_mid=0
len_single=0
for c in sorted(nx.connected_components(Gr), key=len, reverse=True):
    if len(c)!=1 and len(c)!=len_lcc:
        len_mid+=len(c)
len_single=L1-len_lcc-len_mid
print("size of lcc in 0.01%:",len_lcc)
print("size of mid in 0.01%:",len_mid)
print("size of single in 0.01%:",len_single)

#0.1%
rate=0.001
l=int(rate*len(Gn2))
s=[]
for i in sorted_degree2:
    if len(s)<l:
        s.append(i[0])
Gr=nx.Graph()
Gr=Gn2.copy()
for i in s:
    Gr.remove_node(i)
c=max(nx.connected_components(Gr),key=len)
len_lcc=len(c)
len_mid=0
len_single=0
for c in sorted(nx.connected_components(Gr), key=len, reverse=True):
    if len(c)!=1 and len(c)!=len_lcc:
        len_mid+=len(c)
len_single=L1-len_lcc-len_mid
print("size of lcc in 0.1%:",len_lcc)
print("size of mid in 0.1%:",len_mid)
print("size of single in 0.1%:",len_single)

#1%
rate=0.01
l=int(rate*len(Gn2))
s=[]
for i in sorted_degree2:
    if len(s)<l:
        s.append(i[0])
Gr=nx.Graph()
Gr=Gn2.copy()
for i in s:
    Gr.remove_node(i)
c=max(nx.connected_components(Gr),key=len)
len_lcc=len(c)
len_mid=0
len_single=0
for c in sorted(nx.connected_components(Gr), key=len, reverse=True):
    if len(c)!=1 and len(c)!=len_lcc:
        len_mid+=len(c)
len_single=L1-len_lcc-len_mid
print("size of lcc in 1%:",len_lcc)
print("size of mid in 1%:",len_mid)
print("size of single in 1%:",len_single)

#5%
rate=0.05
l=int(rate*len(Gn2))
s=[]
for i in sorted_degree2:
    if len(s)<l:
        s.append(i[0])
Gr=nx.Graph()
Gr=Gn2.copy()
for i in s:
    Gr.remove_node(i)
c=max(nx.connected_components(Gr),key=len)
len_lcc=len(c)
len_mid=0
len_single=0
for c in sorted(nx.connected_components(Gr), key=len, reverse=True):
    if len(c)!=1 and len(c)!=len_lcc:
        len_mid+=len(c)
len_single=L1-len_lcc-len_mid
print("size of lcc in 5%:",len_lcc)
print("size of mid in 5%:",len_mid)
print("size of single in 5%:",len_single)

#10%
rate=0.1
l=int(rate*len(Gn2))
s=[]
for i in sorted_degree2:
    if len(s)<l:
        s.append(i[0])
Gr=nx.Graph()
Gr=Gn2.copy()
for i in s:
    Gr.remove_node(i)
c=max(nx.connected_components(Gr),key=len)
len_lcc=len(c)
len_mid=0
len_single=0
for c in sorted(nx.connected_components(Gr), key=len, reverse=True):
    if len(c)!=1 and len(c)!=len_lcc:
        len_mid+=len(c)
len_single=L1-len_lcc-len_mid
print("size of lcc in 10%:",len_lcc)
print("size of mid in 10%:",len_mid)
print("size of single in 10%:",len_single)

#20%
rate=0.2
l=int(rate*len(Gn2))
s=[]
for i in sorted_degree2:
    if len(s)<l:
        s.append(i[0])
Gr=nx.Graph()
Gr=Gn2.copy()
for i in s:
    Gr.remove_node(i)
c=max(nx.connected_components(Gr),key=len)
len_lcc=len(c)
len_mid=0
len_single=0
for c in sorted(nx.connected_components(Gr), key=len, reverse=True):
    if len(c)!=1 and len(c)!=len_lcc:
        len_mid+=len(c)
len_single=L1-len_lcc-len_mid
print("size of lcc in 20%:",len_lcc)
print("size of mid in 20%:",len_mid)
print("size of single in 20%:",len_single)

#30%
rate=0.3
l=int(rate*len(Gn2))
s=[]
for i in sorted_degree2:
    if len(s)<l:
        s.append(i[0])
Gr=nx.Graph()
Gr=Gn2.copy()
for i in s:
    Gr.remove_node(i)
c=max(nx.connected_components(Gr),key=len)
len_lcc=len(c)
len_mid=0
len_single=0
for c in sorted(nx.connected_components(Gr), key=len, reverse=True):
    if len(c)!=1 and len(c)!=len_lcc:
        len_mid+=len(c)
len_single=L1-len_lcc-len_mid
print("size of lcc in 30%:",len_lcc)
print("size of mid in 30%:",len_mid)
print("size of single in 30%:",len_single)





#CDF
S1=eg.constraint(G1)
S1=sorted(S1.items(), key=lambda d: d[1])
SHS1=list()
for i in S1:
    if len(SHS1)<5000:
        SHS1.append(i[0])
    else:
        break

S2=eg.constraint(G2)
S2=sorted(S2.items(), key=lambda d: d[1])
SHS2=list()
for i in S2:
    if len(SHS2)<5000:
        SHS2.append(i[0])
    else:
        break

#a)old shs 在old/new graph里的constraint CDF
data1={
    'old_constraint':[],
    'new_constraint':[]
}
dict1=eg.constraint(G1,SHS1)
for i in dict1:
    data1['old_constraint'].append(dict1[i])
dict2=eg.constraint(G2,SHS1)
for i in dict2:
    data1['new_constraint'].append(dict2[i])
test=pd.DataFrame(data1)
test.to_csv('/users/sds/downloads/old_shs_constraint.csv')

#b)new shs 在old/new graph里的constraint CDF
data2={
    'old_constraint':[],
    'new_constraint':[]
}
dict1=eg.constraint(G1,SHS2)
for i in dict1:
    data2['old_constraint'].append(dict1[i])
dict2=eg.constraint(G2,SHS2)
for i in dict2:
    data2['new_constraint'].append(dict2[i])
test=pd.DataFrame(data2)
test.to_csv('/users/sds/downloads/new_shs_constraint.csv')

#c)old shs 在old/new graph里的effective size CDF
data3={
    'old_effective_size':[],
    'new_effective_size':[]
}
dict1=eg.effective_size(G1,SHS1)
for i in dict1:
    data3['old_effective_size'].append(dict1[i])
dict2=eg.effective_size(G2,SHS1)
for i in dict2:
    data3['new_effective_size'].append(dict2[i])
test=pd.DataFrame(data3)
test.to_csv('/users/sds/downloads/old_shs_effective_size.csv')

#d)new shs 在old/new graph里的effective size CDF      
data4={
    'old_effective_size':[],
    'new_effective_size':[]
}
dict1=eg.effective_size(G1,SHS2)
for i in dict1:
    data4['old_effective_size'].append(dict1[i])
dict2=eg.effective_size(G2,SHS2)
for i in dict2:
    data4['new_effective_size'].append(dict2[i])
test=pd.DataFrame(data4)
test.to_csv('/users/sds/downloads/new_shs_effective_size.csv')

test=pd.DataFrame(SHS1)
test.to_csv('/users/sds/downloads/SHS1.csv')

test=pd.DataFrame(SHS2)
test.to_csv('/users/sds/downloads/SHS2.csv')



