#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from gurobipy import *


# #### 数据导入模块

# In[2]:


f=r'data_v1.xlsx'
data_ferry=pd.read_excel(f,sheet_name="Sheet2-ferry")
data_port=pd.read_excel(f,sheet_name="Sheet3-1")


# #### 参数设定

# In[3]:


# time1-time2 这段时间不能出海
#早上5:00-6:00
time1=0
time2=6


# In[4]:


m =3 #轮船数
n = 5 #港口数

q = 115 #时刻数
lam = 1 # 轮船成本系数
niu = 10 # 客户成本系数

F = range(0,m) #轮船集合
P = range(0,n) #港口集合
Q = range(0,q) #时刻集合

scalar=1


# In[5]:


#参数list初始化
cap = [0] * m #轮船集合
home_port = [0] * m # 轮船的母港
berth = [0] * n #每个港口的可容纳的轮船数


# In[6]:


w=[[0 for k in P] for f in F]   #w[ferry][port]=waiting time
i=0
j=0
for index,row in data_port.iterrows():
    port,time=row
    port=int(port[-1])
    j+=1
    w[i][port-1]=int(time/10)
    if j%5==0:
        i+=1


# In[7]:


for i in F:
    cap[i] = data_ferry["Cf"][i]  #cap[f] capacity
    home_port[i] = data_ferry['hf'][i]-1  #home_port[f] home port

for i in P:
    berth[i] = 2


# In[8]:


for i in F:
    cap[i]=cap[i]*scalar


# In[9]:


data_ferry.head()


# #### 节点定义

# In[10]:


data_demand=pd.read_excel(f,sheet_name="Sheet5-1")
# 列更名+删除无用行
data_demand.columns=['port','cus_port','series','volume','arrival_time']


# In[11]:


# 根据sheet5生成所有node
V=[[] for i in P] # 对每个港口都生成node   V[port][time] 嵌套的list 
# V中node的格式：(初始port,时刻上的序号)
for i in P:
    for j in Q:        
        node=(i,j) 
        V[i].append(node)


# In[12]:


VV=[]               #VV 所有node
for v in V:
    for i in v:
        VV.append(i)


# In[13]:


U=VV[:]            #U 所有node并上 虚拟点
for i in P:
    U.append((i,q))


# In[14]:


# 对demand列表循环，将客户需求、客户目的地、客户到达时间更新
D=[{} for t in P]                   #D[aim port][VV 中的有效的node]=demand
arr_time=[{} for t in P]            #arr_time[aim port][VV 中的有效的node]=一个具体的时刻
for t in P:
    for v in VV:
        D[t][v]=0
        arr_time[t][v]=q
for index,row in data_demand.iterrows():  
    i, a, j, b, c = row
    D[a-1][(i-1,j-1)]=b
    arr_time[a-1][(i-1,j-1)]=c-1

# 定义虚拟节点，其格式和其他节点一致
for i in P:
    # 先求在虚拟节点的客户净需求量
    virtual_demand=-data_demand[data_demand['cus_port']==(i+1)]['volume'].sum()
    D[i][(i,q)]=virtual_demand
    for t in P:
        if t!=i:
            D[i][(t,q)]=0


# #### 弧定义 

# In[15]:


def ceil(x):
    if int(x)==x:
        return int(x)
    else:
        return int(x)+1


# In[16]:


# 先获取每条船可以往返的港口+单程时间
data_time=pd.read_excel(f,sheet_name='Sheet4-time')
# 排除不能到达的港口
data_time=data_time[data_time['tour_time']!=999]


# ##### 更换例子要改

# In[17]:


# 定义轮船的arc
ferry_type=['small','small','large'] # 通过船的类型确定往返港口和单程时间
#ferry_type=['small','large'] # 通过船的类型确定往返港口和单程时间
E=[[] for f in F] # 定义每条轮船的edge集合                     E[ferry]=[]对于ferry f 的feasible arc
for f in F: # 对每条船都开始生成可行的弧
    temp_data=data_time[data_time['type']==ferry_type[f]] # 生成每种船对应的data
    for index,row in temp_data.iterrows():
        from_port, to_port, tour_time=row[1:]
        for i in Q:
            for j in range(i+1,q):                
                if from_port==to_port and j>i+1: # 生成同港口的弧
                    break
                try:
                    tour=max(1,ceil(float(tour_time/10)))
                except:
                    continue
                if i+tour!=j: # 从from_port(i)到to_port(j)是否可行的判断  
                    continue
                E[f].append((V[from_port-1][i],V[to_port-1][j]))


# In[18]:


# 定义客户流的destination arc
Dest=[[] for k in P]                   #Dest[aim port]=相同港口的0-114的点 到相同港口115的 边
for k in P:
    for i in Q:
        Dest[k].append((V[k][i],(k,q)))


# In[19]:


# 定义客户流的infeasible arc              
Infeas=[]                                #Infeas  所有这样交叉的 边  
Infeas_port=[[] for k in P]             #Infeas_port[aim port]=不通港口的114 到 目的港口115的 边
for k in P:
    for h in P:
        if k==h:
            continue
        Infeas.append((V[h][q-1],(k,q)))
        Infeas_port[k].append((V[h][q-1],(k,q)))


# In[20]:


# 定义约束中需要的，符合一个图的边集
A=[[] for k in P]    #A[aim port] all service arc + infeas + dest 
EE=[]
for f in F:
    for a in E[f]:
        EE.append(a)   #EE 没有剔除重复arc的service arc
Euni=list(set(EE))   
for k in P:
    Atemp=[]
    Atemp=Euni+Infeas_port[k]+Dest[k]
    A[k]=list(set(Atemp))


# #### 成本相关定义

# #### 更换例子要改

# In[21]:


fer_types=set(data_ferry['f'].iloc[:3])
cost={}
for fer_type in fer_types:
    temp_data=data_ferry[data_ferry['f']==fer_type]
    cost[fer_type]={'inport':float(temp_data['gf_inport']),'enroute':float(temp_data['gf_enroute'])}
    
# 定义每条船在每条弧上的成本
new_ferry_type=['small1','small2','large']
#new_ferry_type=['small1','large1']
g=[{} for f in F]      #g[ferry][arc] cost for ferry 
for f in F:
    for e in E[f]:
        node1, node2=e
        if node1[0]==node2[0]:
            g[f][e]=cost[new_ferry_type[f]]['inport']*(node2[1]-node1[1])/6
        else:
            g[f][e]=cost[new_ferry_type[f]]['enroute']*(node2[1]-node1[1])/6
            
# 定义客户流在每条弧上的成本
M=10e5 # 定义大M成本
C=[{} for k in P]
for k in P:
    for f in F:
        for e in E[f]:
            node1, node2=e
            C[k][e]=(node2[1]-node1[1])*10   #C[aim port][arc] cost for comsumers
        for h in Dest[k]:
            C[k][h]=0
        for e in Infeas:
            C[k][e]=M


# #### 其他集合定义

# In[22]:


I=[{} for f in F]             #I[f][VV node ]=[前序 node]  only for service arc
O=[{} for f in F]             #O[f][VV node ]=[后序 node]  only for service arc
for f in F:
    for v in VV:
        I[f][v]=[]
        O[f][v]=[]
        for t in E[f]:
            if t[1]==v:
                I[f][v].append(t[0])
            if t[0]==v:
                O[f][v].append(t[1]) 


# In[23]:


b=[{}for f in F]                #b[f][VV node]=flow balance require
for f in F:
    for v in VV:
        hf=data_ferry['hf'][f]-1
        if v==(hf,0):
            b[f][v]=-1
        elif v==(hf,q-1):
            b[f][v]=1
        else:
            b[f][v]=0


# In[24]:


Iserv=[{} for f in F]   #Iserv[f][VV node]=剔除了相同port 的 能真正运客的 前序点
for f in F:
    for v in VV:
        port=v[0]
        Iserv[f][v]=I[f][v][:]
        for t in Iserv[f][v]:
            if t[0]==port:
                Iserv[f][v].remove(t)


# In[25]:


I_nof_serv={}                   #I_nof_serv[VV node]和ferry 无关的 和当前port不一样的 前序点
for v in VV:
    I_nof_serv[v]=[]
    for f in F:
        if v in Iserv[f].keys():
            for i in Iserv[f][v]:
                I_nof_serv[v].append(i)
            I_nof_serv[v]=list(set(I_nof_serv[v]))


# In[26]:


delta=[{} for f in F]     #dealta[f][VV node]=向后的arc

for f in F:
    for v in VV:
        delta[f][v]=[]
        time=v[1]
        port=v[0]
        r=min(w[f][port],q-1-time)
        for i in range(1,r+1):
            if time+i<=q-1:
                delta[f][v].append(((port,time+i-1),(port,time+i)))


# In[27]:


beta=[2,2,2,2,1]  #port's capacity 


# In[28]:


d=[{}for t in P]        #demand d[aim port][VV node]
for t in P:
    for ki in VV:
        d[t][ki]={}
        for v in U:
            if v==ki:
                d[t][ki][v]=D[t][ki]
            elif v==(t,q):
                d[t][ki][v]=-D[t][ki]
            else:
                d[t][ki][v]=0


# In[29]:


II=[{}for t in P]               #II[aim port][U node]  for客户流平衡方程
OO=[{}for t in P]               #OO[aim port][U node]  
for t in P:
    for v in U:
        II[t][v]=[] 
        OO[t][v]=[]
        for i in A[t]:
            if i[1]==v:
                '''if v[1]==115 and v[0]!=t:
                    continue'''
                II[t][v].append(i[0])
            if i[0]==v:
                if v[1]==q-1 and i[1][0]!=t:  #修改过2--114
                    continue
                OO[t][v].append(i[1])


# In[30]:


FF={}            #FF[arc]  哪些ferry 可以走这条 arc
for arc in Euni:
    FF[arc]=[] # FF表示轮船图中包含arc的轮船
    for f in F:
        if arc in E[f]:
            FF[arc].append(f)        


# ### Model Definition

# In[31]:


model=Model()
y={}
for f in F:
    for e in E[f]:
        y[f,e]=model.addVar(vtype=GRB.BINARY,name="y"+str(f)+str(e))


# In[32]:


x={}
for k in P:
    for t in P:
        if k==t:
            continue
        for i in Q:
            if i==q-1:
                continue
            for e in A[k]:
                x[V[t][i],k,e]=model.addVar(vtype=GRB.INTEGER,
                                            name="x-"+"t:"+str(t)+",i:"+str(i)+",k:"+str(k)+",e:"+str(e))  # V是出发点， k是目标port， e是feasible arc


# In[33]:


z={}
for k in P:
    for t in P:
        if k==t:
            continue
        for i in Q:
            if i==q-1:
                continue
            for j in range(i+1,q):
                z[V[t][i],k,V[k][j]]=model.addVar(vtype=GRB.BINARY, name="z-"+"t"+str(t)+"i"+str(i)+"k"+str(k)+"j"+str(j))   # 第一个t,i 是出发点  k是目标port 第二个V是到达点



# In[34]:
# ##### 目标函数

model.setObjective(lam*quicksum(g[f][e]*y[f,e] for f in F for e in E[f])+niu*quicksum(C[k][a]*x[V[t][i],k,a] for k in P for t in P if t!=k for i in Q 
                                                                                 if i!=q-1 for a in A[k]),GRB.MINIMIZE)
# In[35]:
# ##### 1. 轮船流入流出约束

model.addConstrs((quicksum(y[f,(v1,v)] 
                           for v1 in I[f][v])-
                  quicksum(y[f,(v,v2)] 
                           for v2 in O[f][v]) == b[f][v]
                  for v in VV for f in F),name="c1-"+"v"+str(v)+"f"+str(f))

# In[36]:
# ##### 2. 轮船必须停留一段时间的约束

model.addConstrs((len(delta[f][ki]) * quicksum(y[f,(v1,ki)] for v1 in Iserv[f][ki]) <= quicksum(y[f,a] for a in delta[f][ki])
             for ki in VV for f in F), name="c2-"+"ki"+str(ki)+"f"+str(f)  )

# In[37]:
# ##### 3. 同一个港口最多停留的轮船数约束

model.addConstrs((quicksum(y[f,(V[k][i],V[k][i+1])] for f in F) 
                  <= beta[k] for i in range(0,q-1) for k in P), 
                name="c3-"+"k"+str(k))

# In[38]:

# ##### 4. 客户的流入流出约束
model.addConstrs((quicksum(x[a,t,(u1,v)] for u1 in II[t][v])-
                  quicksum(x[a,t,(v,u2)] for u2 in OO[t][v]) ==
                  -d[t][a][v] for v in U for t in P
            for a in VV if a[0]!=t if a[1]!=q-1),name="c4-"+"v"+str(v)+"t"+str(t)+"a"+str(a))


# In[39]:
# ##### 5. 客户流量<=轮船容量约束

model.addConstrs((quicksum(x[a,t,(uu,v)] for a in VV if (a[1]<=uu[1] and a[1]<=v[1]) for t in P if a[0]!=t) <= 
            quicksum(y[f,(uu,v)]*cap[f] for f in FF[(uu,v)]) for (uu,v) in Euni if uu[0] != v[0] ) , name="c5-" )


# In[40]:

# ##### 6. 客户转移时间
model.addConstrs((quicksum(x[V[k][i],pp,(v,V[k][i])] for v in I_nof_serv[(k,i)]) <= 
            x[V[k][i],pp,(V[k][i],V[k][i+1])]  for k in P for i in Q if i!=q-1 for pp in P if pp!=k),
                name="c6-")

# In[41]:
#  ##### 7. 硬时间窗
M=10**8

model.addConstrs( quicksum(x[V[k][i],p,(v,V[p][j])] 
                           for v in I_nof_serv[(p,j)]) 
                 <= M*z[V[k][i],p,V[p][j]] 
            for k in P for i in Q if i!=q-1 for p in P if p!=k for j in range(i+1,q))

model.addConstrs(j <= arr_time[p][(k,i)]+M*(1-z[V[k][i],p,V[p][j]]) for k in P for i in Q if i!=q-1 for p in P if p!=k for j in range(i+1, q))


# In[42]:
# ##### 8.新约束

model.addConstrs(y[f,(u,v)]== 0 for f in F for (u,v) in E[f] if (u[0]!=v[0] and u[1] in  range(time1,time2)))


# In[43]:

model.Params.timelimit = 10*60*60

model.optimize()

print(model.objVal)

# In[44]
# ### 输出

route=[[]for f in F]
for f in F:
    #home_port
    start=(home_port[f],0)
    route[f].append(start)
    for i in range(q-1):   
        pre=route[f][i]
        no_stop=0  
        for v in O[f][pre]:
            if y[f,(pre,v)].x==1:
                route[f].append(v)
                no_stop=1
        if no_stop==0:
            break

res_y=[route[0],route[1],route[2]]
df=pd.DataFrame(res_y)  

f2=r'res_y_model1.xlsx'
df.to_excel(f2)


# In[45]:

#出发点k,出发时间i,去往p，经过的arc e,数量d
D=range(42)
res_x=[{} for i in D]
for index,row in data_demand.iterrows():
    k,t,i,b,a = row
    k=k-1
    t=t-1
    i=i-1
    a=a-1
    start=(k,i)
    nodes=[start]
    while len(nodes)!=0:
        pre_nodes=nodes[:]
        nodes=[]
        for pre in pre_nodes:
            for v in OO[t][pre]:
                if x[V[k][i],t,(pre,v)].x !=0:
                    res_x[index-1][(pre,v)]=x[V[k][i],t,(pre,v)].x
                    nodes.append(v)

result=[]
for k in res_x[0].keys():
    res={}
    res['arc'+str(0)]=k
    res['quantity'+str(0)]=res_x[0][k]
    result.append(res)
data_x=pd.DataFrame(result)


for d in range(1,42):
    result=[]
    for k in res_x[d].keys():
        res={}
        res['arc'+str(d)]=k
        res['quantity'+str(d)]=res_x[d][k]
        result.append(res)
    data=pd.DataFrame(result)    
    data_x=data_x.join(data)


f3=r'res_x_model1.xlsx'
data_x.to_excel(f3)

