#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix


# In[13]:


#파일 불러오고 NA값 제거 
df = pd.read_csv('C://Users//leejiwon//Desktop//datasets//HouseVotes84.csv')
df = df.dropna() #NA값 보유 행 전체 삭제 


# In[14]:


y = df['Class']
X = df.drop('Class', axis = 1)
for col in X.columns:
    #X1[col] = X1[col].apply(lambda x: 1 if x== 'y' else 0)
    X.loc[X[col] == 'y', col] = 1
    X.loc[X[col] == 'n', col] = 0


# In[15]:


# alpha : 라플라스 수정 옵션
bnb = BernoulliNB(alpha=0).fit(X, y)


# In[16]:


#left: democrat, right: republican 
print(np.round(bnb.predict_proba(X[0:10])), 5)


# In[17]:


#오분류표 행렬
print(confusion_matrix(y, bnb.predict(X)))


# In[3]:


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
tr_iris, ts_iris, tr_target, ts_target =     train_test_split(iris.data, iris.target, test_size = 0.5, #test data와 train data를 반씩 나눠서 할당
    random_state = 1)

clf = KNeighborsClassifier(n_neighbors = 3).fit(tr_iris, tr_target) #3-nn 적용
print("error: {:.3f}".format(1 - clf.score(ts_iris, ts_target))) #오분류율 계산


# In[ ]:




