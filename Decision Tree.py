#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

print('클래스 레이블: ', np.unique(y))

# 붓꽃 데이터를 불러와 X, y에 할당 후 species의 각 정수 레이블을 반환 


# In[3]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size = 0.3, random_state = 1, stratify = y)

# train_test_split 메서드를 사용해 train data와 test data를 랜덤하게 분할 
# 이때, test_size = 0.3이므로 test:train = 3:7
# stratify = y이므로 원래 비율과 같도록 계층화 추출


# In[4]:


print('y의 레이블 카운트:', np.bincount(y))
print('y_train의 레이블 카운트:', np.bincount(y_train))
print('y_test의 레이블 카운트:', np.bincount(y_test))

# 위와 같이 split한 결과 y를 train과 test에 7:3 비율로 분할한 결과 
# bincount는 등장횟수를 계산


# In[5]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train) 
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test) 

# 특성 표준화 
# StandardScaler() 스케일링 클래스로 평균이 0, 표준편차가 1이되도록 변환
# train, test data 모두 표준화


# In[6]:


from matplotlib.colors import ListedColormap

# 결정경계 그리기
def plot_decision_regions(X, y, classifier, test_idx = None, resolution = 0.02):
   
    # 마커와 컬러맵 설정
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # 결정경계
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha = 0.3, cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x = X[y == cl, 0],
                   y = X[y == cl, 1],
                   alpha = 0.8,
                   c = colors[idx], 
                   label = cl,
                   edgecolor = 'black')


# # 3.6 결정 트리 학습
# ## 3.6.1 정보이득 최대화: 자원을 최대로 이용

# In[7]:


import matplotlib.pyplot as plt
import numpy as np

def gini(p): #지니 불순도
    return p * (1 - p) + (1 - p) * (1 - (1 - p))

def entropy(p): #엔트로피
    return - p * np.log2(p) - (1 - p) * np.log2((1 - p))

def error(p): #분류오차
    return 1 - np.max([p, 1 - p])

x = np.arange(0.0, 1.0, 0.01) #0.0~1.0까지 0.01의 간격으로 배열 생성
ent = [entropy(p) if p != 0 else None for p in x]
sc_ent = [e * 0.5 if e else None for e in ent] #엔트로피 * 1/2를 추가로 정의 #scale이 조정된 앤트로피
err = [error(i) for i in x]


# In[8]:


fig = plt.figure() #시각화
ax = plt.subplot(111)
for i, lab, ls, c, in zip([ent, sc_ent, gini(x), err], 
                          ['Entropy', 'Entropy (scaled)', 
                           'Gini Impurity', 'Misclassification Error'],
                          ['-', '-', '--', '-.'],
                          ['black', 'lightgray', 'red', 'green', 'cyan']):
    line = ax.plot(x, i, label=lab, linestyle=ls, lw=2, color=c)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
          ncol=5, fancybox=True, shadow=False)

ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
plt.ylim([0, 1.1])
plt.xlabel('p(i=1)')
plt.ylabel('Impurity Index')
plt.show()

# 지니 불순도가 엔트로피와 분류오차의 중간에 위치함
# 위에서 정의한 sc_ent (scaled Entropy)와 지니 불순도의 그래프가 거의 겹침
# 특성 공간을 사각 격자로 나눈 모습


# In[11]:


from sklearn.tree import DecisionTreeClassifier #결정 트리
tree = DecisionTreeClassifier(criterion = 'gini', #지니 불순도를 분할 조건으로 삼음 (잘못 분류될 확률을 최소화)
                             max_depth = 4, #최대 깊이 = 4
                             random_state = 1)

tree.fit(X_train, y_train) #train 데이터셋에 적용
X_combined = np.vstack((X_train, X_test)) 
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined, y_combined,
                      classifier = tree, test_idx = range(105, 150))

plt.xlabel('petal length[cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc = 'upper left')
plt.tight_layout()
plt.show()


# In[ ]:




