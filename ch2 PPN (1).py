#!/usr/bin/env python
# coding: utf-8

# ## 머신러닝 중간대체과제1 
# ### 경제학부 이지원
# ### ==========================================================================================================

# # 파이썬으로 퍼셉트론 학습 알고리즘 구현하기 

# ## 객체지향 퍼셉트론 API

# In[68]:


import numpy as np

class Perceptron(object): #퍼셉트론 분류기 
    def __init__(self, eta = 0.01, n_iter = 50, random_state = 1):
        self.eta = eta #학습률
        self.n_iter = n_iter #훈련데이터 반복 횟수 
        self.random_state = random_state #가중치 랜덤 초기화를 위한 난수 생성 시드 
        
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size = 1 + X.shape[1]) #가중치 
        self.errors_ = [] #분류 오류를 누적하여 저장 
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, X): #최종 입력 계산 
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X): #단위 계단함수를 사용해 클래스 레이블 반환 
        return np.where(self.net_input(X) >= 0.0, 1, -1)


# 퍼셉트론 클래스 생성을 위해 학습률, 에폭수 등의 하이퍼파라미터를 지정해주고 초기화
# fit 메서드를 이용해 가중치 및 오차 매개변수를 지정 후 학습을 진행 
# 학습을 통해 매개변수를 갱신 
# 입력을 X * w[1] + w[0]으로 지정
# 입력값을 토대로 예측을 실행하는데 임계값인 0보다 크면 1을, 아니면 -1을 예측함


# In[69]:


v1 = np.array([1, 2, 3])
v2 = 0.5 * v1
np.arccos(v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


# ## 붓꽃 데이터셋에서 퍼셉트론 훈련하기 

# ### 붓꽃 데이터셋 읽기 

# In[70]:


import pandas as pd

df = pd.read_csv('https://archive.ics.uci.edu/ml/'
        'machine-learning-databases/iris/iris.data', header=None)
df.tail()

# 붓꽃 데이터 파일을 읽어와 아래에서부터 5개 행을 출력


# ### 붓꽃 데이터 그래프 그리기 

# In[71]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt 
import numpy as np

#setosa와 versicolor를 선택
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

#꽃받침 길이와 꽃잎 길이 추출 
X = df.iloc[0:100, [0, 2]].values

#산점도 시각화 
plt.scatter(X[:50, 0], X[:50, 1],
           color = 'red', marker = 'o', label = 'setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
           color = 'blue', marker = 'x', label = 'versicolor')

plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc = 'upper left')

plt.show()

# matplotlib 라이브러리를 이용해 붓꽃 데이터의 산점도를 그래프로 시각화
# 붓꽃의 species 중 versicolor와 setosa만 추출해 pedal length와 sepal length의 분포를 그림


# ### 퍼셉트론 모델 훈련하기 

# In[72]:


ppn = Perceptron(eta = 0.1, n_iter = 10)

ppn.fit(X, y)

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker = 'o')
plt.xlabel('Epochs')
plt.ylabel('Number of errors')

plt.show()

# 퍼셉트론 모델을 ppn에 지정하고 이 분류기를 기준으로 훈련데이터 학습
# 에폭에 따른 error의 개수를 그래프로 시각화
# 에폭을 거듭할 수록 점점 줄어드는 errors


# ### 결정 경계 그래프 함수 

# In[73]:


from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution = 0.02):
    
    #마커와 컬러맵 설정 
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    #결정 경계 그리기 
    x1_min, x1_max = X[:, 0].min() - 1, X[: ,0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    
    #산점도 그리기 
    plt.contourf(xx1, xx2, Z, alpha = 0.3, cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x = X[y == cl, 0], 
                    y = X[y == cl, 1],
                    alpha = 0.8, 
                    c = colors[idx],
                    marker = markers[idx],
                    label = cl,
                    edgecolor = 'black')


# In[74]:


plot_decision_regions(X, y, classifier = ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc = 'upper left')

plt.show()

# 위에서 그린 붓꽃 데이터의 산점도를 perceptron 분류방법으로 학습하여 경계를 나누어줌 
# 두 타입을 나눠주는 결정경계 생성 


# ## 적응형 선형 뉴런과 학습의 수렴

# ### 파이썬으로 아달린 구현하기 

# In[75]:


class AdalineGD(object): #적응형 선형 뉴런 분류기 
    def __init__(self, eta = 0.01, n_iter = 50, random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size = 1 + X.shape[1])
        self.cost_ = []
        
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
        return self
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, X):
        return X
    
    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
    
# 아달린 함수는 활성화 함수로 선형 함수를 사용 f(X) = X
# 최종 입력값 X * w[1] + w[0]에 대하여 활성화함수를 거쳐 계단함수 형태로 레이블을 반환 
# 임계값인 0보다 크면 1을 아니면 -1을 반환함 


# In[76]:


fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 4))

ada1 = AdalineGD(n_iter = 10, eta = 0.01).fit(X, y)
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker = 'o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.01')

ada2 = AdalineGD(n_iter = 10, eta = 0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker = 'o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.0001')

plt.show()

# 아달린 + 경사하강법을 이용하여 학습시킨 경과를 에폭별로 시각화 
# 에폭에 따른 변화를 보여주는데 학습률이 0.0001로 작을 때 더 제곱합 오차가 줄어드는 것을 알 수 있음 


# ### 특성 스케일을 조정하여 경사 하강법 결과 향상시키기

# In[77]:


# 특성을 표준편차와 평균을 이용하여 표준화해줌 
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()


# In[78]:


ada = AdalineGD(n_iter = 15, eta = 0.01)
ada.fit(X_std, y)

plot_decision_regions(X_std, y, classifier = ada)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length[standardized]]')
plt.legend(loc = 'upper left')
plt.tight_layout()
plt.show()

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker = 'o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')

plt.tight_layout()
plt.show()

# 특성 스케일을 표준화한 뒤에 얻은 결과로
# 결정경계가 더 정확하게 구분되고 에폭별 제곱합 오차도 더 빠르게 작아지는 것을 알 수 있음 


# ## 대규모 머신 러닝과 확률적 경사 하강법

# In[79]:


class AdalineSGD(object): #적응형 선형뉴런 분류기 (Adaptive Linear Neuron classifier) 
    def __init__(self, eta = 0.01, n_iter = 10, shuffle = True, random_state = None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state

    # AdalineSGD 클래스를 정의
    # 하이퍼파라미터들을 초기화시키는데 가중치는 위에서 학습한 그대로를 사용함 
        
    def fit(self, X, y): # 훈련 데이터 학습 
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        return self
    
    def partial_fit(self, X, y): #가중치 초기화 없이 훈련데이터 학습 
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
            else:
                self._update_weights(X, y)
            return self
        
    def _shuffle(self, X, y): #훈련데이터 섞기 
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    
    def _initialize_weights(self, m): # 랜덤한 작은 수로 가중치 초기화 
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc = 0.0, scale = 0.01, size = 1 + m)
        self.w_initialized = True
        
    def _update_weights(self, xi, target): #아달린을 적용하여 가중치 갱신 
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error ** 2
        return cost
    
    def net_input(self, X): #최종 입력 계산 
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, X): #선형 활성화 계산 
        return X
    
    def predict(self, X): # 계단함수를 사용하여 클래스 레이블 반환 
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)


# In[80]:


ada = AdalineSGD(n_iter = 15, eta = 0.01, random_state = 1)
ada.fit(X_std, y)

plot_decision_regions(X_std, y, classifier = ada)
plt.title('Adaline - Stochastic Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc = 'upper left')

plt.tight_layout()
plt.show()

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker = 'o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')

plt.tight_layout()
plt.show()

# 경사하강법(SGD)를 적용한 분류


# In[ ]:




