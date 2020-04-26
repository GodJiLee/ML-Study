#!/usr/bin/env python
# coding: utf-8

# #### 중간대체과제3_2017280063_이지원

# # 3장. 사이킷런을 타고 떠나는 머신 러닝 분류 모델 투어

# ## 사이킷런 첫걸음

# In[1]:


from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

print('클래스 레이블: ', np.unique(y))

# 붓꽃 데이터를 불러와 X, y에 할당 후 species의 각 정수 레이블을 반환


# In[2]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size = 0.3, random_state = 1, stratify = y)

# train_test_split 메서드를 사용해 train data와 test data를 랜덤하게 분할 
# 이때, test_size = 0.3이므로 test:train = 3:7
# stratify = y이므로 원래 비율과 같도록 계층화 추출 


# In[3]:


print('y의 레이블 카운트:', np.bincount(y))
print('y_train의 레이블 카운트:', np.bincount(y_train))
print('y_test의 레이블 카운트:', np.bincount(y_test))

# 위와 같이 split한 결과 y를 train과 test에 7:3 비율로 분할한 결과 
# bincount는 등장횟수를 계산


# In[4]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train) 
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test) 

# 특성 표준화 
# StandardScaler() 스케일링 클래스로 평균이 0, 표준편차가 1이되도록 변환
# train, test data 모두 표준화


# ## 사이킷런으로 퍼셉트론 훈련하기

# In[5]:


from sklearn.linear_model import Perceptron 

ppn = Perceptron(max_iter = 40, eta0 = 0.1, tol = 1e-3, random_state = 1)
ppn.fit(X_train_std, y_train)

# 퍼셉트론 모델로 훈련 
# 옵션으로는 40회 반복, 학습률 0.1, 허용 오차값 1e-3, 랜덤 시드 생성


# In[6]:


y_pred = ppn.predict(X_test_std)
print('잘못 분류된 샘플 개수: %d' % (y_test != y_pred).sum())

# 위에서 진행한 표준화된 데이터셋으로 예측을 진행한 결과 
# 잘못 분류된 샘플 개수 출력 


# In[7]:


from sklearn.metrics import accuracy_score
print('정확도: %.2f' % accuracy_score(y_test, y_pred))

# 분류 정확도 계산 
# 0.98의 정확도로 분류됨 


# In[8]:


print('정확도:%.2f' % ppn.score(X_test_std, y_test))

# 사이킷런의 score 메서드를 사용해 predict + accuracy_score 메서드 연결 계산도 가능


# In[9]:


from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

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
        
    # 테스트 샘플에는 작은 원을 덮어 씌워서 그림
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        
        plt.scatter(X_test[:, 0], # 산점도 그리기 
                   X_test[:, 1],
                   c = '',
                   edgecolor = 'black',
                   alpha = 1.0,
                   linewidth = 1,
                   marker = 'o',
                   s = 100,
                   label = 'test set')


# In[10]:


# 표준화된 훈련데이터를 사용해 퍼셉트론 모델 훈련 
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

# 퍼셉트론 분류기로 결정 경계 그리기 
plot_decision_regions(X = X_combined_std, y = y_combined, 
                     classifier = ppn, test_idx = range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc = 'upper left')

plt.tight_layout()
plt.show()

# 0, 1, 2의 레이블을 가진 iris species들을 종류에 따라 분류한 결과 
# 선형 결정경계로는 완벽하게 분류하지 못하는 모습
# 퍼셉트론은 비선형 구분이 필요한 데이터셋에 수렴하지 못한다는 한계를 가짐


# # 로지스틱 회귀를 사용한 클래스 확률 모델링

# ## 로지스틱 회귀의 이해와 조건부 확률

# In[11]:


import matplotlib.pyplot as plt
import numpy as np

# 시그모이드 함수의 모습을 -7에서 7까지 시각화 
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

z = np.arange(-7, 7, 0.1)
phi_z = sigmoid(z)

plt.plot(z, phi_z)
plt.axvline(0.0, color = 'k')
plt.ylim(-0.1, 1.1)
plt.xlabel('z')
plt.ylabel('$\phi (z)$')

# y축의 눈금과 격자선
plt.yticks([0.0, 0.5, 1.0])
ax = plt.gca()
ax.yaxis.grid(True)

plt.tight_layout()
plt.show()

# 0.5에서 중간값을 가지고 z값이 커질 수록 1에, 작아질 수록 0에 가까워지는 s자 형의 그래프
# 0~1 사이의 값을 가지므로 확률로써 적용할 수 있음


# ## 로지스틱 비용 함수의 가중치 학습하기

# In[12]:


def cost_1(z):
    return -np.log(sigmoid(z)) # y = 1일 때 비용함수 

def cost_0(z):
    return -np.log(1 - sigmoid(z)) # y = 0일 때 비용함수 

z = np.arange(-10, 10, 0.1)
phi_z = sigmoid(z)

c1 = [cost_1(x) for x in z]
plt.plot(phi_z, c1, label = 'J(w) if y = 1')

c0 = [cost_0(x) for x in z]
plt.plot(phi_z, c0, linestyle = '--', label = 'J(w) if y = 0')

# 샘플이 1개일 경우 분류 비용을 시각화 
plt.ylim(0.0, 5.1)
plt.xlim([0,1])
plt.xlabel('$\phi$(z)')
plt.ylabel('J(w)')
plt.legend(loc = 'best')
plt.tight_layout()
plt.show()

# y = 1일 때의 J(w)는 phi(z)->1일 수록 비용 최소화ㅜ
# y = 0일 때의 J(w)는 phi(z)->0일 수록 비용 최소화
# 즉, 정확히 예측할 수록 비용이 줄어드는 반면, 잘못 예측할 수록 비용이 크게 증가하는 양상


# ## 아달린 구현을 로지스틱 회귀 알고리즘으로 변경

# In[13]:


class LogisticRegressionGD(object):
    """경사 하강법을 사용한 로지스틱 회귀 분류기
    매개변수
    -------------------
    eta : float
        학습률(0.0과 1.0 사이)
    n_iter : int
        훈련 데이터셋 반복 횟수
    random_state = int
        가중치 무작위 초기화를 위한 난수 생성기 시드
        
    속성
    -------------------
    w_ : 1d-array
        학습된 가중치
    cost_ : list
        에포크마다 누적된 로지스틱 비용 함수 값
    """

    # 매개변수 초기화 
    def __init__(self, eta = 0.05, n_iter = 100, random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit(self, X, y):
        """ 훈련 데이터 학습
        
        매개변수
        ------------------
        X : {array-like}, shape = [n_sample, n_feature]
            n_samples개의 샘플과 n_features개의 특성으로 이루어진 훈련 데이터
        y : array-like, shape = [n_samples]
            타깃값
            
        반환값
        ------------------
        self : object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size = 1 + X.shape[1])
        self.cost_ = []
        
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            
            # 오차 제곱합 대신 로지스틱 비용을 계산
            cost = -y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output)))
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """최종 입력 계산"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        """로지스틱 시그모이드 활성화 계산"""
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        """단위 계단 함수를 사용하여 클래스 레이블을 반환합니다"""
        return np.where(self.net_input(X) >= 0.0, 1, 0)
        # return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0) 와 동일 
        
# 아달린 구현을 로지스틱 회귀 알고리즘으로 변경한 것으로 기존 항등함수(선형함수)였던
# activation function을 sigmoid 함수로 변형


# In[14]:


X_train_01_subset = X_train[(y_train == 0) | (y_train == 1)]
y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]

# 로지스틱 회귀 + SGD로 분류
lrgd = LogisticRegressionGD(eta=0.05, n_iter=1000, random_state=1)
lrgd.fit(X_train_01_subset,
         y_train_01_subset)

# 결정경계 그리기
plot_decision_regions(X=X_train_01_subset, 
                      y=y_train_01_subset,
                      classifier=lrgd)

#시각화
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()

# 로지스틱 회귀 방식으로
# petal 너비와 길이에 따라 레이블 0,1을 가지는 species를 분류


# ## 사이킷런을 사용해 로지스틱 회귀 모델 훈련하기

# In[15]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(solver = 'liblinear', multi_class = 'auto', C = 100.0, random_state = 1)
lr.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined, 
                     classifier = lr, test_idx = range(105, 150))

plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc = 'upper left')
plt.tight_layout()
plt.show()

# 표준화된 훈련 데이터셋에 대해 로지스틱 회귀 분류 모델로 경계를 나눈 결과
# 이전과 달리 다중 분류도 지원
# 여기서 C란 규제의 강도를 조정할 수 있는 매개변수 


# In[16]:


# predict_proba 메서드를 사용해 훈련 샘플이 특정 클래스에 속할 확률을 계산
# 결과 배열에서 각 행은 각 species의 클래스 소속 확률
# 1행은 3번째 클래스에 속할 확률이 가장 높음 
lr.predict_proba(X_test_std[:3, :])


# In[17]:


# 확률이므로 열을 모두 더하면 1이 됨
lr.predict_proba(X_test_std[:3, :]).sum(axis = 1)


# In[18]:


# argmax 함수를 사용해 행 중 최댓값을 예측 레이블로 할당
lr.predict_proba(X_test_std[:3, :]).argmax(axis = 1)


# In[19]:


#predict 메서드로 더 빠르게 확인 가능
lr.predict(X_test_std[:3, :])


# In[20]:


# 입력데이터의 차원을 2차원으로 맞춰주기 위해 형상 변환
lr.predict(X_test_std[0, :].reshape(1, -1))


# ## 규제를 사용해 과대적합 피하기 

# In[21]:


weights, params = [], []

# 10^-5 ~ 10^5까지 C를 조정
for c in np.arange(-5, 5):
    lr = LogisticRegression(solver = 'liblinear', multi_class = 'auto', C = 10.**c, random_state = 1)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10.**c)
    
weights = np.array(weights)
plt.plot(params, weights[:, 0],
        label = 'petal length')
plt.plot(params, weights[:, 1], linestyle = '--',
        label = 'petal width')
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.legend(loc = 'upper left')
plt.xscale('log')
plt.show()

# C가 줄어들 수록 규제강도가 증가하여 가중치 절댓값이 작아짐 (overfitting 억제)
# 이때 C는 L2 규제에서 규제의 강도를 조정하는 하이퍼파라미터 람다의 역수


# ## 서포트 벡터 머신을 사용한 최대 마진 분류

# In[22]:


from sklearn.svm import SVC #SVM 모델을 훈련하기 위한 클래스

# 서포트 벡터 머신을 활용한 분류
svm = SVC(kernel = 'linear', C = 1.0, random_state = 1)
svm.fit(X_train_std, y_train) #훈련 데이터에 svm 적용

plot_decision_regions(X_combined_std, #결정경계 그리기
                     y_combined,
                     classifier = svm,
                     test_idx = range(105, 150))

#시각화
plt.scatter(svm.dual_coef_[0, :], svm.dual_coef_[1, :])
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc = 'upper left')
plt.tight_layout()
plt.show()


# In[23]:


svm.coef_ # 공분산 배열


# In[24]:


svm.dual_coef_, svm.dual_coef_.shape


# ### 사이킷런의 다른 구현

# In[25]:


from sklearn.linear_model import SGDClassifier #SVC 대신 확률적 경사 하강법(SGD)를 적용한 분류기

ppn = SGDClassifier(loss = 'perceptron') #퍼셉트론 분류기
lr = SGDClassifier(loss = 'log') 
svm = SGDClassifier(loss = 'hinge')


# ## 커널 SVM을 사용해 비선형 문제 풀기 

# In[26]:


import matplotlib.pyplot as plt
import numpy as np

#선형적으로 분류할 수 없는 데이터셋
np.random.seed(1)
X_xor = np.random.randn(200, 2) #200개의 데이터를 2개의 클래스로 분류
y_xor = np.logical_xor(X_xor[:, 0] > 0, #XOR(NAND + OR)
                       X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)

#산점도 그리기
plt.scatter(X_xor[y_xor == 1, 0],
            X_xor[y_xor == 1, 1],
            c='b', marker='x',
            label='1')
plt.scatter(X_xor[y_xor == -1, 0],
            X_xor[y_xor == -1, 1],
            c='r',
            marker='s',
            label='-1')

plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.legend(loc='best')
plt.tight_layout()
plt.show()


# ### 커널 기법을 사용해 고차원 공간에서 분할 초평면 찾기

# In[27]:


#고차원으로 투영 후 선형적으로 구분되도록 하는 커널 방식 적용
svm = SVC(kernel = 'rbf', random_state = 1, gamma = 0.10, C = 10.0) #방사기저함수 방식의 커널 사용
svm.fit(X_xor, y_xor)

#결정경계 그리기
plot_decision_regions(X_xor, y_xor, classifier = svm) #서포트벡터머신을 분류기로 사용

plt.legend(loc = 'upper left')
plt.tight_layout()
plt.show()


# In[28]:


svm = SVC(kernel = 'rbf', random_state = 1, gamma = 0.2, C = 1.0) #감마는 결정경계를 더 세밀하게 만들어주는 매개변수 
                                                                    #여기서는 0.2로 비교적 작은 값 선택
svm.fit(X_train_std, y_train) #훈련 데이터셋에 대해 rbf 커널 SVM

#결정 경계 그리기
plot_decision_regions(X_combined_std, y_combined,
                      classifier = svm, test_idx = range(105, 150))
plt.scatter(svm.dual_coef_[0,:], svm.dual_coef_[1,:])
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc = 'upper left')
plt.tight_layout()
plt.show()

# 비교적 부드러운 형태의 결정경계


# In[29]:


svm = SVC(kernel = 'rbf', random_state = 1, gamma = 100.0, C = 1.0) #감마 값을 크게 만들어준 경우
svm.fit(X_train_std, y_train)

#결정경계 그리기
plot_decision_regions(X_combined_std, y_combined, classifier = svm, test_idx = range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc = 'upper left')
plt.tight_layout()
plt.show()

# 위에서 감마 값을 작게 만들어준 경우에 비해 결정경계가 더 세밀하게 구분됨
# 이에 따라 과대적합의 가능성 또한 높아짐 (새로운 데이터에 대해서는 일반화하기 힘듦)


# ## 결정 트리 학습
# ### 정보 이득 최대화-자원을 최대로 활용하기 

# In[30]:


import matplotlib.pyplot as plt
import numpy as np


def gini(p): #지니 불순도
    return p * (1 - p) + (1 - p) * (1 - (1 - p))


def entropy(p): #엔트로피
    return - p * np.log2(p) - (1 - p) * np.log2((1 - p))


def error(p): #분류오차
    return 1 - np.max([p, 1 - p])

x = np.arange(0.0, 1.0, 0.01)

ent = [entropy(p) if p != 0 else None for p in x]
sc_ent = [e * 0.5 if e else None for e in ent] # 엔트로피 * 1/2를 추가로 정의
err = [error(i) for i in x]

fig = plt.figure()
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


# ## 결정 트리 만들기

# In[41]:


from sklearn.tree import DecisionTreeClassifier #결정 트리

tree = DecisionTreeClassifier(criterion = 'gini', #지니 불순도를 분할 조건으로 삼음 (잘못 분류될 확률을 최소화)
                             max_depth = 4, #최대 깊이 = 4
                             random_state = 1)
tree.fit(X_train, y_train) #train 데이터셋에 적용

X_combined = np.vstack((X_train, X_test))
y_comvined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined, y_combined,
                      classifier = tree, test_idx = range(105, 150))

plt.xlabel('petal length[cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc = 'upper left')
plt.tight_layout()
plt.show()

# 특성 공간을 사각 격자로 나눈 모습


# ### 랜덤 포레스트로 여러 개의 결정 트리 연결하기

# In[47]:


from sklearn.ensemble import RandomForestClassifier 

forest = RandomForestClassifier(criterion='gini', #불순도 지표 = 지니 불순도
                                n_estimators=25, #앙상블을 위해 25개의 결정 트리 사용
                                random_state=1,
                                n_jobs=2) #모델 훈련을 병렬화
forest.fit(X_train, y_train)

plot_decision_regions(X_combined, y_combined, 
                      classifier=forest, test_idx=range(105, 150))

plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# 랜덤포레스트는 여러 개의 결정 트리를 평균내기 때문에 개별 트리의 분산이 높은 문제를 보완할 수 있음
# 이외에도 잡음으로부터 안정적이기 때문에 하이퍼파라미터 튜닝과 가지치기를 할 필요가 없어 효율적임


# ## K-최근접 이웃: 게으른 학습 알고리즘

# In[48]:


from sklearn.neighbors import KNeighborsClassifier #KNN 알고리즘

knn = KNeighborsClassifier(n_neighbors = 5, # 최근접 이웃의 개수 지정
                          p = 2, # p = 1이면 맨해튼, 2면 유클리디안 거리
                          metric = 'minkowski') # 유클리디안 거리와 맨해튼 거리를 일반화 한 거리
knn.fit(X_train_std, y_train)

#결정경계 그리기
plot_decision_regions(X_combined_std, y_combined, classifier = knn, test_idx = range(105, 150))

plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc = 'upper left')
plt.tight_layout()
plt.show()

# 5개의 이웃을 지정했으므로 비교적 부드러운 결정경계를 얻음

