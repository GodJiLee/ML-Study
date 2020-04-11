#!/usr/bin/env python
# coding: utf-8

# # Chapter 6 학습 관련 기술들
# ## 6.1 매개변수 갱신
# ##### 손실함수를 최소화하는 과정인 "최적화" 
# ##### 실제로는 매개변수의 공간이 넓고 복잡하기 때문에 순식간에 최소값을 찾는 일은 불가능함
# ##### SGD : 확률적 경사 하강법은 매개변수의 기울기를 이용해서 최소값을 찾는 방법
# 이보다 더 효율적인 방법도 존재
# ## 6.1.1 모험가 이야기
# 손실함수의 최솟값을 찾는 문제를 '깊은 산골짜기를 탐험하는 모험가'에 비유함
# ## 6.1.2 확률적 경사 하강법(SGD)

# \begin{equation*} W := W - \eta \frac{\partial L}{\partial W} \end{equation*}\begin{equation*} W : 갱신할 매개변수 \end{equation*}\begin{equation*} \frac{\partial L}{\partial W} : 손실 함수의 기울기 \end{equation*}\begin{equation*} \eta : 학습률, 미리 정해서 사용 \end{equation*}

# In[5]:


import sys, os #시스템, 운영체제와 상호작용하는 파이썬 함수as
sys.path.append(os.pardir) #부모 경로 지정
os.chdir('C:\\Users\\leejiwon\\Desktop\\프로그래밍\\deep\\deep-learning-from-scratch-master\\deep-learning-from-scratch-master')
import numpy as np #넘파이 불러오기
from dataset.mnist import load_mnist #mnist 데이터셋에서 load_mnist 불러오기
from common.layers import *
from common.gradient import numerical_gradient
from common.functions import *
from collections import OrderedDict
from typing import TypeVar, Generic

(x_train, t_train), (x_test, t_test) =     load_mnist(flatten = True, normalize = False)


# In[6]:


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size) #첫 번째 매개변수이므로 입력층, 은닉층 1에 대한 뉴런 수 적용
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) #두 번째 매개변수이므로 은닉층, 출력층에 대한 뉴런 수 적용
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
    
        a1 = np.dot(x, W1) + b1 #예측함수
        z1 = sigmoid(a1) #시그모이드에 적용 (활성화함수)
        a2 = np.dot(z1, W2) + b2 #2번째 예측함수
        y = softmax(a2) #소프트맥스에 적용 (확률값)
        
        return y
        
    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t): 
        y = self.predict(x) #손실함수
        
        return cross_entropy_error(y, t) #교차 엔트로피 오차 
    def accuracy(self, x, t): #정확도 
        y = self.predict(x)
        y = np.argmax(y, axis=1)                             
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # x : 입력 데이터, t : 정답 레이블
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
    
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
        
        batch_num = x.shape[0]
        
        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        
        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads


# In[7]:


class SGD: #확률적경사하강법 클래스 정의
    def __init__(self, lr=0.01): #인스턴스변수
        self.lr = lr
    
    def update(self, params, grads): #SGD 동안 반복할 구문
        for key in params.keys(): #params는 딕셔너리 변수
            params[key] -= self.lr * grads[key] #기울기에 따른 갱신


# network = TwoLayerNet(...)
# optimizer = SGD() #최적화 매커니즘으로 SGD 사용 #은닉층과 입력층에 대한 정의 필요
# 
# for i in range(10000):
#     ...
#     x_batch, t_batch = get_mini_batch(...) #미니배치
#     grads = network.gradient(x_batch, t_batch) #기울기
#     params = network.params 
#     optimizer.update(params, grads)

# ### * SGD 대신 이용할 수 있는 다양한 프레임워크들
# #### https://github.com/Lasagne/Lasagne/blob/master/lasagne/updates.py

# ## 6.1.3 SGD의 단점

# \begin{equation*} f(x,y) = \frac{1}{20} x^2 + y^2 \end{equation*}

# In[13]:


# 그림 6-1의 함수 시각화
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
X = np.arange(-10, 10, 0.5)
Y = np.arange(-10, 10, 0.5)
XX, YY = np.meshgrid(X, Y)
ZZ = (1 / 20) * XX**2 + YY**2

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(XX, YY, ZZ, rstride=1, cstride=1, cmap='hot');


# In[14]:


# 그림 6-1 f(x, y) = (1/20) * x**2 + y**2 등고선
plt.contour(XX, YY, ZZ, 100, colors='k')
plt.ylim(-10, 10)
plt.xlim(-10, 10)


# ##### 특징 : y축 방향은 가파른데, x축 방향은 완만함
# ##### 기울기에 따라 최저점이 대부분 (0,0)을 가리키지 않음

# In[65]:


def _numerical_gradient_no_batch(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x) # x와 형상이 같은 배열을 생성
    
    for idx in range(x.size):
        tmp_val = x[idx]
        
        # f(x+h) 계산
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)
        
        # f(x-h) 계산
        x[idx] = tmp_val - h 
        fxh2 = f(x) 
        
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val # 값 복원
        
    return grad


# In[67]:


# 그림 6-2의 기울기 정보
from mpl_toolkits.mplot3d import Axes3D

def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)
        
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)
        
        return grad

def function_2(x):
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)
     
x0 = np.arange(-10, 10, 1)
x1 = np.arange(-10, 10, 1)
X, Y = np.meshgrid(x0, x1)
    
X = X.flatten()
Y = Y.flatten()

grad = numerical_gradient(function_2, np.array([(1/(20**0.5))*X, Y]) )
    
plt.figure()
plt.quiver(X, Y, -grad[0], -grad[1],  angles="xy",color="#666666")#,headwidth=10,scale=40,color="#444444")
plt.xlim([-10, 10])
plt.ylim([-5, 5])
plt.xlabel('x0')
plt.ylabel('x1')
plt.grid()
plt.legend()
plt.draw()
plt.show()


# ##### 특징 : 대부분 최저점인 (0,0)을 가리키고 있지 않음
# ##### 이대로 SGD를 적용하게 되면 '비등방성 함수'의 성질에 따라 비효율적인 경로 [그림 6-3]을 그리며 최저점을 탐색하게 됨

# ## 6.1.4 모멘텀

# \begin{equation*} v := \alpha v - \eta 
#     \frac{\partial{L}}{\partial{W}} 
#     \end{equation*}

# \begin{equation*} W := W + v \end{equation*}\begin{equation*} W : 갱신할 매개변수 \end{equation*}\begin{equation*} \frac{\partial L}{\partial W} : 손실 함수의 기울기 \end{equation*}\begin{equation*} \eta : 학습률, 미리 정해서 사용 \end{equation*}

# ##### 모멘텀 : 물리에서 말하는 '운동량' 
# > 공이 바닥을 구르듯 기울기 방향으로 가중되는 움직임
# ##### av항은 물리에서 '지면 마찰', '공기 저항'과 같은 역할
# > 기울기 영향을 받지 않을 때 서서히 변화하는 역할

# In[69]:


class Momentum:
    def __init__(self, lr = 0.01, momentum = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None #초기화 값은 아무것도 지정하지 않음
        
    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val) #update에서 매개변수와 같은 구조의 데이터를 딕셔너리 변수로 저장함
                
        for key in params.keys(): #위의 식 구현
            self.v[key] = self.momentum*self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


# ##### 최적 갱신 경로에서 모멘텀이 SGD 방법보다 지그재그 정도가 덜함
# ##### x축 방향으로 빠르게 다다가기 때문

# ## 6.1.5 AdaGrad

# ##### 신경망학습에서는 학습률을 잘 설정해야 함 
# > 너무 작으면 거의 갱신되지 않고 너무 크면 발산하기 때문
# ##### 적정한 학습률을 정하기 위해 '학습률 감소' 기법 사용
# #### AdaGrad : 각각의 매개변수에 따른 맞춤값 지정

# \begin{equation*} h := h + \frac{\partial{L}}{\partial{W}} \odot \frac{\partial{L}}{\partial{W}} \end{equation*}

# \begin{equation*} W := W - \eta \frac{1}{\sqrt{h}} \frac{\partial{L}}{\partial{W}} \end{equation*}\begin{equation*} W : 갱신할 매개변수 \end{equation*}\begin{equation*} \frac{\partial L}{\partial W} : 손실 함수의 기울기 \end{equation*}\begin{equation*} \eta : 학습률, 미리 정해서 사용 \end{equation*}

# ##### h는 기존 기울기의 제곱수, 이를 학습률에 반영하여 너무 크게 갱신되었던 값에 대해서는 학습률을 낮춤
# #### AdaGrad는 너무 매몰차기 때문에 이를 개선한 RMSProp 방법을 사용하기도 함 
# > 이전의 갱신 값은 서서히 잊음 : 지수이동평균 (EMA)

# In[71]:


class AdaGrad:
    def __init__(self, lr = 0.01):
        self.lr = lr
        self.h = None
    
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
                
        for key in params.key():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


# ##### 두 번째 식의 h 값에 작은 수를 더해줌으로써 0으로 나눠지는 일을 막음
# #### AdaGrad로 최적 경로를 구하게 되면 SGD, 모멘텀 기법에 비해 더 효율적으로 최적값에 도달하는 것을 알 수 있음
# > 크게 갱신되는 값(y)에 대해 갱신 강도를 빠르게 작아지도록 만들기 때문

# ## 6.1.6 Adam

# ##### 모멘텀과 AdaGrad 갱신방법의 융합버전
# > 매개변수 공간을 효율적으로 탐색하며 하이퍼파라미터의 편향을 보정하는 기능
# ##### 갱신 경로를 보면 모멘텀과 같이 그릇 바닥을 구르듯 갱신되며 모멘텀보다 더 완만한 경사로 갱신됨
# ##### 하이퍼파라미터를 3개 설정함 (학습률, 1차 모멘텀용 계수, 2차 모멘텀용 계수) _자세히 다루지 않음

# In[73]:


class Adam:

    """Adam (http://arxiv.org/abs/1412.6980v8)"""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        
        self.iter += 1 #Adam 수식에 대한 자세한 부분은 책에서 다루지 않음
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)         
        
        for key in params.keys():
            #self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            #self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
            
            #unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias #편향을 조정해주는 기능
            #unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
            #params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)


# ## 6.1.7 어느 갱신 방법을 이용할 것인가?

# ##### SGD, 모멘텀, AdaGrad, Adam 네 기법에 대한 최적 경로 비교
# > 풀어야 할 문제, 하이퍼파라미터 설정에 따라 최적의 방법이 달라짐 (각자의 장단이 있음)
# ##### 이 책에서는 SGD 와 Adam 방법을 사용

# In[75]:


import matplotlib.pyplot as plt
from collections import OrderedDict
from common.optimizer import *

def f(x, y):
    return x**2 / 20.0 + y**2

def df(x, y):
    return x / 10.0, 2.0*y

init_pos = (-7.0, 2.0) #초깃값 설정
params = {} #디렉토리 매개변수
params['x'], params['y'] = init_pos[0], init_pos[1]
grads = {}
grads['x'], grads['y'] = 0, 0


optimizers = OrderedDict() #4가지 최적화 방법 정의
optimizers["SGD"] = SGD(lr=0.95)
optimizers["Momentum"] = Momentum(lr=0.1)
optimizers["AdaGrad"] = AdaGrad(lr=1.5)
optimizers["Adam"] = Adam(lr=0.3)

idx = 1

for key in optimizers:
    optimizer = optimizers[key]
    x_history = []
    y_history = []
    params['x'], params['y'] = init_pos[0], init_pos[1]
    
    for i in range(30):
        x_history.append(params['x'])
        y_history.append(params['y'])
        
        grads['x'], grads['y'] = df(params['x'], params['y'])
        optimizer.update(params, grads)
    

    x = np.arange(-10, 10, 0.01)
    y = np.arange(-5, 5, 0.01)
    
    X, Y = np.meshgrid(x, y) 
    Z = f(X, Y)
    
    # 외곽선 단순화
    mask = Z > 7
    Z[mask] = 0
    
    # 그래프 그리기
    plt.subplot(2, 2, idx)
    idx += 1
    plt.plot(x_history, y_history, 'o-', color="red")
    plt.contour(X, Y, Z)
    plt.ylim(-10, 10)
    plt.xlim(-10, 10)
    plt.plot(0, 0, '+')
    #colorbar()
    #spring()
    plt.title(key)
    plt.xlabel("x")
    plt.ylabel("y")
    
plt.show()


# ##### 이번 데이터셋에 대해서는 Adam이 가장 효율적으로 갱신되는 것을 알 수 있음

# ## 6.1.8 MNIST 데이터셋으로 본 갱신 방법 비교

# In[77]:


# 손글씨 숫자 인식 데이터에 대한 네 기법의 학습 진도 비교
# 각 층이 100개의 뉴런으로 구성된 5층 신경망에서 ReLU 함수를 활성화함수로 사용

import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.util import smooth_curve
from common.multi_layer_net import MultiLayerNet
#from common.optimizer import *

# 0. MNIST 데이터 읽기==========
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

train_size = x_train.shape[0]
batch_size = 128
max_iterations = 2000

# 1. 실험용 설정==========
optimizers = {}
optimizers['SGD'] = SGD()
optimizers['Momentum'] = Momentum()
optimizers['AdaGrad'] = AdaGrad()
optimizers['Adam'] = Adam()
#optimizers['RMSprop'] = RMSprop()

networks = {}
train_loss = {}
for key in optimizers.keys():
    networks[key] = MultiLayerNet(
        input_size=784, hidden_size_list=[100, 100, 100, 100],
        output_size=10)
    train_loss[key] = []    

# 2. 훈련 시작==========
for i in range(max_iterations):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    for key in optimizers.keys():
        grads = networks[key].gradient(x_batch, t_batch)
        optimizers[key].update(networks[key].params, grads)
    
        loss = networks[key].loss(x_batch, t_batch)
        train_loss[key].append(loss)
    
    #출력 설정
    if i % 100 == 0:
        print( "===========" + "iteration:" + str(i) + "===========")
        for key in optimizers.keys():
            loss = networks[key].loss(x_batch, t_batch)
            print(key + ":" + str(loss))

# 3. 그래프 그리기==========
markers = {"SGD": "o", "Momentum": "x", "AdaGrad": "s", "Adam": "D"}
x = np.arange(max_iterations)
for key in optimizers.keys():
    plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 1)
plt.legend()
plt.show()


# ##### 하이퍼파라미터 설정과 신경망 구조에 따라 달라질 수 있지만, 일반적으로 SGD가 나머지 세 방법에 비해 속도, 정확도 면에서 효율성이 떨어짐

# # 6.2 가중치의 초깃값

# ##### 가중치 초깃값 설정에 따라 학습의 성패가 갈림
# > 권장 초깃값 설정

# ## 6.2.1 초깃값을 0으로 하면?

# ##### 가중치 감소 : 오버피팅을 억제해 범용 성능을 높이는 테크닉 
# > 가중치 매개변수의 값이 작아지도록 학습 
# ##### 작은 가중치를 위해 애초에 가중치를 작게 설정함 (0.01 * np.random.randn(10,100))
# > 하지만 0으로 설정하면 오차역전파법에 의해 모든 가중치가 똑같이 갱신되는 문제 발생 
# ##### 가중치 대칭 문제를 해결하기 위해 random하게 설정함

# ## 6.2.2 은닉층의 활성화값 분포

# ##### 활성화함수: 시그모이드, 신경망: 5층 
# > 가중치 초기값에 따른 활성화값의 변화

# In[41]:


import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def ReLU(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

input_data = np.random.randn(1000, 100) #1000개의 데이터 중 100개 임의 추출
node_num = 100 #각 은닉층의 노드 (뉴련) 개수
hidden_layer_size = 5 #은닉층 5개
activations = {} #활성화 결과 저장

x = input_data

def get_activation(hidden_layer_size, x, w, a_func = sigmoid):
    for i in range(hidden_layer_size):
        if i != 0:
            x = activations[i-1]
            
        a = np.dot(x, w)
        
        #활성화함수 ReLU, tanh로 바꿔서 실험
        z = a_func(a)
        # z = ReLU(a), z = tanh(a)

        activations[i] = z 
    return activations

#초깃값을 다양하게 바꿔서 실험
w = np.random.randn(node_num, node_num) * 1 #표준편차가 1인 정규분포 (변경 대상)

z = sigmoid
# z = ReLU
# z = tanh

activations = get_activation(hidden_layer_size, x, w, z)


# In[42]:


# 히스토그램 그리기
def get_histogram(activations):
    
    for i, a in activations.items():
        plt.subplot(1, len(activations) , i + 1)
        plt.title(str(i + 1) + "-layer")
        if i != 0: plt.yticks([], [])
            #plt.xlim(0.1, 1)
            #plt.ylim(0, 7000)
        plt.hist(a.flatten(), 30, range = (0,1))
    plt.show()
    
get_histogram(activations)


# ##### 각 층의 활성화값 분포 
# > 0과 1에 치중되어 분포함: 해당 값들에서 기울기 값이 0으로 수렴함
# ##### 가중치 매개변수를 0으로 지정했을 때의 문제와 동일
# > 기울기 소실 (gradient vanishing)

# In[43]:


# 가중치 표준편차를 0.01로 바꾸었을 때 
w = np.random.randn(node_num, node_num) * 0.01

z = sigmoid
# z = ReLU
# z = tanh

activations = get_activation(hidden_layer_size, x, w, z)


# In[44]:


# 히스토그램 그리기
get_histogram(activations)


# ##### 활성화 값들이 0.5에 치우쳐진 모습 
# > 기울기 소실 문제는 발생하지 않지만 대부분의 데이터가 한 값에 치중되어 있기 때문에 표현력을 제한하는 문제 발생: 다수의 뉴련이 거의 같은 값을 출력하는 상황
# ##### 활성화 값은 적당하게 다양한 데이터가 되어야 

# * 권장 가중치 초깃값 Xavier 초깃값
# > 앞 층의 노드의 개수 (n) 이 커질 수록 가중치는 좁은 분포(1 / np.sqrt(n))를 가짐 

# In[45]:


# 가중치 표준편차를 (1 / np.sqrt(n)로 바꾸었을 때
# 앞 층의 노드 수는 100개로 단순화
w = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num)

z = sigmoid
# z = ReLU
# z = tanh

activations = get_activation(hidden_layer_size, x, w, z)


# In[46]:


# 히스토그램 그리기    
get_histogram(activations)


# ##### 앞선 방식보다 넓게 분포되는 모습 
# > 표현력 제한 없이 효율적인 학습 가능
# ##### tanh함수(쌍곡선 함수)를 사용하면 층을 거듭할 수록 일그러지는 문제 해결, 정규분포화 됨
# > sigmoid: 0.05에서 대칭, tanh: 0에서 대칭 > 활성화 함수로 더 적합

# ## 6.2.3 ReLU를 사용할 때의 가중치 초깃값

# ##### 선형함수 (sigmoid, tanh)의 경우 Xavier 초깃값 사용, 비선형함수 (ReLU)의 경우 2 / np.sqrt(n) 정규분포인 He 초깃값 사용

# In[47]:


# 표준편차가 0.01을 정규분포를 가중치 초깃값으로 사용할 때
w = np.random.randn(node_num, node_num) * 0.01
z = ReLU
activations = get_activation(hidden_layer_size, x, w, z)
get_histogram(activations)

# 아주 작은 활성화 값을 가짐 > 학습이 거의 이루어지지 않음


# In[48]:


# Xavier 초깃값을 사용할 때
w = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num)
activations = get_activation(hidden_layer_size, x, w, z)
get_histogram(activations)

# 층이 거듭될 수록 한 값에 치중되는 모슴 > 기울기 소실 문제


# In[49]:


# He 초깃값을 사용할 때
w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num)
activations = get_activation(hidden_layer_size, x, w, z)
get_histogram(activations)

# ReLU함수의 권장 초깃값으로 적정, 기울기 소실, 표현력 제한 문제 없이 고르게 분포함


# ## 6.2.4 MNIST 데이터셋으로 본 가중치 초깃값 비교

# ##### 실제 데이터셋으로 초깃값에 따른 학습 결과 비교

# In[86]:


import sys, os #시스템, 운영체제와 상호작용하는 파이썬 함수as
sys.path.append(os.pardir) #부모 경로 지정
os.chdir('C:\\Users\\leejiwon\\Desktop\\프로그래밍\\deep\\deep-learning-from-scratch-master\\deep-learning-from-scratch-master')

import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.util import smooth_curve
from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD

# 0. MNIST 데이터 읽기 ==================
(x_train, t_train), (x_test, t_test) = load_mnist(normalize = True)

train_size = x_train.shape[0]
batch_size = 128
max_iterations = 2000

# 1. 실험용 설정 ==================
weight_init_types = {'std = 0.01': 0.01, 'Xavier': 'sigmoid', 'He' : 'relu'}
optimizer = SGD(lr = 0.01)

networks = {}
train_loss = {}
for key, weight_type in weight_init_types.items():
    networks[key] = MultiLayerNet(input_size = 784, hidden_size_list = [100, 100, 100, 100], 
                                 output_size = 10, weight_init_std = weight_type)
    train_loss[key] = []

# 2. 훈련 시작 ===================
for i in range(max_iterations):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    for key in weight_init_types.keys():
        grads = networks[key].gradient(x_batch, t_batch)
        optimizer.update(networks[key].params, grads)
    
        loss = networks[key].loss(x_batch, t_batch)
        train_loss[key].append(loss)
    
    if i % 100 == 0:
        print("===========" + "iteration:" + str(i) + "===========")
        for key in weight_init_types.keys():
            loss = networks[key].loss(x_batch, t_batch)
            print(key + ":" + str(loss))
            
# 3. 그래프 그리기===============
markers = {'std = 0.01' : 'o', 'Xavier' : 's', 'He' : 'D'}
x = np.arange(max_iterations)
for key in weight_init_types.keys():
    plt.plot(x, smooth_curve(train_loss[key]), marker = markers[key], markevery = 100, label = key)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 2.5)
plt.legend()
plt.show()


# ##### 뉴런 개수 100개, 5층 신경망, ReLU를 활성화 함수로 사용한 학습
# > 표준 편차 0.01 : 학습 거의 진행되지 않음, Xavier보다 He 초깃값이 학습진도가 더 빠름

# # 6.3 배치 정규화

# ##### 각 층의 활성화값 분포를 적절히 떨어뜨려 효율적인 학습이 가능하도록 '강제'하는 방법

# ## 6.3.1 배치 정규화 알고리즘

# ##### 배치 정규화가 주목받는 이유 
# > 1) 학습 속도 개선 2) 초깃값 의존도 낮음 3) 오버피팅 억제

# ##### 배치 정규화를 실행하기 위해 활성함수 층 앞 or 뒤에 '배치 정규화 계층' 삽입
# > 미니 배치를 단위로 평균이 0, 분산이 1이 되도록 정규화

# \begin{equation*} \mu_{B} := \frac{1}{m} \sum^{m}_{i=1} x_{i} \end{equation*}\begin{equation*} \sigma^{2}_{B} := \frac{1}{m} \sum^{m}_{i=1} (x_{i} - \mu_{B})^{2} \end{equation*}\begin{equation*} x_{i} := \frac{x_{i}-\mu_{B}}{\sqrt{\sigma^{2}_{B}+\epsilon}} \end{equation*}

# ##### 기호 엡실론은 분모가 0이 되지 않게 하기위한 작은 상수 
# > 이런 일련의 과정을 통해 분포가 덜 치우치고 효율적인 학습이 가능하도록 함 

# ##### 배치 정규화 계층에 확대, 이동 작업을 수행함 
# > 아래 식에서 감마가 확대, 베타가 이동을 나타내며 초깃값은 (1, 0) : 원본 그대로에서 시작

# \begin{equation*} y_{i} = \gamma \hat{x_{i}} + \beta \end{equation*}

# ##### 이를 신경망에서 순전파에 적용해보면 계산 그래프에 의해 표현 가능 [그림 6-17]
# ##### https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html 에서 계산그래프 알고리즘 확인 가능
# > 역전파는 다소 복잡하므로 생략

# ## 6.3.2 배치 정규화의 효과

# In[88]:


import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.optimizer import SGD, Adam

(x_train, t_train), (x_test, t_test) = load_mnist(normalize = True)

# 학습 데이터를 줄임====================
x_train = x_train[:1000]
t_train = t_train[:1000]

max_epochs = 20
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.01

def __train(weight_init_std):
    bn_network = MultiLayerNetExtend(input_size = 784, hidden_size_list = [100, 100, 100, 100, 100], 
                                    output_size = 10, weight_init_std = weight_init_std, use_batchnorm = True)
    network = MultiLayerNetExtend(input_size = 784, hidden_size_list = [100, 100, 100, 100, 100],
                                 output_size = 10, weight_init_std = weight_init_std)
    optimizer = SGD(lr = learning_rate)
    
    train_acc_list = []
    bn_train_acc_list = []
    
    iter_per_epoch = max(train_size / batch_size, 1)
    epoch_cnt = 0
    
    for i in range(1000000000):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        
        for _network in (bn_network, network):
            grads = _network.gradient(x_batch, t_batch)
            optimizer.update(_network.params, grads)
            
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            bn_train_acc = bn_network.accuracy(x_train, t_train)
            train_acc_list.append(train_acc)
            bn_train_acc_list.append(bn_train_acc)
            
            print("epoch:" + str(epoch_cnt) + " / " + str(train_acc) + " - " + str(bn_train_acc))
            
            epoch_cnt += 1
            if epoch_cnt >= max_epochs:
                break
            
    return train_acc_list, bn_train_acc_list

# 그래프 그리기===================
weight_scale_list = np.logspace(0, -4, num = 16)
x = np.arange(max_epochs)

for i, w in enumerate(weight_scale_list):
    print( "=============== " + str(i + 1) + "/16" + " ================")
    train_acc_list, bn_train_acc_list = __train(w)
    
    plt.subplot(4, 4, i + 1)
    plt.title("W: " + str(w))
    if i == 15:
        plt.plot(x, bn_train_acc_list, label = 'Batch Normalization', markevery = 2)
        plt.plot(x, train_acc_list, linestyle = "--", label = 'Normal(without BatchNorm)', 
                markevery = 2)
    else:
        plt.plot(x, bn_train_acc_list, markevery =2)
        plt.plot(x, train_acc_list, linestyle = '--', markevery =2)
        
    plt.ylim(0, 1.0)
    if i % 4:
        plt.yticks([])
    else:
        plt.ylabel("accuracy")
    
    if i < 12:
        plt.xticks([])
    else: 
        plt.xlabel("epochs")
    plt.legend(loc = 'lower right')

plt.show()


# ##### 거의 모든 경우 배치 정규화를 사용할 때 학습 진도가 빠르며 가중치 의존도가 낮음

# # 6.4 바른 학습을 위해 

# ##### 오버피팅: 훈련데이터에만 지나치에 적응되어 그 외의 데이터에는 적절히 대응하지 못하는, 범용 능력이 떨어지는 문제

# ## 6.4.1 오버피팅

# ##### 오버피팅이 발생하는 경우 : 1) 매개변수가 많고 표현력이 높은 경우, 2) 훈련데이터가 적은 경우 
# > 오버피팅 문제가 발생하는 상황을 만들기 위해 기존 MNIST 데이터셋에서 학습데이터 수를 300개로 줄이고, 복잡한 7층 네트워크를 사용함 (활성화함수는 ReLU, 층별 뉴런 수는 100개)

# In[7]:


from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize = True) #데이터 불러오기
x_train = x_train[:300]
t_train = t_train[:300] #오버피팅을 위해 학습데이터 수를 줄임


# In[8]:


import numpy as np
import matplotlib.pyplot as plt
from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD

#가중치 감쇠 설정 
weight_decay_lambda = 0 #가중치 감쇠 사용하지 않음
#weight_decay_lambda = 0.1 #사용하는 경우

network = MultiLayerNet(input_size = 784, hidden_size_list = [100, 100, 100, 100, 100, 100], 
                       output_size = 10, weight_decay_lambda = weight_decay_lambda)
optimizer = SGD(lr = 0.01) #학습률이 0.01인 SGD로 매개변수 갱신 
max_epochs = 201
train_size = x_train.shape[0]
batch_size = 100

train_loss_list = []
train_acc_list = [] #에폭단위 정확도 저장 #에폭은 모든 훈련 데이터를 한 번씩 본 주기  
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)
epoch_cnt = 0

for i in range(1000000000):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask] #x, t 배치 획득
    
    grads = network.gradient(x_batch, t_batch) #기울기 산출
    optimizer.update(network.params, grads) #활성화함수 업데이트
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc) #정확도 계산
        
        print("epoch:" + str(epoch_cnt) + ", train acc:" + str(train_acc) + "test acc" + str(test_acc))
        
        epoch_cnt += 1
        if epoch_cnt >= max_epochs:
            break


# In[9]:


# 그래프 그리기
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, train_acc_list, marker = 'o', label = 'train', markevery = 10) #훈련데이터 표기 지정
plt.plot(x, test_acc_list, marker = 's', label = 'test', markevery = 10) #시험 데이터 표기 지정
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc = "lower right") #주석 표기 위치 설정
plt.show()


# ##### 100 epoch 이상부터 훈련데이터는 100%의 정확도를 보이는 반면, 시험데이터에 대해서는 적절한 학습이 이루어지지 못함 
# > 훈련데이터를 줄이고 계층을 복잡하게 만들어 오버피팅 문제가 발생함 (범용성을 잃음)

# ## 6.4.2 가중치 감소

# ##### 오버피팅 억제 방법 중 하나, 애초에 오버피팅이 큰 가중치에 의해 발생했으므로 이에 패널티를 주고자 하는 아이디어
# > 기존 손실함수에 가중치의 제곱노름을 더해줌 (책에서는 L2노름=L2법칙을 사용, L2 노름은 아래 식과 같음)

# \begin{equation*} \sqrt{W_{1}^{2}+W_{2}^{2} + ... + W_{n}^{2}} \end{equation*}

# ##### 제곱 노름을 적용한 가중치 감소는 1/2 λ (W**2) 
# > 여기서 람다는 하이퍼파라미터로 패널티 경중을 설정(크게 잡을 수록 큰 패널티 부과), 1/2는 가중치 감소의 미분값에 대한 조정치 

# In[23]:


# 람다값을 0.1로 설정한 경우 
#weight_decay_lambda = 0 #가중치 감쇠 사용하지 않음
weight_decay_lambda = 0.1 #사용하는 경우

network = MultiLayerNet(input_size = 784, hidden_size_list = [100, 100, 100, 100, 100, 100], 
                       output_size = 10, weight_decay_lambda = weight_decay_lambda)
optimizer = SGD(lr = 0.01) #학습률이 0.01인 SGD로 매개변수 갱신 
max_epochs = 201
train_size = x_train.shape[0]
batch_size = 100

train_loss_list = []
train_acc_list = [] #에폭단위 정확도 저장 #에폭은 모든 훈련 데이터를 한 번씩 본 주기  
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)
epoch_cnt = 0

for i in range(1000000000):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask] #x, t 배치 획득
    
    grads = network.gradient(x_batch, t_batch) #기울기 산출
    optimizer.update(network.params, grads) #활성화함수 업데이트
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc) #정확도 계산
        
        print("epoch:" + str(epoch_cnt) + ", train acc:" + str(train_acc) + "test acc" + str(test_acc))
        
        epoch_cnt += 1
        if epoch_cnt >= max_epochs:
            break

#그래프 그리기
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, train_acc_list, marker = 'o', label = 'train', markevery = 10) #훈련데이터 표기 지정
plt.plot(x, test_acc_list, marker = 's', label = 'test', markevery = 10) #시험 데이터 표기 지정
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc = "lower right") #주석 표기 위치 설정
plt.show()


# ##### 가중치 감소를 적용하여 오버피팅이 어느정도 억제됨
# > 하지만 훈련데이터에 대한 정확도 역시 함께 감소되었음

# ## 6.4.3 드롭아웃

# ##### 신경망이 더 복잡해지는 경우 가중치 감소만으로는 오버피팅 문제를 해결할 수 없음
# > further한 기법으로 드롭아웃이 존재함 
# ##### 드롭아웃 : 훈련 데이터에 대해서만 데이터 흐름에 있어 은닉층의 뉴런을 임의로 삭제하여 신호를 전달하지 못하도록 하는 방법
# > 시험 데이터에 대해서는 적용하지 않고 훈련 때 삭제하지 않은 비율을 곱해서 출력해줌 

# ##### http://chainer.org/ 에서 더 자세한 드롭아웃 구현 확인 가능

# In[17]:


#드롭아웃 구현
class Dropout:
    def __init__(self, dropout_ratio=0.5): #드롭아웃 비율을 0.5로 지정
        self.dropout_ratio = dropout_ratio 
        self.mask = None
        
    def forward(self, x, train_flg=True): #순전파 계산 #중요한 부분
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio #무작위 삭제 비율 > 드롭아웃 비율
            # x와 형상이 같은 배열 생성
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio) #드롭아웃하지 않은 비율
        
    def backward(self, dout): #역전파(미분값) 계산
        #ReLU함수와 비슷한 매커니즘 (True일 때만 통과)
        return dout * self.mask


# ##### 역전파에서는 순전파에서 통과된 뉴런만 신호를 받을 수 있도록 지정
# > ReLU 함수의 성질과 동일 

# In[ ]:


# MNIST 데이터를 이용한 구현
# trainer 클래스를 구현 > 네트워크 학습을 대신해줌
# 7층의 네트워크 계층, 앞 실험과 같은 조건
import numpy as np
import matplotlib.pyplot as plt 
from dataset.mnist import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.trainer import Trainer

(x_train, t_train), (x_test, t_test) = load_mnist(normalize = True)

#오버피팅 재현을 위해 학습 데이터 줄임
x_train = x_train[:300]
t_train = t_train[:300]

#드롭아웃 사용 유무와 비율 설정 ==============
use_dropout = True #사용하지 않을 때는 False
dropout_ratio = 0.2
#=============================================

network = MultiLayerNetExtend(input_size = 784, hidden_size_list = [100, 100, 100, 100, 100, 100],
                             output_size =10, use_dropout = use_dropout, dropout_ration = dropout_ratio)

trainer = Trainer(network, x_train, t_train, x_test, t_test, 
                 epochs = 301, mini_batch_size = 100, 
                 optimizer = 'sgd', optimizer_param = {'lr' : 0.01}, verbose = False)
trainer.train()

#그래프 그리기 ===================
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, marker = 'o', label = 'train', markevery = 10)
plt.plot(x, test_acc_list, marker = 's', label = 'test', markevery = 10)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc = 'lower right')
plt.show()


# ##### 가중치 감소와 마찬가지로 오버피팅을 억제하며 훈련데이터의 정확도가 낮아지는 결과

# * 앙상블 학습: 개별적으로 학습시킨 여러 모델의 출력을 평균 내어 추론하는 방식 
#     > 드롭아웃과 매우 비슷한 매커니즘 1) 무작위 삭제 == 매번 다른 모델을 학습시킴 2) 삭제 비율을 곱해줌 == 평균 작업
# 
# ##### 드롭아웃은 앙상블 학습을 하나의 네트워크로 나타낸 것이라고 생각할 수 있음

# # 6.5 적절한 하이퍼파라미터 값 찾기

# ##### 하이퍼파라미터 : 인간이 직접 설정해줘야 하는 매개변수
# > 각 층의 뉴런 수, 배치 크기, 학습률, 가중치 감소 등 

# ## 6.5.1 검증 데이터

# ##### 학습에 사용되는 데이터셋은 대게 오버피팅과 범용성능을 테스트하기 위해 시험데이터와 훈련데이터를 나눠서 세팅함 
# ##### 하이퍼파라미터의 성능을 평가하기 위해서는 검증 데이터 (validation data)를 따로 할당해 줘야 함
# > 자체적으로 지정해두지 않은 경우도 있음 (훈련데이터의 일부를 직접 할당해야 함)

# In[20]:


# coding: utf-8
import numpy as np


def smooth_curve(x):
    """손실 함수의 그래프를 매끄럽게 하기 위해 사용
    
    참고：http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html
    """
    window_len = 11
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[5:len(y)-5]


def shuffle_dataset(x, t):
    """데이터셋을 뒤섞는다.
    Parameters
    ----------
    x : 훈련 데이터
    t : 정답 레이블
    
    Returns
    -------
    x, t : 뒤섞은 훈련 데이터와 정답 레이블
    """
    permutation = np.random.permutation(x.shape[0])
    x = x[permutation,:] if x.ndim == 2 else x[permutation,:,:,:]
    t = t[permutation]

    return x, t

def conv_output_size(input_size, filter_size, stride=1, pad=0):
    return (input_size + 2*pad - filter_size) / stride + 1


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """다수의 이미지를 입력받아 2차원 배열로 변환한다(평탄화).
    
    Parameters
    ----------
    input_data : 4차원 배열 형태의 입력 데이터(이미지 수, 채널 수, 높이, 너비)
    filter_h : 필터의 높이
    filter_w : 필터의 너비
    stride : 스트라이드
    pad : 패딩
    
    Returns
    -------
    col : 2차원 배열
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """(im2col과 반대) 2차원 배열을 입력받아 다수의 이미지 묶음으로 변환한다.
    
    Parameters
    ----------
    col : 2차원 배열(입력 데이터)
    input_shape : 원래 이미지 데이터의 형상（예：(10, 1, 28, 28)）
    filter_h : 필터의 높이
    filter_w : 필터의 너비
    stride : 스트라이드
    pad : 패딩
    
    Returns
    -------
    img : 변환된 이미지들
    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]


# In[21]:


(x_train, t_train), (x_test, t_test) = load_mnist()

#훈련데이터 뒤섞음 
x_train, t_train = shuffle_dataset(x_train, t_train) #데이터셋 안 치우침 문제 해결

#20%를 검증 데이터로 분할
validation_rate = 0.20
validation_num = int(x_train.shape[0] * validation_rate)

#검증 데이터셋 획득!
x_val = x_train[:validation_num]
t_val = t_train[:validation_num]
x_train = x_train[validation_num:]
t_train = t_train[validation_num:]


# ## 6.5.2 하이퍼파라미터 최적화

# ##### 최적값이 존재하는 범위를 조금씩 좁혀가는 방법 선택
# > 1) 대략적인 범위 설정 2) 샘플링 3) 정확도 평가 4) 작업 반복, 값 획득
# ##### 신경망 학습에는 그리드 서치보다 샘플링이 더 적합함 
# * '대략적인 범위'는 로그 스케일로 지정 (10의 거듭제곱 꼴)
# * 시간이 오래걸리는 학습 단계이므로 최대한 거를 데이터는 걸러서 에폭의 크기를 작게 만드는 것이 중요함 

# ##### 위의 최적화 방법은 실용적이지만 과학적인 방법은 아님
# > 베이즈 최적화로 과학적 접근 가능 Practical Bayesian Optimization of Machine Learning Algorithms 참고

# ## 6.5.3 하이퍼 파라미터 최적화 구현하기 

# In[22]:


# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.util import shuffle_dataset
from common.trainer import Trainer

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 결과를 빠르게 얻기 위해 훈련 데이터를 줄임
x_train = x_train[:500]
t_train = t_train[:500]

# 20%를 검증 데이터로 분할
validation_rate = 0.20
validation_num = int(x_train.shape[0] * validation_rate)
x_train, t_train = shuffle_dataset(x_train, t_train)
x_val = x_train[:validation_num]
t_val = t_train[:validation_num]
x_train = x_train[validation_num:]
t_train = t_train[validation_num:]


def __train(lr, weight_decay, epocs=50):
    network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100],
                            output_size=10, weight_decay_lambda=weight_decay)
    trainer = Trainer(network, x_train, t_train, x_val, t_val,
                      epochs=epocs, mini_batch_size=100,
                      optimizer='sgd', optimizer_param={'lr': lr}, verbose=False)
    trainer.train()

    return trainer.test_acc_list, trainer.train_acc_list


# 하이퍼파라미터 무작위 탐색======================================
optimization_trial = 100
results_val = {}
results_train = {}
for _ in range(optimization_trial):
    # 탐색한 하이퍼파라미터의 범위 지정===============
    weight_decay = 10 ** np.random.uniform(-8, -4) #가중치감소 범위지정
    lr = 10 ** np.random.uniform(-6, -2) #학습률 범위지정
    # ================================================

    val_acc_list, train_acc_list = __train(lr, weight_decay)
    print("val acc:" + str(val_acc_list[-1]) + " | lr:" + str(lr) + ", weight decay:" + str(weight_decay))
    key = "lr:" + str(lr) + ", weight decay:" + str(weight_decay)
    results_val[key] = val_acc_list
    results_train[key] = train_acc_list

# 그래프 그리기========================================================
print("=========== Hyper-Parameter Optimization Result ===========")
graph_draw_num = 20
col_num = 5
row_num = int(np.ceil(graph_draw_num / col_num))
i = 0

for key, val_acc_list in sorted(results_val.items(), key=lambda x:x[1][-1], reverse=True):
    print("Best-" + str(i+1) + "(val acc:" + str(val_acc_list[-1]) + ") | " + key)

    plt.subplot(row_num, col_num, i+1)
    plt.title("Best-" + str(i+1))
    plt.ylim(0.0, 1.0)
    if i % 5: plt.yticks([])
    plt.xticks([])
    x = np.arange(len(val_acc_list))
    plt.plot(x, val_acc_list)
    plt.plot(x, results_train[key], "--")
    i += 1

    if i >= graph_draw_num:
        break

plt.show()


# ##### Best 1~ Best 5까지의 범위를 보면 학습률은 0.001~0.1, 가중치 감소 계수는 10^-8~10^-6까지의 범위를 가짐
# > 이 범위 내에서 학습을 반복하여 최적값을 찾아낼 수 있음 
