#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np

x = np.array([1,2,3])
W = np.array([[1,2,3],[4,5,6]])

print(x.__class__) #클래스 이름 표시
print(x.shape)
print(x.ndim)
print(W.shape) 
print(W.ndim)


# In[15]:


W = np.array([[1,2,3],[4,5,6]])
X = np.array([[0,1,2],[3,4,5]])

print(W + X)
print(W * X)


# In[16]:


A = np.array([[1,2],[3,4]])
b = np.array([10,20])
A * b


# In[17]:


a = np.array([1,2,3])
b = np.array([4,5,6])
np.dot(a,b)


# In[18]:


A = np.array([[1,2],[3,4]])
B = np.array([[5,6],[7,8]])
np.matmul(A,B)


# In[19]:


def sigmoid(x):
    return 1 / (1 + np.exp(-x)) #시그모이드 함수 정의

x = np.random.randn(10, 2)
W1 = np.random.randn(2, 4)
b1 = np.random.randn(4)
W2 = np.random.randn(4, 3)
b2 = np.random.randn(3)

h = np.matmul(x, W1) + b1 #선형
a = sigmoid(h) #활성화함수
s = np.matmul(a, W2) + b2 #비션형


# In[20]:


class Sigmoid:
    def __init__(self):
        self.params = []
        
    def forward(self, x): #주 변환처리 담당
        return 1 / (1 + np.exp(-x))


# In[21]:


class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
    
    def forward(self, x):
        W, b = self.params
        out = np.matmul(x, W) + b
        return out


# In[22]:


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size
        
        #가중치와 편향 초기화
        W1 = np.random.randn(I, H)
        b1 = np.random.randn(H)
        W2 = np.random.randn(H, O)
        b2 = np.random.randn(O)
        
        #계층 생성
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]
        
        #모든 가중치를 리스트에 모은다.
        self.params = []
        for layer in self.layers:
            self.params += layer.params
        
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x


# In[23]:


x = np.random.randn(10, 2)
model = TwoLayerNet(2, 4, 3)
s = model.predict(x)


# In[24]:


# "+"연산자는 리스트들을 결합해줌
a = ['A', 'B']
a += ['C', 'D']
a


# In[25]:


import numpy as np
D, N = 8, 7
x = np.random.randn(1, D) #입력 
y = np.repeat(x, N, axis = 0) #repeat 노드의 순전파 
dy = np.random.randn(N, D) #무작위 기울기 
dx = np.sum(dy, axis = 0, keepdims = True) #역전파 


# In[26]:


D, N = 8, 7
x = np.random.randn(N, D) #입력
y = np.sum(x, axis = 0, keepdims = True) #순전파 

dy = np.random.randn(1, D) #무작위 기울기 
dx = np. repeat(dy, N, axis = 0) #역전파 


# In[27]:


class MatMul:
    def __init__(self, W):
        self.params = [W] #매개변수 저장
        self.grads = [np.zeros_like(W)] #기울기 저장
        self.x = None
        
    def forward(self, x): #순전파
        W, = self.params 
        out = np.matmul(x, W) #matmul (행렬 곱) 실행
        self.x = x
        return out
    
    def backward(self, dout): #역전파
        W, = self.params
        dx = np.matmul(dout, W.T) #W의 전치행렬 곱
        dW = np.matmul(self.x.T, dout) #x의 전치행렬 곱
        self.grads[0][...] = dW #기울기 깊은 복사
        return dx


# In[28]:


a = np.array([1, 2, 3])
b = np.array([4, 5, 6])


# In[29]:


class Sigmoid: #시그모이드 계층 정의
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None
        
    def forward(self, x): #순전파 
        out = 1 / (1 + np.exp(-x)) #시그모이드 함수 정의
        self.out = out
        return out
    def backward(self, dout): #역전파
        dx = dout * (1.0 - self.out) * self.out #시그모이드 함수 미분값과 출력값 쪽의 미분값의 곱
        return dx 


# In[30]:


class Affine:
    def __init__(self, W, b):
        self.params = [W, b] #매개변수 저장
        self.grads = [np.zero_like(W), np.zero_like(b)] #기울기 저장
        self.x = None
        
    def forward(self, x): #순전파
        W, b = self.params
        out = np.matmul(x, W) + b #편향은 repeat 모듈에 의해 더해짐 
        self.x = x
        return out
    
    def backwarad(self, dout): #역전파
        W, b = self.params
        dx = np.matmul(dout, W.T) #W의 전치행렬과의 행렬곱
        dW = np.matmul(self.x.T, dout) #x의 전치행렬과의 행렬곱
        db = np.sum(dout, axis = 0) #repeat과 np.sum은 역관계
        
        self.grads[0][...] = dW #깊은 복제로 기울기 저장 
        self.grads[1][...] = db
        return dx


# In[31]:


class SGD: #매개변수 갱신
    def __init__(self, lr = 0.01):
        sefl.lr = lr #에타 정의
        
    def update(self, params, grads): #매개변수 갱신 처리
        for i in range(len(params)): 
            params[i] -= self.lr * grads[i] #SGD 수식


# In[32]:


#model = TwoLayerNet(...)
#optimizer = SGD() #최적화 모듈

#for i in range(10000):
    #...
    #x_batch, t_batch = get_mini_batch(...) #미니배치 획득
    #loss = model.forward(x_batch, t_batch) #손실함수
    #model.backward()
    #optimizer.update(model.params, model.grads) #갱신
    #...


# In[33]:


import sys
sys.path.append('C:\\Users\\leejiwon\\Documents\\GitHub\\deep-learning-from-scratch-2') #부모 디렉터리의 파일을 가져올 수 있도록 설정
from dataset import spiral #나선형 데이터
import matplotlib.pyplot as plt

x, t = spiral.load_data()
print('x', x.shape) #300개의 샘플 데이터, 2차원
print('t', t.shape) #300개의 샘플 데이터, 3차원

# 데이터점 플롯
N = 100
CLS_NUM = 3
markers = ['o', 'x', '^']
for i in range(CLS_NUM):
    plt.scatter(x[i*N:(i+1)*N, 0], x[i*N:(i+1)*N, 1], s=40, marker=markers[i])
plt.show()


# In[34]:


import sys
sys.path.append('C:\\Users\\leejiwon\\Documents\\GitHub\\deep-learning-from-scratch-2')
import numpy as np
from common.layers import Affine, Sigmoid, SoftmaxWithLoss

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size
        
        #가중치와 편향 초기화
        W1 = 0.01 * np.random.randn(I, H) #작은 무작위 값으로 초기화
        b1 = np.zeros(H) #0벡터로 초기화
        W2 = 0.01 * np.random.randn(H, O)
        b2 = np.zeros(O)
        
        #계층 생성
        self.layers = [
            Affine(W1, b1), #완전연결계층
            Sigmoid(),
            Affine(W2, b2)
        ]
        self.loss_layer = SoftmaxWithLoss() #softmax와 crossentropyerror
        
        #모든 가중치와 기울기를 리스트에 모은다.
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layers.params
            self.grads += layer.grads


# In[4]:


def predict(self, x): #추론
    for layer in self.layers:
        x = layer.forward(x)
    return x

def forward(self, x, t): #순전파
    score = self.predict(x)
    loss = self.loss_layer.forward(score, t)
    return loss

def backward(self, dout = 1):
    dout = self.loss_layer.backward(dout) #softmaxwithloss 클래스
    for layer in reversed(self.layers):
        dout = layer.backward(dout)
    return dout


# In[36]:


import sys
sys.path.append('C:\\Users\\leejiwon\\Documents\\GitHub\\deep-learning-from-scratch-2')
import numpy as np
from common.optimizer import SGD #최적화 모듈로 SGD 선택
from dataset import spiral #나선형 데이터셋 임포트
import matplotlib.pyplot as plt
from two_layer_net import TwoLayerNet

#1. 하이퍼파라미터 설정
max_epoch = 300
batch_size = 30
hidden_size = 10
learning_rate = 1.0

#2. 데이터 읽기, 모델과 옵티마이저 생성
x, t = spiral.load_data()
model = TwoLayerNet(input_size = 2, hidden_size = hidden_size, output_size =3) #신경망 생성
optimizer = SGD(lr = learning_rate) #갱신 방법 설정

#학습에 사용하는 변수
data_size = len(x)
max_iters = data_size // batch_size 
total_loss = 0
loss_count = 0
loss_list = []

for epoch in range(max_epoch):
    #3. 데이터 뒤섞기
    idx = np.random.permutation(data_size)
    x = x[idx]
    t = t[idx]
    
    for iters in range(max_iters):
        batch_x = x[iters * batch_size : (iters + 1) * batch_size]
        batch_t = t[iters * batch_size : (iters + 1) * batch_size]
        
        #4. 기울기를 구해 매개변수 갱신
        loss = model.forward(batch_x, batch_t)
        model.backward()
        optimizer.update(model.params, model.grads) #갱신!
        
        total_loss += loss
        loss_count += 1 #반복문
        
        #5. 정기적으로 학습 결과 출력
        if (iter + 1) % 10 == 0:
            avg_loss = total_loss / loss_count
            print('| 에폭 %d | 반복 %d / %d | 손실 %.2f'
                 % (epoch + 1, iter + 1, max_iters, avg_loss))
            loss_list.append(avg_loss)
            total_loss, loss_count = 0, 0

# 학습 결과 플롯
plt.plot(np.arange(len(loss_list)), loss_list, label='train')
plt.xlabel('반복 (x10)')
plt.ylabel('손실')
plt.show()

# 경계 영역 플롯
h = 0.001
x_min, x_max = x[:, 0].min() - .1, x[:, 0].max() + .1
y_min, y_max = x[:, 1].min() - .1, x[:, 1].max() + .1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X = np.c_[xx.ravel(), yy.ravel()]
score = model.predict(X)
predict_cls = np.argmax(score, axis=1)
Z = predict_cls.reshape(xx.shape)
plt.contourf(xx, yy, Z)
plt.axis('off')

# 데이터점 플롯
x, t = spiral.load_data()
N = 100
CLS_NUM = 3
markers = ['o', 'x', '^']
for i in range(CLS_NUM):
    plt.scatter(x[i*N:(i+1)*N, 0], x[i*N:(i+1)*N, 1], s=40, marker=markers[i])
plt.show()


# In[10]:


import numpy as np
np.random.permutation(10) #0~9까지의 인덱스로부터 무작위 순서를 생성해줌


# In[11]:


np.random.permutation(10)


# In[44]:


import sys
sys.path.append('C:\\Users\\leejiwon\\Documents\\GitHub\\deep-learning-from-scratch-2')
from common.optimizer import SGD
from common.trainer import Trainer
from dataset import spiral
from two_layer_net import TwoLayerNet

max_epoch = 300
batch_size = 30
hidden_size = 10
learning_rate = 1.0

x, t = spiral.load_data()
model = TwoLayerNet(input_size = 2, hidden_size = hidden_size, output_size = 3)
optimizer = SGD(lr = learning_rate)

trainer = Trainer(model, optimizer)
trainer.fit(x, t, max_epoch, batch_size, eval_interval = 10)
trainer.plot()


# In[45]:


import numpy as np
a = np.random.randn(3)
a.dtype #데이터 타입 확인


# In[46]:


b = np.random.randn(3).astype(np.float32)
b.dtype


# In[47]:


c = np.random.randn(3).astype('f')
c.dtype


# In[48]:


import cupy as cp
x = cp.arnage(6).reshape(2, 3).astype('f')
x


# In[49]:


x.sum(axis = 1)


# In[ ]:




