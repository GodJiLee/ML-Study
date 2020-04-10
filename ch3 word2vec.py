#!/usr/bin/env python
# coding: utf-8

# # chapter3. word2vec

# ##### 이 장의 목표: 추론기반기법에 사용되는 신경망 word2vec의 구조를 이해하고 구현하기 
# > 이 장에서는 효율을 포기하고 단순한 word2vec을 구현하게 될 것

# ## 3.1 추론 기반 기법과 신경망

# ##### 분포 가설을 이용하여 단어를 벡터로 분류하는 1) 통계기반기법 2) 추론기반기법
# > 1)의 한계 및 2)의 이점을 소개

# ### 3.1.1 통계 기반 기법의 문제점

# ##### 알고리즘: 동시발생 행렬 생성 > SVD 적용 > 밀집벡터 얻기
# > 이 방법은 대규모 말뭉치를 다룰 때 상당한 컴퓨팅 자원을 들여야 한다는 한계가 있다. 
# 
# ##### 통계기반기법 : 1회처리 (배치학습), 추론기반기법 : 일부, 순차적 학습 (미니배치학습)
# > 더불어 추론기반기법은 신경망의 특성에 따라 GPU 이용, 병렬 처리가 가능하므로 처리속도가 빠르다.

# ### 3.1.2 추론 기반 기법 개요

# ##### [그림 3-3]
# > 분포가설 ('단어의 의미는 주변 단어에 의해 형성된다')에 기반한 맥락 정보를 입력, 동시발생 가능성 출력
# >> 학습은 말뭉치 데이터를 사용하여 단어의 분산 표현을 올바르게 추측하도록 함

# ### 3.1.3 신경망에서의 단어 처리

# ##### 신경망에서의 단어 처리를 위해 단어를 원-핫 벡터로 변환
# > 총 어휘 수만큼의 벡터를 마련한 후 단어의 해당 인덱스가 자리하는 지점만 1로, 나머지는 0으로 표시
# >> [그림 3-5] 와 같이 입력층의 뉴런을 고정벡터로 마련하면 신경망 처리가 가능

# ##### [그림 3-6]
# > 위의 단어벡터 입력데이터를 완전연결계층(은닉층 노드 수: 3)에 대입한 모습
# >> 완전연결계층이므로 입력층의 모든 노드가 은닉층의 모든 노드에 화살표로 연결됨
# >>> 이때, 화살표의 의미는 가중치를 의미 ([그림 3-7]과 같은 모습)
# 
# ##### 이 경우 편향은 따로 더해주지 않기 때문에 MatMul계층(내적)과 같은 구현
# > MatMul 모듈은 p51에서 자세히 설명

# In[2]:


#행렬곱을 이용한 완전연결계층 구현
import numpy as np

c = np.array([[1, 0, 0, 0, 0, 0, 0]]) #입력
W = np.random.randn(7,3) #가중치
h = np.matmul(c, W) #중간 노드 #c와 W 내적
print(h)


# ##### 원핫 벡터의 특징상 [그림 3-8]과 같이 가중치 행렬은 첫 행만 사용됨
# > 그럼에도 행렬곱을 수행할 때의 비효율성에 대한 개선법은 다음 장에서 설명

# In[4]:


#MatMul 계층을 이용한 구현
import sys
sys.path.append('C:\\Users\\leejiwon\\Documents\\GitHub\\deep-learning-from-scratch-2')
import numpy as np
from common.layers import MatMul #MatMul 계층 불러오기

c = np.array([[1, 0, 0, 0, 0, 0, 0]]) #미니배치를 고려하여 2차원으로 설계 #사용하는 데이터는 0번째 차원
W = np.random.randn(7, 3)
layer = MatMul(W)
h = layer.forward(c) #순전파 메서드 호출
print(h)


# ## 3.2. 단순한 word2vec

# ##### word2vec 구현
# > word2vec은 프로그램명, CBOW와 skip-gram 신경망이 여기서 사용되는데 다음 절에서는 CBOW에 대해 먼저 설명
# >> CBOW : Continuous Bag Of Words #연속적 말뭉치

# ### 3.2.1 CBOW 모델의 추론 처리

# ##### CBOW 모델은 맥락(주변단어)로부터 타깃(중앙단어)를 추론하는 신경망
# > [그림 3-9]와 같이 원핫표현된 맥락을 입력으로 받아 가중치가 Win인 완전연결계층을 거쳐 은닉층이 되고, 다시 가중치가 Wout인 계층을 거쳐 출력된다. 
# >> 입력층이 2개인 이유는 참고하고자하는 맥락이 2개이기 때문 (N개라면 N개의 입력층 생성)

# ##### 은닉층
# > 입력층이 [그림 3-9]와 같이 2개 이상일 경우, 각 입력층에서 계산된 값들의 평균이 된다.
# 
# ##### 출력층
# > 출력층의 뉴런 각각은 단어에 Match되며, 의미는 각 단어가 출현할 경우의 수(점수)이다.
# >> 소프트맥스 함수를 거쳐 확률값이 된다. 

# ##### 위 완전연결계층에서의 가중치는 단어의 분산표현을 의미한다. 
# > [그림 3-10]과 같이 각 행의 가중치가 각 단어에 Match된다. 
# >> 이 가중치들은 학습이 진행될 수록 타깃 단어를 잘 추론하는 쪽으로 갱신된다. 
# 
# ##### 이때 은닉층의 뉴런수가 입력층의 뉴런수에 비해 작게 만들어 정보를 '밀집'벡터로 표현한다.
# > 은닉층에 담긴 정보는 '인코딩'정보, 이를 '디코딩'하여 인간이 해석할 수 있는 정보로 복원한다. 

# ##### 같은 CBOW 모델을 계층 관점에서 해석
# > [그림 3-11]과 같이 가중치 매개변수를 MatMul 계층 안으로 넣어줌 (편향을 더해주지 않는 완전연결계층은 MatMul 계층의 역전파로 표현 가능)
# >> 1) 각 단어를 원-핫 벡터로 표현 2) MatMul 계층을 거친 입력 데이터를 더해 평균(0.5)내줌 3) 다시 MatMul (이번엔 가중치 Wout으로 변경)계층을 거쳐 단어발생점수 출력

# In[7]:


# CBOW모델의 파이썬 구현 # [그림 3-11] 그대로 구현
import sys
sys.path.append('C:\\Users\\leejiwon\\Documents\\GitHub\\deep-learning-from-scratch-2')
import numpy as np
from common.layers import MatMul

#샘플 맥락 데이터
c0 = np.array([[1, 0, 0, 0, 0, 0, 0]])
c1 = np.array([[0, 0, 1, 0, 0, 0, 0]])

#가중치 초기화
W_in = np.random.randn(7, 3)
W_out = np.random.randn(3, 7) #행렬곱을 고려한 형상

#계층 생성
in_layer0 = MatMul(W_in)
in_layer1 = MatMul(W_in)
out_layer = MatMul(W_out)

#순전파
h0 = in_layer0.forward(c0) #은닉층 1
h1 = in_layer1.forward(c1) #은닉층 2
h = 0.5 * (h0 + h1) #평균
s = out_layer.forward(h) #출력층

print(s)


# ##### CBOW 모델은 활성화함수를 하나도 사용하지 않는 비교적 간단한 모델 
# > 핵심: 입력층이 2개 이상 존재할 수 있으며, 각 입력층들은 가중치를 공유한다. 

# ### 3.2.2 CBOW 모델의 학습

# ##### CBOW 모델에서 출력한 점수에 소프트맥스 함수를 적용시키면 확률값이 도출됨
# > 이때 확률값은 맥락 속에서 중심단어를 추정하는 확률
# >> [그림 3-12]에서 'you'와 'goodbye'사이 'say'의 출현 확률이 가장 높게 나타나는 것을 알 수 있음
# ##### CBOW 모델의 학습은 가중치를 조정하는 일
# > Win과 Wout을 올바른 예측이 가능하도록 조정
# >> 이로써 얻어지는 단어의 분산표현은 학습 데이터의 성격에 따라 달라질 수 있음 (스포츠면 vs 예능면 데이터)

# * 신경망으로 생각했을 때 과정 [그림3-13]
# > 소프트맥스와 교차엔트로피오차 계층을 사용
# >> 확률값과 정답레이블과의 오차값을 구함
# >>> 이를 기준으로 손실함수를 도출하여 학습

# ##### [그림 3-14]에서는 소프트맥스와 교차엔트로피오차 계층을 하나의 계층 'softmax with loss'로 표현

# ### 3.2.3 word2vec의 가중치와 분산 표현

# ##### word2vec에서 사용되는 신경망에는 Win과 Wout 두 가중치가 존재함
# > Win은 단어의 분산표현을 행 방향으로, Wout은 열 방향으로 저장함 [그림 3-15]
# >> 최종적으로 분산표현에 적용할 가중치를 p130에 A, B, C로 결정 가능
# >>> word2vec 모델은 A안을 선택하며 앞으로 책에서도 이를 적용할 것임

# ## 3.3 학습 데이터 준비

# ##### 말뭉치로는 동일하게 "You say goodbye and I say hello."를 사용

# ### 3.3.1 맥락과 타깃

# ##### 신경망의 입력인 '맥락'과 정답레이블인 '타깃'을 말뭉치로부터 도출해내는 과정
# > [그림 3-16]과 같이 샘플 데이터에서 맥락과 타깃을 모든 단어에 대해 뽑아냄 
# >> 이때, 맥락의 수는 여러 개일 수 있지만, 타깃은 오직 한 개여야 함 (contexts로 표기하는 이유)
# 
# ##### 이 알고리즘을 함수로 구현
# > 2장에서 공부한 단어-> ID, ID -> 단어 작업을 거쳐야 함
# >> preprocess() 함수를 사용하겠음

# In[1]:


import sys
sys.path.append('C:\\Users\\leejiwon\\Documents\\GitHub\\deep-learning-from-scratch-2')
from common.util import preprocess

text = 'You say goodbye and I say hello'
corpus, word_to_id, id_to_word = preprocess(text)
print(corpus)


# In[2]:


print(id_to_word)


# ##### 도출된 ID 배열(corpus)로부터 [그림 3-17]과 같이 맥락, 타깃을 반환하는 함수
# > 맥락은 2차원 배열, 0번째 차원에는 0,2, 1번째 차원에는 1,3의 맥락 정보가 저장됨 (타깃도 마찬가지)
# >> 이와 같이 맥락, 타깃을 만드는 함수 구현 (create_contexts_target(corpus, window_size)

# In[30]:


# github 상에 보충 코드 참고
import os
from common.np import *

def preprocess(text):
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')

    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = np.array([word_to_id[w] for w in words])
    return corpus, word_to_id, id_to_word

def cos_similarity(x, y, eps=1e-8):
    '''코사인 유사도 산출
    :param x: 벡터
    :param y: 벡터
    :param eps: '0으로 나누기'를 방지하기 위한 작은 값
    :return:
    '''
    nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
    ny = y / (np.sqrt(np.sum(y ** 2)) + eps)
    return np.dot(nx, ny)

def most_similar(query, word_to_id, id_to_word, word_matrix, top = 5):
    '''유사 단어 검색
    :param query: 쿼리(텍스트)
    :param word_to_id: 단어에서 단어 ID로 변환하는 딕셔너리
    :param id_to_word: 단어 ID에서 단어로 변환하는 딕셔너리
    :param word_matrix: 단어 벡터를 정리한 행렬. 각 행에 해당 단어 벡터가 저장되어 있다고 가정한다.
    :param top: 상위 몇 개까지 출력할 지 지정
    '''
    if query not in word_to_id:
        print('%s(을)를 찾을 수 없습니다.' % query)
        return

    print('\n[query] ' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    # 코사인 유사도 계산
    vocab_size = len(id_to_word)

    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    # 코사인 유사도를 기준으로 내림차순으로 출력
    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(' %s: %s' % (id_to_word[i], similarity[i]))
        
        count += 1
        if count >= top:
            return

def convert_one_hot(corpus, vocab_size):
    '''원핫 표현으로 변환
    :param corpus: 단어 ID 목록(1차원 또는 2차원 넘파이 배열)
    :param vocab_size: 어휘 수
    :return: 원핫 표현(2차원 또는 3차원 넘파이 배열)
    '''
    N = corpus.shape[0]
    if corpus.ndim == 1:
        one_hot = np.zeros((N, vocab_size), dtype=np.int32)
        for idx, word_id in enumerate(corpus):
            one_hot[idx, word_id] = 1

    elif corpus.ndim == 2:
        C = corpus.shape[1]
        one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
        for idx_0, word_ids in enumerate(corpus):
            for idx_1, word_id in enumerate(word_ids):
                one_hot[idx_0, idx_1, word_id] = 1

    return one_hot

def create_co_matrix(corpus, vocab_size, window_size=1):
    '''동시발생 행렬 생성
    :param corpus: 말뭉치(단어 ID 목록)
    :param vocab_size: 어휘 수
    :param window_size: 윈도우 크기(윈도우 크기가 1이면 타깃 단어 좌우 한 단어씩이 맥락에 포함)
    :return: 동시발생 행렬
    '''
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1
    return co_matrix

def ppmi(C, verbose=False, eps = 1e-8):
    '''PPMI(점별 상호정보량) 생성
    :param C: 동시발생 행렬
    :param verbose: 진행 상황을 출력할지 여부
    :return:
    '''
    M = np.zeros_like(C, dtype = np.float32)
    N = np.sum(C)
    S = np.sum(C, axis = 0)
    total = C.shape[0] * C.shape[1]
    cnt = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[j] * S[i]) + eps)
            M[i, j] = max(0, pmi)

            if verbose:
                cnt += 1
                if cnt % (total // 100) == 0:
                    print('%.1f%% 완료' % (100 * cnt / total))
    return M


# In[31]:


def create_contexts_target(corpus, window_size=1):
    '''맥락과 타깃 생성
    :param corpus: 말뭉치(단어 ID 목록)
    :param window_size: 윈도우 크기(윈도우 크기가 1이면 타깃 단어 좌우 한 단어씩이 맥락에 포함)
    :return:
    '''
    target = corpus[window_size:-window_size]
    contexts = []

    for idx in range(window_size, len(corpus)-window_size):
        cs = []
        for t in range(-window_size, window_size + 1):
            if t == 0:
                continue
            cs.append(corpus[idx + t])
        contexts.append(cs)

    return np.array(contexts), np.array(target)


# In[32]:


contexts, target = create_contexts_target(corpus, window_size = 1)
print(contexts)


# In[33]:


print(target)


# ##### 이 함수는 corpus (ID로 변환한 배열)과 window_size(좌,우로 몇 개의 단어를 포함할 건지)를 인수로 가짐
# > 결과값(맥락, 타깃)은 넘파이 배열로 return

# ##### 이렇게 만들어진 맥락, 타깃 정보를 CBOW 모델에게 넘겨주게 됨
# > but 아직까지는 각 배열이 단어 ID 형태
# >> 원핫 표현으로 변환

# ### 3.3.2 원핫 표현으로 변환

# ##### [그림 3-18]과 같이, 형상을 바꿔줌
# > 원핫 표현으로의 변환을 수행하는 함수는 convert_one_hot()
# >> 인수로 단어 ID 목록과 어휘 수를 받음 

# In[1]:


# 데이터 준비과정 정리
import sys 
sys.path('C:\\Users\\leejiwon\\Documents\\GitHub\\deep-learning-from-scratch-2')
from common.util import preprocess, create_contexts_target, convert_one_hot

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)

contexts, target = create_contexts_target(corpus, window_size = 1)

vocab_size = len(word_to_id)
target = convert_one_hot(target, vocab_size)
contexts = convert_one_hot(contexts, vocab_size)


# ## 3.4 CBOW 모델 구현

# ##### [그림 3-19]는 앞에서 본 신경망
# > Softmax with Loss 계층을 사용함
# >> 이 장에서는 이 신경망을 간추린 SimpleCBOW 신경망을 구현 (다음 장에서 CBOW 클래스 구현)

# In[37]:


# 초기화 메서드
import sys
sys.path.append('C:\\Users\\leejiwon\\Documents\\GitHub\\deep-learning-from-scratch-2')
import numpy as np
from common.layers import MatMul, SoftmaxWithLoss

class SimpleCBOW:
    def __init__(self, vocab_size, hidden_size): #어휘 수와 은닉층의 뉴런 수를 인수로 받음
        V, H = vocab_size, hidden_size #초기화
        
        # 가중치 초기화
        W_in = 0.01 * np.random.randn(V, H).astype('f') #무작위 작은 수로 가중치 초기화 (부동소수점 32비트)
        W_out = 0.01 * np.random.randn(H, V).astype('f')
        
        # 계층 생성
        self.in_layer0 = MatMul(W_in)
        self.in_layer1 = MatMul(W_in) # 단어의 수(윈도우 크기)만큼 만들어 줘야 함 #같은 값을 공유
        self.out_layer = MatMul(W_out)
        self.loss_layer = SoftmaxWithLoss()
        
        # 모든 가중치와 기울기를 리스트에 모은다.
        layers = [self.in_layer0, self.in_layer1, self.out_layer]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads
            
        # 인스턴스 변수에 단어의 분산 표현을 저장한다. 
        self.word_vecs = W_in


# ##### 위 코드에서는 같은 가중치를 여러 개 공유하고 있음
# > 이렇게 되면 Adam, Momentum 등의 최적화 기법을 사용할 때 문제가 될 수 있음
# >> 따라서 이 중복을 없애는 간단한 구현이 github 본 코드에는 적용되어 있음

# ##### 신경망의 순전파 구현
# > 인수로 맥락(contexts)와 타깃(target)을 받고 손실(loss)를 반환

# In[38]:


def forward(self, contexts, target):
    h0 = self.in_layer0.forward(contexts[:, 0])
    h1 = self.in_layer1.forward(contexts[:, 1])
    h = (h0 + h1) * 0.5
    score = self.in_layer1.forward(contexts[:, 1])
    loss = self.loss_layer.forward(score, target)
    return loss


# ##### 역전파 구현
# > [그림 3-20]과 같이 기울기를 순전파 반대 방향으로 전파해줌

# In[40]:


def backward(self, dout = 1):
    ds = self.loss_layer.backward(dout) # softmaxwithloss 계층 미분값 > MatMul계층으로 전달
    da = self.out_layer.backward(ds) # Matmul 계층 미분값
    da *= 0.5 # 곲 역전파
    self.in_layer1.backward(da)
    self.in_layer0.backward(da) #합 역전파
    return None


# ##### 일전에 기울기 값을 grads 리스트에 모두 저장해두었으므로, 순전파/역전파 메서드를 실행해 주는 것 만으로도 기울기 각각을 갱신시킬 수 있음

# ### 3.4.1 학습 코드 구현

# ##### CBOW 모델의 학습 과정
# > 학습 데이터를 신경망에 입력, 기울기를 구하고 이를 토대로 가중치를 갱신
# >> 학습 과정에 Trainer 클래스를 사용할 것 
# >>> 1장에서 언급되었던 '미니배치 선택' > '신경망에 입력 후 기울기 산출' > '최적화 함수에 전달 후 매개변수 갱신' (깔끔한 코드 유지 가능)

# In[ ]:


import sys
sys.path.append('C:\\Users\\leejiwon\\Documents\\GitHub\\deep-learning-from-scratch-2')
from common.trainer import Trainer
from common.optimizer import Adam
from simple_cbow import SimpleCBOW 
from common.util import preprocess, create_contexts_target, convert_one_hot

window_size = 1
hidden_size = 5
batch_size = 3
max_epoch = 1000

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)

vacab_size = len(word_to_id)
contexts, target = create_contexts_target(corpus, window_size)
target = convert_one_hot(target, vocab_size)
contexts = convert_one_hot(contexts, vocab_size)

model = SimpleCBOW(vocab_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot()


# ## 3.5 word2vec 보충

# ##### word2vec의 CBOW 모델을 확률 관점에서 살펴보는 절

# ### 3.5.1 CBOW 모델과 확률

# ##### 확률의 표기법
# > 사건 A가 일어날 확률 = P(A), 사건 A와 B가 동시에 일어날 '동시확률' = P(A, B)
# ##### > 사후확률(조건부확률) : 사건 B가 일어났을 때 사건 A가 일어날 확률 = P(A|B)
# #####
# ##### CBOW모델에 대하여 확률 표기법으로 기술해보면,
# > [그림 3-22]와 같이 말뭉치를 각각 Wt로 표기하고 t번째 단어에 대해 윈도우크기가 1인 맥락을 생각한다면,
# >> 타깃이 wt가 될 확률은 [식 3.1] (wt-1과 wt+1이 일어났을 때-맥락, wt가 일어날-타깃 확률)

# #####
# ##### 이 식을 이용하여 CBOW모델의 손실함수를 표현하면 
# > 1장, p41의 [식 1.7] 교차엔트로피오차 식에 따라, tk 값에는 정답레이블이 원핫벡터로 표현되었으므로 1이, yk에는 정답일 때의 확률값 P(Wt|Wt-1, Wt+1)만 적용됨
# >> 결과값은 [식 3.2]와 같이 확률에 log를 취해주고 음의 값을 부여한 꼴 (음의 로그가능도라 표현)
# #####
# ##### 이를 말뭉치 전체로 확장하면 [식 3.3]과 같이 평균작업만 취해주면 됨
# > CBOW 모델은 이 손실함수를 최소화 하는 모델

# ### 3.5.2 skip-gram 모델

# In[ ]:





# ### 3.5.3 통계 기반 vs 추론 기반

# In[ ]:




