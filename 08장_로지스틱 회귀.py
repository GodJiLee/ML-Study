#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install watermark')


# In[2]:


get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-u -d -v -p numpy,pandas,sklearn,nltk')


# In[42]:


print(os.getcwd())


# In[55]:


import os
import tarfile

if not os.path.isdir('aclImdb_v1'):

    with tarfile.open('aclImdb_v1.tar.gz', 'r:gz') as tar:
        tar.extractall()


# In[4]:


get_ipython().system('pip install pyprind')


# In[84]:


import pyprind
import pandas as pd
import os

basepath = "aclImdb_v1\\aclImdb"

labels = {'pos': 1, 'neg': 0}
pbar = pyprind.ProgBar(50000)
df = pd.DataFrame()
for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        path = os.path.join(basepath, s, l)
        for file in sorted(os.listdir(path)):
            with open(os.path.join(path, file), 
                      'r', encoding='utf-8') as infile:
                txt = infile.read()
            df = df.append([[txt, labels[l]]], 
                           ignore_index=True)
            pbar.update()
df.columns = ['review', 'sentiment']


# In[86]:


import numpy as np

np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))


# In[87]:


df.to_csv('movie_data.csv', index=False, encoding='utf-8')


# In[88]:


df = pd.read_csv('movie_data.csv', encoding='utf-8')
df.head(3)


# In[71]:


df.shape


# In[72]:


import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer()
docs = np.array([
        'The sun is shining',
        'The weather is sweet',
        'The sun is shining, the weather is sweet, and one and one is two'])
bag = count.fit_transform(docs)

print(count.vocabulary_)


# In[11]:


print(bag.toarray())


# In[19]:


np.set_printoptions(precision=2)


# In[20]:


from sklearn.feature_extraction.text import TfidfTransformer

tfidf = TfidfTransformer(use_idf=True, 
                         norm='l2', 
                         smooth_idf=True)
print(tfidf.fit_transform(count.fit_transform(docs))
      .toarray())


# In[21]:


tf_is = 3
n_docs = 3
idf_is = np.log((n_docs+1) / (3+1))
tfidf_is = tf_is * (idf_is + 1)
print('tf-idf of term "is" = %.2f' % tfidf_is)


# In[22]:


tfidf = TfidfTransformer(use_idf=True, norm=None, smooth_idf=True)
raw_tfidf = tfidf.fit_transform(count.fit_transform(docs)).toarray()[-1]
raw_tfidf


# In[23]:


l2_tfidf = raw_tfidf / np.sqrt(np.sum(raw_tfidf**2))
l2_tfidf


# In[90]:


df.loc[0, 'review'][-50:] ## 끝에서부터 50글자 출력


# In[91]:


import re   ## 정규식 라이브러리
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',  
                           text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text


# In[92]:


preprocessor(df.loc[0, 'review'][-50:])


# In[27]:


preprocessor("</a>This :) is :( a test :-)!")


# In[76]:


df['review'] = df['review'].apply(preprocessor)


# In[77]:


df['review'].map(preprocessor)


# In[30]:


from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()

def tokenizer(text):
    return text.split()

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


# In[31]:


tokenizer('runners like running and thus they run')


# In[32]:


tokenizer_porter('runners like running and thus they run')


# In[33]:


import nltk

nltk.download('stopwords')


# In[103]:


from nltk.corpus import stopwords

stop = stopwords.words('english')
[w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:]
if w not in stop]


# In[102]:


X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values


# In[36]:


from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)

param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              {'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'vect__use_idf':[False],
               'vect__norm':[None],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              ]

lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', LogisticRegression(solver='liblinear', random_state=0))])

gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                           scoring='accuracy',
                           cv=5,
                           verbose=1,
                           n_jobs=1)


# In[106]:


param_grid


# In[79]:


gs_lr_tfidf.fit(X_train, y_train)


# In[80]:


print('최적의 매개변수 조합: %s ' % gs_lr_tfidf.best_params_)
print('CV 정확도: %.3f' % gs_lr_tfidf.best_score_)


# In[81]:


clf = gs_lr_tfidf.best_estimator_
print('테스트 정확도: %.3f' % clf.score(X_test, y_test))


# In[ ]:




