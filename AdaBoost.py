#!/usr/bin/env python
# coding: utf-8

# In[102]:


import numpy as np
import pandas as pd


# In[103]:


data = {"x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "y": [1, 1, 1, -1, 1, -1, -1, 1, -1, -1], "h1(xi)": [-1, -1, -1, -1, 1, -1, -1, 1, -1, -1], 
        "h2(xi)": [1, 1, 1, -1, 1, 1, 1, 1, 1, -1], "h3(xi)": [1, 1, 1, 1, -1, -1, -1, -1, -1, -1]}
dataset = pd.DataFrame(data = data)


# In[104]:


dataset

 
# In[105]:


evaluation1 = np.where(dataset["h1(xi)"] == dataset["y"], 1, 0)
weight1 = 1 / len(dataset)

err1 = sum(evaluation1 * weight1)

alpha1 = 0.5 * np.log((1 - err1) / err1)

weight_update1 = np.exp(alpha1 * dataset["h1(xi)"] * dataset["y"])

Z1 = sum(weight1 * weight_update1)


# In[106]:


evaluation2 = np.where(dataset["h2(xi)"] == dataset["y"], 1, 0)
weight2 = (weight1 * weight_update1) / Z1

err2 = sum(evaluation2 * weight2)

alpha2 = 0.5 * np.log((1 - err2) / err2)

weight_update2 = np.exp(alpha2 * dataset["h2(xi)"] * dataset["y"])

Z2 = sum(weight2 * weight_update2)


# In[107]:


evaluation3 = np.where(dataset["h3(xi)"] == dataset["y"], 1, 0)
weight3 = (weight2 * weight_update2) / Z2

err3 = sum(evaluation3 * weight3)

alpha3 = 0.5 * np.log((1 - err3) / err3)

weight_update3 = np.exp(alpha3 * dataset["h3(xi)"] * dataset["y"])

Z3 = sum(weight3 * weight_update3)


# In[108]:


print(Z1, Z2, Z3)

