#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sp
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import scipy.stats as st
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from yellowbrick.cluster import KElbowVisualizer
from termcolor import colored

from warnings import filterwarnings
filterwarnings("ignore")

from sklearn import set_config
set_config(print_changed_only = False)

print(colored("Required libraries were succesfully imported...",
              color = "green", attrs = ["bold", "dark"]))


# In[4]:


df = pd.read_csv("Mall_Customers.csv")


# In[5]:


df.head(n = 7)


# In[6]:


df.index = df.iloc[:, 0]
df.head(n = 7)


# In[7]:


dff = df.drop(["CustomerID"], axis = 1)
dff.index.name = None

print(colored("'CustomerID' variable was succesfully dropped...",
              color = "green", attrs = ["bold", "dark"]))

dff.head(n = 7)


# In[8]:


dff.rename(columns = {"Annual Income (k$)": "annual_income",
                      "Spending Score (1-100)": "spending_score",
                      "Age": "age", "Gender": "gender"}, inplace = True)


# In[9]:


# Checking Null Values
print("There is {} null values in this dataset".format(dff.isnull().sum().sum()))


# In[10]:


# Checking Duplicate Values
print("There is {} duplicated values in this dataset".format(dff.duplicated().sum()))


# In[11]:


dff.describe()


# In[12]:


dff.info()


# # Data Visualization and Exploratory Data Analysis

# In[13]:


sns.set(rc={"axes.facecolor":"#727877", "figure.facecolor":"#ffffff"})

plt.figure(figsize = [10, 7], clear = False)


# In[14]:


sns.barplot(x = dff["gender"].value_counts().index, 
            y = dff["gender"].value_counts(), palette = ["#D63913", "#E9F709"],
            saturation = 1).set(title = "The number of classes of 'gender' variable");


# In[16]:


fig, axes = plt.subplots(1, 3, figsize = (30, 8))

sns.barplot(ax = axes[0], x = "gender", y = "age", data = dff, saturation = 1, palette = ["#D63913", "#E9F709"])
axes[0].set_title("Gender and age")

sns.barplot(ax = axes[1], x = "gender", y = "annual_income", data = dff, saturation = 1, palette = ["#D63913", "#E9F709"])
axes[1].set_title("Annual income by gender")

sns.barplot(ax = axes[2], x = "gender", y = "spending_score", data = dff, saturation = 1, palette = ["#D63913", "#E9F709"])
axes[2].set_title("Spending score by gender");


# In[17]:


# The graphs above visualize the relationship between 'gender' variable and 'age','annual_income' and 'spending_score' variables.

# There are no significant differences between the genders (male and female) in the mentioned parameters


# In[18]:


sns.set(rc={"axes.facecolor":"#727877", "figure.facecolor":"#ffffff"})

sns.displot(data = dff, x = "age", hue = "gender", kind = "kde", height = 6.5,  aspect = 1.8, clip = (0, None),
    palette = ["#D63913", "#000000"]).set(title = "density of the classes of 'gender' variable by age");


# In[19]:


sns.displot(data = dff, x = "spending_score", hue = "gender", kind = "kde", height = 6.5, aspect = 1.8,
            clip = (0, None), palette = ["#D63913", "#000000"]).set(title = "density of the classes of 'gender' variable by spending score");


# In[20]:


# Histogram


# In[21]:


dff.hist(figsize = (15, 10), bins = 20, backend = None, legend = True);


# In[22]:


# Lmplot


# In[23]:


sns.set(rc={"axes.facecolor":"#727877", "figure.facecolor":"#ffffff"})

sns.lmplot(x = "spending_score", y = "annual_income", hue = "gender", data = dff, height = 6.5, aspect = 1.6,
          palette = ["#D63913", "#E9F709"]);


# In[24]:


# The graph above shows the relationship between the annual income 
# and annual expenditure of women and men separately.


# In[25]:


# Visualization with plotly express...


# In[26]:


px.scatter(dff, x = "age", y = "spending_score", color = "annual_income", template = "plotly_dark", opacity = 1)


# In[27]:


fig = px.histogram(dff, x = "age", y = "spending_score", marginal = "violin", color = "gender", template = "plotly_dark",
                   text_auto = True, width = None, height = None, nbins = 100, hover_data  = dff.columns)
fig.show()


# In[28]:


fig = px.histogram(dff, x = "age", y = "annual_income", marginal = "box", color = "gender", template = "plotly_dark",
                   text_auto = True, width = None, height = None, nbins = 100, hover_data  = dff.columns)
fig.show()


# In[29]:


fig = px.histogram(dff, x = "annual_income", y = "spending_score", marginal = "rug", color = "gender", template = "plotly_dark",
                   text_auto = True, width = None, height = None, nbins = 100, hover_data  = dff.columns)
fig.show()


# In[30]:


fig = px.density_heatmap(dff, x = "age", y = "annual_income", z = "spending_score", color_continuous_scale = "electric", text_auto = True,
                         title = "Density heatmap of relationship between 'annual_income' and 'spending_score' variables by 'age' variable")
fig.show()


# In[31]:


# Pearson correlation coefficient between variables...


# In[32]:


list = (dff[["age","annual_income"]].corr(),
        dff[["age", "spending_score"]].corr(),
        dff[["annual_income","spending_score"]].corr())
for corr in list:
    print(corr, "\n\n")


# In[33]:


# Here we see that there is not correlation between variables.


# In[34]:


print("Pearson correlation coefficient between 'age' and 'annual_income' variables: \n",
      st.pearsonr(dff["age"], dff["annual_income"])[0])


# In[35]:


print("Pearson correlation coefficient between 'age' and 'spending_score' variables: \n",
      st.pearsonr(dff["age"], dff["spending_score"])[0])


# In[36]:


print("Pearson correlation coefficient between 'annual_income' and 'spending_score' variables: \n",
      st.pearsonr(dff["annual_income"], dff["spending_score"])[0])


# In[37]:


print(colored("For 'annual_income' variable:", color = "blue", attrs = ["bold", "dark"]))
test_statistic, pvalue = st.shapiro(dff["annual_income"])
print("Test statistic = %.11f, p-value = %.11f" % (test_statistic, pvalue))

print("_______________________________________________________")

print(colored("\nFor 'spending_score' variable:", color = "blue", attrs = ["bold", "dark"]))
test_statistic, pvalue = st.shapiro(dff["spending_score"])
print("Test statistic = %.11f, p-value = %.11f" % (test_statistic, pvalue))


# In[38]:


st.spearmanr(dff["annual_income"], dff["spending_score"])


# In[39]:


# Encode 'gender' variable...

# 1 = Male
# 0 = Female


# In[40]:


lbe = LabelEncoder()
lbe.fit_transform(dff["gender"])
dff["gender"] = lbe.fit_transform(dff["gender"])


# In[41]:


kmeans = KMeans(n_clusters = 2, random_state = 11)
k_fit = kmeans.fit(dff)


# In[42]:


plt.figure(figsize = [10, 7], clear = False)

clusters = k_fit.labels_
centers = k_fit.cluster_centers_

plt.scatter(dff.iloc[:, 1],
            dff.iloc[:, 2],
            c = clusters,
            s = 30,
            cmap = "viridis")

plt.scatter(centers[:, 1],
           centers[:, 2],
           c = "black",
           s = 200,
           alpha = 0.8);


# In[43]:


kmeans = KMeans(n_clusters = 3, random_state = 11)
k_fit = kmeans.fit(dff)
clusters = k_fit.labels_
centers = k_fit.cluster_centers_


# In[46]:


plt.rcParams["figure.figsize"] = (20, 10)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(dff.iloc[:, 1], dff.iloc[:, 2], dff.iloc[:, 3]);

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(dff.iloc[:, 1], dff.iloc[:, 2], dff.iloc[:, 3], c = clusters)
ax.scatter(centers[:, 1], centers[:, 2], centers[:, 3],
           marker = '*',
           c = "#8F931F",
           s = 1000);


# In[47]:


kmeans = KMeans(random_state = 11)
visualizer = KElbowVisualizer(kmeans, k = (2, 20), locate_elbow = True)
visualizer.fit(dff)
visualizer.poof();


# In[48]:


kmeans = KMeans(n_clusters = 6, random_state = 11)
k_fit = kmeans.fit(dff)
clusters = k_fit.labels_
data = pd.DataFrame({"Customer ID": dff.index, "Clusters": (k_fit.labels_ + 1)})

data.head(n = 7)


# In[ ]:




