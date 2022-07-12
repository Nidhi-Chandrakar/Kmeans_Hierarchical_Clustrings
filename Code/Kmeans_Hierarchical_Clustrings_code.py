#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing libraries 
import numpy as np
import pandas as pd
import sklearn
import pandas as pd

# For K means
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

#For Agglomerative Clustering
from sklearn.cluster import AgglomerativeClustering 

import warnings
warnings.filterwarnings('ignore')

#plot and maps
import seaborn as sns
import chart_studio.plotly as py #For World Map
import plotly.graph_objs as go
import matplotlib.lines as lines
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')



#From Normalising data
from sklearn import preprocessing
import pandas as pd


# In[2]:


#Reading Happinesss-data file
df = pd.read_csv("/Users/himan/OneDrive/Desktop/Data Mining/Course_work/Happiness-Data.csv")


# In[3]:


#Identifying any missing values – n.a, NA, na, n/a,? etc
missing_value_formats = ["n.a.","?","NA","n/a", "na", "--"]
df = pd.read_csv("/Users/himan/OneDrive/Desktop/Data Mining/Course_work/Happiness-Data.csv", na_values = missing_value_formats)
print(df.isnull().sum())


# In[4]:


#Finding and dropping dupliactes
df.duplicated().sum()


# In[5]:


df.columns = df.columns.str.replace(' ','_')
df.columns = df.columns.str.replace(':','')
df.columns = df.columns.str.replace('+','')
df.head(5)


# In[6]:


import seaborn as sns

plt.rcParams['figure.figsize'] = (20, 15)
sns.heatmap(df.corr(), cmap = 'copper', annot = True)

plt.show()


# In[7]:


df.dtypes


# In[8]:


#Function to identify highly correlated columns
def corr(dataset,threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j]) > threshold:
                col_name = corr_matrix.columns[i]
                col_corr.add(col_name)
    return col_corr


features = df.select_dtypes(include=[np.number])
x= features


# In[9]:


#Find features having correlation greater than 90%
Highly_correlated_columns = corr(features, 0.90)
Highly_correlated_columns


# In[10]:


#dropping highly correlated columns
df = df.drop(['upperwhisker','lowerwhisker','Explained_by_Log_GDP_per_capita','Explained_by_Social_support','Explained_by_Healthy_life_expectancy','Explained_by_Freedom_to_make_life_choices','Explained_by_Generosity','Explained_by_Perceptions_of_corruption','Ladder_score_in_Dystopia'], axis=1)
df.head(5)


# In[11]:


import seaborn as sns

plt.rcParams['figure.figsize'] = (10, 7)
sns.heatmap(df.corr(), cmap = 'copper', annot = True)

plt.show()


# In[12]:


#Plotting world map as per the ladder score - happiness score
fig = go.Figure(data=go.Choropleth(
    locations=df['Country_name'],
    locationmode='country names',
    z = df['Ladder_score'],
    text = df['Country_name'],
    colorbar = {'title':'Happiness score'},
    colorscale='Greens'
))

fig.show()


# In[13]:


#dropping categorical columns
df1 = df.drop(labels=['Country_name', 'Regional_indicator'], axis=1)


# In[14]:


def box_plot(df):
    plt.figure(figsize= (20,15))
    plt.subplot(3,3,1)
    sns.boxplot(x= df.Ladder_score, color='lightgreen')

    plt.subplot(3,3,2)
    sns.boxplot(x= df.Standard_error_of_ladder_score, color='lightgreen')


    plt.subplot(3,3,3)
    sns.boxplot(x= df.Logged_GDP_per_capita, color='lightgreen')

    plt.subplot(3,3,4)
    sns.boxplot(x= df.Social_support, color='lightgreen')
    
    plt.subplot(3,3,5)
    sns.boxplot(x= df.Healthy_life_expectancy, color='lightgreen')
    
    plt.subplot(3,3,6)
    sns.boxplot(x= df.Freedom_to_make_life_choices, color='lightgreen')
    
    plt.show()
    
    
box_plot(df1)


# In[15]:


def box_plot(df):
    
    plt.figure(figsize= (20,15))
    
    
    plt.subplot(3,3,1)
    sns.boxplot(x= df.Generosity, color='lightgreen')
    
    plt.subplot(3,3,2)
    sns.boxplot(x= df.Perceptions_of_corruption, color='lightgreen')
    
    plt.subplot(3,3,4)
    sns.boxplot(x= df.Dystopia__residual, color='lightgreen')
    
    
    plt.show()
    
box_plot(df1)


# In[16]:


df1 = df1[df1['Ladder_score']>3]
#df1 = df1[df1['Standard_error_of_ladder_score']<0.12]
df1 = df1[df1['Freedom_to_make_life_choices']>0.5]
df1 = df1[df1['Generosity']<0.4]
df1 = df1[df1['Dystopia__residual']>1.2]


# In[17]:


#plotting the data attributes
df1.plot.bar()


# In[18]:


#Normalize the data attributes 
scaler = preprocessing.MinMaxScaler()
names = df1.columns
d = scaler.fit_transform(df1)
scaled_df = pd.DataFrame(d, columns=names)
scaled_df_AH = scaled_df
scaled_df.head()


# In[19]:


#Plotting the normalized data attributes 
scaled_df.plot.bar()


# In[20]:


# Create a PCA instance: pca
pca_4PCs = PCA(n_components=4)
principalComponents = pca_4PCs.fit_transform(scaled_df)


# In[21]:


pca_4PCs.explained_variance_ratio_


# In[22]:


pca_4PCs.explained_variance_ratio_.sum()


# In[23]:


# Create a PCA instance: pca
pca = PCA(n_components=5)
principalComponents = pca.fit_transform(scaled_df)

# Plot the explained variances
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_, color='lightgreen')
plt.xlabel('PCA features')
plt.ylabel('variance %')
plt.xticks(features)


# In[24]:


pca.explained_variance_ratio_


# In[25]:


pca.explained_variance_ratio_.sum()


# In[26]:


print('Eigen Values ', pca.explained_variance_ratio_)


# In[27]:


print('Eigen vectors ', pca.components_)


# In[28]:


# Save components to a DataFrame
PCA_components = pd.DataFrame(principalComponents)
PCA_components.head(5)


# In[29]:


Sum_of_squared_distances = []
K = range(1,15)
for clusters in K:
    km = KMeans(n_clusters=clusters)
    km = km.fit(principalComponents)
    Sum_of_squared_distances.append(km.inertia_)


# In[30]:


plt.plot(K, Sum_of_squared_distances,'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()


# In[31]:


from sklearn.metrics import silhouette_score


n_clusters = [2,3,4,5,6]


for K in n_clusters:
    cluster = KMeans (n_clusters= K, random_state= 10)
    predict = cluster.fit_predict(scaled_df)
    score = silhouette_score(scaled_df, predict, random_state= 10)
    print ("For n_clusters = {}, silhouette score is {})".format(K, score))


# In[32]:


# Clustering using KMeans
import numpy as np
model = KMeans(n_clusters = 2, init = "k-means++")
label = model.fit_predict(PCA_components.iloc[:,:4])
centers = np.array(model.cluster_centers_)
uniq = np.unique(label)

# colors for plotting

colors = ['blue', 'green']
# assign a color to each features (note that we are using features as target)
features_colors = [ colors[label[i]] for i in range(len(PCA_components.iloc[:,:4])) ]
T=PCA_components.iloc[:,:2]

# plot the PCA cluster components
plt.scatter(T[0], T[1],
            c=features_colors, marker='o',
            alpha=0.4
        )


# In[33]:


clusterer = KMeans(n_clusters=2, random_state=10)
cluster_labels = clusterer.fit_predict(PCA_components)
print(cluster_labels)


# In[34]:


scaled_df['k_means_cluster'] = label
scaled_df['k_means_cluster'].value_counts()
#scaled_df.head(5)


# In[35]:


scaled_df.head(5)
scaled_df.insert(0,'Country_name',df.iloc[:,0])


# In[36]:


scaled_df.sort_values('Ladder_score', ascending=False)


# In[37]:


#scaled_df.insert(0,'Country_name',df.iloc[:,0])

fig = go.Figure(data=go.Choropleth(
    locations=scaled_df['Country_name'],
    locationmode='country names',
    z = scaled_df['k_means_cluster'],
    text = df['Country_name'],
   colorbar = {'title':'Dystopia'},
    colorscale=('lightblue','lightGreen')
))

fig.show()


# In[62]:


scaled_df['k_means_cluster'].value_counts()


# In[38]:



continuous = ['Logged_GDP_per_capita',
 'Social_support',
 'Healthy_life_expectancy',
 'Freedom_to_make_life_choices',
 'Generosity',
 'Perceptions_of_corruption']


# In[39]:


background_color = '#fbfbfb'
fig = plt.figure(figsize=(12, 6), dpi=150,facecolor=background_color)
gs = fig.add_gridspec(2, 3)
gs.update(wspace=0.2, hspace=0.5)


plot = 0
for row in range(0, 2):
    for col in range(0, 3):
        locals()["ax"+str(plot)] = fig.add_subplot(gs[row, col])
        locals()["ax"+str(plot)].set_facecolor(background_color)
        locals()["ax"+str(plot)].tick_params(axis='y', left=False)
        locals()["ax"+str(plot)].get_yaxis().set_visible(False)
        locals()["ax"+str(plot)].set_axisbelow(True)
        for s in ["top","right","left"]:
            locals()["ax"+str(plot)].spines[s].set_visible(False)
        plot += 1

plot = 0

Yes = scaled_df[scaled_df['k_means_cluster'] == 1]
No = scaled_df[scaled_df['k_means_cluster'] == 0]

for variable in continuous:
        sns.kdeplot(Yes[variable], ax=locals()["ax"+str(plot)], color='lightgreen', shade=True, linewidth=1.5, alpha=0.9, zorder=3, legend=False)
        sns.kdeplot(No[variable],ax=locals()["ax"+str(plot)], color='lightblue', shade=True,linewidth=1.5, alpha=0.9, zorder=3, legend=False)
        locals()["ax"+str(plot)].grid(which='major', axis='x', zorder=0, color='gray', linestyle=':', dashes=(1,5))
        locals()["ax"+str(plot)].set_xlabel(variable, fontfamily='monospace')
        plot += 1
        
Xstart, Xend = ax0.get_xlim()
Ystart, Yend = ax0.get_ylim()

ax0.text(Xstart, Yend+(Yend*0.5), 'Differences between happy & unhappy countries', fontsize=17, fontweight='bold', fontfamily='sansserif',color='#323232')

plt.show()


# In[40]:


# plot the centroids
plt.scatter(centers[:, 0], centers[:, 1],
            marker='x', s=100,
            linewidths=3, c=colors
        )

# store the values of PCA component in variable: for easy writing
xvector =  pca.components_[0] * max(T[0])
yvector =  pca.components_[1] * max(T[1])
columns = df1.columns


# In[41]:


# plot the 'name of individual features' along with vector length
for i in range(len(columns)):
    # plot arrows
    plt.arrow(0, 0, xvector[i], yvector[i],
                color='b', width=0.005,
                head_width=0.08, alpha=0.5
            )
    # plot name of features
    plt.text(xvector[i], yvector[i], list(columns)[i], color='r', alpha=0.75)

plt.scatter(T[0], T[1], 
            c=features_colors, marker='o',
            alpha=0.4)

#plot the centroids
plt.scatter(centers[:, 0], centers[:, 1],
            marker='x', s=100,
            linewidths=3, c=colors )            
plt.show()


# In[42]:


df1.head(5)


# In[43]:


scaled_df_AH=scaled_df_AH.drop(columns = 'Country_name')
scaled_df_AH.head(5)


# In[44]:


scaled_df_AH=scaled_df_AH.drop(columns = 'k_means_cluster')
scaled_df_AH.head(5)


# In[45]:


scaled_df_new = scaled_df_AH


# In[46]:


scaled_df_new.head(4)


# In[47]:


silhouette_scores = [] 

for n_cluster in range(2, 8):
    silhouette_scores.append( 
        silhouette_score(scaled_df_AH, AgglomerativeClustering(n_clusters = n_cluster).fit_predict(scaled_df_AH))) 
    
# Plotting a bar graph to compare the results 
k = [2, 3, 4, 5, 6,7] 
plt.bar(k, silhouette_scores) 
plt.xlabel('Number of clusters', fontsize = 10) 
plt.ylabel('Silhouette Score', fontsize = 10) 
plt.show() 


# In[48]:


import scipy.cluster.hierarchy as shc

plt.figure(figsize=(20, 7))
plt.title("Dendograms - Single linkage")
dend = shc.dendrogram(shc.linkage(principalComponents, method='single'))


# In[49]:


import scipy.cluster.hierarchy as shc

plt.figure(figsize=(20, 7))
plt.title("Dendograms - complete linkage")
dend = shc.dendrogram(shc.linkage(principalComponents, method='complete'))


# In[50]:


import scipy.cluster.hierarchy as shc

plt.figure(figsize=(20, 7))
plt.title("Dendograms - centroid linkage")
dend = shc.dendrogram(shc.linkage(principalComponents, method='centroid'))


# In[51]:


#Dendogram

import scipy.cluster.hierarchy as shc
plt.figure(figsize=(20,11))
plt.title("Dendograms")
dg = shc.dendrogram(shc.linkage(principalComponents, method='ward'))
plt.axhline(y=5, color='b', linestyle='--')


# In[52]:


from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
Hierical_label = cluster.fit_predict(principalComponents)
Hierical_label


# In[53]:


plt.figure(figsize=(10, 7))
plt.scatter(principalComponents[:,0], principalComponents[:,1],c=cluster.labels_, cmap='rainbow')


# In[54]:


#scaled_df_AH.head(5)


# In[55]:


T=PCA_components 
colors = ['red', 'purple']
features_colors = [ colors[Hierical_label[i]] for i in range(len(T)) ]
xvector =  pca.components_[0] * max(T[0])
yvector =  pca.components_[1] * max(T[1])
columns = scaled_df_AH.columns




# assign a color to each features (note that we are using features as target)
for i in range(len(columns)):
    # plot arrows
    plt.arrow(0, 0, xvector[i], yvector[i],
                color='b', width=0.005,
                head_width=0.08, alpha=0.5
            )
    # plot name of features
    plt.text(xvector[i], yvector[i], list(columns)[i], color='b', alpha=0.75)

plt.scatter(T[0], T[1], 
            c=features_colors, marker='o',
            alpha=0.4)

#plot the centroids
plt.scatter(centers[:, 0], centers[:, 1],
            marker='x', 
            linewidths=3, c=colors )            
plt.show()


# In[56]:


scaled_df_AH['Hierical_cluster'] = Hierical_label


# In[57]:


scaled_df_AH.sort_values('Ladder_score', ascending=False)


# In[58]:


scaled_df_AH.insert(0,'Country_name',df.iloc[:,0])

fig = go.Figure(data=go.Choropleth(
    locations=scaled_df_AH['Country_name'],
    locationmode='country names',
    z = scaled_df_AH['Hierical_cluster'],
    text = df['Country_name'],
   colorbar = {'title':'Dystopia'},
    colorscale=('red','purple')
))

fig.show()


# In[64]:


background_color = '#fbfbfb'
fig = plt.figure(figsize=(12, 6), dpi=150,facecolor=background_color)
gs = fig.add_gridspec(2, 3)
gs.update(wspace=0.2, hspace=0.5)


plot = 0
for row in range(0, 2):
    for col in range(0, 3):
        locals()["ax"+str(plot)] = fig.add_subplot(gs[row, col])
        locals()["ax"+str(plot)].set_facecolor(background_color)
        locals()["ax"+str(plot)].tick_params(axis='y', left=False)
        locals()["ax"+str(plot)].get_yaxis().set_visible(False)
        locals()["ax"+str(plot)].set_axisbelow(True)
        for s in ["top","right","left"]:
            locals()["ax"+str(plot)].spines[s].set_visible(False)
        plot += 1

plot = 0

Yes = scaled_df_AH[scaled_df_AH['Hierical_cluster'] == 1]
No = scaled_df_AH[scaled_df_AH['Hierical_cluster'] == 0]

for variable in continuous:
        sns.kdeplot(Yes[variable], ax=locals()["ax"+str(plot)], color='red', shade=True, linewidth=1.5, alpha=0.9, zorder=3, legend=False)
        sns.kdeplot(No[variable],ax=locals()["ax"+str(plot)], color='purple', shade=True,linewidth=1.5, alpha=0.9, zorder=3, legend=False)
        locals()["ax"+str(plot)].grid(which='major', axis='x', zorder=0, color='gray', linestyle=':', dashes=(1,5))
        locals()["ax"+str(plot)].set_xlabel(variable, fontfamily='monospace')
        plot += 1
        
Xstart, Xend = ax0.get_xlim()
Ystart, Yend = ax0.get_ylim()

ax0.text(Xstart, Yend+(Yend*0.5), 'Differences between happy & unhappy countries', fontsize=17, fontweight='bold', fontfamily='sansserif',color='#323232')
ax0.text(Xstart, Yend+(Yend*0.25), 'RED:UTOPIAN COUNTRIES                PURPLE: DYSTOPIAN COUNTRIES', fontsize=12, fontweight='light', fontfamily='monospace',color='gray')
plt.show()


# In[60]:


scaled_df_AH['Hierical_cluster'].value_counts()

