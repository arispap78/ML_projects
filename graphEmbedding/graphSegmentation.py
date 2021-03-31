from time import time
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.manifold import LocallyLinearEmbedding,Isomap
import mainGraph
# import warnings filters
from warnings import simplefilter
from warnings import filterwarnings


# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
filterwarnings('ignore')

#get the dataset
with open(r'C:\Users\papan\Downloads\segmentation_dataset\segmentation.test') as f:
    data_test = f.read()
    # take only the data
    df = pd.DataFrame([x.split(',') for x in data_test.split('\n')[5:2105]])

#separate the data in two parts,one with the features and the other with the labels
y =df[0]
X = df.drop([0],axis=1)

#create a min_max_scaler
scaler = MinMaxScaler()

#to categorize the labels
enc = OrdinalEncoder()
y=enc.fit_transform(df[[0]])
#reshape the dataframe
y=y[:,0]
#convert the string values to float
X = X.applymap(float)

#scaling the values
X =pd.DataFrame(scaler.fit_transform(X))

#the number of the neighbors for the computing of the graph embedding
n_neighbors=10

#reduction of dimentionality in 2 components and visualization
# Isomap
print ("Computing Isomap embedding")
t0 = time()
X_iso = Isomap(n_neighbors=n_neighbors,n_components=2).fit_transform(X)
print ("Done.")
mainGraph.plot_embedding(X_iso,y,"Isomap projection with n_neighbors "+str(n_neighbors)+" (time %.2fs)" %(time() - t0))

# Locally linear embedding
print("Computing LLE embedding")
LLE = LocallyLinearEmbedding(n_neighbors=n_neighbors,n_components=2, method='standard')
t0 = time()
X_lle = LLE.fit_transform(X)
print ("Done. Reconstruction error: %g" % LLE.reconstruction_error_)
mainGraph.plot_embedding(X_lle,y,"Locally Linear Embedding with n_neighbors "+str(n_neighbors)+" (time %.2fs)" %(time() - t0))

#reduction of dimentionality but higher than 2 components
X_lle=pd.DataFrame(LocallyLinearEmbedding(n_neighbors=n_neighbors,n_components=2, method='standard').fit_transform(X))
X_iso=pd.DataFrame(Isomap(n_neighbors=n_neighbors,n_components=2).fit_transform(X))

#number of clusters
num_clusters=7

# List of different values of affinity
affinity = ['rbf', 'nearest_neighbors']
# List of the graph embedding algorithms
data=[[X_iso,"ISOMAP Embedding"],[X_lle,"Locally Linear Embedding"]]

#method for visualization of the clustering,returns the cluster labels
cluster_label=mainGraph.plot_cluster(data,affinity,num_clusters,X,y)

#for the cluster labels of each method
for label in cluster_label:
    print("with "+label[2]+" reduction of dimensions and "+label[1]+" affinity of the spectral clustering")
    # accuracy rate with the nearest centroid
    mainGraph.centroic(X, label[0])
    # accuracy rate with the nearest neighbors
    mainGraph.gridsearch_KN(X, label[0])
