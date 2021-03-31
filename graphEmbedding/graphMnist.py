from time import time
import pandas as pd
import mainGraph
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import LocallyLinearEmbedding,Isomap
# import warnings filters
from warnings import simplefilter
from warnings import filterwarnings


# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
filterwarnings('ignore')

#reading data from two seperated files,the training and the testing sets
train_data=pd.read_csv(r'C:\Users\papan\Downloads\computer intelligence\Papanastasiou_Nektarios_ergasia1\datasets_papanastasiou\mnist_train.csv')
test_data=pd.read_csv(r'C:\Users\papan\Downloads\computer intelligence\Papanastasiou_Nektarios_ergasia1\datasets_papanastasiou\mnist_test.csv')
# join the sets for handling the splitting
data_mnist=pd.concat([train_data, test_data])

#create a min_max_scaler
scaler = MinMaxScaler()

#take a part of the data
data_mnist=data_mnist.iloc[:2000]
#separate the data in two parts,one with the features and the other with the labels
y = data_mnist["label"]
X = data_mnist.drop(["label"],axis=1)

#scaling the values
X =pd.DataFrame(scaler.fit_transform(X))

#the number of the neighbors for the computing of the graph embedding
n_neighbors = 5

#reduction of dimentionality in 2 components and visualization
# Isomap
print ("Computing Isomap embedding")
t0 = time()
X_iso = Isomap(n_neighbors=n_neighbors,n_components=2).fit_transform(X)
print ("Done.")
mainGraph.plot_embedding(X_iso,y,"Isomap projection with n_neighbors "+str(n_neighbors)+" (time %.2fs)" %(time() - t0))

# Locally linear embedding
print("Computing LLE embedding")
t0 = time()
X_lle = LocallyLinearEmbedding(n_neighbors=n_neighbors,n_components=2, method='standard').fit_transform(X)
print ("Done.")
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
    print(label[2]+" reduction of dimensions and "+label[1]+" affinity of the spectral clustering")
    #accuracy rate with the nearest centroid
    mainGraph.centroic(X, label[0])
    # accuracy rate with the nearest neighbors
    mainGraph.gridsearch_KN(X, label[0])



