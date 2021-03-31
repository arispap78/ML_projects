import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import main


#reading data from two seperated files,the training and the testing sets
train_data=pd.read_csv(r'C:\Users\papan\Downloads\mnist\mnist_train.csv')
test_data=pd.read_csv(r'C:\Users\papan\Downloads\mnist\mnist_test.csv')
# join the sets for handling the splitting
data_mnist=pd.concat([train_data, test_data])

#method for making odd-even labels for classification
main.mnist_oddeven(data_mnist)

#find the number of features which keep more than 90% of the information
pca = PCA(n_components=0.90, svd_solver='full')
#create a min_max_scaler
scaler = MinMaxScaler()

#take a part of the data
#data_mnist=data_mnist.iloc[:40000]
#separate the data in two parts,one with the features and the other with the labels
y = data_mnist["label"]
X = data_mnist.drop(["label"],axis=1)

#scaling the values
X =pd.DataFrame(scaler.fit_transform(X))
#create a dataframe with only the principal components
X=pd.DataFrame(pca.fit_transform(X))

#print the array with the rate of variance of each principal component
#print (pca.explained_variance_ratio_.cumsum())

#the method for the metrics of SVC
main.parameters_svm(X,y)
#the method for the Nearest Centroid classifier
main.centroic(X,y)
#method for the K-Neighbors Classifier
main.gridsearch_KN(X,y)
