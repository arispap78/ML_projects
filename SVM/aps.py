from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.decomposition import PCA
import main


#reading data from two seperated files,the training and the testing sets
train_data=pd.read_csv(r'C:\Users\papan\Downloads\aps_failure_training_set.csv')
test_data=pd.read_csv(r'C:\Users\papan\Downloads\aps_failure_test_set.csv')
#join the sets for handling the splitting
aps=pd.concat([train_data, test_data])

#one-hot encoding of the label-column
aps_data=pd.get_dummies(aps, columns=['class'], drop_first=True)
#rename the one-hot encoding column with its previous name
aps_data.rename(columns = {"class_pos": "class"}, inplace = True)
#insert interpolated values to the NaN elements
aps_data=aps_data.interpolate()
#insert 0 to the NaN elements
aps_data=aps_data.replace("na", 0)

#create a min_max_scaler
scaler = MinMaxScaler()
#find the number of features which keep more than 90% of the information
pca = PCA(n_components=0.90, svd_solver='full')

#separate the data in two parts,one with the features and the other with the labels
y=aps_data['class']
X=aps_data.drop(['class'],axis=1)

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