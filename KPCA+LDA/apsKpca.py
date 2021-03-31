from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.decomposition import PCA,KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import mainKpca


#reading data from two seperated files,the training and the testing sets
train_data=pd.read_csv(r'C:\Users\papan\Downloads\computer intelligence\Papanastasiou_Nektarios\datasets_papanastasiou\aps_failure_training_set.csv')
test_data=pd.read_csv(r'C:\Users\papan\Downloads\computer intelligence\Papanastasiou_Nektarios\datasets_papanastasiou\aps_failure_test_set.csv')
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

#take a part of the data
aps_data=aps_data.iloc[:40000]
#separate the data in two parts,one with the features and the other with the labels
y=aps_data['class']
X=aps_data.drop(['class'],axis=1)

#scaling the values
X =pd.DataFrame(scaler.fit_transform(X))

#find the best parameters for the kernelPCA
parameters_knn,parameters_ced=mainKpca.kpca_parameters(X,y)
#create a kernelPCA
kpca = KernelPCA(kernel=parameters_knn[0],n_components=parameters_knn[1],  gamma=parameters_knn[2])
#kpca = KernelPCA(kernel=parameters_ced[0],n_components=parameters_ced[1],  gamma=parameters_ced[2])
#find the number of features which keep more than 90% of the information
pca = PCA(n_components=5, svd_solver='full')
#create a LDA
lda=LinearDiscriminantAnalysis()

# the list of the methods to be compared
dim_red_methods = {'PCA':pca,'LDA':lda,'KernelPCA':kpca,'KernelPCA+LDA':[kpca,lda]}
#print and plot te results
mainKpca.plot_results_bin(dim_red_methods,X,y)
