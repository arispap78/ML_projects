import pandas as pd
from sklearn.decomposition import PCA,KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import MinMaxScaler
import mainKpca


#reading data from two seperated files,the training and the testing sets
train_data=pd.read_csv(r'C:\Users\papan\Downloads\computer intelligence\Papanastasiou_Nektarios\datasets_papanastasiou\mnist_train.csv')
test_data=pd.read_csv(r'C:\Users\papan\Downloads\computer intelligence\Papanastasiou_Nektarios\datasets_papanastasiou\mnist_test.csv')
# join the sets for handling the splitting
data_mnist=pd.concat([train_data, test_data])

#create a min_max_scaler
scaler = MinMaxScaler()

#take a part of the data
data_mnist=data_mnist.iloc[:20000]
#separate the data in two parts,one with the features and the other with the labels
y = data_mnist["label"]
X = data_mnist.drop(["label"],axis=1)

#scaling the values
X =pd.DataFrame(scaler.fit_transform(X))

#find the best parameters for the kernelPCA
parameters_knn,parameters_ced=mainKpca.kpca_parameters(X,y)
#create a kernelPCA
kpca = KernelPCA(kernel=parameters_knn[0],n_components=parameters_knn[1],  gamma=parameters_knn[2])
#kpca = KernelPCA(kernel=parameters_ced[0],n_components=parameters_ced[1],  gamma=parameters_ced[2])
#create a PCA
pca = PCA(n_components=87, svd_solver='full')
#create a LDA to find the number of components for a specific variance
lda=LinearDiscriminantAnalysis(n_components=None)
lda.fit(X,y)
# Create array of explained variance ratios
lda_var_ratios = lda.explained_variance_ratio_
#the number of the components for 90% variance
components=mainKpca.select_components_lda(lda_var_ratios, 0.90)
#create a LDA with the number of components for 90% variance
lda=LinearDiscriminantAnalysis(n_components=components)

# the list of the methods to be compared
dim_red_methods = {'PCA':pca,'LDA':lda,'KernelPCA':kpca,'KernelPCA+LDA':[kpca,lda]}
#print and plot te results
mainKpca.plot_results(dim_red_methods,X,y)

