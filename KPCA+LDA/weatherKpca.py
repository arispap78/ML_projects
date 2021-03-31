from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.decomposition import PCA,KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import mainKpca
from sklearn.preprocessing import OrdinalEncoder


#reading data from two seperated files,the training and the testing sets
data=pd.read_csv(r'C:\Users\papan\Downloads\computer intelligence\Papanastasiou_Nektarios\datasets_papanastasiou\weatherAUS.csv')

#drop the columns with the string values
data.drop(["Date", "Location"], axis = 1, inplace = True)

#to categorize columns
enc = OrdinalEncoder()
#create a min_max_scaler
scaler = MinMaxScaler()

#one-hot encoding of the columns with "yes","no" values
data=pd.get_dummies(data,columns=['RainToday','RainTomorrow'],drop_first=True)
#rename the one-hot encoding columns with their previous names
data.rename(columns = {"RainToday_Yes":"RainToday","RainTomorrow_Yes":"RainTomorrow"}, inplace = True)
#insert interpolated values to the NaN elements
data=data.interpolate()
#insert values of next elements to the NaN elements
data=data.fillna(method ='bfill')
#to_categorical values
data[["WindDir9am","WindDir3pm","WindGustDir"]] = enc.fit_transform(data[["WindDir9am","WindDir3pm","WindGustDir"]])

#take a part of the data
data=data.iloc[:70000]
#separate the data in two parts,one with the features and the other with the labels
y=data['RainTomorrow']
X=data.drop(['RainTomorrow'],axis=1)
#scaling the values
X =pd.DataFrame(scaler.fit_transform(data))

#find the best parameters for the kernelPCA
parameters_knn,parameters_ced=mainKpca.kpca_parameters(X,y)
#create a kernelPCA
kpca = KernelPCA(kernel=parameters_knn[0],n_components=parameters_knn[1],  gamma=parameters_knn[2])
#kpca = KernelPCA(kernel=parameters_ced[0],n_components=parameters_ced[1],  gamma=parameters_ced[2])
#find the number of features which keep more than 90% of the information
pca = PCA(n_components=0.90, svd_solver='full')
#create a LDA
lda=LinearDiscriminantAnalysis()

# the list of the methods to be compared
dim_red_methods = {'PCA':pca,'LDA':lda,'KernelPCA':kpca,'KernelPCA+LDA':[kpca,lda]}
#print and plot te results
mainKpca.plot_results_bin(dim_red_methods,X,y)

