from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.decomposition import PCA
import main
from sklearn.preprocessing import OrdinalEncoder


#reading data from two seperated files,the training and the testing sets
data=pd.read_csv(r'C:\Users\papan\Downloads\weatherAUS.csv')

#drop the columns with the string values
data.drop(["Date", "Location"], axis = 1, inplace = True)

#to categorize columns
enc = OrdinalEncoder()
#find the number of features which keep more than 90% of the information
pca = PCA(n_components=0.90, svd_solver='full')
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

#take a part of the data
#data=data.iloc[:100000]
#separate the data in two parts,one with the features and the other with the labels
y=data['RainTomorrow']
X=data.drop(['RainTomorrow'],axis=1)

#to_categorical values
X[["WindDir9am","WindDir3pm","WindGustDir"]] = enc.fit_transform(X[["WindDir9am","WindDir3pm","WindGustDir"]])
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
