from sklearn.decomposition import KernelPCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
# import warnings filters
from warnings import simplefilter
from warnings import filterwarnings


# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
filterwarnings('ignore')

#compute the classification of nearest neighbor
def gridsearch_KN(X_train, y_train,X_test, y_test):
    #values of parameters of the gridsearch
    param_grid = {'n_neighbors':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,15,16,17,18,19,20]}
    #the classifier
    grid = GridSearchCV(KNeighborsClassifier(), param_grid)

    # fitting the model for grid search
    grid.fit(X_train, y_train)
    # Predicting the test set result
    grid_predictions = grid.predict(X_test)
    #number of neighbors
    neighbors=grid.best_params_
    #the accuracy rate
    accuracy=accuracy_score(y_test, grid_predictions)

    print("\n---------THE METRICS OF K-NEIGHBORS CLASSIFIER---------\n")
    # print best parameter after tuning
    print("the number of neighbors:\n" + str(neighbors))
    # report of the metrics
    print("the accuracy rate is: \n"+str(accuracy))
    # Return the the number of neighbors and accuracy
    return neighbors,accuracy

#compute the classification of nearest centroid
def centroid(X_train, X_test, y_train, y_test):
    # the classifier
    modelCN=NearestCentroid()
    # train the model
    modelCN.fit(X_train, y_train)
    # Predicting the test set result
    y_pred = modelCN.predict(X_test)

    print("\n---------THE METRICS OF NEAREST CENTROID CLASSIFIER---------\n")
    # the accuracy score
    print("the accuracy rate is: \n"+str(accuracy_score(y_test, y_pred)))
    # Return the accuracy
    return accuracy_score(y_test, y_pred)


# find the number of components which keep a required percentage of the information(goal_var) for LDA
def select_components_lda(var_ratio, goal_var: float) -> int:
    # initialize variance
    total_variance = 0.0
    # initialize number of features-components
    n_components = 0
    # For the explained variance of each feature:
    for explained_variance in var_ratio:
        # Add the explained variance to the total
        total_variance += explained_variance
        # Add one more component
        n_components +=1
        # If we reach our goal level of explained variance
        if total_variance >= goal_var:
            # break the loop
            break
    # Return the number of components
    return n_components

#compute and plot the results for multiclass dataset
def plot_results(dim_red_methods,X,y):
    # for all the dimensions reduction methods
    for j, (name, model) in enumerate(dim_red_methods.items()):
        plt.subplot(2, 2, j + 1, aspect='auto')
        # split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.60, random_state=4)
        # the name of the method
        print(name)
        # if the method is combination of methods(KPCA+LDA)
        if isinstance(model, list):
            X_train = pd.DataFrame(model[0].fit_transform(X_train))
            X_test = pd.DataFrame(model[0].fit_transform(X_test))
            X_train = pd.DataFrame(model[1].fit_transform(X_train,y_train))
            X_test = pd.DataFrame(model[1].fit_transform(X_test,y_test))
            # Fit the method using the fitted model
            X_tr = model[0].transform(X)
            X_tr = model[1].transform(X_tr)
        # if the method is the LDA
        elif (name=="LDA"):
            X_train = pd.DataFrame(model.fit_transform(X_train,y_train))
            X_test = pd.DataFrame(model.fit_transform(X_test,y_test))
            # Fit the method using the fitted model
            X_tr = model.transform(X)
        else:
            X_train = pd.DataFrame(model.fit_transform(X_train))
            X_test = pd.DataFrame(model.fit_transform(X_test))
            # Fit the method using the fitted model
            X_tr = model.transform(X)
        #get the number of neighbors and the accuracy rate from the method gridsearch_KN
        neighbors,accuracy=gridsearch_KN(X_train, y_train,X_test, y_test)
        # get the accuracy rate from the method centroid
        accuracy_ced = centroid(X_train, X_test, y_train, y_test)
        # Plot the projected points and show the evaluation score
        plt.scatter(X_tr[:, 0], X_tr[:, 1], c=y, s=20, cmap='Set1')
        plt.title("{}, KNN ({}) Test accuracy = {:.2f}\n {}, Nearest Centroid  Test accuracy = {:.2f}".format(name,neighbors,accuracy,name, accuracy_ced))
        #show a colorbar for each class
        plt.colorbar()
    plt.show()

#compute and plot the results for two-class dataset
def plot_results_bin(dim_red_methods,X,y):
    #for all the dimensions reduction methods
    for j, (name, model) in enumerate(dim_red_methods.items()):
        plt.subplot(2, 2, j + 1, aspect='auto')
        # split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.60, random_state=4)
        #the name of the method
        print(name)
        #if the method is combination of methods(KPCA+LDA)
        if isinstance(model, list):
            X_train = pd.DataFrame(model[0].fit_transform(X_train))
            X_test = pd.DataFrame(model[0].fit_transform(X_test))
            X_train = pd.DataFrame(model[1].fit_transform(X_train,y_train))
            X_test = pd.DataFrame(model[1].fit_transform(X_test,y_test))
            # Fit the method using the fitted model
            X_tr = model[0].transform(X)
            X_tr = model[1].transform(X_tr)
        #if the method is the LDA
        elif (name=="LDA"):
            X_train = pd.DataFrame(model.fit_transform(X_train,y_train))
            X_test = pd.DataFrame(model.fit_transform(X_test,y_test))
            # Fit the method using the fitted model
            X_tr = model.transform(X)
        else:
            X_train = pd.DataFrame(model.fit_transform(X_train))
            X_test = pd.DataFrame(model.fit_transform(X_test))
            # Fit the method using the fitted model
            X_tr = model.transform(X)
        #get the number of neighbors and the accuracy rate from the method gridsearch_KN
        neighbors,accuracy=gridsearch_KN(X_train, y_train,X_test, y_test)
        # get the accuracy rate from the method centroid
        accuracy_ced = centroid(X_train, X_test, y_train, y_test)
        # Plot the projected points and show the evaluation score only for these two methods
        if (name=='KernelPCA' or name=='PCA'):
            plt.scatter(X_tr[:, 0], X_tr[:, 1], c=y, s=20, cmap='Set1')
            plt.title("{}, KNN ({}) Test accuracy = {:.2f}\n {}, Nearest Centroid  Test accuracy = {:.2f}".format(name,neighbors,accuracy,name, accuracy_ced))
    plt.show()

def kpca_parameters(X,y):
    # pca kernels that will be used
    krnl = ['poly', 'rbf','sigmoid']
    # number of components
    compo = [20.50,80]
    # gamma values for the rbf,poly,sigmoid kernels
    dev_gamma = [0.001,0.01,0.1,1]
    #for the best accuracy
    best_score=0
    best_score_ced=0
    #the best parameters
    best_kernel=""
    best_components=0
    best_gamma=0.0
    best_kernel_ced = ""
    best_components_ced = 0
    best_gamma_ced = 0.0
    #for each kernel
    for kernel in krnl:
        #for each number of components
        for comp in compo:
            #for each gamma value
            for gama in dev_gamma:
                # create a kernelPCA
                kpca = KernelPCA(n_components=comp, kernel=kernel, gamma=gama)
                # split the data
                X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.60, random_state=4)
                X_train = pd.DataFrame(kpca.fit_transform(X_train))
                X_test = pd.DataFrame(kpca.fit_transform(X_test))
                # get the number of neighbors and the accuracy rate from the method gridsearch_KN
                neighbors, accuracy = gridsearch_KN(X_train, y_train, X_test, y_test)
                # get the accuracy rate from the method centroid
                accuracy_ced=centroid(X_train, X_test, y_train, y_test)
                #if the score is higher
                if best_score < accuracy:
                    best_score = accuracy
                    best_kernel=kernel
                    best_components=comp
                    best_gamma=gama
                # if the score is higher
                elif best_score_ced<accuracy_ced:
                    best_score_ced = accuracy_ced
                    best_kernel_ced = kernel
                    best_components_ced = comp
                    best_gamma_ced = gama
    #print the results
    print("---------------------------------------------------------------------------\n")
    print("the best accuracy rate for knn is "+str(best_score))
    print("for kernel: " + best_kernel + " for Components: " + str(best_components) + " for gamma: " + str(
        best_gamma))
    print("the best accuracy rate for centroid is " + str(best_score_ced))
    print("for kernel: " + best_kernel_ced + " for Components: " + str(best_components_ced) + " for gamma: " + str(
        best_gamma_ced))
    return [best_kernel,best_components,best_gamma],[best_kernel_ced,best_components_ced,best_gamma_ced]

