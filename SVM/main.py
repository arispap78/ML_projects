from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
# import warnings filter
from warnings import simplefilter
from warnings import filterwarnings

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
filterwarnings('ignore')

def parameters_svm(X,y):

    #3 different percentages of splitting the data to training and testing sets
    data_split = np.arange(0.6, 1, 0.15)
    #svm kernels that will be used
    krnl = ['poly', 'linear', 'rbf', 'sigmoid']
    #values of C parameter
    dev_C = [0.001, 0.01, 0.1, 1, 10]
    #gamma values for the rbf,poly,sigmoid kernels
    dev_gamma = [0.0001, 0.001, 0.01, 0.1]
    #the degree of the polynomial kernel
    dev_deg = np.arange(2, 5, 1)

    #lists with the parameters of the best accuracy rate
    linear_list = ["linear", 0, 0, 0]
    poly_list = ["poly", 0, 0, 0, 0, 0]
    rbf_list = ["rbf", 0, 0, 0, 0]
    sigmoid_list = ["sigmoid", 0, 0, 0, 0]
    #for each possible combination of the parameters
    for dat in data_split:
        # the best accuracy rates from each kernel
        best_score_linear = 0
        best_score_poly = 0
        best_score_rbf = 0
        best_score_sigmoid = 0
        #split the data to testing and training set
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=dat, random_state=4)
        #for each kernel
        for a in krnl:
            #for each value of C
            for b in dev_C:
                if a=="linear":
                    #the SV classifier
                    modelSVM1 = SVC(C=b, kernel=a)
                    #train the model
                    modelSVM1.fit(X_train, y_train)
                    # Predicting the test set result
                    y_pred1 = modelSVM1.predict(X_test)
                    # Creating the Confusion matrix
                    cm1 = confusion_matrix(y_test, y_pred1)
                    #report of the metrics
                    cr1=classification_report(y_test, y_pred1)
                    # the accuracy score
                    ac1 = accuracy_score(y_test, y_pred1)
                    #if there is a max accurascy score
                    if best_score_linear < ac1:
                        best_score_linear = ac1
                        print("---------------------------------------------------------------------------\n")
                        print("for kernel: " + a + " for C: " + str(b) + " for data_split: " + str(format(dat, '.2f')))
                        print("the accuracy_linear is: \n" + str(format(best_score_linear, '.4f')))
                        print("the confusion matrix: \n" + str(cm1))
                        print("the classification metrics: \n" + str(cr1))
                        #save the values in the list
                        linear_list[1]=format(dat, '.2f')
                        linear_list[2]=b
                        linear_list[3]=format(best_score_linear, '.4f')
                elif a=="poly":
                    #for each degree of the polynomial
                    for de in dev_deg:
                        # the SV classifier
                        modelSVM2 = SVC(C=b, kernel=a, degree=de)
                        # train the model
                        modelSVM2.fit(X_train, y_train)
                        # Predicting the test set result
                        y_pred2 = modelSVM2.predict(X_test)
                        # Creating the Confusion matrix
                        cm2 = confusion_matrix(y_test, y_pred2)
                        # report of the metrics
                        cr2 = classification_report(y_test, y_pred2)
                        # the accuracy score
                        ac2 = accuracy_score(y_test, y_pred2)
                        # if there is a max accurascy score
                        if best_score_poly < ac2:
                            best_score_poly = ac2
                            print("---------------------------------------------------------------------------\n")
                            print("for kernel: " + a + " for C: " + str(b) + " for data_split: " + str(format(dat, '.2f')) + " for degree: " + str(de))
                            print("the accuracy_poly is: \n" + str(format(best_score_poly, '.4f')))
                            print("the confusion matrix: \n"+str(cm2))
                            print("the classification metrics: \n"+str(cr2))
                            # save the values in the list
                            poly_list[1] = format(dat, '.2f')
                            poly_list[2] = b
                            poly_list[3] = de
                            poly_list[4] = format(best_score_poly, '.4f')
                elif a=="rbf":
                    for g in dev_gamma:
                        # the SV classifier
                        modelSVM3 = SVC(C=b, kernel=a, gamma=g)
                        # train the model
                        modelSVM3.fit(X_train, y_train)
                        # Predicting the test set result
                        y_pred3 = modelSVM3.predict(X_test)
                        # Creating the Confusion matrix
                        cm3 = confusion_matrix(y_test, y_pred3)
                        # report of the metrics
                        cr3 = classification_report(y_test, y_pred3)
                        # the accuracy score
                        ac3 = accuracy_score(y_test, y_pred3)
                        # if there is a max accurascy score
                        if best_score_rbf < ac3:
                            best_score_rbf = ac3
                            print("---------------------------------------------------------------------------\n")
                            print("for kernel: " + a + " for C: " + str(b) + " for gamma: " + str(
                                g) + " for data_split: " + str(format(dat, '.2f')))
                            print("the accuracy_rbf is: \n" + str(format(best_score_rbf, '.4f')))
                            print("the confusion matrix: \n"+str(cm3))
                            print("the classification metrics: \n"+str(cr3))
                            # save the values in the list
                            rbf_list[1] = format(dat, '.2f')
                            rbf_list[2] = b
                            rbf_list[3] = g
                            rbf_list[4] = format(best_score_rbf, '.4f')
                else:
                    for g1 in dev_gamma:
                        # the SV classifier
                        modelSVM4 = SVC(C=b, kernel=a, gamma=g1)
                        # train the model
                        modelSVM4.fit(X_train, y_train)
                        # Predicting the test set result
                        y_pred4 = modelSVM4.predict(X_test)
                        # Creating the Confusion matrix
                        cm4 = confusion_matrix(y_test, y_pred4)
                        # report of the metrics
                        cr4 = classification_report(y_test, y_pred4)
                        # the accuracy score
                        ac4 = accuracy_score(y_test, y_pred4)
                        # if there is a max accurascy score
                        if best_score_sigmoid < ac4:
                            best_score_sigmoid = ac4
                            print("---------------------------------------------------------------------------\n")
                            print("for kernel: " + a + " for C: " + str(b) + " for gamma: " + str(
                                g1) + " for data_split: " + str(format(dat, '.2f')))
                            print("the accuracy_sigmoid is: \n" + str(format(best_score_sigmoid, '.4f')))
                            print("the confusion matrix: \n"+str(cm4))
                            print("the classification metrics: \n"+str(cr4))
                            # save the values in the list
                            sigmoid_list[1] = format(dat, '.2f')
                            sigmoid_list[2] = b
                            sigmoid_list[3] = g1
                            sigmoid_list[4] = format(best_score_sigmoid, '.4f')
        #Formation of the print of the results(for a quick review)
        print("THE BEST PARAMETERS FOR SPLIT OF DATA: "+str((dat*100))+"%\n")
        print("for kernel: " + linear_list[0] + ", for C: " + str(linear_list[2]) + ", for data_split: " + str(linear_list[1]) +
            ", the accuracy rate is: "+linear_list[3])
        print("for kernel: " + poly_list[0] + ", for C: " + str(poly_list[2]) + ", for data_split: " + str(poly_list[1]) + ", for degree: " + str(poly_list[3])+
            ", the accuracy rate is: "+poly_list[4])
        print("for kernel: " + rbf_list[0] + ", for C: " + str(rbf_list[2]) + ", for gamma: " + str(
            rbf_list[3]) + ", for data_split: " + str(rbf_list[1]) +
            ", the accuracy rate is: "+rbf_list[4])
        print("for kernel: " + sigmoid_list[0] + ", for C: " + str(sigmoid_list[2]) + ", for gamma: " + str(
            sigmoid_list[3]) + ", for data_split: " + str(sigmoid_list[1]) +
            ", the accuracy rate is: "+sigmoid_list[4])


#method for making binary classification on mnist dataset
def mnist_duo(data):
    # list of the ten datasets with different classification
    svms = {}
    #for all the digits of the label column
    for i in range(0, 10):
        #copy the dataframe
        svms[i]=data.copy(deep=True)
        #if the label digit is not right
        svms[i].label[svms[i].label != i] = 'not ' + str(i)
        #if the label digit is right
        svms[i].label[svms[i].label == i] = str(i)
    return svms

#method for making odd-even labels for classification on mnist dataset
def mnist_oddeven(data):
    # for all the digits of the label column
    for i in range(0, 10):
        #if the label digit is odd
        data.label[data.label%2 != 0] = 1
        # if the label digit is even
        data.label[data.label%2 == 0] = 0

def gridsearch_KN(X,y):
    # split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.60, random_state=4)
    #values of parameters of the gridsearch
    param_grid = {'n_neighbors':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,15,16,17,18,19,20]}
    #the classifier
    grid = GridSearchCV(KNeighborsClassifier(), param_grid)

    # fitting the model for grid search
    grid.fit(X_train, y_train)
    # Predicting the test set result
    grid_predictions = grid.predict(X_test)

    print("\n---------THE METRICS OF K-NEIGHBORS CLASSIFIER---------\n")
    # print best parameter after tuning
    print("the number of neighbors:\n" + str(grid.best_params_))
    # Creating the Confusion matrix
    print("the confusion matrix: \n" + str(confusion_matrix(y_test, grid_predictions)))
    # report of the metrics
    print("the accuracy rate is: \n"+str(accuracy_score(y_test, grid_predictions)))
    # print classification report
    print("the classification metrics: \n"+str(classification_report(y_test, grid_predictions)))

def centroic(X,y):
    # split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.60, random_state=4)
    # the SV classifier
    mnist_modelCN=NearestCentroid()
    # train the model
    mnist_modelCN.fit(X_train, y_train)
    # Predicting the test set result
    y_pred = mnist_modelCN.predict(X_test)

    print("\n---------THE METRICS OF NEAREST CENTROID CLASSIFIER---------\n")
    # Creating the Confusion matrix
    print("the confusion matrix: \n"+str(confusion_matrix(y_test, y_pred)))
    # report of the metrics
    print("the classification metrics: \n"+str(classification_report(y_test, y_pred)))
    # the accuracy score
    print("the accuracy rate is: \n"+str(accuracy_score(y_test, y_pred)))







