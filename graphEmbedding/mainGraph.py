import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, homogeneity_score, classification_report, accuracy_score
import pylab as pl
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier

#plot the data after graph empbedding
def plot_graph(X,labels,clusters,graphEmb,affinity):
    # Building the label to colour mapping
    colours = {}
    colours[0] = 'b'
    colours[1] = 'y'
    colours[2] = 'r'
    colours[3] = 'g'
    colours[4] = 'c'
    colours[5] = 'k'
    colours[6] = 'm'
    colours[7] = 'orange'
    colours[8] = 'pink'
    # Building the colour vector for each data point
    cvec = [colours[label] for label in labels]
    # Plotting the clustered scatter plot
    b = plt.scatter(X[0], X[1], color='b');
    y = plt.scatter(X[0], X[1], color='y');
    r = plt.scatter(X[0], X[1], color='r');
    g = plt.scatter(X[0], X[1], color='g');
    c = plt.scatter(X[0], X[1], color='c');
    po = plt.scatter(X[0], X[1], color='orange');
    k = plt.scatter(X[0], X[1], color='k');
    m = plt.scatter(X[0], X[1], color='m');
    pi = plt.scatter(X[0], X[1], color='pink');
    plt.figure(figsize=(9, 9))
    plt.scatter(X[0], X[1], c=cvec)
    if(clusters==2):
        plt.legend((b, y), ('Cluster 1', 'Cluster 2'))
    elif(clusters==3):
        plt.legend((b, y, r), ('Cluster 1', 'Cluster 2', 'Cluster 3'))
    elif (clusters == 4):
        plt.legend((b, y, r, g), ('Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'))
    elif (clusters == 5):
        plt.legend((b, y, r, g, c), ('Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4','Cluster 5'))
    elif (clusters == 6):
        plt.legend((b, y, r, g, c, m), ('Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4','Cluster 5','Cluster 6'))
    elif (clusters == 7):
        plt.legend((b, y, r, g, c, m, k), ('Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4','Cluster 5','Cluster 6','Cluster 7'))
    elif (clusters == 8):
        plt.legend((b, y, r, g, c, m, k, po), ('Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4','Cluster 5','Cluster 6','Cluster 7','Cluster 8'))
    elif (clusters == 9):
        plt.legend((b, y, r, g, c, m, k, po, pi), (
        'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5', 'Cluster 6', 'Cluster 7', 'Cluster 8','Cluster 9'))
    plt.title('Clustering Model with '+graphEmb+' embedding and affinity '+affinity)
    plt.show()

#method for visualization of the clustering,returns the cluster labels
def plot_cluster(data,affinity,num_clusters,X,y):
    cluster_labels=[]
    for data_embedding in data:
        # List of Silhouette Scores
        s_scores = []
        # List of Homogeneity Scores
        h_scores = []
        for affin in affinity:
            print("1")
            # Building the clustering model
            spectral_model = SpectralClustering(n_clusters=num_clusters,n_components=10, affinity=affin)
            print("2")
            # Training the model and Storing the predicted cluster labels
            labels = spectral_model.fit_predict(data_embedding[0])
            print("3")
            plot_graph(data_embedding[0], labels, num_clusters, data_embedding[1], affin)
            # Evaluating the performance
            s_scores.append(silhouette_score(X, labels))
            h_scores.append(homogeneity_score(y, labels))
            cluster_labels.append([labels,affin,data_embedding[1]])
        print(s_scores)
        print(h_scores)
        # comparing performances
        # Plotting a Bar Graph to compare the models for the Silhouette Score
        plt.bar(affinity, s_scores)
        plt.xlabel('Affinity')
        plt.ylabel('Silhouette Score')
        plt.title('Comparison of different Clustering Models with ' + data_embedding[1] + '\n based on Silhouette Score')
        plt.show()
        # Plotting a Bar Graph to compare the models for the Homogeneity Score
        plt.bar(affinity, h_scores)
        plt.xlabel('Affinity')
        plt.ylabel('Homogeneity Score')
        plt.title('Comparison of different Clustering Models with ' + data_embedding[1] + '\n based on Homogeneity Score')
        plt.show()
    return cluster_labels


# Scale and visualize the embedding vectors
def plot_embedding(X,y, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    pl.figure()
    ax = pl.subplot(111)
    for i in range(X.shape[0]):
        pl.text(X[i, 0], X[i, 1], str(y[i]),
                color=pl.cm.Set1(y[i] / 10.),
                fontdict={'weight': 'bold', 'size': 9})
    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
    pl.xticks([]), pl.yticks([])
    if title is not None:
        pl.title(title)
    pl.show()

#accuracy rate with the nearest centroid
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
    # report of the metrics
    print("the classification metrics: \n"+str(classification_report(y_test, y_pred)))
    # the accuracy score
    print("the accuracy rate is: \n"+str(accuracy_score(y_test, y_pred)))

# accuracy rate with the nearest neighbor
def gridsearch_KN(X,y):
    # split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.60, random_state=4)
    #values of parameters of the gridsearch
    param_grid = {'n_neighbors':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
    #the classifier
    grid = GridSearchCV(KNeighborsClassifier(), param_grid)

    # fitting the model for grid search
    grid.fit(X_train, y_train)
    # Predicting the test set result
    grid_predictions = grid.predict(X_test)

    print("\n---------THE METRICS OF K-NEIGHBORS CLASSIFIER---------\n")
    # print best parameter after tuning
    print("the number of neighbors:\n" + str(grid.best_params_))
    # report of the metrics
    print("the accuracy rate is: \n"+str(accuracy_score(y_test, grid_predictions)))
    # print classification report
    print("the classification metrics: \n"+str(classification_report(y_test, grid_predictions)))
