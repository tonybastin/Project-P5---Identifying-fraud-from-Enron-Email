
def best_estimator_finder(clf, parameters, features, labels):
    
    from sklearn.grid_search import GridSearchCV
    from sklearn.cross_validation import train_test_split
    
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=42) 
    
    clf_grid = GridSearchCV(clf, parameters)
    clf_grid.fit(features_train, labels_train)

    #clf.best_score_
    #clf.grid_scores_ 
    return (clf_grid.best_params_)
                
    
def estimator_evaluator(clf, features, labels, num_iter ):
    
    from numpy import mean
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    from sklearn.cross_validation import train_test_split
                
    precision_values = []
    recall_values = []
    accuracy_values = []
    for i in range(num_iter):
        features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=42) 
        clf.fit(features_train, labels_train)    
        pred = clf.predict(features_test)
        precision_values.append(precision_score(labels_test, pred))
        recall_values.append(recall_score(labels_test, pred))
        accuracy_values.append(accuracy_score(labels_test, pred))
        
    print ("Mean values over {} iterations ".format(num_iter))
    print ("Accuracy : ",mean(accuracy_values))
    print ("Precision : ",mean(precision_values))
    print ("Recall : ",mean(recall_values)) 
             
             
              
              
