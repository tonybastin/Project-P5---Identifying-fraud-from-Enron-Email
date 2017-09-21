
def best_estimator_finder(clf, parameters, features, labels):
    
    from sklearn.grid_search import GridSearchCV
    from sklearn.cross_validation import train_test_split
    
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=42) 
    
    clf_grid = GridSearchCV(clf, parameters, scoring = 'precision') # , 'recall'
    clf_grid.fit(features_train, labels_train)

    print ("Best estimator : ",clf_grid.best_score_)
    print ("Best estimator values : ",clf_grid.best_params_)
    #print ("Estimator grid: ",clf_grid.grid_scores_) 
    
    return (clf_grid.best_params_)

###############################################   
def estimator_evaluator(clf, dataset, feature_list, num_iter ):
    
    from numpy import mean
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    from feature_format import featureFormat, targetFeatureSplit
    from sklearn.cross_validation import train_test_split
    
                
    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    
    from sklearn.preprocessing  import MinMaxScaler
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)
    
    precision_values = []
    recall_values = []
    accuracy_values = []
    for i in range(num_iter):
        features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=i) 
        clf.fit(features_train, labels_train)    
        pred = clf.predict(features_test)
        precision = precision_score(labels_test, pred)
        recall = recall_score(labels_test, pred)
        accuracy = accuracy_score(labels_test, pred)
        if precision != 0 and recall != 0:
            precision_values.append(precision)
            recall_values.append(recall)
            accuracy_values.append(accuracy)
        
    print ("Mean values over {} iterations ".format(num_iter))
    print ("Accuracy : ",mean(accuracy_values))
    print ("Precision : ",mean(precision_values))
    print ("Recall : ",mean(recall_values)) 
                             
    
##################
def estimator_evaluator1(clf, dataset, feature_list, folds ):
    from feature_format import featureFormat, targetFeatureSplit
    from sklearn.cross_validation import StratifiedKFold
    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    cv = StratifiedKFold(labels, n_folds= folds, random_state = 30)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        
        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print ("Warning: Found a predicted label not == 0 or 1.")
                print ("All predictions should take value 0 or 1.")
                print ("Evaluating performance for processed predictions:")
                break
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        print (clf)
        print (PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5))
        print (RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives))
        print ("")
    except:
        print ("Got a divide by zero when trying out:", clf)
        print ("Precision or recall may be undefined due to a lack of true positive predicitons.")    

    
    
    
    
 
             
              
              
