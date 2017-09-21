#!/usr/bin/python
import os
os.chdir('C:\\Users\\TONY BASTIN\\Google Drive\\Udacity\\P8 ML\\Project P5 - Identifying fraud from Enron Email\\final_project')
os.getcwd()

import sys
import pickle
from matplotlib import pyplot as plt
from numpy import mean
import pprint
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
from data_exploration import initial_data_exploration, plot_data_exploration, \
        find_oulier, find_person_missing_features
from feature_creation import create_feature, best_features
from estimator_finder_evaluator import best_estimator_finder, \
        estimator_evaluator
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

# boolean, represented as integer
poi_label =['poi']

# Units of financial features are in US dollars
financial_features = ['salary', 'deferral_payments', 'total_payments', 
                      'loan_advances', 'bonus', 'restricted_stock_deferred', 
                      'deferred_income', 'total_stock_value', 'expenses', 
                      'exercised_stock_options', 'other', 
                      'long_term_incentive', 'restricted_stock',
                      'director_fees'] 

# Units are generally number of emails messages; ‘email_address’ is removed
email_features = ['to_messages', 'from_poi_to_this_person',
                  'from_messages', 'from_this_person_to_poi',
                  'shared_receipt_with_poi'] 

# Features used for Machine learning                  
all_features_list = poi_label + financial_features + email_features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)
    
 
### Task 2: Remove outliers

# See "final_project/data_exploration.py" for functions for initial data 
# exploration, make scatter plots and finding outliers etc.

# Untag below to print out names, POI's and their numbers 
#initial_data_exploration(data_dict)  
# Noted unsual names "TOTAL" and "THE TRAVEL AGENCY IN THE PARK" in the name list

# Untag below to make scatter plots for 'bonus' vs 'salary' 
#plot_data_exploration(data_dict,'bonus' , 'salary')
# Noted a person with salary greater than 1e7 which is a big oulier
#find_oulier(data_dict, "salary", 1e7)
# Noted that the outlier is "TOTAL"

# Untag below to make scatter plots for 
#'from_poi_to_this_person' vs 'from_this_person_to_poi' 
#plot_data_exploration(data_dict,'from_poi_to_this_person', \
#                      'from_this_person_to_poi' )
# Noted a person wih 'from_poi_to_this_person' emails greater than 500
#find_oulier(data_dict, 'from_poi_to_this_person', 500)
# Noted nothing wrong with 'LAVORATO JOHN J'
# Noted a POI wih 'from_this_person_to_poi' emails greater than 500
#find_oulier(data_dict, 'from_this_person_to_poi', 500)
# Noted nothing wrong with 'DELAINEY DAVID W'

# Untag below to find if any person have less than 10% features filled
#find_person_missing_features(data_dict)
# Noted that 'LOCKHART EUGENE E' has less than 10% features filled

# From above exploration, it was deiced to remove the following persons from 
# the data : 'TOTAL', 'THE TRAVEL AGENCY IN THE PARK', 'LOCKHART EUGENE E'
for key in ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK', 'LOCKHART EUGENE E']:
    data_dict.pop(key,0)


### Task 3: Create new feature(s)

# Untag to create two new features "fraction_from_poi" and "fraction_to_poi"
# "fraction_from_poi" is "from_poi_to_this_person"/"from_messages"
# "fraction_to_poi" is "from_this_person_to_poi"/"to_messages"      
create_feature(data_dict)

# Update "fraction_from_poi" and "fraction_to_poi" to "all_features_list"
all_features_list += ["fraction_from_poi", "fraction_to_poi"]

# Select, print and store 10 best features using SelectKBest
best_features_and_scores = best_features(data_dict, all_features_list, 17)

# Update "my_features_list" with "poi" and the best 10 features
my_features_list = poi_label + list (best_features_and_scores.keys()) 
                  # + ["fraction_from_poi"]

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, my_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)



### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Scale features using standard scaler
from sklearn.preprocessing  import MinMaxScaler
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

# Logistic regression
from sklearn.linear_model import LogisticRegression
parameters_log = {'C': [0.05, 0.5, 1, 5, 10, 10**2, 500, 10**3, 10**5],
              'tol':[10**-1, 10**-2, 10**-4, 10**-5],
              'class_weight':['balanced'] } 
clf_log = LogisticRegression()        
        
# Decision tree classifier
from sklearn import tree
parameters_dt = {'criterion': ['gini', 'entropy'], 
                        'min_samples_split': [2,5,8,10,15,20,25,50]}
clf_dt = tree.DecisionTreeClassifier()


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Logistic regression

# Untag below to check for best parameters using GridSearchCV()
best_estimator_finder(clf_log, parameters_log, features, labels)
#clf_log = LogisticRegression(C = 0.05, class_weight = 'balanced', tol= 0.1)
#estimator_evaluator(clf_log, features, labels, 1000 )
from sklearn.pipeline import Pipeline
clf_log = Pipeline(steps=[("scaler", scaler),
                      ("clf", LogisticRegression(C = 5, class_weight = 'balanced', tol= 0.1))])
test_classifier(clf_log, my_dataset, my_features_list, folds = 1000)

# Decision tree classifier
best_estimator_finder(clf_dt, parameters_dt, features, labels)
clf_dt = tree.DecisionTreeClassifier(criterion = 'entropy', min_samples_split= 8)
#estimator_evaluator(clf_dt, features, labels, 1000 )
test_classifier(clf_dt, my_dataset, my_features_list, folds = 1000)

    
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.


clf = clf_log
dump_classifier_and_data(clf, my_dataset, my_features_list)