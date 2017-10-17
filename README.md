# Identifying fraud from Enron Email
by Tony Bastin

### Objective

This objective of this project is to build a Machine Learning Model to predict the Person of Interest 'POI' in the Enron financial and email dataset. The implementation of the machine learning algorithms is achieved using Python Scikit-learn libraries.

### About the dataset

In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives. The dataset consists of the from ENRON emails and financial data publicly available for research.  

### Steps involved and Code discussion

The major steps involved are discussed below and the main code is written in the ['poi_id.dy'](https://github.com/tonybastin/Project-P5---Identifying-fraud-from-Enron-Email/blob/master/final_project/poi_id.py) file.

1) Remove outliers from dataset:
The dataset was investigated for outliers using exploratory data analysis and other functions.
(Details of the functions made can be seen in [data_exploration.py](https://github.com/tonybastin/Project-P5---Identifying-fraud-from-Enron-Email/blob/master/final_project/data_exploration.py) and the plots here ([1](https://github.com/tonybastin/Project-P5---Identifying-fraud-from-Enron-Email/blob/master/final_project/bonus%20Vs%20salary.png),[2](https://github.com/tonybastin/Project-P5---Identifying-fraud-from-Enron-Email/blob/master/final_project/from_poi_to_this_person%20Vs%20from_this_person_to_poi.png)))

2) Two new features were modelled from the existing features.

3) A feature selection technique called 'SelectKBest' was used to identify the best 5 features and the details of the function **best_features()** can be seen [here](https://github.com/tonybastin/Project-P5---Identifying-fraud-from-Enron-Email/blob/master/final_project/feature_creation.py)).

4) Afte that a 'MinMaxScale'r' was used to transform the data.

5) The best parameters for 'Logistic Regression' and 'Decision Tree' were tuned using GridSearchCV and is implemented in the [best_estimator_finder()](https://github.com/tonybastin/Project-P5---Identifying-fraud-from-Enron-Email/blob/master/final_project/estimator_finder_evaluator.py) .

6) Logistic Regression has been identified as the best machine learning model by checking the precion and recall scores using the [estimator_evaluator()](https://github.com/tonybastin/Project-P5---Identifying-fraud-from-Enron-Email/blob/master/final_project/estimator_finder_evaluator.py).

# Further info

A detailed discussion of the above steps with values and graphs can be seen in [Q&A.md](https://github.com/tonybastin/Project-P5---Identifying-fraud-from-Enron-Email/blob/master/Q%26A.md).

## License

The code given in the repository is a public domain work, dedicated using [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). Feel free to do whatever you want with it.
