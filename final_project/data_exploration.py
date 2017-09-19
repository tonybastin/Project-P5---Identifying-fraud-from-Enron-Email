# Data set exploration
from matplotlib import pyplot as plt

# Section 1 - General function to print out names, POI's and their numbers
def initial_data_exploration(data_dict):

    # Number of persons and their names
    print ("Number of persons in the dataframe : ", len(data_dict))
    print (" ")
    print ("Name of persons in the dataset : \n")
    person_names = list(data_dict.keys())
    person_names.sort()
    person_names
    num = 1
    for name in person_names :
        print("{}) {}".format(num, name))
        num+=1
    
    # Number of POI's and their names    
    num_poi = 0
    for name, value in data_dict.items() :
        if value['poi'] == 1:
            num_poi += 1  
    print("Number of person of inteterst's (POI) in dataset : ", num_poi)
    print (" ")
    print ("Name of POI's in the dataset : \n")
    num = 1
    for name, value in data_dict.items() :
        if value['poi'] == 1:
            print("{}) {}".format(num, name)) 
            num+=1
    return None


# Section 2 
def plot_data_exploration(data_dict, x_label , y_label):
    
    for name, value in data_dict.items() :
        x = value[x_label] 
        y = value[y_label]
        if value['poi'] == 1:
            plt.scatter(x, y, color = "red")
        else:
            plt.scatter(x, y, color = "blue")
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
            
    
# Checking for o

def find_oulier(data_dict, feature, feature_value):
    
    outlier_names = []
    for name, value in data_dict.items():    
        if float(value[feature]) > feature_value:
            outlier_names.append(name)
    print ("Ouliers in '{}' is/are : {}".format(feature, outlier_names))
    
    
def find_person_missing_features(data_dict):
    # Percentage feature available for each person    
    person_missing_feature = {}
    for name, value in data_dict.items() :
        count = 0
        for feature, feature_value in value.items():
            if feature_value == "NaN":
                count +=1
        person_missing_feature[name] = round((1 - float(count/21))*100,3)    
    
    # Print the name of persons having feature
    print ("Persons having less than 10% features is/are : ")    
    num = 1
    for name in person_missing_feature :
        if person_missing_feature[name] < 10 :
            print("{}) {}".format(num, name)) 
            num+=1   
    
    return None
