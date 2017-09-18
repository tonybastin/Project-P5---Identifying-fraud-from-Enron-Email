# Data set exploration

# Section 1 - General
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
for name, key in data_dict.items() :
    if key['poi'] == 1:
        num_poi += 1  
print("Number of person of inteterst's (POI) in dataset : ", num_poi)
print (" ")
print ("Name of POI's in the dataset : \n")
num = 1
for name, key in data_dict.items() :
    if key['poi'] == 1:
        print("{}) {}".format(num, name)) 
        num+=1

# Section 2 