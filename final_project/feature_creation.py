def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """

       
    if poi_messages == "NaN":
        poi_messages = 0
        
    if all_messages == "NaN":
        all_messages = 1
    
    fraction = round(float(poi_messages/all_messages),3)

    return fraction


def create_feature(data_dict):
    #new_feature_dict = {}
    for name in data_dict:
    
        data_point = data_dict[name]
    
        #print
        from_poi_to_this_person = data_point["from_poi_to_this_person"]
        to_messages = data_point["to_messages"]
        fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
        #print (fraction_from_poi,",",name)
        data_point["fraction_from_poi"] = fraction_from_poi
    
    
        from_this_person_to_poi = data_point["from_this_person_to_poi"]
        from_messages = data_point["from_messages"]
        fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
        #print (fraction_to_poi,",",name)
        #new_feature_dict[name]={"from_poi_to_this_person":fraction_from_poi,
        #                   "from_this_person_to_poi":fraction_to_poi}
        data_point["fraction_to_poi"] = fraction_to_poi
    
    return None
