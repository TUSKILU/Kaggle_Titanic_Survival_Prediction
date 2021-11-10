
import numpy as np
from sklearn.preprocessing import OrdinalEncoder


#=========fun for encode name
def encode_name(namelist):
    word2id={}
    for i in range(len(namelist)):
        
        if namelist[i] in word2id.keys():
            namelist[i]=word2id[namelist[i]]
        else:
            word2id[namelist[i]]= len(word2id)
            namelist[i]=word2id[namelist[i]]
            

def clean_data(Data, label= False):
    # Training data => label = True / Testing data => label = False 
    
    #Create new column "family size"= Sibsp+parch
    Data["Family_size"]=Data["SibSp"]+Data["Parch"]
    #drop not considered data 12 -6 +1 = 7
    Data_cleaned=Data.drop(columns=["PassengerId","Embarked","Cabin","SibSp","Parch"])
    if label: # for Training data drop Survived
        Data_cleaned=Data_cleaned.drop(columns=["Survived"])

#----------------Start cleaning data
# encode Sex    
    Data_sex= Data_cleaned[["Sex"]]
    ordinal_encoder = OrdinalEncoder()
    train_sex_encoded= ordinal_encoder.fit_transform(Data_sex)
    Data_cleaned["Sex"]= train_sex_encoded

#------------------------------------
# Age
    # - Normalization
    amean=Data_cleaned["Age"].mean()
    astd=Data_cleaned["Age"].std()
    for i in range(len(Data_cleaned["Age"])):
        if np.isnan(Data_cleaned["Age"][i]):
            #print(train_data_cleaned["Age"][i])
            Data_cleaned["Age"][i]=amean
    Data_cleaned["Age"][i]=(Data_cleaned["Age"][i]-amean)/astd
#Ticket
    # remove non number content
    for i in range(len(Data_cleaned["Ticket"])):
        try:
            Data_cleaned["Ticket"][i] = float(Data_cleaned["Ticket"][i].split()[-1])
        except ValueError:
            Data_cleaned["Ticket"][i] = np.nan
    # - Normalization        
    tmean=Data_cleaned["Ticket"].mean()
    tstd=Data_cleaned["Ticket"].std()
    for i in range(len(Data_cleaned["Ticket"])):
        if np.isnan(Data_cleaned["Ticket"][i]):
            #print(train_data_cleaned["Age"][i])
            Data_cleaned["Ticket"][i]=tmean
        Data_cleaned["Ticket"][i]=(Data_cleaned["Ticket"][i]-tmean)/tstd
#Fare
    # - Normalization
    fmean=Data_cleaned["Fare"].mean()
    fstd= Data_cleaned["Fare"].std()
    for i in range(len(Data_cleaned["Fare"])):
        if np.isnan(Data_cleaned["Fare"][i]):
            #print(train_data_cleaned["Age"][i])
            Data_cleaned["Fare"][i]=fmean
        Data_cleaned["Fare"][i]=(Data_cleaned["Fare"][i]-fmean)/fstd
    
#Name
    # - Remove titles
    for i  in range(len(Data_cleaned['Name'])):
        line = Data_cleaned['Name'][i].replace('(','').replace(')','').split()
        #print(f'Index {i} name={data_name[i]} ')
        if line[2] == "Mr." or line[2] == "Mrs." or line[2] == "Miss.":
            Data_cleaned['Name'][i]=line[3]
        else:     
            Data_cleaned['Name'][i]=line[2]
    # - encode name
    encode_name(Data_cleaned['Name'])
    #change type of train_data_cleaned to float
    Data_cleaned.astype("float64")
    # - Normalization
    nmean=Data_cleaned["Name"].mean()
    nstd=Data_cleaned["Name"].std()
    Data_cleaned['Name']=(Data_cleaned['Name']-nmean)/nstd
    return Data_cleaned
    
