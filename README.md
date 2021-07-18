# Kaggle_Titanic_Survival_Prediction

This is a well-known and basic challenge from Kaggle.
<a href="https://www.kaggle.com/c/titanic">Titanic - Machine Learning from Disaster</a>.
All contains are Written by myself
titanic.py => main function
titanic_clean.py => function for cleaning data

# Works
## Data description
<b>survival</b>	Survival	0 = No, 1 = Yes  
<b>pclass</b>	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd  
<b>sex</b>	Sex	  
<b>Age</b>	Age in years  	
<b>sibsp</b>	# of siblings / spouses aboard the Titanic	  
<b>parch</b>	# of parents / children aboard the Titanic	  
<b>ticket</b>	Ticket number	  
<b>fare</b>	Passenger fare	  
<b>cabin</b>	Cabin number	  
<b>embarked</b>  

# Data Engineering
The Feature Engineer I have done here are as follows:

- Create new feature: "family size" from the combination of features "Sibsp" and "parch"
- Remove non number features in "Ticket"  
- Extract name First Name from feature "Name"
- Encode features "Sex" and "Name"
- Normalize all features inside the dataset
 


## Model Used

<b>Randomforest</b> from lib Scikit
<b>GridSearchCV</b> from lib Scikit -> to find best estimator

## result

<b>Score of 0.78468</b>
<b>Ranking of 9015</b>


