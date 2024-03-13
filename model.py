#from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle


df=pd.read_csv('Churn_data.csv')
#df.rename(columns={"Exited":"Churned"},inplace=True)
#df['Churned'].replace({0:'No',1:'Yes'},inplace=True)


from sklearn.model_selection import train_test_split
x=df.drop(columns=['Churned'])
y=df['Churned']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report


clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(x_train,y_train)
clf.predict(x_test)
pickle.dump(clf,open('model.pklz','wb'))