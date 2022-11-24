from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle

df =pd.read_csv('Iris.csv')

print(df.head())

#Dependent and Independent variables

X= df[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]
y=df['Species']

X_train,X_test, y_train, y_test = train_test_split(X,y,random_state=42,test_size=0.3)

#feature scaling
sc= StandardScaler()
X_train =sc.fit_transform(X_train)
X_test =sc.transform(X_test)

classifier= RandomForestClassifier()

classifier.fit(X_train,y_train)

#pickle file
pickle.dump(classifier,open("model.pkl","wb"))