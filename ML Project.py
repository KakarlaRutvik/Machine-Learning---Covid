#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

data = pd.read_csv('covid_data.csv')

data.dropna(inplace=True)
label_encoder = LabelEncoder()
data['sex'] = label_encoder.fit_transform(data['sex'])
data['age'] = label_encoder.fit_transform(data['age'])
data['vaccine'] = label_encoder.fit_transform(data['vaccine'])
data['health'] = label_encoder.fit_transform(data['health'])
data['status'] = label_encoder.fit_transform(data['status'])

X = data.drop('status', axis=1)
y = data['status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(20,10))
plot_tree(clf, filled=True, feature_names=list(X.columns), class_names=['Alive', 'Dead'])
plt.show()


# In[7]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('Covid_data.csv')
data.dropna(inplace=True)
label_encoder = LabelEncoder()
data['sex'] = label_encoder.fit_transform(data['sex'])
data['age'] = label_encoder.fit_transform(data['age'])
data['vaccine'] = label_encoder.fit_transform(data['vaccine'])
data['health'] = label_encoder.fit_transform(data['health'])
data['status'] = label_encoder.fit_transform(data['status'])

X = data.drop('status', axis=1)
y = data['status']

clf = DecisionTreeClassifier()
clf.fit(X, y)

sex_mapping = {'Male': 0, 'Female': 1}
vaccine_mapping = {'Covaxin': 0, 'Covishield': 1}
health_mapping = {'None': 0, 'Asthma': 1, 'Diabetes': 2, 'Hypertension': 3, 'Heart Disease': 4}

#Note:- Please enter the input in the same format of given option. Or esle it will end up showing invalid input. 

sex = input("Enter sex (Male/Female): ")
age = int(input("Enter age: "))
vaccine = input("Enter vaccine (Covaxin/Covishield): ")
health = input("Enter health condition (None/Asthma/Diabetes/Hypertension/Heart Disease): ")

sex_encoded = sex_mapping.get(sex, -1)
vaccine_encoded = vaccine_mapping.get(vaccine, -1)
health_encoded = health_mapping.get(health, -1)

if sex_encoded == -1 or vaccine_encoded == -1 or health_encoded == -1:
    print("Invalid input!")
else:
    predicted_status = clf.predict([[sex_encoded, age, vaccine_encoded, health_encoded]])
    if predicted_status == 0:
        print("Predicted status: Alive")
    else:
        print("Predicted status: Dead")


# In[8]:


data


# In[ ]:




