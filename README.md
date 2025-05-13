# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas
2. Import Decision tree classifier
3. Fit the data in the model
4. Find the accuracy score

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: S KANUSHA SREE
RegisterNumber:  212224040149
*/
```
```
import pandas as pd
df=pd.read_csv("/content/Employee.csv")
print("data.head():")
df.head()
```
```
print("data.info()")
df.info()
```
```
print("data.isnull().sum()")
df.isnull().sum()
```
```
print("data value counts")
df["left"].value_counts()
```
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
print("data.head() for Salary:")
df["salary"]=le.fit_transform(df["salary"])
df.head()
```
```
print("x.head():")
x=df[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
```
```
y=df["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
y_pred
```
```
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
```
```
print("Data prediction")
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
```
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plot_tree(dt,filled=True,feature_names=x.columns,class_names=['salary' , 'left'])
plt.show()
```
## Output:
![image](https://github.com/user-attachments/assets/d01a6a6d-dfb5-4a3f-b57f-bc18a0a01a1b)

![image](https://github.com/user-attachments/assets/cdf7d654-7050-4cf4-bcfc-0217ed2d3785)

![image](https://github.com/user-attachments/assets/502b2a6c-8faf-4ad5-be19-f89a97c5cd5e)

![image](https://github.com/user-attachments/assets/6af080ac-c27e-464f-a26b-977ef4b94210)

![image](https://github.com/user-attachments/assets/680e5e5e-79ea-4a84-b538-0f7a61e0849c)

![image](https://github.com/user-attachments/assets/d30c36bb-2457-4ddf-8ce0-c80dda3d8c0d)

![image](https://github.com/user-attachments/assets/40033eb1-f44f-4f84-8fe5-eaf64ec4b049)

![image](https://github.com/user-attachments/assets/b1010d0b-8d30-4e89-bd22-9ea409ee3d9b)

![image](https://github.com/user-attachments/assets/23016ff5-6247-42a9-82f1-8bcc640a53d9)

![image](https://github.com/user-attachments/assets/88835ccd-e81c-42d0-851d-acd86a8a37df)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
