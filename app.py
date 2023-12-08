import streamlit as st
import pandas as pd
import numpy as np
st.title('Wine Quality ML web App')
df = pd.read_csv('Processed_Wine_dataset.csv')
df.head()
cols=['quality','Id']
x = df.drop(cols,axis=1)
y = df['quality']
print(x.shape)
print(y.shape)
print(type(x))
print(type(y))
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=22)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

def cls_eval(ytest,ypred):
    cm = confusion_matrix(ytest,ypred)
    print(cm)
    print('Accuracy Score',accuracy_score(ytest,ypred))
    print(classification_report(ytest,ypred,zero_division=0))
    
def mscore(model):
    print('Training Score',model.score(x_train,y_train)) 
    print('Testing Score',model.score(x_test,y_test))    

classifiers = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'RF with entropy': RandomForestClassifier(n_estimators=80,criterion='entropy',min_samples_split=15,max_depth=4),
    'RF with gini': RandomForestClassifier(n_estimators=80,criterion='gini',min_samples_split=15,max_depth=8)
}
model=[]
results = {}
for clf_name, clf in classifiers.items():
    m=clf.fit(x_train, y_train)
    print("Training :",clf_name)
    y_pred = clf.predict(x_test)
    print("Predicting using :",clf_name)
    cm = confusion_matrix(y_test,y_pred)
    print('Confusion Matrix :\n',cm)
    print('\nClassification Report:\n\n',classification_report(y_test,y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    print(clf_name, "accuracy :",accuracy)
    print('Training Score',clf.score(x_train,y_train))  
    print('Testing Score',clf.score(x_test,y_test))        
    results[clf_name] = accuracy
    model.append(m)
    print("")

st.sidebar.header("Select the ML model you want to use")
Drop_options = ["Random Forest with Gini Classifier", "Random Forest with Entropy Classifier","Logistic Regression", "Decision Tree Classifier"]
Model_choice = st.sidebar.selectbox("Drop_options", options=Drop_options)

Fixed_acidity = st.slider("Fixed Acidity", min_value=4.6, max_value=15.9, step=0.1)
st.text(f"Fixed acidity value: {Fixed_acidity}")
volatile_acidity = st.slider("Volatile acidity", min_value=0.12, max_value=1.58, step=0.05)
st.text(f"Volatile acidity value: {volatile_acidity}")
citric_acid = st.slider("Citric Acid", min_value=0.0, max_value=1.0,step=0.1)
st.text(f"citric acid value: {citric_acid}")
residual_sugar = st.slider("Residual Sugar", min_value=0.9, max_value=15.5,step=0.1)
st.text(f"Residual sugar value: {residual_sugar}")
chlorides = st.slider("Chlorides", min_value=0.012, max_value=0.611,step=0.05)
st.text(f"Chlorides value: {chlorides}")
free_sulfur_dioxide = st.slider("Free sulfur dioxide", min_value=1, max_value=68,step=1)
st.text(f"Free sulfur dioxide value: {free_sulfur_dioxide}")
total_sulfur_dioxide = st.slider("Total sulfur dioxide", min_value=6, max_value=289,step=1)
st.text(f"Total sulfur dioxide value: {total_sulfur_dioxide}")
density = st.slider("density", min_value=0.990070, max_value=1.003690,step=0.005)
st.text(f"density value: {density}")
pH = st.slider("pH", min_value=2.74, max_value=4.01,step=0.02)
st.text(f"pH value: {pH}")
sulphates = st.slider("Sulphates", min_value=0.33, max_value=2.0,step=0.05)
st.text(f"Sulphates value: {sulphates}")
alcohol = st.slider("alcohol", min_value=8.4, max_value=14.9,step=0.1)
st.text(f"alcohol value: {alcohol}")
input={'fixed acidity':fixed_acidity,
       'volatile acidity':volatile_acidity,
       'citric acid':citric_acid,
       'Residual sugar':residual_sugar,
       'Chlorides':chlorides,
       'free sulfur dioxide':free_sulfur_dioxide,
       'total sulfur dioxide':total_sulfur_dioxide,
       'Density':density,
       'pH value':pH,
       'Sulphates':sulphates,
       'Alcohol':alcohol}
input_X=pd.DataFrame(input, index=['value'])
st.text("Input value of features:")
input_X.T

if Model_choice=="Random Forest with Gini Classifier":
    ypred=model[3].predict(input_X)
    st.text(f"Category of Wine Quality is: {ypred}")
elif Model_choice=="Random Forest with Entropy Classifier":  
    ypred=model[2].predict(input_X)
    st.text(f"Category of Wine Quality is: {ypred}")
elif Model_choice=="Logistic Regression":
    ypred=model[0].predict(input_X)
    st.text(f"Category of Wine Quality is: {ypred}")
else :
    ypred=model[1].predict(input_X)
    st.text(f"Category of Wine Quality is: {ypred}")
