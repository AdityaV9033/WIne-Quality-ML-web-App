import streamlit as st
import pandas as pd
import numpy as np
st.title('Wine Quality ML web App')
df = pd.read_csv('Processed_Wine_dataset.csv')
df.head()

st.sidebar.header("Select the ML model you want to use")
Drop_options = ["Random Forest Regression", "Bagging with Decision Tree Regressor", "KNN Regression", "Bagging with Linear Regressor"]
Model_choice = st.sidebar.selectbox("Drop_options", options=Drop_options)

Year = st.slider("Model Purchase Year", min_value=1992, max_value=2020, step=1)
st.text(f"Year value: {Year}")
Km_Driven = st.slider("km driven", min_value=1, max_value=806599, step=1)
st.text(f"Km Driven value: {Km_Driven}")
Fuel = st.slider("Fuel type", min_value=0, max_value=5,step=1)
st.text(f"Fuel type value: {Fuel}")
Seller_Type = st.slider("Seller Type", min_value=0, max_value=2,step=1)
st.text(f"Seller_Type value: {Seller_Type}")
Transmission = st.slider("Transmission", min_value=0, max_value=1,step=1)
st.text(f"Transmission value: {Transmission}")
Owner = st.slider("Owner", min_value=0, max_value=4,step=1)
st.text(f"Owner value: {Owner}")
Model_Name = st.slider("Model_Name", min_value=0, max_value=28,step=1)
st.text(f"Model_Name value: {Model_Name}")
input={'year':Year,
       'km_driven':Km_Driven,
       'fuel':Fuel,
       'seller_type':Seller_Type,
       'transmission':Transmission,
       'owner':Owner,
       'Model_Name':Model_Name}
input_X=pd.DataFrame(input, index=['value'])
st.text("Input value of features:")
input_X.T
scaled_data={}
for i_name, i in input.items():
    mean_value = X[i_name].mean()
    std_value = X[i_name].std()
    S_data = (i - mean_value) / std_value
    scaled_data[i_name]=S_data

X1_scaled=pd.DataFrame(scaled_data, index=['value'])
if Model_choice=="Random Forest Regression":
    ypred=model[5].predict(X1_scaled)
    st.text(f"Selling price of Car is: {ypred}")
elif Model_choice=="Bagging with Decision Tree Regressor":  
    ypred=model[6].predict(X1_scaled)
    st.text(f"Selling price of Car is: {ypred}")
elif Model_choice=="Bagging with Linear Regressor":
    ypred=model[7].predict(X1_scaled)
    st.text(f"Selling price of Car is: {ypred}")
else :
    ypred=model[3].predict(X1_scaled)
    st.text(f"Selling price of Car is: {ypred}")
