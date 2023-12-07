import streamlit as st
import pandas as pd
import numpy as np
st.title('Wine Quality ML web App')
df = pd.read_csv('Processed_Wine_dataset.csv')
df.head()
