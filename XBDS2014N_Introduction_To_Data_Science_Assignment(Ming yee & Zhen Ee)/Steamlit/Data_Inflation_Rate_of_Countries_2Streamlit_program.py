#!/usr/bin/env python
# coding: utf-8
# %%

# ### Import Libraries

# %%


#Import Libraries
import yfinance as yf
import streamlit as st
import numpy as np #Basic operations
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
import pandas as pd #For dataframe manipulations
import matplotlib.pyplot as plt #For data visualization
import seaborn as sns
import plotly.express as px #For plotting graphs
import plotly.io as pio
from sklearn.utils import shuffle
from category_encoders import TargetEncoder, OneHotEncoder
from sklearn.model_selection import GridSearchCV, train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, mean_absolute_error, mean_squared_error


# ### Read the data
df= pd.read_csv("D:/KDU BLENDER LEARNING/Introduction to Data Science/New datasetr/API_FP.CPI.TOTL.ZG_DS2_en_csv_v2_4250827.csv", index_col=0)


st.title("Data Inflation Rate of  Countries")


st.subheader("Data that we used")
df[df.isna().any(axis=1)]






st.subheader("Data Select")
AgGrid(df)





st.subheader("World Map Data Visualization")
data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])

st.map(data)
