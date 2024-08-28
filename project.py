#importing Dependencies
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import streamlit as st

# Loading the data
data = pd.read_csv("Country Quater Wise Visitors.csv")

#CSS
st.markdown(
    """
    <style>
    
    .stApp {
        background-color: #FFC0CB;
    }
   
    </style>
    """,
    unsafe_allow_html=True
)


# Data Preprocessing

# Reshaping the data to have year-wise columns
data_long = pd.melt(data, id_vars=["Country of Nationality"], var_name="Year_Quarter", value_name="Visitors")

data_long[['Year', 'Quarter']] = data_long['Year_Quarter'].str.extract(r'(\d{4})\s(.*)')
data_long['Year'] = data_long['Year'].astype(int)

# Calculating the total annual visitors by country and year
data_annual = data_long.groupby(['Country of Nationality', 'Year'])['Visitors'].sum().reset_index()

data_annual['GrowthRate'] = data_annual.groupby('Country of Nationality')['Visitors'].pct_change()
data_annual['GrowthRate'] = data_annual['GrowthRate'].replace([np.inf, -np.inf], np.nan)  # Replace inf with NaN
data_annual['GrowthRate'].fillna(0, inplace=True)  # Replace NaN with 0

# Droping NaN values 
data_annual = data_annual.dropna()


# One-hot encoding
encoder = OneHotEncoder(sparse_output=False)
country_encoded = encoder.fit_transform(data_annual[['Country of Nationality']].astype(str))

X = np.hstack((country_encoded, data_annual[['Year']]))

# Target variable
y = data_annual['GrowthRate']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Applying Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluating the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = np.mean(np.abs(y_test - y_pred))
rmse = np.sqrt(mse)
r2 = model.score(X_test, y_test)

print(f'Mean Squared Error: {mse}')
# Function to predict the growth rate for a given country and year
def predict_growth_rate(country, year):
    country_encoded = encoder.transform([[country]])[0]
    features = np.hstack((country_encoded, [year]))
    predicted_growth = model.predict([features])[0]
    return predicted_growth



st.title("Growth Rate Predictor")

# Selectbox
country = st.selectbox("Select a country", data_annual['Country of Nationality'].unique())

year = st.slider("Select a year", 2014, 2050)

# Adding button
if st.button("Predict Growth Rate"):
    country_encoded = encoder.transform([[country]])[0]
    features = np.hstack((country_encoded, [year]))
    predicted_growth = model.predict([features])[0] *100
    st.write(f"Predicted growth rate for {country} in {year}: {predicted_growth:.2f}%")
    st.snow()  
    st.success("Prediction successful!")
    st.warning("Remember this is a machine learning model, do not trust it completely.")