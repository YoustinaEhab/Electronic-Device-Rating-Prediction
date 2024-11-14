import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, chi2,f_regression
from sklearn.model_selection import train_test_split
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

# Write your test file below
data = pd.read_csv('')

data[['Os','Os_Type']]=data['os'].str.split(' ', expand=True)
data.drop(columns=['os'],inplace=True)

# Load the Feature_Encoder function
cols = ('processor_name', 'Os_Type', 'Touchscreen', 'msoffice', 'ram_type','brand','processor_brand')

lbl_encoder = pickle.load(open('MS1_TestScript_Files/feature_encoder.pkl', 'rb'))

for c in cols:
    data[c] = lbl_encoder[c].transform(list(data[c].values))

df = pd.DataFrame(data)

# Remove the units (GB) and convert to numerical values
df['ram_gb'] = df['ram_gb'].str.replace(' GB', '').astype(int)
df['ssd'] = df['ssd'].str.replace(' GB', '').astype(int)
df['hdd'] = df['hdd'].str.replace(' GB', '').astype(int)
df['graphic_card_gb'] = df['graphic_card_gb'].str.replace(' GB', '').astype(int)

#Remove the bits in Os column
df['Os'] = df['Os'].str.replace('-bit', '').astype(int)

df['warranty'] = df['warranty'].str.replace('No warranty', '0').str.replace('year', '').str.replace('s', '')
df['warranty'] = df['warranty'].astype('int32')
# print(df['warranty'].unique())

df['rating'] = df['rating'].str.replace('star', '').str.replace('s', '')
df['rating'] = df['rating'].astype('int32')
# print(df['rating'].unique())

df['processor_gnrtn'] = df['processor_gnrtn'].str.replace('Not Available', '0').str.replace('th', '').str.strip()
df['processor_gnrtn'] = df['processor_gnrtn'].astype('int32')

modes=pickle.load(open('MS1_TestScript_Files/data_with_mode.pkl', 'rb'))

# List of columns where you want to replace null values with mode
columns_to_replace = ['brand', 'processor_brand', 'processor_name', 'processor_gnrtn',
                      'ram_gb', 'ram_type', 'ssd', 'hdd', 'graphic_card_gb', 'weight',
                      'warranty', 'Touchscreen', 'msoffice', 'Price', 'rating',
                      'Number of Ratings', 'Number of Reviews', 'Os', 'Os_Type']

# Replace null values with mode for each column
for col in columns_to_replace:
    mode_value = modes[col]
    df[col].fillna(mode_value, inplace=True)

one_hot_encoded_loaded = pickle.load(open('MS1_TestScript_Files/one_hot_encoding.pkl', 'rb'))
df = pd.concat([df, one_hot_encoded_loaded], axis=1)
df = df.drop(columns = ['weight'])

# Load the preprocessed DataFrame
x = pickle.load(open('MS1_TestScript_Files/feature_selection.pkl', 'rb'))
y =df['rating']  # Label

# Load the preprocessed DataFrame
selected_features = pickle.load(open('MS1_TestScript_Files/Chi-Square.pkl', 'rb'))
# print("Selected features:", selected_features)

# # Load Linear Regression model
# lr_loaded = pickle.load(open('linear_regression_model.pkl', 'rb'))
# y_pred = lr_loaded.predict(x)

# mse_train = mean_squared_error(y, y_pred)
# print("Mean Square Error on Train Set:", mse_train)

# r2_test = r2_score(y, y_pred)
# print("R2 Score on Test Set:", r2_test)

poly_features = pickle.load(open('MS1_TestScript_Files/polynomial_model.pkl', 'rb'))

X_poly = poly_features.transform(x)

# Load the polynomial regression model
poly_reg = pickle.load(open('MS1_TestScript_Files/polynomial_regression_model.pkl', 'rb'))

# Make predictions
y_prediction = poly_reg.predict(X_poly)
#print(y.isna().sum(),'\n')
#print(y_prediction.isna().sum(),'\n')

test_mse = mean_squared_error(y, y_prediction)
print("Mean Square Error on Test Set:", test_mse)

r2_test = r2_score(y, y_prediction)
print("R2 Score on Test Set:", r2_test)

# # Save Lasso Regression model
# lasso_reg = pickle.load(open('lasso_regression_model.pkl', 'rb'))

# # Make predictions
# y_pred_l = lasso_reg.predict(x)

# # Evaluate the model
# test_mse = mean_squared_error(y, y_pred_l)
# print("Mean Square Error on Test Set:", test_mse)

# r2_test = r2_score(y, y_pred_l)
# print("R2 Score on Test Set:", r2_test)
