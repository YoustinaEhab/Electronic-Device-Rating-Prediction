import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, chi2,f_regression
from sklearn.model_selection import train_test_split
import pickle

def Feature_Encoder(X,cols):
    lbl_encoders={}
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(X[c].values))
        X[c] = lbl.transform(list(X[c].values))
        lbl_encoders[c]=lbl
    return lbl_encoders,X

def featureScaling(X, a, b):
    X = np.array(X)
    scaler = MinMaxScaler(feature_range=(a, b))
    normalized_X = scaler.fit_transform(X)
    return normalized_X

"""#read data

"""

data = pd.read_csv('ElecDeviceRatingPrediction.csv')

"""# Feature Engineering"""

data[['Os','Os_Type']]=data['os'].str.split(' ', expand=True)
data.drop(columns=['os'],inplace=True)
# print(data)

"""#LabelEncoder to convert string to numerical values"""

# LabelEncoder to convert string to numerical values
cols = ('processor_name', 'Os_Type', 'Touchscreen', 'msoffice', 'ram_type','brand','processor_brand')
lbl_encoder,data=Feature_Encoder(data, cols)

with open('feature_encoder.pkl', 'wb') as file:
    pickle.dump(lbl_encoder, file)

# print(data)

"""# Remove any string from data"""

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


modes={'brand': 1, 'processor_brand': 1, 'processor_name': 2, 'processor_gnrtn': 11, 'ram_gb': 8, 'ram_type': 1, 'ssd': 512, 'hdd': 0, 'graphic_card_gb': 0, 'weight': 'Casual', 'warranty': 1, 'Touchscreen': 0, 'msoffice': 0, 'Price': 64990, 'rating': 4, 'Number of Ratings': 0, 'Number of Reviews': 0, 'Os': 64, 'Os_Type': 2}

# Save the DataFrame to a pickle file
with open('data_with_mode.pkl', 'wb') as file:
    pickle.dump(modes, file)

columns_to_replace = ['brand', 'processor_brand', 'processor_name', 'processor_gnrtn',
                      'ram_gb', 'ram_type', 'ssd', 'hdd', 'graphic_card_gb', 'weight',
                      'warranty', 'Touchscreen', 'msoffice', 'Price', 'rating',
                      'Number of Ratings', 'Number of Reviews', 'Os', 'Os_Type']

# Replace null values with mode for each column
for col in columns_to_replace:
    mode_value = modes[col]
    df[col].fillna(mode_value, inplace=True)

"""# Perform one-hot encoding for ( weight ) column"""

one_hot_encoded = pd.get_dummies(df['weight'])
one_hot_encoded = one_hot_encoded.astype(int)

# Save the DataFrame to a pickle file
with open('one_hot_encoding.pkl', 'wb') as file:
    pickle.dump(one_hot_encoded, file)

df = pd.concat([df, one_hot_encoded], axis=1)
df = df.drop(columns = ['weight'])
# print(df)

"""# Feature Scaling"""

# scaled_df = featureScaling(df.values, 0, 1)

# scaled_df = pd.DataFrame(scaled_df, columns=df.columns)
# scaled_df

#from sklearn.preprocessing import MinMaxScaler

# Initialize MinMaxScaler
#scaler = MinMaxScaler()

# Fit scaler to the data and transform the data
#normalized_data = scaler.fit_transform(df)

# Get the list of all column names from headers
#column_names = list(df.columns.values)
#print(column_names)

#normalized_data=pd.DataFrame(normalized_data, columns =column_names)
#normalized_data

"""# Correlation Visualization"""

plt.subplots(figsize=(15, 11))
sns.heatmap(df.corr(), annot=True)
plt.title('Correlation Matrix')
plt.show()

"""# Feature Selection"""

x =df.drop(columns=['rating'])  # Features
y =df['rating']  # Label
corr =df.corr()
#Top 5% Correlation training features with the Value
top_feature = corr.index[abs(corr['rating']) > 0.05]
plt.subplots(figsize=(12, 8))
top_corr = df[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()
top_feature = top_feature.drop('rating')
x = x[top_feature]

with open('feature_selection.pkl', 'wb') as f:
    pickle.dump(x, f)

"""Chi-Square"""

#Chi-Square
# Calculate F-scores for features
selector = SelectKBest(score_func=f_regression, k=10)  # Choose top features
X_new = selector.fit_transform(x, y)

# Get the indices of selected features
selected_indices = selector.get_support(indices=True)
selected_features = x.columns[selected_indices]

# Save the preprocessed DataFrame
with open('Chi-Square.pkl', 'wb') as f:
    pickle.dump(selected_features, f)

print("Selected features:", selected_features)

"""# Splitting data"""

# Split the data into train (60%), test (20%), and validation (20%) sets
X_train, X_temp, y_train, y_temp = train_test_split(x, y, test_size=0.4, random_state=42, shuffle=True)
X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, shuffle=True)

"""# Linear Regression Model"""

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Train the linear regression model on the training set
lr = LinearRegression()
lr.fit(X_train, y_train)

# Save Linear Regression model
with open('linear_regression_model.pkl', 'wb') as file:
    pickle.dump(lr, file)


y_train_pred = lr.predict(X_train)
print("***Linear Regression Model***")
mse_train = mean_squared_error(y_train, y_train_pred)
print("Mean Square Error on Train Set:", mse_train)

# Make predictions on the validation set
y_val_pred = lr.predict(X_validation)
mse_val = mean_squared_error(y_validation, y_val_pred)
print("Mean Square Error on Validation Set:", mse_val)

# Finally, evaluate the model on the test set
y_test_pred = lr.predict(X_test)
mse_test = mean_squared_error(y_test, y_test_pred)
print("Mean Square Error on Test Set:", mse_test)
print()
# Compute R2 score
r2_train = r2_score(y_train, y_train_pred)
r2_val = r2_score(y_validation, y_val_pred)
r2_test = r2_score(y_test, y_test_pred)

print("R2 Score on Train Set:", r2_train)
print("R2 Score on Validation Set:", r2_val)
print("R2 Score on Test Set:", r2_test, "\n")

"""# Linear Regression Plotting"""

# Plot each feature against the rating column
for feature in x.columns:
    plt.figure(figsize=(8, 6))
    plt.scatter(x[feature], y, color='blue')
    plt.xlabel(feature)
    plt.ylabel('Rating')
    plt.title(f'{feature} vs Rating')
    plt.show()

# Plot actual vs predicted values on the validation set
plt.figure(figsize=(8, 6))
plt.scatter(y_validation, y_val_pred, color='blue')
plt.plot([min(y_validation), max(y_validation)], [min(y_validation), max(y_validation)], color='red')  # Perfect fit line
plt.xlabel('Actual Rating')
plt.ylabel('Predicted Rating')
plt.title('Actual vs Predicted Ratings on Validation Set')
# plt.show()

# Plot actual vs predicted values on the test set
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Perfect fit line
plt.xlabel('Actual Rating')
plt.ylabel('Predicted Rating')
plt.title('Actual vs Predicted Ratings on Test Set')
# plt.show()

"""# Polynomial Regression Model"""

from sklearn.preprocessing import PolynomialFeatures

# Polynomial features transformation
poly_features = PolynomialFeatures(degree=2)
poly_features.fit(X_train)

X_train_poly = poly_features.transform(X_train)
X_val_poly = poly_features.transform(X_validation)
X_test_poly = poly_features.transform(X_test)

# Save the polynomial regression model
with open('polynomial_model.pkl', 'wb') as f:
    pickle.dump(poly_features.fit(X_train), f)

# fit the transformed features to Linear Regression
poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)

# Save the polynomial regression model
with open('polynomial_regression_model.pkl', 'wb') as f:
    pickle.dump(poly_reg, f)

# Make predictions
y_train_prediction = poly_reg.predict(X_train_poly)
y_val_prediction = poly_reg.predict(X_val_poly)
y_test_prediction = poly_reg.predict(X_test_poly)

# Evaluate the model
train_mse = mean_squared_error(y_train, y_train_prediction)
val_mse = mean_squared_error(y_validation, y_val_prediction)
test_mse = mean_squared_error(y_test, y_test_prediction)
print("***Polynomial Regression Model***")
print("Mean Square Error on Train Set:", train_mse)
print("Mean Square Error on Validation Set:", val_mse)
print("Mean Square Error on Test Set:", test_mse)
print()

# Compute R2 score
r2_train = r2_score(y_train, y_train_prediction)
r2_val = r2_score(y_validation, y_val_prediction)
r2_test = r2_score(y_test, y_test_prediction)

print("R2 Score on Train Set:", r2_train)
print("R2 Score on Validation Set:", r2_val)
print("R2 Score on Test Set:", r2_test,"\n")

"""# Polynomial Regression Plotting"""

# Plot actual vs predicted values on the validation set
plt.figure(figsize=(8, 6))
plt.scatter(y_validation, y_val_prediction, color='blue')
plt.plot([min(y_validation), max(y_validation)], [min(y_validation), max(y_validation)], color='red')  # Perfect fit line
plt.xlabel('Actual Rating')
plt.ylabel('Predicted Rating')
plt.title('Actual vs Predicted Ratings on Validation Set')
# plt.show()

# Plot actual vs predicted values on the test set
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_prediction, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Perfect fit line
plt.xlabel('Actual Rating')
plt.ylabel('Predicted Rating')
plt.title('Actual vs Predicted Ratings on Test Set')
# plt.show()

"""# Lasso Regression Model"""

from sklearn.linear_model import Lasso

# Fit Lasso regression model
alpha = 0.01  # Regularization strength
lasso_reg = Lasso(alpha=alpha)
lasso_reg.fit(X_train, y_train)

# Save Lasso Regression model
with open('lasso_regression_model.pkl', 'wb') as file:
    pickle.dump(lasso_reg, file)

# Make predictions
y_train_pred = lasso_reg.predict(X_train)
y_val_pred = lasso_reg.predict(X_validation)
y_test_pred = lasso_reg.predict(X_test)

# Evaluate the model
train_mse = mean_squared_error(y_train, y_train_pred)
val_mse = mean_squared_error(y_validation, y_val_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
print("***Lasso Regression Model***")
print("Mean Square Error on Train Set:", train_mse)
print("Mean Square Error on Validation Set:", val_mse)
print("Mean Square Error on Test Set:", test_mse)

print()

# Compute R2 score
r2_train = r2_score(y_train, y_train_pred)
r2_val = r2_score(y_validation, y_val_pred)
r2_test = r2_score(y_test, y_test_pred)

print("R2 Score on Train Set:", r2_train)
print("R2 Score on Validation Set:", r2_val)
print("R2 Score on Test Set:", r2_test,"\n")

"""# Lasso Regression Plotting"""

# Plot actual vs predicted values on the validation set
plt.figure(figsize=(8, 6))
plt.scatter(y_validation, y_val_pred, color='blue')
plt.plot([min(y_validation), max(y_validation)], [min(y_validation), max(y_validation)], color='red')  # Perfect fit line
plt.xlabel('Actual Rating')
plt.ylabel('Predicted Rating')
plt.title('Actual vs Predicted Ratings on Validation Set')
# plt.show()

# Plot actual vs predicted values on the test set
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Perfect fit line
plt.xlabel('Actual Rating')
plt.ylabel('Predicted Rating')
plt.title('Actual vs Predicted Ratings on Test Set')
# plt.show()