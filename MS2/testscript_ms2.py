from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn import preprocessing, svm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
import pickle

# Write your test file below
data = pd.read_csv('')

data[['Os','Os_Type']]=data['os'].str.split(' ', expand=True)
data.drop(columns=['os'],inplace=True)

# Load the Feature_Encoder function
cols = ('processor_name', 'Os_Type', 'Touchscreen', 'msoffice', 'ram_type','brand','processor_brand','rating')
#df['rating']=df['rating'].fillna('Good Rating', inplace=True)

lbl_encoder = pickle.load(open('MS2_test_Script/MS2_feature_encoder.pkl', 'rb'))

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

df['processor_gnrtn'] = df['processor_gnrtn'].str.replace('Not Available', '0').str.replace('th', '').str.strip()
df['processor_gnrtn'] = df['processor_gnrtn'].astype('int32')

# modes=df.mode().iloc[0]
# print(dict(modes))
modes={'brand': 1, 'processor_brand': 1, 'processor_name': 2, 'processor_gnrtn': 11, 'ram_gb': 8, 'ram_type': 1, 'ssd': 512, 'hdd': 0, 'graphic_card_gb': 0, 'weight': 'Casual', 'warranty': 1, 'Touchscreen': 0, 'msoffice': 0, 'Price': 64990, 'rating': 'Good Rating', 'Number of Ratings': 0, 'Number of Reviews': 0, 'Os': 64, 'Os_Type': 2}

# Save the DataFrame to a pickle file
modes=pickle.load(open('MS2_test_Script/MS2_data_with_mode.pkl', 'rb'))

columns_to_replace = ['brand', 'processor_brand', 'processor_name', 'processor_gnrtn',
                      'ram_gb', 'ram_type', 'ssd', 'hdd', 'graphic_card_gb', 'weight',
                      'warranty', 'Touchscreen', 'msoffice', 'Price', 'rating',
                      'Number of Ratings', 'Number of Reviews', 'Os', 'Os_Type']

# Replace null values with mode for each column
for col in columns_to_replace:
    mode_value = modes[col]
    df[col].fillna(mode_value, inplace=True)

#print(df.isna().sum(), "\n")
#print(df['rating'].isna().sum(), "\n")
one_hot_encoded_loaded = pickle.load(open('MS2_test_Script/MS2_one_hot_encoding.pkl', 'rb'))
df = pd.concat([df, one_hot_encoded_loaded], axis=1)
df = df.drop(columns = ['weight'])
x = pickle.load(open('MS2_test_Script/MS2_feature_selection.pkl', 'rb'))
y =df['rating']# Label

#print(y)
rf = pickle.load(open('MS2_test_Script/MS2_random_forest.pkl', 'rb'))
y_pred_rf = rf.predict(x)
#print(y_pred_rf.isna().sum(), "\n")
#print(y.isna().sum(), "\n")
accuracy_rf = accuracy_score(y, y_pred_rf)
print("Accuracy:", accuracy_rf)
