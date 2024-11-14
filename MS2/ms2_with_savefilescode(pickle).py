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

"""# Reading Dataset"""

data = pd.read_csv('D:\PC1\Youstina\Machine Learning\Final Project\CS_23\MS2\ElecDeviceRatingPrediction_Milestone2.csv')

"""# Label Encoding"""

def Feature_Encoder(X,cols):
    lbl_encoders={}
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(X[c].values))
        X[c] = lbl.transform(list(X[c].values))
        lbl_encoders[c]=lbl
    return lbl_encoders,X

"""# Preprocessing"""

data[['Os','Os_Type']]=data['os'].str.split(' ', expand=True)
data.drop(columns=['os'],inplace=True)
# print(data, "\n")

cols = ('processor_name', 'Os_Type', 'Touchscreen', 'msoffice', 'ram_type','brand','processor_brand','rating')
lbl_encoder,data=Feature_Encoder(data, cols)

with open('MS2_feature_encoder.pkl', 'wb') as file:
    pickle.dump(lbl_encoder, file)
# print(data, "\n")

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
with open('MS2_data_with_mode.pkl', 'wb') as file:
    pickle.dump(modes, file)

columns_to_replace = ['brand', 'processor_brand', 'processor_name', 'processor_gnrtn',
                      'ram_gb', 'ram_type', 'ssd', 'hdd', 'graphic_card_gb', 'weight',
                      'warranty', 'Touchscreen', 'msoffice', 'Price', 'rating',
                      'Number of Ratings', 'Number of Reviews', 'Os', 'Os_Type']

# Replace null values with mode for each column
for col in columns_to_replace:
    mode_value = modes[col]
    df[col].fillna(mode_value, inplace=True)

# print(df, "\n")

one_hot_encoded = pd.get_dummies(df['weight'])
one_hot_encoded = one_hot_encoded.astype(int)

# Save the DataFrame to a pickle file
with open('MS2_one_hot_encoding.pkl', 'wb') as file:
    pickle.dump(one_hot_encoded, file)

df = pd.concat([df, one_hot_encoded], axis=1)
df = df.drop(columns = ['weight'])

# print(df, "\n")

# # # Select numerical columns (you can adjust this based on your DataFrame)

# numerical_cols = df.select_dtypes(include=[np.number])

# # Calculate Z-scores for each numerical column
# z_scores = (numerical_cols - numerical_cols.mean()) / numerical_cols.std()

# # Define threshold for outlier detection (e.g., Z-score > 3 or < -3)
# threshold = 3

# # Find outliers based on Z-scores
# outliers_mask = (z_scores > threshold) | (z_scores < -threshold)

# with open('MS2_detect_outliers.pkl', 'wb') as f:
#     pickle.dump(outliers_mask, f)

# # Replace outliers with mode of their respective columns
# for col in numerical_cols.columns:
#     col_mode = df[col].mode()[0]
#     df.loc[outliers_mask[col], col] = col_mode

x =df.drop(columns=['rating'])
y =df['rating']

# from sklearn.preprocessing import MinMaxScaler

# #Initialize MinMaxScaler
# scaler = MinMaxScaler()

# #Fit scaler to the data and transform the data
# normalized_data = scaler.fit_transform(x)
# column_names = list(x.columns.values)
# # print(column_names)

# normalized_data=pd.DataFrame(normalized_data, columns =column_names)
# with open('MS2_feature_scaling.pkl', 'wb') as f:
#     pickle.dump(normalized_data, f)

# #Get the list of all column names from headers
# column_names = list(x.columns.values)
# # print(column_names)

# normalized_data=pd.DataFrame(normalized_data, columns =column_names)
# normalized_data

"""# Splitting Data"""

# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle = True)

"""# Feature Selection"""

fs = SelectKBest(score_func=f_classif, k=10)
X_train_selected=fs.fit(X_train, y_train)
X_train_fs = fs.transform(X_train)
X_test_fs = fs.transform(X_test)

# Get indices of the top k features
selected_indices = fs.get_support(indices=True)
# Get the names of the top k features
selected_features = x.columns[selected_indices]
xx = x.iloc[:, selected_indices]

with open('MS2_feature_selection.pkl', 'wb') as f:
    pickle.dump(xx, f)

X_train = X_train.iloc[:, selected_indices]
X_test = X_test.iloc[:, selected_indices]

# what are scores for the features
for i in range(len(fs.scores_)):
 print('Feature %s: %f' % (df.columns[i], fs.scores_[i]))
# plot the scores
plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
plt.show()

# Print the names of the top k features
print("Selected Features:")
for feature in selected_features:
     print(feature)

"""# SVM Model"""

C = 0.01 # SVM regularization parameter
linear_svc = svm.SVC(kernel='linear', C=C).fit(X_train_fs, y_train)
predictions = linear_svc.predict(X_train_fs)
accuracy = np.mean(predictions == y_train)
print("***SVM Model***")
print("Accuracy (Linear SVC):", accuracy)

"""# Logistic Regression"""

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
print("***Logisitc Regression Model***")
print("Accuracy: ", accuracy_logreg)
#print("Accuracy: {:.2f}%".format(acc * 100))

# evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_logreg))
print("\nClassification Report:\n", classification_report(y_test, y_pred_logreg))

"""# Random Forest"""

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

with open('MS2_random_forest.pkl', 'wb') as f:
    pickle.dump(rf, f)

y_pred_rf = rf.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("***Random Forest Model***")
print("Accuracy:", accuracy_rf)

# evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))

# Initialize KNN classifier with k=3 (you can adjust this value)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)

# Train the classifier
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("***KNN Model***")
print("Accuracy:", accuracy)

from sklearn.ensemble import VotingClassifier

# Initialize Voting Classifier with soft voting
voting_clf = VotingClassifier(estimators=[('knn', knn), ('rf', rf), ('lr', logreg)], voting='soft')

# Train the Voting Classifier
voting_clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = voting_clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("***Voting Ensemble Model***")
print("Accuracy:", accuracy)

from sklearn.ensemble import StackingClassifier

# Initialize StackingClassifier with Logistic Regression as the meta-model
stacking_model = StackingClassifier(estimators=[('knn', knn), ('rf', rf), ('lr', logreg)], final_estimator=LogisticRegression())

# Train the stacking model
stacking_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = stacking_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("***Stacking Ensemble Model***")
print("Accuracy:", accuracy)