#Importing the libraries
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

#Fetching the diabetes dataset from excel to pandas dataframe
diabetes_dataset = pd.read_csv('./dataSet/diabetes.csv')

#View the dataset in dataframe
# print(diabetes_dataset.head(5))

#Getting the statistical measures of the dataset
# print(diabetes_dataset.describe())

#Number of rows and columns in the dataset
# print(diabetes_dataset.shape)

#Categories the outcome types
# print(diabetes_dataset['Outcome'].value_counts())

#Mean for outcomes
# print(diabetes_dataset.groupby('Outcome').mean())

#Features
X = diabetes_dataset.drop('Outcome',axis=1)
Y = diabetes_dataset['Outcome']

#Standardization the data as the values are in different ranges as per respective features
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

#Train Test split
x_train,x_test,y_train,y_test = train_test_split(X_scaled,Y,random_state=42,test_size=0.2,stratify=Y)

#Model Training
svm = SVC(kernel='rbf')
svm.fit(x_train, y_train)

#Model Prediction
y_pred = svm.predict(x_test)
y_train_pred = svm.predict(x_train)

#Model Evaluation
print(f'Accuracy Score of the train dataset {accuracy_score(y_train_pred,y_train)}')
print(f'Accuracy Score of the test dataset {accuracy_score(y_pred,y_test)}')

#Making a predictive system
def predictionResponse(prediction):
    if prediction[0] == 0:
        print('The person is not diabetic')
    elif prediction[0] == 1:
        print('The person is diabetic')
    else:
        print('No Prediction')
        
input_data = (6,148,72,35,0,33.6,0.627,50)
input_arr = np.asarray(input_data)
input_arr = input_arr.reshape(1,-1)
input_arr = sc.transform(input_arr)

prediction = svm.predict(input_arr)
predictionResponse(prediction)

input_data = (1,85,66,29,0,26.6,0.351,31)
input_arr = np.asarray(input_data)
input_arr = input_arr.reshape(1,-1)
input_arr = sc.transform(input_arr)

prediction = svm.predict(input_arr)
predictionResponse(prediction)