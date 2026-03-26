#Importing the libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

#Data Collection and Data Processing

#Loading the dataset to pandas dataframe
sonar_data = pd.read_csv('./dataset/sonarData.csv', header=None)

#Number of rows and columns
# print(sonar_data.shape)

#Describe the data --> Statistical Measures of the data
# print(sonar_data.describe()) 

#Categories of data we have as per the data to check the imbalance
# print(sonar_data[60].value_counts())

#Get mean for each column to see the values 
# print(sonar_data.groupby(60).mean())

#Seperating data and labels
X = sonar_data.drop(columns=[60])
Y = sonar_data[60]

#Training and Test Data
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42,stratify=Y)

#Model Training
lr = LogisticRegression()
lr.fit(x_train, y_train)

#Model Evaluation
pred_test = lr.predict(x_test)
pred_accuracy_score = accuracy_score(pred_test, y_test)
print(f'Logistic Regression Accuracy Score : {pred_accuracy_score}')

#SVM Model Training
svm = SVC()
sc = StandardScaler()
x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.transform(x_test)
svm.fit(x_train_scaled,y_train)

pred_test = svm.predict(x_test_scaled)
pred_accuracy_score = accuracy_score(pred_test, y_test)
print(f'SVM Accuracy Score: {pred_accuracy_score}')

#Making the Predictive System for RockvsMine
input_data = (0.0181,0.0146,0.0026,0.0141,0.0421,0.0473,0.0361,0.0741,0.1398,0.1045,0.0904,0.0671,0.0997,0.1056,0.0346,0.1231,0.1626,0.3652,0.3262,0.2995,0.2109,0.2104,0.2085,0.2282,0.0747,0.1969,0.4086,0.6385,0.7970,0.7508,0.5517,0.2214,0.4672,0.4479,0.2297,0.3235,0.4480,0.5581,0.6520,0.5354,0.2478,0.2268,0.1788,0.0898,0.0536,0.0374,0.0990,0.0956,0.0317,0.0142,0.0076,0.0223,0.0255,0.0145,0.0233,0.0041,0.0018,0.0048,0.0089,0.0085)

#Changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

#Reshape the numpy array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction_lr = lr.predict(input_data_reshaped)

input_M = (0.0283,0.0599,0.0656,0.0229,0.0839,0.1673,0.1154,0.1098,0.1370,0.1767,0.1995,0.2869,0.3275,0.3769,0.4169,0.5036,0.6180,0.8025,0.9333,0.9399,0.9275,0.9450,0.8328,0.7773,0.7007,0.6154,0.5810,0.4454,0.3707,0.2891,0.2185,0.1711,0.3578,0.3947,0.2867,0.2401,0.3619,0.3314,0.3763,0.4767,0.4059,0.3661,0.2320,0.1450,0.1017,0.1111,0.0655,0.0271,0.0244,0.0179,0.0109,0.0147,0.0170,0.0158,0.0046,0.0073,0.0054,0.0033,0.0045,0.0079)
input_M_numpy_arr = np.asarray(input_M)
input_M_reshaped = input_M_numpy_arr.reshape(1,-1)

prediction_svm = svm.predict(input_M_reshaped)

# print(f'Prediction of L.R. : {prediction_lr}')
# print(f'Prediction of SVM : {prediction_svm}')

def returnPrediction(prediction):
    if(prediction[0]=='R'):
        print('Rock')
    elif(prediction[0]=='M'):
        print('Mine')
    else:
        print('No Prediction')

returnPrediction(prediction_lr)
returnPrediction(prediction_svm)